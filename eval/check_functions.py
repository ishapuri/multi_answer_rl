from eval.eval_utils import compute_pass_n, get_brier, get_ece, get_auroc, exact_match_score
import numpy as np
from math_verify import verify, parse
import re
from vllm import LLM, SamplingParams
import gc
from transformers import AutoTokenizer

def confidence_extractor(response, **kwargs):
    """Extracts the confidence from the completions
    If a float is found within confidence tags, it is processed as follows:
    If the float is between 0 and 1, it is returned as is.
    If the float is between 1 and 100, it is divided by 100 and returned.
    If float is not directly found, the first number in the string is extracted and processed as above.
    If no float is found, 0 is returned.    
    """
    conf_pattern = r"<confidence>(.*?)</confidence>"
    # Get all <confidence>...</confidence> occurrences
    conf_matches = re.findall(conf_pattern, response, re.DOTALL | re.MULTILINE)
    # Get the last confidence, if exists
    last_confidence = conf_matches[-1] if conf_matches else ""
    if last_confidence == "":
        return 0, 0.0
    else:
        try:
            confidence = float(last_confidence)
            if confidence > 1 and confidence <= 100:
                return 1, confidence/100
            elif confidence >= 0 and confidence <= 1:
                return 1, confidence
            else:
                return 0, 0.0
        except:
            # extract the first number in the string
            first_number = re.search(r'-?\d+(?:\.\d+)?', last_confidence)
            if first_number:
                first_number = float(first_number.group())
                if first_number >= 0 and first_number <= 1:
                    return 1, first_number
                elif first_number > 1 and first_number <= 100:
                    return 1, first_number/100
                else:
                    return 0, 0.0
            else:
                return 0, 0.0


def gen_correctness_reward(completions, answer, **kwargs):
    """Reward function that checks if the answer is correct or not
    The answer must be present within the answer tags.
    For math datasets, the correctness is checked using huggingface math-verify.
    For factual datasets, the correctness is checked using exact match.
    Supports both single answer and multiple answers (list).
    Returns a tuple: (matches, which_answers) where:
    - matches: list of correctness labels (0 or 1)
    - which_answers: list of indices (-1 if incorrect, otherwise index of matched gold answer)

    """
    ans_pattern = r"<answer>(.*?)</answer>"
    completion_contents = [completion[0]["content"]
                           for completion in completions]
    eval_contents = [e for e in answer]
    matches = []
    which_answers = []

    for content, e in zip(completion_contents, eval_contents):
        # Get all <answer>...</answer> occurrences
        ans_matches = re.findall(ans_pattern, content,
                                 re.DOTALL | re.MULTILINE)
        # Get the last answer, if exists
        last_answer = ans_matches[-1] if ans_matches else ""
        
        # Handle multiple correct answers
        if isinstance(e, (list, tuple)):
            gold_list = e
        else:
            gold_list = [e]
        
        label = 0
        matched_idx = -1
        for idx, gold in enumerate(gold_list):
            attempt = parse(last_answer)
            label = verify(gold, attempt)
            if label == 0:
                label = exact_match_score(last_answer, gold)
            if label:  # If any gold answer matches, break
                matched_idx = idx
                break
        matches.append(float(label))
        which_answers.append(matched_idx)

    return matches, which_answers




# --- NEW: minimal helpers for multi-answer parsing/scoring ---

_MULTI_ANS_RE = re.compile(r"<answer(\d+)>(.*?)</answer\1>", re.DOTALL | re.MULTILINE)
_MULTI_CONF_RE = re.compile(r"<confidence(\d+)>(.*?)</confidence\1>", re.DOTALL | re.MULTILINE)

def _safe_float_01(x):
    try:
        v = float(x)
        if 0 <= v <= 1:
            return v
        if 1 < v <= 100:
            return v / 100.0
    except Exception:
        pass
    # fallback: first number in string
    m = re.search(r"-?\d+(?:\.\d+)?", str(x))
    if not m:
        return None
    v = float(m.group())
    if 0 <= v <= 1:
        return v
    if 1 < v <= 100:
        return v / 100.0
    return None

def _extract_multi_answers_and_confidences(text):
    """Returns (answers_by_index, confidences_by_index)
       where each is a dict {int_index: value} with 1-based indices.
    """
    answers = {int(i): a.strip() for i, a in _MULTI_ANS_RE.findall(text)}
    confs_raw = {int(i): c.strip() for i, c in _MULTI_CONF_RE.findall(text)}
    confs = {i: _safe_float_01(c) for i, c in confs_raw.items()}
    return answers, confs

def _candidate_correct_labels(candidates, ground_truth):
    """Compute y_i for each candidate using your current math/fact checker.
    Supports both single ground_truth and multiple ground_truth (list)."""
    # Handle multiple correct answers
    if isinstance(ground_truth, (list, tuple)):
        gold_list = ground_truth
    else:
        gold_list = [ground_truth]
    
    ys = []
    for ans in candidates:
        label = 0
        for gold in gold_list:
            attempt = parse(ans)
            lab = verify(gold, attempt)
            if lab == 0:
                lab = exact_match_score(ans, gold)
            if lab:  # If any gold answer matches, break
                label = lab
                break
        ys.append(1.0 if label else 0.0)
    return ys

def _candidate_correct_labels_with_indices(candidates, ground_truth):
    """Compute y_i and matched indices for each candidate.
    Returns (ys, matched_indices) where:
    - ys: list of correctness labels (0.0 or 1.0)
    - matched_indices: list of gold answer indices (-1 if incorrect, otherwise index of matched gold answer)
    """
    # Handle multiple correct answers
    if isinstance(ground_truth, (list, tuple)):
        gold_list = ground_truth
    else:
        gold_list = [ground_truth]
    
    ys = []
    matched_indices = []
    for ans in candidates:
        label = 0
        matched_idx = -1
        for idx, gold in enumerate(gold_list):
            attempt = parse(ans)
            lab = verify(gold, attempt)
            if lab == 0:
                lab = exact_match_score(ans, gold)
            if lab:  # If any gold answer matches, break
                label = lab
                matched_idx = idx
                break
        ys.append(1.0 if label else 0.0)
        matched_indices.append(matched_idx)
    return ys, matched_indices








def confidence_verifier(local_dataset, config, format_fn="confidence_format", format_pattern="tabc", **kwargs):
    label_dict = {f"{config.name}-evals": []}
    evals = []
    which_answers = []
    c_lengths = []
    confidence_levels = []
    conf_format_levels = []
    metrics = {}
    n = config.n
    correctness_fn = gen_correctness_reward

    print("format_pattern: ", format_pattern)

    if f"{config.name}-class_output" in local_dataset.column_names:
        #If classification outputs are present (these come from classifier/probe)
        class_outputs = local_dataset[f"{config.name}-class_output"]
    else:
        class_outputs = None

    # Check if multi-answer format early
    is_multi_rlcr  = (str(format_pattern).lower() == "multi_answer_short")
    is_multi_rlvr  = (str(format_pattern).lower() == "multi_answer_rlvr_short")
    do_multi_block = (is_multi_rlcr or is_multi_rlvr)

    ### CHECK CORRECTNESS ###

    for i in range(len(local_dataset)):
        eval_list, which_ans_list, c_len_list, conf_list, conf_format_list = [], [], [], [], []
        # Support both "answer" and "answers" columns
        if "answers" in local_dataset[i]:
            answer = local_dataset[i]["answers"]
        else:
            answer = local_dataset[i]["answer"]
        
        if do_multi_block:
            # For multi-answer formats, only process the first completion (output_0)
            # and track each candidate separately to match candidate_correctness structure
            j = 0
            pred_response = local_dataset[i][f"{config.name}-output_{j}"]
            conf_format, conf_level = confidence_extractor(pred_response)
            conf_format_list.append(conf_format)
            c_len_list.append(len(pred_response))
            conf_list.append(conf_level)
            
            answers_by_idx, confs_by_idx = _extract_multi_answers_and_confidences(pred_response)
            
            if not answers_by_idx:
                # No multi tags; treat as all incorrect
                # We still need to match the structure, but with empty/zero values
                eval_list = []
                which_ans_list = []
            else:
                # Build ordered candidate list
                max_i = max(answers_by_idx.keys())
                candidates = [answers_by_idx.get(k, "").strip() for k in range(1, max_i + 1)]
                
                # Check correctness and get matched indices for each candidate
                ys, matched_indices = _candidate_correct_labels_with_indices(candidates, answer)
                
                # Track each candidate separately (matching candidate_correctness structure)
                for y, midx in zip(ys, matched_indices):
                    eval_list.append(1 if y == 1.0 else 0)
                    which_ans_list.append(midx)
        else:
            # For single-answer formats, process all n completions
            for j in range(n):
                pred_response = local_dataset[i][f"{config.name}-output_{j}"]
                conf_format, conf_level = confidence_extractor(pred_response)
                conf_format_list.append(conf_format)
                c_len_list.append(len(pred_response))
                conf_list.append(conf_level)
                
                # For single-answer formats, use the existing logic
                pred = [{"role": "assistant", "content": pred_response}]
                args = {"completions": [pred], "answer": [answer]}
                actual_correctness, matched_idx = correctness_fn(**args)
                actual_correctness = actual_correctness[0]
                matched_idx = matched_idx[0]
                
                if actual_correctness == 1:
                    eval_list.append(1)
                else:
                    eval_list.append(0)
                which_ans_list.append(matched_idx)
            else:
                # For single-answer formats, use the existing logic
                pred = [{"role": "assistant", "content": pred_response}]
                args = {"completions": [pred], "answer": [answer]}
                actual_correctness, matched_idx = correctness_fn(**args)
                actual_correctness = actual_correctness[0]
                matched_idx = matched_idx[0]
                
                if actual_correctness == 1:
                    eval_list.append(1)
                else:
                    eval_list.append(0)
                which_ans_list.append(matched_idx)

        evals.append(eval_list)
        which_answers.append(which_ans_list)
        c_lengths.append(c_len_list)
        confidence_levels.append(conf_list)
        conf_format_levels.append(conf_format_list)

    ### END OF CHECK CORRECTNESS ###
     

    # --- NEW: Multi-answer metrics (minimal & isolated) ---

    multi_pass_at_1, multi_pass_at_2, multi_pass_at_3 = [], [], []
    per_example_set_brier = []  # only used for RLCR
    multi_candidate_correctness = []  # Store correctness for each candidate answer

    if do_multi_block:
        # We compute per-example from the FIRST completion (index 0) of each item.
        # (This mirrors typical n=1 inference; keeps changes minimal and deterministic.)
        for i in range(len(local_dataset)):
            pred_response = local_dataset[i][f"{config.name}-output_0"]
            # Support both "answer" and "answers" columns
            if "answers" in local_dataset[i]:
                gt_list = local_dataset[i]["answers"]
            else:
                gt_list = local_dataset[i]["answer"]
            # Pass full gt_list to support multiple ground truth answers for ambiguous datasets
            gt = gt_list

            answers_by_idx, confs_by_idx = _extract_multi_answers_and_confidences(pred_response)

            if not answers_by_idx:
                # No multi tags; treat as all incorrect / zero-confs
                multi_pass_at_1.append(0.0); multi_pass_at_2.append(0.0); multi_pass_at_3.append(0.0)
                if is_multi_rlcr: per_example_set_brier.append(0.0)
                multi_candidate_correctness.append([])  # Empty list for no candidates
                continue

            # Build ordered candidate list
            max_i = max(answers_by_idx.keys())
            # answers in numbered order (1..K)
            candidates = [answers_by_idx.get(k, "").strip() for k in range(1, max_i + 1)]

            # Labels - _candidate_correct_labels handles both single and multiple ground truth answers
            ys = _candidate_correct_labels(candidates, gt)
            
            # Store correctness labels for each candidate (in original order 1..K)
            multi_candidate_correctness.append([float(y) for y in ys])

            # Ordering: RLCR rank by confidence desc; RLVR keep numeric order
            order = list(range(len(candidates)))
            if is_multi_rlcr:
                confs = [confs_by_idx.get(k+1, None) for k in range(len(candidates))]
                # Use -inf for missing confidence so ranked last
                confs_for_sort = [(-c if c is not None else float("inf")) for c in confs]
                order = sorted(range(len(candidates)), key=lambda k: confs_for_sort[k])

            # Compute pass@k on ordered candidates
            def _pass_at_k(k):
                idxs = order[:min(k, len(order))]
                return 1.0 if any(ys[t] == 1.0 for t in idxs) else 0.0

            multi_pass_at_1.append(_pass_at_k(1))
            multi_pass_at_2.append(_pass_at_k(2))
            multi_pass_at_3.append(_pass_at_k(3))

            # Set Brier (RLCR only): sum_i (p_i - y_i)^2 over all candidates
            if is_multi_rlcr:
                ps = []
                for k in range(len(candidates)):
                    p = confs_by_idx.get(k+1, None)
                    ps.append(0.0 if p is None else p)
                set_brier_sum = float(np.sum((np.array(ps) - np.array(ys)) ** 2))
                per_example_set_brier.append(set_brier_sum)

        # Aggregate new metrics
        metrics["pass@1"] = float(np.mean(multi_pass_at_1)) if multi_pass_at_1 else metrics.get("pass@1", 0.0)
        metrics["pass@2"] = float(np.mean(multi_pass_at_2)) if multi_pass_at_2 else metrics.get("pass@2", 0.0)
        metrics["pass@3"] = float(np.mean(multi_pass_at_3)) if multi_pass_at_3 else metrics.get("pass@3", 0.0)
        metrics["accuracy"] = metrics["pass@1"]

        if is_multi_rlcr:
            # Average of per-example sums (scalar like other metrics)
            metrics["set_brier"] = float(np.mean(per_example_set_brier)) if per_example_set_brier else 0.0
        
        # Add candidate correctness to label_dict for dataset storage
        label_dict[f"{config.name}-candidate_correctness"] = multi_candidate_correctness
    # --- END NEW multi-answer block ---


    ### COMPUTE PASS@K ###
    # Skip traditional pass@k computation for multi-answer formats since we've already computed it above
    if not do_multi_block:
        if n not in config.pass_k_vals:
            config.pass_k_vals.append(n)
        if 1 not in config.pass_k_vals:
            config.pass_k_vals.append(1)
        for k in config.pass_k_vals:
            if k <= n:
                pass_k = compute_pass_n(evals, k)
                metrics[f"pass@{k}"] = pass_k

    ### END OF COMPUTE PASS@K ###

    if class_outputs is not None:
        #If classification outputs are present (these come from classifier/probe), then we use the corresponding confidence levels
        if type(class_outputs[0]) == list:
            confidence_levels = [[c[1]] for c in class_outputs]
            print("Overriding confidence levels with classification outputs")
        else:
            confidence_levels = [ [c] for c in class_outputs]

    # take mean of c_lengths
    c_length_mean = np.mean(np.array(c_lengths))

    label_dict[f"{config.name}-evals"] = evals
    label_dict[f"{config.name}-which_answers"] = which_answers
    label_dict[f"{config.name}-c_lengths"] = c_lengths
    label_dict[f"{config.name}-confidence_levels"] = confidence_levels
    label_dict[f"{config.name}-conf_format_adherence"] = conf_format_levels

    # Flatten evals and confidence_levels for metrics computation
    # For multi-answer formats, lists can have different lengths, so flatten manually
    if do_multi_block:
        correctness_list = []
        expanded_conf_levels = []
        for i in range(len(evals)):
            eval_list = evals[i]
            conf_list = confidence_levels[i] if i < len(confidence_levels) else []
            num_candidates = len(eval_list)
            
            # Add correctness values
            correctness_list.extend(eval_list)
            
            # Expand confidence to match number of candidates
            # Only add confidence if we have candidates (to keep lengths matching)
            if num_candidates > 0:
                if len(conf_list) > 0:
                    # Repeat the confidence for each candidate
                    expanded_conf_levels.extend([conf_list[0]] * num_candidates)
                else:
                    # No confidence available, use 0.0 for each candidate
                    expanded_conf_levels.extend([0.0] * num_candidates)
            # If num_candidates == 0, we skip adding confidence (to match correctness)
        
        # Ensure arrays have the same length before creating numpy arrays
        min_len = min(len(correctness_list), len(expanded_conf_levels))
        if len(correctness_list) != len(expanded_conf_levels):
            print(f"Warning: Mismatch in array lengths: correctness={len(correctness_list)}, confidence={len(expanded_conf_levels)}. Truncating to {min_len}.")
            correctness_list = correctness_list[:min_len]
            expanded_conf_levels = expanded_conf_levels[:min_len]
        
        correctness_array = np.array(correctness_list) if len(correctness_list) > 0 else np.array([], dtype=float)
        confidence_array = np.array(expanded_conf_levels) if len(expanded_conf_levels) > 0 else np.array([], dtype=float)
        
        # Final safety check
        assert len(correctness_array) == len(confidence_array), f"Array length mismatch: correctness={len(correctness_array)}, confidence={len(confidence_array)}"
    else:
        correctness_array = np.array(evals).flatten()
        confidence_array = np.array(confidence_levels).flatten()
    if not do_multi_block: 
        metrics["brier_score"] = get_brier(correctness_array, confidence_array) 
    metrics["ece"] = get_ece(correctness_array, confidence_array)
    metrics["auroc"] = get_auroc(correctness_array, confidence_array)

    if not do_multi_block:
        metrics["accuracy"] = metrics["pass@1"]
    metrics["completion length"] = c_length_mean
    # For confidence level, use the already-computed confidence_array for multi-answer formats
    if do_multi_block:
        metrics["confidence level"] = np.mean(confidence_array) if len(confidence_array) > 0 else 0.0
    else:
        metrics["confidence level"] = np.mean(np.array(confidence_levels))
    metrics["confidence format adherence"] = np.mean(
        np.array(conf_format_levels))

    print("\n\n___________________________")
    print(f"Metrics of {config.name} =")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    return label_dict, metrics




def llm_confidence_verifier(
    local_dataset,
    config,
    judge_model="meta-llama/Llama-3.1-8B-Instruct",
    format_fn="confidence_format",
    format_pattern="tabc",
    **kwargs
):
    label_dict = {f"{config.name}-evals": []}
    evals = []
    which_answers = []
    c_lengths = []
    confidence_levels = []
    conf_format_levels = []
    metrics = {}
    n = config.n

    # --- Multi-answer flags (mirror confidence_verifier) ---
    is_multi_rlcr  = (str(format_pattern).lower() == "multi_answer_short")
    is_multi_rlvr  = (str(format_pattern).lower() == "multi_answer_rlvr_short")
    do_multi_block = (is_multi_rlcr or is_multi_rlvr)

    if f"{config.name}-class_output" in local_dataset.column_names:
        class_outputs = local_dataset[f"{config.name}-class_output"]
    else:
        class_outputs = None

    # FIRST EXTRACT OUT ALL ANSWERS FROM THE MODEL OUTPUTS (single-answer style).
    extracted_answers = []
    for i in range(len(local_dataset)):
        q_spec_ans = []
        for j in range(n):
            pred = local_dataset[i][f"{config.name}-output_{j}"]
            ans_pattern = r"<answer>(.*?)</answer>"
            # Get all <answer>...</answer> occurrences
            ans_matches = re.findall(
                ans_pattern, pred, re.DOTALL | re.MULTILINE
            )
            # Get the last answer, if exists
            last_answer = ans_matches[-1] if ans_matches else ""
            if last_answer == "":
                last_answer = "I don't know"
            q_spec_ans.append(last_answer)
        extracted_answers.append(q_spec_ans)

    ####### DO LLM AS JUDGE SETUP #######
    sys_prompt = """
    You are a judge that will be given a question, ground truth answers and a model generated answer. There might be multiple ground truth answers. 
    The model generated answer is correct if it matches any of the ground truth answers.
    You will need to determine if the model generated answer is correct or not. 
    Your response should be a single word: 'YES' if the answer is correct and 'NO' if it is not.
    """

    prompts = []
    chosen_key = "question" if "question" in local_dataset.column_names else "problem"
    tokenizer = AutoTokenizer.from_pretrained(judge_model, trust_remote_code=True)
    
    # Generate prompts for each completion (as before)
    for i in range(len(local_dataset)):
        for j in range(n):
            # Support both "answer" and "answers" columns
            if "answers" in local_dataset[i]:
                gt_answers = local_dataset[i]["answers"]
            else:
                gt_answers = local_dataset[i]["answer"]
            prompt = f"""
            Question: {local_dataset[i][chosen_key]}
            Ground Truth Answers: {gt_answers}
            Model Generated Answer: {extracted_answers[i][j]}
            """
            processed_prompt = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ]
            tokenized_prompt = tokenizer.apply_chat_template(
                processed_prompt, truncation=False, add_generation_prompt=True
            )
            decoded_prompt = tokenizer.decode(tokenized_prompt)
            prompts.append(decoded_prompt)

    # Setup LLM and send prompts
    sampling_params = SamplingParams(n=1, temperature=0, max_tokens=20)
    llm = LLM(model=judge_model, gpu_memory_utilization=0.8)
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    ####### END OF LLM AS JUDGE SETUP #######

    ####### AGGREGATE RESPONSES FOR COMPLETIONS #######
    responses = []
    for output in outputs:
        text_r = output.outputs[0].text
        if "yes" in text_r.lower():
            responses.append(1)
        else:
            responses.append(0)

    agg_responses = []
    # group responses into chunks of size n (one list per dataset example)
    for i in range(0, len(responses), n):
        agg_responses.append(responses[i : i + n])

    # Raw single-answer accuracy under judge (for logging)
    accuracy = np.mean(responses)
    print(f"Accuracy of {config.name} (single-answer judge) = {accuracy}")

    # Build evals / lengths / confidence metadata (same as before)
    for i in range(len(local_dataset)):
        eval_list, which_ans_list, c_len_list, conf_list, conf_format_list = [], [], [], [], []
        # Support both "answer" and "answers" columns
        if "answers" in local_dataset[i]:
            gt_answers = local_dataset[i]["answers"]
        else:
            gt_answers = local_dataset[i]["answer"]
        # Handle multiple correct answers
        if isinstance(gt_answers, (list, tuple)):
            gold_list = gt_answers
        else:
            gold_list = [gt_answers]
        
        if do_multi_block:
            # For multi-answer formats, only process the first completion (output_0)
            # and track each candidate separately to match candidate_correctness structure
            j = 0
            pred_response = local_dataset[i][f"{config.name}-output_{j}"]
            conf_format, conf_level = confidence_extractor(pred_response)
            conf_format_list.append(conf_format)
            c_len_list.append(len(pred_response))
            conf_list.append(conf_level)
            
            answers_by_idx, confs_by_idx = _extract_multi_answers_and_confidences(pred_response)
            
            if not answers_by_idx:
                # No multi tags; treat as all incorrect
                eval_list = []
                which_ans_list = []
            else:
                # Build ordered candidate list
                max_i = max(answers_by_idx.keys())
                candidates = [answers_by_idx.get(k, "").strip() for k in range(1, max_i + 1)]
                
                # Check correctness and get matched indices for each candidate
                ys, matched_indices = _candidate_correct_labels_with_indices(candidates, gt_answers)
                
                # Track each candidate separately (matching candidate_correctness structure)
                for y, midx in zip(ys, matched_indices):
                    eval_list.append(1 if y == 1.0 else 0)
                    which_ans_list.append(midx)
        else:
            # For single-answer formats, process all n completions
            for j in range(n):
                pred_response = local_dataset[i][f"{config.name}-output_{j}"]
                conf_format, conf_level = confidence_extractor(pred_response)
                conf_format_list.append(conf_format)
                c_len_list.append(len(pred_response))
                conf_list.append(conf_level)
                
                # For single-answer formats, use LLM judge results
                actual_correctness = agg_responses[i][j]
                eval_list.append(1 if actual_correctness == 1 else 0)
                
                # Find which answer matched (if correct)
                matched_idx = -1
                if actual_correctness == 1:
                    # Extract the answer from the response
                    ans_pattern = r"<answer>(.*?)</answer>"
                    ans_matches = re.findall(ans_pattern, pred_response, re.DOTALL | re.MULTILINE)
                    last_answer = ans_matches[-1] if ans_matches else ""
                    
                    # Check which gold answer matches
                    for idx, gold in enumerate(gold_list):
                        attempt = parse(last_answer)
                        label = verify(gold, attempt)
                        if label == 0:
                            label = exact_match_score(last_answer, gold)
                        if label:
                            matched_idx = idx
                            break
                which_ans_list.append(matched_idx)

        evals.append(eval_list)
        which_answers.append(which_ans_list)
        c_lengths.append(c_len_list)
        confidence_levels.append(conf_list)
        conf_format_levels.append(conf_format_list)

    # --- Multi-answer metrics with LLM judge per candidate ---
    multi_pass_at_1, multi_pass_at_2, multi_pass_at_3 = [], [], []
    per_example_set_brier = []  # only used for RLCR
    multi_candidate_correctness = []  # Store correctness for each candidate

    if do_multi_block:
        # For each example, we look only at the FIRST completion (output_0),
        # parse its multi answers, and judge each candidate via the LLM.
        for i in range(len(local_dataset)):
            pred_response = local_dataset[i][f"{config.name}-output_0"]
            # Support both "answer" and "answers" columns
            if "answers" in local_dataset[i]:
                gt_list = local_dataset[i]["answers"]
            else:
                gt_list = local_dataset[i]["answer"]
            question = local_dataset[i][chosen_key]
            gt = gt_list  # allow list; prompt text handles multiple GTs

            answers_by_idx, confs_by_idx = _extract_multi_answers_and_confidences(pred_response)

            if not answers_by_idx:
                # No multi tags; treat as all incorrect / zero-confs
                multi_pass_at_1.append(0.0)
                multi_pass_at_2.append(0.0)
                multi_pass_at_3.append(0.0)
                if is_multi_rlcr:
                    per_example_set_brier.append(0.0)
                multi_candidate_correctness.append([])
                continue

            # Build ordered candidate list 1..K
            max_i = max(answers_by_idx.keys())
            candidates = [answers_by_idx.get(k, "").strip() for k in range(1, max_i + 1)]

            # Build prompts for each candidate answer for this example
            cand_prompts = []
            for cand in candidates:
                prompt = f"""
                Question: {question}
                Ground Truth Answers: {gt}
                Model Generated Answer: {cand}
                """
                processed_prompt = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ]
                tokenized_prompt = tokenizer.apply_chat_template(
                    processed_prompt, truncation=False, add_generation_prompt=True
                )
                decoded_prompt = tokenizer.decode(tokenized_prompt)
                cand_prompts.append(decoded_prompt)

            # Judge all candidates for this example in a small batch
            cand_outputs = llm.generate(cand_prompts, sampling_params=sampling_params)
            ys = []
            for out in cand_outputs:
                text_r = out.outputs[0].text
                ys.append(1.0 if "yes" in text_r.lower() else 0.0)

            multi_candidate_correctness.append(ys)

            # Ordering: RLCR rank by confidence desc; RLVR keep numeric order
            order = list(range(len(candidates)))
            if is_multi_rlcr:
                confs = [confs_by_idx.get(k + 1, None) for k in range(len(candidates))]
                # Use -confidence (descending), None → last
                confs_for_sort = [(-c if c is not None else float("inf")) for c in confs]
                order = sorted(range(len(candidates)), key=lambda k: confs_for_sort[k])

            def _pass_at_k(k):
                idxs = order[: min(k, len(order))]
                return 1.0 if any(ys[t] == 1.0 for t in idxs) else 0.0

            multi_pass_at_1.append(_pass_at_k(1))
            multi_pass_at_2.append(_pass_at_k(2))
            multi_pass_at_3.append(_pass_at_k(3))

            # Set Brier (RLCR only): sum_i (p_i - y_i)^2 over all candidates
            if is_multi_rlcr:
                ps = []
                for k in range(len(candidates)):
                    p = confs_by_idx.get(k + 1, None)
                    ps.append(0.0 if p is None else p)
                ps_arr = np.array(ps, dtype=float)
                ys_arr = np.array(ys, dtype=float)
                set_brier_sum = float(np.sum((ps_arr - ys_arr) ** 2))
                per_example_set_brier.append(set_brier_sum)

        # Aggregate multi-answer metrics
        metrics["pass@1"] = float(np.mean(multi_pass_at_1)) if multi_pass_at_1 else 0.0
        metrics["pass@2"] = float(np.mean(multi_pass_at_2)) if multi_pass_at_2 else 0.0
        metrics["pass@3"] = float(np.mean(multi_pass_at_3)) if multi_pass_at_3 else 0.0
        metrics["accuracy"] = metrics["pass@1"]

        if is_multi_rlcr:
            metrics["set_brier"] = float(np.mean(per_example_set_brier)) if per_example_set_brier else 0.0

        # Save candidate-level correctness judged by LLM
        label_dict[f"{config.name}-candidate_correctness"] = multi_candidate_correctness

    # Compute traditional pass@k ONLY if not in multi-answer mode
    if not do_multi_block:
        if n not in config.pass_k_vals:
            config.pass_k_vals.append(n)
        if 1 not in config.pass_k_vals:
            config.pass_k_vals.append(1)
        for k in config.pass_k_vals:
            if k <= n:
                pass_k = compute_pass_n(evals, k)
                metrics[f"pass@{k}"] = pass_k

    if class_outputs is not None:
        if isinstance(class_outputs[0], list):
            confidence_levels = [[c[1]] for c in class_outputs]
            print("Overriding confidence levels with class outputs")
        else:
            confidence_levels = [[c] for c in class_outputs]

    # Flatten evals and confidence_levels for metrics computation
    # For multi-answer formats, lists can have different lengths, so flatten manually
    if do_multi_block:
        correctness_list = []
        expanded_conf_levels = []
        for i in range(len(evals)):
            eval_list = evals[i]
            conf_list = confidence_levels[i] if i < len(confidence_levels) else []
            num_candidates = len(eval_list)
            
            # Add correctness values
            correctness_list.extend(eval_list)
            
            # Expand confidence to match number of candidates
            # Only add confidence if we have candidates (to keep lengths matching)
            if num_candidates > 0:
                if len(conf_list) > 0:
                    # Repeat the confidence for each candidate
                    expanded_conf_levels.extend([conf_list[0]] * num_candidates)
                else:
                    # No confidence available, use 0.0 for each candidate
                    expanded_conf_levels.extend([0.0] * num_candidates)
            # If num_candidates == 0, we skip adding confidence (to match correctness)
        
        # Ensure arrays have the same length before creating numpy arrays
        min_len = min(len(correctness_list), len(expanded_conf_levels))
        if len(correctness_list) != len(expanded_conf_levels):
            print(f"Warning: Mismatch in array lengths: correctness={len(correctness_list)}, confidence={len(expanded_conf_levels)}. Truncating to {min_len}.")
            correctness_list = correctness_list[:min_len]
            expanded_conf_levels = expanded_conf_levels[:min_len]
        
        correctness_array = np.array(correctness_list) if len(correctness_list) > 0 else np.array([], dtype=float)
        confidence_array = np.array(expanded_conf_levels) if len(expanded_conf_levels) > 0 else np.array([], dtype=float)
        
        # Final safety check
        assert len(correctness_array) == len(confidence_array), f"Array length mismatch: correctness={len(correctness_array)}, confidence={len(confidence_array)}"
    else:
        correctness_array = np.array(evals).flatten()
        confidence_array = np.array(confidence_levels).flatten()

    # Only compute standard brier_score when NOT multi-answer
    if not do_multi_block:
        metrics["brier_score"] = get_brier(correctness_array, confidence_array)
    metrics["ece"] = get_ece(correctness_array, confidence_array)
    metrics["auroc"] = get_auroc(correctness_array, confidence_array)

    # take mean of c_lengths
    c_length_mean = np.mean(np.array(c_lengths))

    label_dict[f"{config.name}-evals"] = evals
    label_dict[f"{config.name}-which_answers"] = which_answers
    label_dict[f"{config.name}-c_lengths"] = c_lengths
    label_dict[f"{config.name}-confidence_levels"] = confidence_levels
    label_dict[f"{config.name}-conf_format_adherence"] = conf_format_levels

    if not do_multi_block:
        metrics["accuracy"] = metrics["pass@1"]
    metrics["completion length"] = c_length_mean
    # For confidence level, use the already-computed confidence_array for multi-answer formats
    if do_multi_block:
        metrics["confidence level"] = np.mean(confidence_array) if len(confidence_array) > 0 else 0.0
    else:
        metrics["confidence level"] = np.mean(np.array(confidence_levels))
    metrics["confidence format adherence"] = np.mean(
        np.array(conf_format_levels)
    )

    print(f"Metrics of {config.name} =")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    del llm
    gc.collect()
    return label_dict, metrics





# def llm_confidence_verifier(local_dataset, config, judge_model="meta-llama/Llama-3.1-8B-Instruct", format_fn="confidence_format", **kwargs):
#     label_dict = {f"{config.name}-evals": []}
#     evals = []
#     c_lengths = []
#     confidence_levels = []
#     conf_format_levels = []
#     metrics = {}
#     n = config.n

#     if f"{config.name}-class_output" in local_dataset.column_names:
#         class_outputs = local_dataset[f"{config.name}-class_output"]
#     else:
#         class_outputs = None

#     # FIRST EXTRACT OUT ALL ANSWERS FROM THE MODEL OUTPUTS. 
#     extracted_answers = []
#     for i in range(len(local_dataset)):
#         q_spec_ans = []
#         for j in range(n):
#             pred = local_dataset[i][f"{config.name}-output_{j}"]
#             ans_pattern = r"<answer>(.*?)</answer>"
#             # Get all <answer>...</answer> occurrences
#             ans_matches = re.findall(
#                 ans_pattern, pred, re.DOTALL | re.MULTILINE)
#             # Get the last answer, if exists
#             last_answer = ans_matches[-1] if ans_matches else ""
#             if last_answer == "":
#                 last_answer = "I don't know"
#             q_spec_ans.append(last_answer)
#         extracted_answers.append(q_spec_ans)

#     ####### DO LLM AS JUDGE SETUP #######
#     sys_prompt = """
#     You are a judge that will be given a question,ground truth answers and a model generated answer. There might be multiple ground truth answers. 
#     The model generated answer is correct if it matches any of the ground truth answers.
#     You will need to determine if the model generated answer is correct or not. 
#     Your response should be a single word. 'YES' if the answer is correct and 'NO' if it is not.
#     """

#     prompts = []
#     chosen_key = "question" if "question" in local_dataset.column_names else "problem"
#     tokenizer = AutoTokenizer.from_pretrained(judge_model, trust_remote_code=True)
    
#     #Generate prompts for each example
#     for i in range(len(local_dataset)):
#         for j in range(n):
#             prompt = f"""
#             Question: {local_dataset[i][chosen_key]}
#             Ground Truth Answers: {local_dataset[i]["answer"]}
#             Model Generated Answer: {extracted_answers[i][j]}
#             """
#             processed_prompt = [{'role': 'system', 'content': sys_prompt}, {
#                 'role': 'user', 'content': prompt}]
#             tokenized_prompt = tokenizer.apply_chat_template(
#                 processed_prompt, truncation=False, add_generation_prompt=True)
#             decoded_prompt = tokenizer.decode(tokenized_prompt)
#             prompts.append(decoded_prompt)

#     # Setup LLM and send prompts
#     sampling_params = SamplingParams(n=1, temperature=0, max_tokens=20)
#     llm = LLM(model=judge_model, gpu_memory_utilization=0.8)
#     outputs = llm.generate(prompts, sampling_params=sampling_params)

#     ####### END OF LLM AS JUDGE SETUP #######

#     ####### AGGREGATE RESPONSES #######

#     responses = []
#     for output in outputs:
#         text_r = output.outputs[0].text
#         if "yes" in text_r.lower():
#             responses.append(1)
#         else:
#             responses.append(0)

#     agg_responses = []
#     # agg responses by taking groups of n and making a list of them
#     for i in range(0, len(responses), n):
#         agg_responses.append(responses[i:i+n])

#     ####### END OF AGGREGATE RESPONSES #######

#     # Compute accuracy
#     accuracy = np.mean(responses)
#     print(f"Accuracy of {config.name} = {accuracy}")

#     for i in range(len(local_dataset)):
#         eval_list, c_len_list, conf_list, conf_format_list = [], [], [], []
#         for j in range(n):
#             pred_response = local_dataset[i][f"{config.name}-output_{j}"]
#             pred = [{"role": "assistant", "content": pred_response}]

#             actual_correctness = agg_responses[i][j]
#             conf_format, conf_level = confidence_extractor(pred_response)
#             conf_format_list.append(conf_format)

#             c_len_list.append(len(pred[0]["content"]))
#             conf_list.append(conf_level)
#             if actual_correctness == 1:
#                 eval_list.append(1)
#             else:
#                 eval_list.append(0)

#         evals.append(eval_list)
#         c_lengths.append(c_len_list)
#         confidence_levels.append(conf_list)
#         conf_format_levels.append(conf_format_list)


#     if n not in config.pass_k_vals:
#         config.pass_k_vals.append(n)
#     if 1 not in config.pass_k_vals:
#         config.pass_k_vals.append(1)
#     for k in config.pass_k_vals:
#         if k <= n:
#             pass_k = compute_pass_n(evals, k)
#             metrics[f"pass@{k}"] = pass_k

#     if class_outputs is not None:
#         if type(class_outputs[0]) == list:
#             confidence_levels = [[c[1]] for c in class_outputs]
#             print("Overriding confidence levels with class outputs")
#         else:
#             confidence_levels = [ [c] for c in class_outputs]

#     correctness_array = np.array(evals).flatten()
#     confidence_array = np.array(confidence_levels).flatten()
#     metrics["brier_score"] = get_brier(correctness_array, confidence_array) 
#     metrics["ece"] = get_ece(correctness_array, confidence_array)
#     metrics["auroc"] = get_auroc(correctness_array, confidence_array)

#     # take mean of c_lengths
#     c_length_mean = np.mean(np.array(c_lengths))

#     label_dict[f"{config.name}-evals"] = evals
#     label_dict[f"{config.name}-c_lengths"] = c_lengths
#     label_dict[f"{config.name}-confidence_levels"] = confidence_levels
#     label_dict[f"{config.name}-conf_format_adherence"] = conf_format_levels

#     metrics["accuracy"] = metrics["pass@1"]
#     metrics["completion length"] = c_length_mean
#     metrics["confidence level"] = np.mean(np.array(confidence_levels))
#     metrics["confidence format adherence"] = np.mean(
#         np.array(conf_format_levels))

#     print(f"Metrics of {config.name} =")
#     for k, v in metrics.items():
#         print(f"{k}: {v}")

#     del llm
#     gc.collect()
#     return label_dict, metrics 