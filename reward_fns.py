import math
import re
from math_verify import verify,parse
import numpy as np 
import string

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def _safe_float(x):
    try:
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None



def check_content_has_required_tags(content: str, K: int, checkConfidences: bool = True) -> bool:
    if "<think>" not in content or "</think" not in content:
        #print("in check_content_has_required_tags, f<think> or f</think> not in content")
        return False

    if K == 1: 
        if checkConfidences:
            if (f"<answer>" not in content or f"</answer>" not in content or
                f"<confidence>" not in content or f"</confidence>" not in content or f"<analysis>" not in content or f"</analysis>" not in content):
                return False
        else:
            if (f"<answer>" not in content or f"</answer>" not in content):
                #print("in check_content_has_required_tags, f<answer> or f</answer> not in content")
                return False
    else: 
        for i in range(1, K + 1):
            if checkConfidences:
                if (f"<answer{i}>" not in content or f"</answer{i}>" not in content or
                    f"<confidence{i}>" not in content or f"</confidence{i}>" not in content or f"<analysis>" not in content or f"</analysis>" not in content):
                    return False
            else:
                if (f"<answer{i}>" not in content or f"</answer{i}>" not in content):
                    return False
    return True





def _extract_single_candidate(content: str):
    """
    Extracts a single answer and confidence using <answer>...</answer>
    and <confidence>...</confidence>.
    Returns (answers, confidences) or (None, None) on failure.
    """
    answer_open, answer_close = "<answer>", "</answer>"
    conf_open, conf_close = "<confidence>", "</confidence>"

    answer_start = content.find(answer_open)
    if answer_start == -1:
        return None, None
    answer_start += len(answer_open)

    answer_end = content.find(answer_close, answer_start)
    if answer_end == -1:
        return None, None
    ans = content[answer_start:answer_end].strip()

    conf_start = content.find(conf_open, answer_end + len(answer_close))
    if conf_start == -1:
        return None, None
    conf_start += len(conf_open)

    conf_end = content.find(conf_close, conf_start)
    if conf_end == -1:
        return None, None
    conf = content[conf_start:conf_end].strip()

    return [ans], [conf]



def _extract_flat_candidates(content: str, K: int):
    """
    makes sure the ordering is right: <answer1>... </answer1> then <confidence1>...</confidence1>, blah blah blah
    returns (answers, confidences) or (None, None) on failure
    """
    answers = []
    confidences = []
    idx = 0
    
    for i in range(1, K + 1):
        answer_open, answer_close = f"<answer{i}>", f"</answer{i}>"
        conf_open, conf_close = f"<confidence{i}>", f"</confidence{i}>"

        answer_start = content.find(answer_open, idx)
        if answer_start == -1: 
            return None, None
        answer_start += len(answer_open)
        
        answer_end = content.find(answer_close, answer_start)
        if answer_end == -1: 
            return None, None
        ans = content[answer_start:answer_end].strip()

        conf_start = content.find(conf_open, answer_end + len(answer_close))
        if conf_start == -1: 
            return None, None
            
        conf_start += len(conf_open)
        
        conf_end = content.find(conf_close, conf_start)
        
        if conf_end == -1: 
            return None, None
            
        conf = content[conf_start:conf_end].strip()

        answers.append(ans)
        confidences.append(conf)
        
        idx = conf_end + len(conf_close)

    return answers, confidences



def _extract_last_between(content: str, open_tag: str, close_tag: str) -> str:
    """find last occurrence of <open_tag> ... </close_tag>"""
    last = ""
    idx = 0
    while True:
        start = content.find(open_tag, idx)

        if start == -1: 
            break

        start += len(open_tag)
        end = content.find(close_tag, start)

        if end == -1: 
            break

        last = content[start:end].strip()
        idx = end + len(close_tag)
    return last

def extract_only_answers_rlvr(content: str, K: int):
    """
    Extracts answers in order. Supports both:
      - Numbered tags: <answer1>...</answer1>, <answer2>...</answer2>, etc.
      - Single tag: <answer>...</answer>
    Returns a list of answers, or None if parsing fails.
    """
    answers = []
    idx = 0

    # Case 1: numbered answers (e.g., <answer1>, <answer2>, ...)
    numbered_found = False
    for i in range(1, K + 1):
        answer_open, answer_close = f"<answer{i}>", f"</answer{i}>"

        answer_start = content.find(answer_open, idx)
        if answer_start == -1:
            break  # stop if no more numbered answers
        numbered_found = True
        answer_start += len(answer_open)

        answer_end = content.find(answer_close, answer_start)
        if answer_end == -1:
            return None

        ans = content[answer_start:answer_end].strip()
        answers.append(ans)

        # advance the search index
        idx = answer_end + len(answer_close)

    # Case 2: single unnumbered <answer>...</answer>
    if not numbered_found:
        answer_open, answer_close = "<answer>", "</answer>"
        answer_start = content.find(answer_open)
        if answer_start == -1:
            return None
        answer_start += len(answer_open)

        answer_end = content.find(answer_close, answer_start)
        if answer_end == -1:
            return None

        ans = content[answer_start:answer_end].strip()
        answers.append(ans)

    return answers if answers else None



def format_reward(format_pattern, completions, num_candidates, **kwargs):
    """
    Returns a list of {0.0, 1.0}.
    - 'multi_answer': requires K ordered <answeri>/<confidencei> pairs and numeric confidences in [0,1].
      Also requires <think> and </analysis> to be present.
    - 'ta'/'tac'/'tabc'/'tbac': light presence + (for *c*) numeric confidence in [0,1].
    """
    completion_contents = [c[0]["content"] for c in completions]
    out = []
    fmt = str(format_pattern).lower()

    if fmt == "rlcr_single_answer": #k should be 1

        K = num_candidates #kwargs.get("k_expected", kwargs.get("num_candidates"))

        if not isinstance(K, int) or K <= 0:
            return [0.0 for _ in completions]

        for content in completion_contents:
            if not check_content_has_required_tags(content, K, checkConfidences=True):
                out.append(0.0); continue


            answers, confs = _extract_single_candidate(content)
            
            if answers is None:
                out.append(0.0); continue
            vals = [_safe_float(c) for c in confs]
            ok = all(v is not None and 0.0 <= v <= 1.0 for v in vals)
            out.append(1.0 if ok else 0.0)
           
        return out

    elif fmt == "rlvr_single_answer": #k should be 1

        K = num_candidates #kwargs.get("k_expected", kwargs.get("num_candidates"))
        #print("K", K)

        if not isinstance(K, int) or K <= 0:
            return [0.0 for _ in completions]

        for content in completion_contents:
            if not check_content_has_required_tags(content, K, checkConfidences=False):
                #print("appending 0.0 because not check_content_has_required_tags")
                out.append(0.0); continue


            answers, confs = extract_only_answers_rlvr(content, K), None
            
            if answers is None:
                #print("appending 0.0 because answers is None")
                out.append(0.0); continue
            
            out.append(1.0)
           
        return out

    elif fmt == "multi_answer":
        K = num_candidates #kwargs.get("k_expected", kwargs.get("num_candidates"))
        if not isinstance(K, int) or K <= 0:
            return [0.0 for _ in completions]

        for content in completion_contents:
            #make sure tags are there
            if not check_content_has_required_tags(content, K, checkConfidences=True):
                out.append(0.0); continue
            
            #extract answers and confidences 
            answers, confs = _extract_flat_candidates(content, K)
            if answers is None:
                out.append(0.0); continue
            
            vals = [_safe_float(c) for c in confs]
            ok = all(v is not None and 0.0 <= v <= 1.0 for v in vals)
            out.append(1.0 if ok else 0.0)
        return out

    
    elif fmt == "multi_answer_rlvr":
        K = num_candidates #kwargs.get("k_expected", kwargs.get("num_candidates"))
        if not isinstance(K, int) or K <= 0:
            return [0.0 for _ in completions]
        
        for content in completion_contents:
            if not check_content_has_required_tags(content, K, checkConfidences=False):
                out.append(0.0); continue
            answers = extract_only_answers_rlvr(content, K)
            if answers is None:
                out.append(0.0); continue
            out.append(1.0)
        return out

    def has_tag(tag): 
        return tag in content

    for content in completion_contents:
        if fmt == "ta":
            ok = has_tag("<think>") and has_tag("</think>") and has_tag("<answer>") and has_tag("</answer>")
            out.append(1.0 if ok else 0.0)
        elif fmt in ("tac", "tabc", "tbac"):
            need_analysis = (fmt in ("tabc", "tbac"))
            ok = has_tag("<think>") and has_tag("</think>") and has_tag("<answer>") and has_tag("</answer>")
            if need_analysis:
                ok = ok and has_tag("<analysis>") and has_tag("</analysis>")
            if not ok:
                out.append(0.0); continue
            last_conf = _extract_last_between(content, "<confidence>", "</confidence>")
            v = _safe_float(last_conf)
            out.append(1.0 if (v is not None and 0.0 <= v <= 1.0) else 0.0)
        else:
            out.append(0.0)
    return out


# =========================
# Response constraints reward (UNIQUENESS + SUM <= 1)
# =========================

def response_constraint_reward(completions, num_candidates=None, **kwargs):
    """
    Returns a list of {0.0, 1.0} for multi-candidate only:
      - Exactly K ordered pairs can be extracted
      - All K answers unique 
      - Sum(confidences) <= 1 
    """

    #print("responseConstraintReward num_candidates before kwargs set", num_candidates)
    #num_candidates = kwargs.get("num_candidates")
    #print("responseConstraintReward num_candidates after kwargs set", num_candidates)

    if not isinstance(num_candidates, int) or num_candidates <= 0:
        return [0.0 for _ in completions]

    scores = []
    
    for content in (c[0]["content"] for c in completions):
        
        if num_candidates == 1:
            answers, confs = _extract_single_candidate(content)
        else:
            answers, confs = _extract_flat_candidates(content, num_candidates)
        
        #right number of answers and confidences 
        if answers is None or len(answers) != num_candidates or len(confs) != num_candidates:
            scores.append(0.0); continue

        # uniqueness
        norm = [a.strip().lower() for a in answers]
        if len(set(norm)) != len(norm):
            scores.append(0.0); continue

        vals = [_safe_float(c) for c in confs]
        if any(v is None for v in vals):
            scores.append(0.0); continue
        
        #sum <= 1
        s = float(np.sum(vals))
        scores.append(1.0 if s <= 1.0 else 0.0)
        
    return scores



def combined_format_and_constraint_reward(
    completions,
    format_pattern=None,
    num_candidates=None,
    **kwargs
):
    """
    Returns a list of {0.0, 1.0}: 1.0 iff BOTH
      - response_format_reward(...) == 1.0
      - response_constraint_reward(...) == 1.0
    for the corresponding completion.
    """

    #print("In combined_format_and_constraint_reward as of 4:41 on saturday")
    # 1) Format validity (uses format_pattern and kwargs such as k_expected/num_candidates)
    fmt_scores = format_reward(
        format_pattern,
        completions,
        num_candidates
    )

    # 2) Constraint validity (uniqueness + sum(conf) <= 1 - tol), only meaningful for multi-candidate
    cons_scores = response_constraint_reward(
        completions,
        num_candidates=num_candidates,
    )

    # 3) Element-wise AND
    out = []
    for formatReward, constrainReward  in zip(fmt_scores, cons_scores):
        # print("formatReward", formatReward)
        # print("constrainReward", constrainReward)
        out.append(formatReward*constrainReward)
        # print("out", out)
        # print("--------------------------------")
    return out



# =========================
# Accuracy reward
# =========================


def accuracy_reward(format_pattern, completions, answer=None, num_candidates=None, source=None, **kwargs):
    """
    - If format_pattern == 'multi_answer': evaluate correctness of <answer1>...</answer1>.
    - Else (old stuff): evaluate LAST <answer>...</answer>.
    Uses try/except so bad parses can't mess up ranks.
    Supports both "answer" and "answers" columns (prefers "answers" if both exist).
    """
    # Support both "answer" and "answers" columns
    if answer is None:
        answer = kwargs.get("answers", [])
    elif "answers" in kwargs:
        answer = kwargs["answers"]  # Prefer "answers" if both exist
    
    fmt = str(format_pattern).lower()
    contents = [c[0]["content"] for c in completions]
    correct_answers    = list(answer)
    out = []

    #print("num_candidates after kwargs set", num_candidates)
    
    #num_candidates = kwargs.get("num_candidates")
    # Pre-checks to keep ranks aligned

    #print("num_candidates ", num_candidates)
    fmt_scores = format_reward(format_pattern, completions, num_candidates=num_candidates, **kwargs)
    reward_constraint_scores  = (response_constraint_reward(completions, num_candidates=num_candidates)
                  if fmt == "multi_answer" else [1.0]*len(completions))

    #if format or response constraint is 0, then acc gets zeroed out 
    for content, correct_answer, format_reward_score, reward_constraint_score in zip(contents, correct_answers, fmt_scores, reward_constraint_scores):
        if format_reward_score == 0 or reward_constraint_score == 0:
            out.append(0.0)
            #print("either format reward or reward constraint is 0", format_reward_score, reward_constraint_score)
            continue

        try:
            # Collect candidates depending on the format
            if fmt == "multi_answer":
                # Assume candidates are in tags like <answer1> ... </answer1>, <answer2> ... </answer2>, etc.
                candidates = [
                    _extract_last_between(content, f"<answer{i}>", f"</answer{i}>")
                    for i in range(1, num_candidates + 1)
                ]
                #print("HIHIHIHI candidates", candidates)
            elif fmt == "rlvr_single_answer":
                candidates = [_extract_last_between(content, "<answer>", "</answer>")]
                #print("only one candidate", candidates)
            elif fmt == "multi_answer_rlvr":
                candidates = extract_only_answers_rlvr(content, num_candidates)

            else: #single answer rlcr
                candidates = [_extract_last_between(content, "<answer>", "</answer>")]

            # Check if ANY candidate matches ANY of the correct answers
            correct = False
            for cand in candidates:
                if not cand:
                    continue
                if _is_correct(cand, correct_answer, source):
                    correct = True
                    break

            out.append(1.0 if correct else 0.0)

        except Exception as e:
            print(e)
            out.append(0.0)

    return out







def _is_correct(cand: str, gold, source=None) -> bool:
    """Check if candidate matches gold answer(s).
    gold can be a single string or a list of strings (multiple correct answers).
    Returns True if cand matches any of the gold answers."""
 
    if not cand:
        return False
    
    # Handle multiple gold answers
    if isinstance(gold, (list, tuple)):
        gold_list = gold
    else:
        gold_list = [gold]
    
    try:
        for g in gold_list:
            if source is not None and (source[0] == 'hotpotQA' or source[0] == 'ambigQA' or source[0] == 'medDataset'):
                if exact_match_score(cand, g):
                       #print("exact_match_score is true, cand, gold", cand, gold)
                       return True
            else:
                if verify(g, parse(cand)):
                    return True
    
        #print("no match found, return false, cand, gold", cand, gold)
        return False
    except Exception:
        #print("exception in _is_correct, return false, cand, gold", cand, gold)
        return False


def pass_at_1(format_pattern, completions, answer=None, num_candidates=None, source=None, **kwargs):
    """
    pass@1 reward:
      - RLVR single / RLCR single: 1 iff the (only) answer is correct.
      - RLVR multiple: 1 iff the FIRST candidate is correct.
      - RLCR multiple: 1 iff the HIGHEST-CONFIDENCE candidate is correct.
    Returns a list of {0.0, 1.0} per completion.
    Supports both "answer" and "answers" columns (prefers "answers" if both exist).
    """
    # Support both "answer" and "answers" columns
    if answer is None:
        answer = kwargs.get("answers", [])
    elif "answers" in kwargs:
        answer = kwargs["answers"]  # Prefer "answers" if both exist
    
    fmt = str(format_pattern).lower()
    contents = [c[0]["content"] for c in completions]
    golds = list(answer)

    # Enforce formatting to avoid false positives
    fmt_scores = format_reward(format_pattern, completions, num_candidates=num_candidates, **kwargs)

    out = []
    if fmt in ("rlvr_single_answer", "rlcr_single_answer"):
        for content, gold, fr in zip(contents, golds, fmt_scores):
            if fr == 0:
                out.append(0.0); continue
            cand = _extract_last_between(content, "<answer>", "</answer>")
            out.append(1.0 if _is_correct(cand, gold, source) else 0.0)
        return out

    if fmt == "multi_answer_rlvr":
        K = num_candidates
        for content, gold, fr in zip(contents, golds, fmt_scores):
            if fr == 0:
                out.append(0.0); continue
            cands = extract_only_answers_rlvr(content, K)
            first = cands[0] if cands else ""
            out.append(1.0 if _is_correct(first, gold, source) else 0.0)
        return out

    if fmt == "multi_answer":
        K = num_candidates
        for content, gold, fr in zip(contents, golds, fmt_scores):
            if fr == 0:
                out.append(0.0); continue
            answers, confs = _extract_flat_candidates(content, K)
            if not answers or not confs:
                out.append(0.0); continue

            # Choose highest-confidence candidate
            best_idx, best_conf = -1, -1.0
            for j, conf in enumerate(confs):
                v = _safe_float(conf)
                v = -1.0 if v is None else max(0.0, min(1.0, v))
                if v > best_conf:
                    best_conf, best_idx = v, j

            chosen = answers[best_idx] if best_idx >= 0 else ""
            out.append(1.0 if _is_correct(chosen, gold, source) else 0.0)
        return out

    # Unknown format
    return [0.0 for _ in completions]


def pass_at_i(format_pattern, completions, answer=None, num_candidates=None, i=None, source=None, **kwargs):
    """
    pass@i reward:
      - RLVR single / RLCR single:
           * Only i == 1 supported; else raises ValueError.
      - RLVR multiple:
           * 1 iff ANY of the first i candidates (in order) is correct.
      - RLCR multiple:
           * Rank candidates by confidence (desc) and return 1 iff ANY of the top i is correct.
    Returns a list of {0.0, 1.0} per completion.
    Supports both "answer" and "answers" columns (prefers "answers" if both exist).
    """
    if not isinstance(i, int) or i <= 0:
        raise ValueError("pass_at_i expects i to be a positive integer.")
    
    # Support both "answer" and "answers" columns
    if answer is None:
        answer = kwargs.get("answers", [])
    elif "answers" in kwargs:
        answer = kwargs["answers"]  # Prefer "answers" if both exist

    fmt = str(format_pattern).lower()
    contents = [c[0]["content"] for c in completions]
    golds = list(answer)

    # Enforce formatting to avoid false positives
    fmt_scores = format_reward(format_pattern, completions, num_candidates=num_candidates, **kwargs)

    # SINGLE-ANSWER: only i==1
    if fmt in ("rlvr_single_answer", "rlcr_single_answer"):
        if i != 1:
            raise ValueError("pass_at_i for single-answer formats only supports i=1.")
        return pass_at_1(format_pattern, completions, answer, num_candidates, source=source, **kwargs)

    out = []

    # MULTI-ANSWER RLVR: sequential order, take first i
    if fmt == "multi_answer_rlvr":
        K = num_candidates
        for content, gold, fr in zip(contents, golds, fmt_scores):
            if fr == 0:
                out.append(0.0); continue
            cands = extract_only_answers_rlvr(content, K)
            if not cands:
                out.append(0.0); continue
            top = cands[:min(i, len(cands))]
            ok = any(_is_correct(c, gold, source) for c in top)
            out.append(1.0 if ok else 0.0)
        return out

    # MULTI-ANSWER RLCR: rank by confidence desc, then take top i
    if fmt == "multi_answer":
        K = num_candidates
        for content, gold, fr in zip(contents, golds, fmt_scores):
            if fr == 0:
                out.append(0.0); continue

            answers, confs = _extract_flat_candidates(content, K)
            if not answers or not confs:
                out.append(0.0); continue

            scored = []
            for ans, conf in zip(answers, confs):
                v = _safe_float(conf)
                v = -1.0 if v is None else max(0.0, min(1.0, v))
                scored.append((v, ans))
            scored.sort(key=lambda t: t[0], reverse=True)

            top_i_answers = [a for _, a in scored[:min(i, len(scored))]]
            ok = any(_is_correct(c, gold, source) for c in top_i_answers)
            out.append(1.0 if ok else 0.0)
        return out

    # Unknown format
    return [0.0 for _ in completions]





# =========================
# Brier reward (per-completion, over K candidates)
# =========================


def brier_reward(format_pattern, completions, answer=None, source=None, **kwargs):
    """
    reward = 1 - mean_i (y_i - p_i)^2, where
      y_i is correctness of candidate i,
      p_i is the model's confidence_i 
    Supports both "answer" and "answers" columns (prefers "answers" if both exist).
    """
    # Support both "answer" and "answers" columns
    if answer is None:
        answer = kwargs.get("answers", [])
    elif "answers" in kwargs:
        answer = kwargs["answers"]  # Prefer "answers" if both exist
    
    fmt = str(format_pattern).lower()
    # if fmt != "multi_answer":
    #     # rn i'm not doing non-multi-answer things - when we release the code we can put back the single answer brier score - im tired and dont wanna add it back rn 
    #     return [0.0 for _ in completions]

    K = kwargs.get("k_expected", kwargs.get("num_candidates"))
    if not isinstance(K, int) or K <= 0:
        return [0.0 for _ in completions]

    contents = [c[0]["content"] for c in completions]
    correct_answers    = list(answer)

    format_reward_scores = format_reward(format_pattern, completions, **kwargs)
    reward_constraint_scores  = response_constraint_reward(completions, num_candidates=K)

    rewards = []
    for content, correct_answer, fr, rc in zip(contents, correct_answers, format_reward_scores, reward_constraint_scores):
        if fr == 0 or rc == 0:
            rewards.append(0.0); continue

        if format_pattern == "rlcr_single_answer":
            answers, confs = _extract_single_candidate(content) 
        else:
            answers, confs = _extract_flat_candidates(content, K)
        
        if answers is None:
            rewards.append(0.0); continue

        sq_errs = []
        for cand, conf in zip(answers, confs):
            try:
                p = _safe_float(conf)
                p = 0.0 if p is None else max(0.0, min(1.0, p))
                y = float(bool(_is_correct(cand, correct_answer, source)))
                sq_errs.append((y - p) ** 2)
            except Exception:
                # Treat any failure as wrong answer with p=clip(conf)
                p = _safe_float(conf)
                p = 0.0 if p is None else max(0.0, min(1.0, p))
                sq_errs.append((0.0 - p) ** 2)

        avg_mse = sum(sq_errs) / len(sq_errs) if sq_errs else 1.0
        rewards.append(1.0 - avg_mse)
    return rewards


def mean_confidence_reward(completions, **kwargs):
    """Reward function that extracts the last occurrence of text inside the answer tags and then checks if a label is present there"""
    confidence_pattern = r"<confidence>(.*?)</confidence>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = []

    for content in completion_contents:
        confidence_matches = re.findall(confidence_pattern, content, re.DOTALL | re.MULTILINE)  # Get all <confidence>...</confidence> occurrences
        last_confidence = confidence_matches[-1] if confidence_matches else ""  # Get the last confidence, if exists
        if last_confidence == "":
            matches.append(0.0)
        else:
            try:
                confidence = float(last_confidence)
                #clip confidence to be between 0 and 1
                confidence = max(0.0, min(confidence, 1.0))
            except:
                confidence = 0.0
            matches.append(confidence)
    return matches

def confidence_one_or_zero(completions, **kwargs):
    """Reward function that extracts the last occurrence of text inside the answer tags and then checks if a label is present there"""
    confidence_pattern = r"<confidence>(.*?)</confidence>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = []

    for content in completion_contents:
        confidence_matches = re.findall(confidence_pattern, content, re.DOTALL | re.MULTILINE)  # Get all <confidence>...</confidence> occurrences
        last_confidence = confidence_matches[-1] if confidence_matches else ""  # Get the last confidence, if exists
        if last_confidence == "":
            matches.append(0.0)
        else:
            try:
                confidence = float(last_confidence)
                #clip confidence to be between 0 and 1
                confidence = max(0.0, min(confidence, 1.0))
            except:
                confidence = 0.0
            if abs(confidence - 1) < 0.01 or abs(confidence - 0) < 0.01:
                matches.append(1.0)
            else:
                matches.append(0.0)
    return matches







if __name__ == '__main__':
    s = "    h   ello whatever </think> <answer> The number of non-empty subsets 31 </answer> <confidence> 0.9 </confidence>   \n \n  "
 
    pattern = r".*?</think>\s*<answer>.*?</answer>\s*<confidence>.*?</confidence>\s*\Z" 
    match = re.match(pattern, s, re.DOTALL | re.MULTILINE)
    print(match)
    print(match[0])