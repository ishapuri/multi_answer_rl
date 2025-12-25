TABC_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind, provides the user with the final answer, then analyzes its confidence about the solution and then provides the user with its confidence level. "
    "The confidence level is a number between 0 and 1 (inclusive) enclosed within <confidence> </confidence> tags. The final answer is enclosed between <answer> </answer> tags."
    " The analysis about confidence and uncertainty is enclosed within <analysis> </analysis> tags. The assistant should reason about its confidence in the solution and its uncertainty in the solution within these tags."
    "The final format that must be followed is : <think> reasoning process here </think><answer> final answer here </answer> <analysis> analysis about confidence and uncertainty here</analysis> <confidence> confidence level here (number between 0 and 1) </confidence>"
)

TAC_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and analyzes its confidence about the solution and then provides the user with the final answer as well as its confidence level. "
    "The confidence level is a number between 0 and 1 (inclusive) enclosed within <confidence> </confidence> tags. The final answer is enclosed between <answer> </answer> tags."
    "The final format that must be followed is : <think> reasoning process here </think><answer> final answer here </answer> <confidence> confidence level here (number between 0 and 1) </confidence>"
)

TABC_LONG_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind, provides the user with the final answer, then analyzes its confidence about the solution and then provides the user with its confidence level. "
    "The confidence level is a number between 0 and 1 (inclusive) enclosed within <confidence> </confidence> tags. The final answer is enclosed between <answer> </answer> tags."
    "The analysis about confidence and uncertainty is enclosed within <analysis> </analysis> tags. The assistant should reason about its confidence in the solution and its uncertainty in the solution within these tags."
    "Here are some guidelines for the analysis: "
    "1. Your task is to point out things where the model could be wrong in its thinking, or things where there might be ambiguity in the solution steps, or in the reasoning process itself.\n" 
    "2. You should not suggest ways of fixing the response, your job is only to reason about uncertainties.\n"
    "3. For some questions, the response might be correct. In these cases, It is also okay to have only a small number of uncertainties and then explictly say that I am unable to spot more uncertainties.\n"
    "4. Uncertainties might be different from errors. For example, uncertainties may arise from ambiguities in the question, or from the application of a particular lemma/proof. \n"
    "5. If there are alternate potential approaches that may lead to different answers, you should mention them.\n"
    "6. List out plausible uncertainties, do not make generic statements, be as specific about uncertainties as possible.\n"
    "7. Enclose this uncertainty analysis within <analysis> </analysis> tags.\n"
    "The final format that must be followed is : <think> reasoning process here </think><answer> final answer here </answer> <analysis> analysis about confidence and uncertainty here</analysis> <confidence> confidence level here (number between 0 and 1) </confidence>"
)

GEN_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>. Put the reasoning process in the <think>...</think> tags and the answer in the <answer>...</answer> tags. Do NOT put any sentences or reasoning process within the <answer> </answer> tags - only put the final answer that will be verified with exact match score within the <answer> </answer> tags."
)

DEEPSEEK_VERIFIER_PROMPT = (
    "You are given a question and a solution to it. You have to verify if the solution is correct and enclose your verification reasoning within <analysis> </analysis> tags. Your analysis should be a minimum of 300 characters and should sequentially go through the thinking solution step by step. Here are the guidelines for your analysis - "
    "1. Your analysis should also be in 'I' form as if you wrote the solution and are now verifying it. \n"
    "2. Your goal is not to solve the problem but instead to verify if the steps in the presented solution are correct. \n"
    "3. If there are ambiguities in the solution steps or if step introduces uncertainty, you should mention it in the analysis. \n"
    "4. Go through the solution sequentially in a step-by-step manner. \n"
    "5. The analysis should be 300 characters minimum. \n"
    "6. Enclose this uncertainty analysis within <analysis> </analysis> tags.\n"
)

RLCR_SINGLE_PROMPT_NO_EXTRA_STUFF_IN_ANSWER = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind, provides the user with the final answer, then analyzes its confidence about the solution and then provides the user with its confidence level. "
    "The confidence level is a number between 0 and 1 (inclusive) enclosed within <confidence> </confidence> tags. The final answer is enclosed between <answer> </answer> tags."
    "The analysis about confidence and uncertainty is enclosed within <analysis> </analysis> tags. The assistant should reason about its confidence in the solution and its uncertainty in the solution within these tags."
    "Here are some guidelines for the analysis: "
    "1. Your task is to point out things where the model could be wrong in its thinking, or things where there might be ambiguity in the solution steps, or in the reasoning process itself.\n" 
    "2. You should not suggest ways of fixing the response, your job is only to reason about uncertainties.\n"
    "3. For some questions, the response might be correct. In these cases, It is also okay to have only a small number of uncertainties and then explictly say that I am unable to spot more uncertainties.\n"
    "4. Uncertainties might be different from errors. For example, uncertainties may arise from ambiguities in the question, or from the application of a particular lemma/proof. \n"
    "5. If there are alternate potential approaches that may lead to different answers, you should mention them.\n"
    "6. List out plausible uncertainties, do not make generic statements, be as specific about uncertainties as possible.\n"
    "7. Enclose this uncertainty analysis within <analysis> </analysis> tags.\n"
    "The final format that must be followed is : <think> reasoning process here </think><answer> final answer here </answer> <analysis> analysis about confidence and uncertainty here</analysis> <confidence> confidence level here (number between 0 and 1) </confidence>\n"
    "IMPORTANT: The <answer> </answer> tag must contain ONLY the minimal final answer. Do NOT write a full sentence in <answer>. Do NOT restate the question in <answer>. If any extra words are included, the answer is incorrect."
)


# =========================
# NEW: Distributional Candidates Prompts
# =========================

# NOTE:
# - These prompts instruct the model to produce K candidates with confidences that sum to 1,
#   plus an “Other” option if the gold might not be present.
# - They also constrain the output to a strict tag order so the format reward can enforce structure.
# - The "{K}" placeholder can be formatted at runtime:
#       prompt = get_sys_prompt_formatted("tabc_multi", K=cfg.num_candidates)
#   If you don't format it, the literal "{K}" will appear; prefer using the formatted accessor.

MULTI_ANSWER_PROMPT_TEMPLATE_MEDIUM = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it.\n"
    "You must produce a distribution over exactly {K} distinct candidate answers, with confidences that SUM TO 1.\n"
    "Use the following STRICT format and tag order only (no extra text before/after):\n"
    "<think> your internal reasoning steps here (do not reference the tags) </think>\n"
    "<analysis> explain key uncertainties and why mass is distributed across candidates; do not propose fixes </analysis>\n"
    "\n"
    "<candidates>\n"
    "  <answer><text> candidate_1_text </text><confidence> p1 </confidence></answer>\n"
    "  <answer><text> candidate_2_text </text><confidence> p2 </confidence></answer>\n"
    "  ... exactly {K} total <answer> blocks ...\n"
    "</candidates>\n"
    "REQUIREMENTS:\n"
    "1) Produce EXACTLY {K} candidates in the <candidates> block (no more, no fewer); each candidate must have one <text> and one <confidence>.\n"
    "2) Confidences must be numeric (0 to 1), and after normalization they must SUM TO 1 across the {K} candidates.\n"
    "3) Do NOT include an 'Other' candidate in the list; if you think the correct answer may be absent, reflect that by distributing probability mass conservatively across the {K} items and explaining the uncertainty in <analysis>.\n"
    "4) All candidate texts must be DISTINCT and concise; avoid duplicates and paraphrases.\n"
    "5) Do NOT output any content outside the required tags; preserve the exact order: <think>, <candidates>, <analysis>.\n"
)

MULTI_ANSWER_PROMPT_TEMPLATE_LONG = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it.\n"
    "You will output a calibrated distribution over EXACTLY {K} candidate answers whose confidences sum to 1.\n"
    "STRICT OUTPUT FORMAT (no extra text):\n"
    "<think> detailed step-by-step internal reasoning here </think>\n"
    "<analysis>\n"
    "  Provide a focused uncertainty analysis: where could the reasoning fail, what alternative derivations could shift probability mass, and what evidence supports the current allocation.\n"
    "  Do NOT fix or change the answers here; only analyze uncertainty and calibration.\n"
    "</analysis>\n"
    "\n"
    "<candidates>\n"
    "  <answer><text> candidate_1_text </text><confidence> p1 </confidence></answer>\n"
    "  <answer><text> candidate_2_text </text><confidence> p2 </confidence></answer>\n"
    "  ... exactly {K} total <answer> blocks ...\n"
    "</candidates>\n"
    "RULES:\n"
    "- EXACTLY {K} distinct candidates; each must have one <text> and one <confidence>.\n"
    "- Confidences must be numeric and sum to 1 across all {K} candidates.\n"
    "- Keep candidate texts short and non-redundant. No duplicates.\n"
    "- Output tags in the precise order and no additional prose outside tags.\n"
)

# (Optional) A minimal variant without the long analysis guidance.
# MULTI_ANSWER_PROMPT_TEMPLATE_SHORT = (
#     "A conversation between User and Assistant. The user asks a question, and the Assistant solves it.\n"
#     "Output EXACTLY {K} DISTINCT candidates with confidences that sum to 1.\n"
#     "FORMAT ONLY (no extra text):\n"
#     "<think> reasoning process about different candidate answers here </think>\n"
#     "<answer1> candidate_answer_1 </answer1>\n"
#     "<confidence1> confidence level for candidate 1 here (number between 0 and 1) </confidence1>\n"
#     "<answer2> candidate_answer_2 </answer2>\n"
#     "<confidence2>  confidence level for candidate 2 here </confidence2>\n"
#     "... exactly {K} pairs ...\n"
#     "<analysis> analysis about confidence and uncertainty here</analysis>\n"
# )

MULTI_ANSWER_PROMPT_TEMPLATE_SHORT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it.\n"
    "You must propose multiple possible answers, not just one. For each candidate, think separately about why it could be correct or incorrect.\n"
    "Compare the candidates before assigning confidences, and make sure the confidences sum to less than or equal to 1.\n"
    "Output EXACTLY {K} DISTINCT candidates with confidences that sum to less than or equal to1.\n"
    "FORMAT ONLY (no extra text):\n"
    "<think> reasoning process about different candidate answers here </think>\n"
    "<analysis> analysis about confidence and uncertainty here</analysis>"
    "<answer1> candidate_answer_1 </answer1>\n"
    "<confidence1> confidence level for candidate 1 here (number between 0 and 1) </confidence1>\n"
    "<answer2> candidate_answer_2 </answer2>\n"
    "<confidence2> confidence level for candidate 2 here </confidence2>\n"
    "... exactly {K} pairs ...\n"
    
)


MULTI_ANSWER_PROMPT_TEMPLATE_SHORT_OG_W_ANS_SPECIFICATION = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it.\n"
    "You must propose multiple possible answers, not just one. For each candidate, think separately about why it could be correct or incorrect.\n"
    "Compare the candidates before assigning confidences, and make sure the confidences sum to less than or equal to 1.\n"
    "Output EXACTLY {K} DISTINCT candidates with confidences that sum to less than or equal to1.\n"
    "FORMAT ONLY (no extra text):\n"
    "<think> reasoning process about different candidate answers here </think>\n"
    "<analysis> analysis about confidence and uncertainty here</analysis>"
    "<answer1> candidate_answer_1 </answer1>\n"
    "<confidence1> confidence level for candidate 1 here (number between 0 and 1) </confidence1>\n"
    "<answer2> candidate_answer_2 </answer2>\n"
    "<confidence2> confidence level for candidate 2 here </confidence2>\n"
    "... exactly {K} pairs ...\n"
    f"IMPORTANT: Each <answer{{i}}> </answer{{i}}> tag must contain ONLY a minimal final answer. Do NOT write a full sentence in <answer{{i}}>. Do NOT restate the question in <answer{{i}}>. If any extra words are included, the answer is incorrect. The answer will be graded on exact match with a ground truth answer."
)



MULTI_ANSWER_PROMPT_TEMPLATE_SHORT_AMBIGQA = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant responds.\n"
    "The questions in this dataset are intentionally ambiguous and may admit multiple valid answers under different interpretations.\n"
    "You must propose multiple DISTINCT candidate answers.\n"
    "For each candidate, reason about why it is plausible and what assumptions or interpretation it relies on.\n"
    "Do NOT assume there is a single correct answer.\n"
    "Compare the candidates and assign confidence values representing relative plausibility across interpretations.\n"
    "The confidences must sum to less than or equal to 1.\n"
    "Output EXACTLY {K} DISTINCT candidates.\n"
    "FORMAT ONLY (no extra text):\n"
    "<think> reasoning about different interpretations and candidate answers </think>\n"
    "<analysis> analysis of uncertainty and ambiguity across candidates </analysis>\n"
    "<answer1> candidate_answer_1 </answer1>\n"
    "<confidence1> confidence level for candidate 1 (number between 0 and 1) </confidence1>\n"
    "<answer2> candidate_answer_2 </answer2>\n"
    "<confidence2> confidence level for candidate 2 </confidence2>\n"
    "... exactly {K} pairs ...\n"
    f"IMPORTANT: Each <answer{{i}}> </answer{{i}}> tag must contain ONLY a minimal final answer. Do NOT write a full sentence in <answer{{i}}>. Do NOT restate the question in <answer{{i}}>. If any extra words are included, the answer is incorrect. The answer will be graded on exact match with a ground truth answer."
)


MULTI_ANSWER_RLVR_PROMPT_TEMPLATE_SHORT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it."
    "You must propose multiple possible answers, not just one. For each candidate, think separately about why it could be correct or incorrect."
    "Output EXACTLY {K} DISTINCT candidate answers."
    "FORMAT ONLY (no extra text):"
    "<think> reasoning process about different candidate answers here </think>"
    "<answer1> candidate_answer_1 </answer1>"
    "<answer2> candidate_answer_2 </answer2>"
    "... exactly {K} answers ..."
)

def get_sys_prompt(sys_prompt_name):
    if sys_prompt_name == "gen":
        return GEN_PROMPT
    elif sys_prompt_name == "tac":
        return TAC_PROMPT
    elif sys_prompt_name == "tabc":
        return TABC_PROMPT
    elif sys_prompt_name == "tabc_long":
        return TABC_LONG_PROMPT
    elif sys_prompt_name == "deepseek_verifier":
        return DEEPSEEK_VERIFIER_PROMPT
    elif sys_prompt_name == "rlcr_single_answer_no_extra_stuff_in_answer":
        return RLCR_SINGLE_PROMPT_NO_EXTRA_STUFF_IN_ANSWER
    # New distributional prompts (unformatted; include {K} placeholder)
    elif sys_prompt_name == "multi_answer_short":
        return MULTI_ANSWER_PROMPT_TEMPLATE_SHORT
    elif sys_prompt_name == "multi_answer_short_og_w_ans_specification":
        return MULTI_ANSWER_PROMPT_TEMPLATE_SHORT_OG_W_ANS_SPECIFICATION
    elif sys_prompt_name == "multi_answer_short_ambigqa":
        return MULTI_ANSWER_PROMPT_TEMPLATE_SHORT_AMBIGQA
    elif sys_prompt_name == "multi_answer_medium":
        return MULTI_ANSWER_PROMPT_TEMPLATE_MEDIUM
    elif sys_prompt_name == "multi_answer_long":
        return MULTI_ANSWER_PROMPT_TEMPLATE_LONG
    elif sys_prompt_name == "multi_answer_rlvr_short":
        return MULTI_ANSWER_RLVR_PROMPT_TEMPLATE_SHORT
    else:
        raise ValueError(f"Invalid system prompt name: {sys_prompt_name}")

# Helper: format the {K} placeholder without breaking existing call sites.
def get_sys_prompt_formatted(sys_prompt_name, num_candidates=1):
    """
    Example:
        prompt = get_sys_prompt_formatted("tabc_multi", K=cfg.num_candidates)
    Falls back to the unformatted string if no placeholders are provided.
    """    
    print("SYS PROMPT NAME in get_sys_prompt_formatted: ", sys_prompt_name)
    base = get_sys_prompt(sys_prompt_name)
    try:
        print("what will you return: ", base.format(K=num_candidates))
        return base.format(K=num_candidates)
    except Exception:
        print("Formatting failed")
        # If formatting fails (e.g., K not provided), return base as-is.
        return base