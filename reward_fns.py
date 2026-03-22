import math
import re
import string
import warnings

import numpy as np
from math_verify import verify, parse

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize_answer(s):
    """Lowercase, remove punctuation/articles, and collapse whitespace."""
    s = s.lower()
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def _safe_float(x):
    """Convert x to float, returning None if invalid or non-finite."""
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Tag extraction helpers
# ---------------------------------------------------------------------------

def _extract_between(content: str, open_tag: str, close_tag: str, start: int = 0):
    """
    Return (text, end_pos) for the first open_tag...close_tag after `start`.
    Returns (None, -1) if not found.
    end_pos points to right after close_tag, suitable as the next search start.
    """
    s = content.find(open_tag, start)
    if s == -1:
        return None, -1
    s += len(open_tag)
    e = content.find(close_tag, s)
    if e == -1:
        return None, -1
    return content[s:e].strip(), e + len(close_tag)


def _extract_last_between(content: str, open_tag: str, close_tag: str) -> str:
    """Return the last occurrence of text between open_tag and close_tag."""
    last, pos = "", 0
    while True:
        text, pos = _extract_between(content, open_tag, close_tag, pos)
        if text is None:
            break
        last = text
    return last


def _extract_single_candidate(content: str):
    """
    Extract a single answer and confidence from <answer>...</answer> and
    <confidence>...</confidence> tags. Returns ([answer], [conf]) or (None, None).
    """
    ans, pos = _extract_between(content, "<answer>", "</answer>")
    if ans is None:
        return None, None
    conf, _ = _extract_between(content, "<confidence>", "</confidence>", pos)
    if conf is None:
        return None, None
    return [ans], [conf]


def _extract_flat_candidates(content: str, K: int):
    """
    Extract K (answer, confidence) pairs. Supports two layouts:
    - Interleaved: <answer1>A</answer1><confidence1>p</confidence1>...
    - Split: all <answerI> tags first, then all <confidenceI> tags
      (with optional <analysis> in between, as in the medical RLCR format).
    Returns (answers, confidences) or (None, None).
    """
    # Try interleaved layout first
    answers, confs, pos = [], [], 0
    for i in range(1, K + 1):
        ans, pos = _extract_between(content, f"<answer{i}>", f"</answer{i}>", pos)
        if ans is None:
            break
        conf, pos = _extract_between(content, f"<confidence{i}>", f"</confidence{i}>", pos)
        if conf is None:
            break
        answers.append(ans)
        confs.append(conf)

    if len(answers) == K:
        return answers, confs

    # Fall back to split layout (all answers first, all confidences after analysis)
    answers, confs = [], []
    for i in range(1, K + 1):
        ans, _ = _extract_between(content, f"<answer{i}>", f"</answer{i}>")
        if ans is None:
            return None, None
        answers.append(ans)
    for i in range(1, K + 1):
        conf, _ = _extract_between(content, f"<confidence{i}>", f"</confidence{i}>")
        if conf is None:
            return None, None
        confs.append(conf)

    return (answers, confs) if len(answers) == K else (None, None)


def extract_only_answers_rlvr(content: str, K: int):
    """
    Extract K answers from numbered <answer1>...<answerK> tags, or from a
    single <answer> tag as fallback. Returns a list of strings, or None.
    """
    answers, pos = [], 0
    for i in range(1, K + 1):
        ans, pos = _extract_between(content, f"<answer{i}>", f"</answer{i}>", pos)
        if ans is None:
            break
        answers.append(ans)

    if answers:
        return answers

    # Fallback: single unnumbered <answer> tag
    ans, _ = _extract_between(content, "<answer>", "</answer>")
    return [ans] if ans is not None else None


# ---------------------------------------------------------------------------
# Format validation
# ---------------------------------------------------------------------------

def check_content_has_required_tags(
    content: str, K: int, checkConfidences: bool = True, format_pattern: str = None
) -> bool:
    """
    Return True iff content contains all required structural tags.

    Checks:
    - Balanced, non-empty <think>...</think> block.
    - K answer tags (numbered for K>1, plain for K=1).
    - K confidence tags when checkConfidences=True.
    - <analysis> tags when checkConfidences=True, unless format is 'multi_answer_no_analysis'.
    """
    # Validate <think>...</think>: must be balanced, ordered, and non-empty
    opens  = [m.start() for m in re.finditer(r"<think>",  content)]
    closes = [m.start() for m in re.finditer(r"</think>", content)]
    if not opens or not closes or len(opens) != len(closes):
        return False
    if any(c < o or not content[o + len("<think>"):c].strip() for o, c in zip(opens, closes)):
        return False

    requires_analysis = (
        str(format_pattern).lower() != "multi_answer_no_analysis" if format_pattern else True
    )

    if K == 1:
        if "<answer>" not in content or "</answer>" not in content:
            return False
        if checkConfidences:
            if "<confidence>" not in content or "</confidence>" not in content:
                return False
            if requires_analysis and ("<analysis>" not in content or "</analysis>" not in content):
                return False
    else:
        for i in range(1, K + 1):
            if f"<answer{i}>" not in content or f"</answer{i}>" not in content:
                return False
            if checkConfidences and (
                f"<confidence{i}>" not in content or f"</confidence{i}>" not in content
            ):
                return False
        if checkConfidences and requires_analysis and (
            "<analysis>" not in content or "</analysis>" not in content
        ):
            return False

    return True


# ---------------------------------------------------------------------------
# Format reward
# ---------------------------------------------------------------------------

def format_reward(format_pattern, completions, num_candidates, **kwargs):
    """
    Returns 1.0 if the completion matches the expected structural format, else 0.0.

    Supported format_pattern values:
    - 'multi_answer'             : K candidates with confidences + <analysis>
    - 'multi_answer_no_analysis' : K candidates with confidences, no <analysis>
    - 'multi_answer_rlvr'        : K candidates without confidences
    - 'rlcr_single_answer'       : 1 candidate with confidence
    - 'rlvr_single_answer'       : 1 candidate without confidence
    - 'ta' / 'tac' / 'tabc' / 'tbac': legacy single-answer formats
    """
    fmt = str(format_pattern).lower()
    K   = num_candidates
    contents = [c[0]["content"] for c in completions]

    if fmt in ("multi_answer", "multi_answer_no_analysis"):
        if not isinstance(K, int) or K <= 0:
            return [0.0] * len(completions)
        out = []
        for content in contents:
            if not check_content_has_required_tags(content, K, checkConfidences=True, format_pattern=fmt):
                out.append(0.0); continue
            answers, confs = _extract_flat_candidates(content, K)
            if answers is None:
                out.append(0.0); continue
            vals = [_safe_float(c) for c in confs]
            out.append(1.0 if all(v is not None and 0.0 <= v <= 1.0 for v in vals) else 0.0)
        return out

    if fmt == "multi_answer_rlvr":
        if not isinstance(K, int) or K <= 0:
            return [0.0] * len(completions)
        out = []
        for content in contents:
            if not check_content_has_required_tags(content, K, checkConfidences=False, format_pattern=fmt):
                out.append(0.0); continue
            out.append(1.0 if extract_only_answers_rlvr(content, K) is not None else 0.0)
        return out

    if fmt == "rlcr_single_answer":
        if not isinstance(K, int) or K <= 0:
            return [0.0] * len(completions)
        out = []
        for content in contents:
            if not check_content_has_required_tags(content, K, checkConfidences=True, format_pattern=fmt):
                out.append(0.0); continue
            answers, confs = _extract_single_candidate(content)
            if answers is None:
                out.append(0.0); continue
            vals = [_safe_float(c) for c in confs]
            out.append(1.0 if all(v is not None and 0.0 <= v <= 1.0 for v in vals) else 0.0)
        return out

    if fmt == "rlvr_single_answer":
        if not isinstance(K, int) or K <= 0:
            return [0.0] * len(completions)
        out = []
        for content in contents:
            if not check_content_has_required_tags(content, K, checkConfidences=False, format_pattern=fmt):
                out.append(0.0); continue
            out.append(1.0 if extract_only_answers_rlvr(content, K) is not None else 0.0)
        return out

    # Legacy single-answer formats: ta, tac, tabc, tbac
    out = []
    for content in contents:
        has = lambda tag: tag in content
        ok  = has("<think>") and has("</think>") and has("<answer>") and has("</answer>")
        if fmt in ("tabc", "tbac"):
            ok = ok and has("<analysis>") and has("</analysis>")
        if ok and fmt in ("tac", "tabc", "tbac"):
            v  = _safe_float(_extract_last_between(content, "<confidence>", "</confidence>"))
            ok = v is not None and 0.0 <= v <= 1.0
        out.append(1.0 if ok else 0.0)
    return out


# ---------------------------------------------------------------------------
# Constraint reward  (uniqueness + sum of confidences ≤ 1)
# ---------------------------------------------------------------------------

def response_constraint_reward(completions, num_candidates=None, **kwargs):
    """
    Returns 1.0 iff:
    - All K candidates are unique (case-insensitive).
    - Sum of confidences ≤ 1 (when confidences_sum_to_less_than_1=True, the default).
    """
    if not isinstance(num_candidates, int) or num_candidates <= 0:
        return [0.0] * len(completions)

    enforce_sum = kwargs.get("confidences_sum_to_less_than_1", True)
    scores = []

    for content in (c[0]["content"] for c in completions):
        if num_candidates == 1:
            answers, confs = _extract_single_candidate(content)
        else:
            answers, confs = _extract_flat_candidates(content, num_candidates)

        if answers is None or len(answers) != num_candidates:
            scores.append(0.0); continue

        # Uniqueness
        norm = [a.strip().lower() for a in answers]
        if len(set(norm)) != len(norm):
            scores.append(0.0); continue

        vals = [_safe_float(c) for c in confs]
        if any(v is None for v in vals):
            scores.append(0.0); continue

        if enforce_sum and float(np.sum(vals)) > 1.0:
            scores.append(0.0)
        else:
            scores.append(1.0)

    return scores


def uniqueness_reward(format_pattern, completions, num_candidates=None, **kwargs):
    """
    Returns 1.0 iff all K candidates are distinct (case-insensitive, normalized).
    Single-answer formats trivially return 1.0.
    """
    if not isinstance(num_candidates, int) or num_candidates <= 0:
        return [0.0] * len(completions)

    fmt = str(format_pattern).lower()
    scores = []

    for content in (c[0]["content"] for c in completions):
        if fmt in ("multi_answer", "multi_answer_no_analysis"):
            answers, _ = _extract_flat_candidates(content, num_candidates)
        elif fmt == "multi_answer_rlvr":
            answers = extract_only_answers_rlvr(content, num_candidates)
        else:
            scores.append(1.0); continue  # single-answer: trivially unique

        if answers is None or len(answers) != num_candidates:
            scores.append(0.0); continue

        norm = [a.strip().lower() for a in answers]
        scores.append(1.0 if len(set(norm)) == len(norm) else 0.0)

    return scores


def combined_format_and_constraint_reward(
    completions, format_pattern=None, num_candidates=None, **kwargs
):
    """Returns 1.0 iff both format_reward and response_constraint_reward return 1.0."""
    fmt_scores = format_reward(format_pattern, completions, num_candidates)
    kw = {k: v for k, v in kwargs.items() if k != "num_candidates"}
    con_scores = response_constraint_reward(completions, num_candidates=num_candidates, **kw)
    return [f * c for f, c in zip(fmt_scores, con_scores)]


# ---------------------------------------------------------------------------
# Correctness helper
# ---------------------------------------------------------------------------

def _is_correct(cand: str, gold, source=None) -> bool:
    """
    Return True if cand matches any answer in gold.

    For math datasets (source contains 'math', excluding hotpotQA/ambigQA/medDataset),
    uses math-verify for structural equivalence; falls back to normalized exact match
    only on parse/verify errors. For all other datasets, uses normalized exact match.
    """
    if not cand:
        return False

    gold_list  = list(gold) if isinstance(gold, (list, tuple)) else [gold]
    source_name = (source[0] if isinstance(source, (list, tuple)) else source) if source else None
    use_math = (
        source_name is not None
        and 'math' in str(source_name).lower()
        and source_name not in ('hotpotQA', 'ambigQA', 'medDataset')
    )

    for g in gold_list:
        if use_math:
            try:
                if verify(g, parse(cand)):
                    return True
                # verify returned False — don't fall back to exact match for this gold
            except Exception:
                if exact_match_score(cand, g):
                    return True
        else:
            if exact_match_score(cand, g):
                return True

    return False


# ---------------------------------------------------------------------------
# Accuracy reward
# ---------------------------------------------------------------------------

def accuracy_reward(
    format_pattern, completions, answer=None, num_candidates=None, source=None, **kwargs
):
    """
    Returns 1.0 if any candidate matches a gold answer (binary), or fractional
    credit equal to (# unique correct candidates / K) when more_than_one_correctness_pt=True.

    Requires valid format. For multi-answer RLCR formats, also requires unique candidates
    by default (override with enforceUniqueness=False).

    Accepts both 'answer' and 'answers' dataset columns ('answers' takes priority).
    """
    if "answers" in kwargs:
        answer = kwargs["answers"]
    elif answer is None:
        answer = []

    fmt = str(format_pattern).lower()
    contents = [c[0]["content"] for c in completions]

    # Normalize gold answers to one entry per completion
    if answer is None:
        golds = [[] for _ in completions]
    elif isinstance(answer, (list, tuple)) and len(answer) > 0 and isinstance(answer[0], str):
        # Flat list of valid answers shared across all completions
        golds = [list(answer)] * len(completions)
    else:
        golds = list(answer)
        if len(golds) < len(completions):
            last = golds[-1] if golds else []
            golds += [last] * (len(completions) - len(golds))
        golds = golds[:len(completions)]

    more_than_one    = kwargs.get("more_than_one_correctness_pt", False)
    enforce_uniqueness = kwargs.get(
        "enforceUniqueness", fmt in ("multi_answer", "multi_answer_no_analysis")
    )

    fmt_scores  = format_reward(format_pattern, completions, num_candidates=num_candidates, **kwargs)
    uniq_scores = (
        uniqueness_reward(format_pattern, completions, num_candidates=num_candidates)
        if enforce_uniqueness else [1.0] * len(completions)
    )

    out = []
    for content, gold, fs, us in zip(contents, golds, fmt_scores, uniq_scores):
        if fs == 0 or us == 0:
            out.append(0.0); continue
        try:
            if fmt in ("multi_answer", "multi_answer_no_analysis"):
                answers, _ = _extract_flat_candidates(content, num_candidates)
                candidates  = answers or []
            elif fmt == "multi_answer_rlvr":
                candidates  = extract_only_answers_rlvr(content, num_candidates) or []
            else:
                candidates  = [_extract_last_between(content, "<answer>", "</answer>")]

            if more_than_one:
                n_correct = len({c.strip() for c in candidates if c and _is_correct(c, gold, source)})
                out.append(float(n_correct) / num_candidates)
            else:
                out.append(1.0 if any(_is_correct(c, gold, source) for c in candidates if c) else 0.0)
        except Exception as e:
            print(e)
            out.append(0.0)

    return out


# ---------------------------------------------------------------------------
# Pass@k / num_correct@k metrics
# ---------------------------------------------------------------------------

def _rank_by_confidence(answers, confs):
    """Return answers sorted by their confidence values, highest first."""
    scored = []
    for ans, conf in zip(answers, confs):
        v = _safe_float(conf)
        scored.append((max(0.0, min(1.0, v)) if v is not None else -1.0, ans))
    return [a for _, a in sorted(scored, key=lambda t: t[0], reverse=True)]


def pass_at_1(
    format_pattern, completions, answer=None, num_candidates=None, source=None, **kwargs
):
    """
    Returns 1.0 if the model's primary answer is correct:
    - Single-answer: the answer tag contents.
    - RLVR multi: the first candidate (rank 1 by order).
    - RLCR multi: the highest-confidence candidate.
    """
    if "answers" in kwargs:
        answer = kwargs["answers"]
    elif answer is None:
        answer = []

    fmt        = str(format_pattern).lower()
    K          = num_candidates
    contents   = [c[0]["content"] for c in completions]
    golds      = list(answer)
    fmt_scores = format_reward(format_pattern, completions, num_candidates=K, **kwargs)

    out = []
    for content, gold, fr in zip(contents, golds, fmt_scores):
        if fr == 0:
            out.append(0.0); continue

        if fmt in ("rlvr_single_answer", "rlcr_single_answer"):
            cand = _extract_last_between(content, "<answer>", "</answer>")
        elif fmt == "multi_answer_rlvr":
            cands = extract_only_answers_rlvr(content, K)
            cand  = cands[0] if cands else ""
        elif fmt in ("multi_answer", "multi_answer_no_analysis"):
            answers, confs = _extract_flat_candidates(content, K)
            cand = _rank_by_confidence(answers, confs)[0] if answers and confs else ""
        else:
            cand = ""

        out.append(1.0 if _is_correct(cand, gold, source) else 0.0)

    return out


def pass_at_i(
    format_pattern, completions, answer=None, num_candidates=None, i=None, source=None, **kwargs
):
    """
    Returns 1.0 if any of the top-i candidates is correct:
    - Single-answer: only i=1 is supported.
    - RLVR multi: top-i in sequence order.
    - RLCR multi: top-i by confidence.
    """
    if not isinstance(i, int) or i <= 0:
        raise ValueError("pass_at_i expects i to be a positive integer.")

    if "answers" in kwargs:
        answer = kwargs["answers"]
    elif answer is None:
        answer = []

    fmt = str(format_pattern).lower()

    if fmt in ("rlvr_single_answer", "rlcr_single_answer"):
        if i != 1:
            raise ValueError("pass_at_i for single-answer formats only supports i=1.")
        return pass_at_1(format_pattern, completions, answer, num_candidates, source=source, **kwargs)

    K          = num_candidates
    contents   = [c[0]["content"] for c in completions]
    golds      = list(answer)
    fmt_scores = format_reward(format_pattern, completions, num_candidates=K, **kwargs)

    out = []
    for content, gold, fr in zip(contents, golds, fmt_scores):
        if fr == 0:
            out.append(0.0); continue

        if fmt == "multi_answer_rlvr":
            cands = extract_only_answers_rlvr(content, K) or []
            top   = cands[:i]
        elif fmt in ("multi_answer", "multi_answer_no_analysis"):
            answers, confs = _extract_flat_candidates(content, K)
            top = _rank_by_confidence(answers, confs)[:i] if answers and confs else []
        else:
            top = []

        out.append(1.0 if any(_is_correct(c, gold, source) for c in top) else 0.0)

    return out


def num_correct_at_i(
    format_pattern, completions, answer=None, num_candidates=None, i=None, source=None, **kwargs
):
    """
    Returns the count of correct answers (float) among the top-i candidates:
    - Single-answer: only i=1 is supported.
    - RLVR multi: first i in sequence order.
    - RLCR multi: top i by confidence.
    """
    if not isinstance(i, int) or i <= 0:
        raise ValueError("num_correct_at_i expects i to be a positive integer.")

    if "answers" in kwargs:
        answer = kwargs["answers"]
    elif answer is None:
        answer = []

    fmt = str(format_pattern).lower()

    if fmt in ("rlvr_single_answer", "rlcr_single_answer"):
        if i != 1:
            raise ValueError("num_correct_at_i for single-answer formats only supports i=1.")
        return pass_at_1(format_pattern, completions, answer, num_candidates, source=source, **kwargs)

    K          = num_candidates
    contents   = [c[0]["content"] for c in completions]
    golds      = list(answer)
    fmt_scores = format_reward(format_pattern, completions, num_candidates=K, **kwargs)

    out = []
    for content, gold, fr in zip(contents, golds, fmt_scores):
        if fr == 0:
            out.append(0.0); continue

        if fmt == "multi_answer_rlvr":
            cands = extract_only_answers_rlvr(content, K) or []
            top   = cands[:i]
        elif fmt in ("multi_answer", "multi_answer_no_analysis"):
            answers, confs = _extract_flat_candidates(content, K)
            top = _rank_by_confidence(answers, confs)[:i] if answers and confs else []
        else:
            top = []

        out.append(float(sum(1 for c in top if _is_correct(c, gold, source))))

    return out


# ---------------------------------------------------------------------------
# Brier score reward
# ---------------------------------------------------------------------------

def brier_reward(format_pattern, completions, answer=None, source=None, **kwargs):
    """
    Returns 1 - mean_i((y_i - p_i)^2) per completion, where:
      y_i = 1 if candidate i is correct, else 0
      p_i = model's stated confidence for candidate i

    Requires valid format and constraint (unique answers, sum ≤ 1).
    """
    if "answers" in kwargs:
        answer = kwargs["answers"]
    elif answer is None:
        answer = []

    fmt = str(format_pattern).lower()
    K   = kwargs.get("num_candidates")
    if not isinstance(K, int) or K <= 0:
        return [0.0] * len(completions)

    contents = [c[0]["content"] for c in completions]
    golds    = list(answer)
    kw       = {k: v for k, v in kwargs.items() if k != "num_candidates"}

    fmt_scores = format_reward(format_pattern, completions, num_candidates=K, **kw)
    con_scores = response_constraint_reward(completions, num_candidates=K, **kw)

    rewards = []
    for content, gold, fs, cs in zip(contents, golds, fmt_scores, con_scores):
        if fs == 0 or cs == 0:
            rewards.append(0.0); continue

        is_single = fmt in ("rlcr_single_answer", "rlvr_single_answer", "ta", "tac", "tabc", "tbac") or K == 1
        answers, confs = (
            _extract_single_candidate(content) if is_single
            else _extract_flat_candidates(content, K)
        )

        if answers is None:
            rewards.append(0.0); continue

        sq_errs = []
        for cand, conf in zip(answers, confs):
            p = _safe_float(conf)
            p = max(0.0, min(1.0, p)) if p is not None else 0.0
            y = float(_is_correct(cand, gold, source))
            sq_errs.append((y - p) ** 2)

        rewards.append(1.0 - (sum(sq_errs) / len(sq_errs)))

    return rewards


# ---------------------------------------------------------------------------
# Entropy reward (token-level, from precomputed logits)
# ---------------------------------------------------------------------------

def entropy_from_logits(logits: "torch.Tensor", chunk_size: int = 128) -> "torch.Tensor":
    """
    Compute per-token Shannon entropy (nats) from logits, memory-efficiently.
    Matches the entropy implementation in trainer_utils.py used for wandb logging.
    """
    original_shape = logits.shape[:-1]
    flat = logits.reshape(-1, logits.shape[-1])
    chunks = []
    for chunk in flat.split(chunk_size):
        logps = F.log_softmax(chunk, dim=-1)
        chunks.append(-(torch.exp(logps) * logps).sum(-1))
    return torch.cat(chunks).reshape(original_shape)


def entropy_reward(format_pattern, completions, **kwargs):
    """
    Returns per-completion mean token entropy, normalized to [0, 1] by log(vocab_size),
    with an optional linear decay applied over training steps.

    Higher entropy → higher reward (encourages diverse/exploratory generation).

    Requires kwarg:
        precomputed_entropies (torch.Tensor [n, seq_len]): entropy per token per completion,
            computed during the training forward pass to avoid redundant computation.

    Optional kwargs:
        completion_mask (torch.Tensor [n, seq_len]): masks padding tokens.
        tokenizer: used to get vocab_size (defaults to Qwen3's 151936).
        trainer_state: provides global_step and max_steps for decay scheduling.
        entropy_decay_start_step (int, default=200): step when decay begins.
        entropy_decay_final_factor (float, default=0.0): multiplier at max_steps.
    """
    if torch is None:
        warnings.warn("Entropy reward: torch not available. Returning zeros.")
        return [0.0] * len(completions)

    precomputed = kwargs.get("precomputed_entropies")
    if precomputed is None:
        warnings.warn("Entropy reward: precomputed_entropies not provided. Returning zeros.")
        return [0.0] * len(completions)

    try:
        if not isinstance(precomputed, torch.Tensor):
            precomputed = torch.tensor(precomputed)

        # Compute mean entropy per completion, respecting the completion mask
        mask = kwargs.get("completion_mask")
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask, device=precomputed.device)
            mask = mask.to(precomputed.device)
            # Align seq_len if the trainer and reward fn see different padding
            if precomputed.shape[1] != mask.shape[1]:
                min_len = min(precomputed.shape[1], mask.shape[1])
                precomputed, mask = precomputed[:, :min_len], mask[:, :min_len]
            mean_ent = (precomputed * mask.float()).sum(1) / mask.sum(1).clamp(min=1.0)
        else:
            mean_ent = precomputed.mean(dim=1)

        # Normalize to [0, 1] by dividing by maximum possible entropy = log(vocab_size)
        tokenizer  = kwargs.get("tokenizer")
        vocab_size = getattr(tokenizer, "vocab_size", 151936) if tokenizer else 151936
        normalized = (mean_ent / math.log(vocab_size)).clamp(0.0, 1.0)

        # Linear decay: full reward before decay_start_step, then linear ramp to final_factor
        decay = 1.0
        state = kwargs.get("trainer_state")
        if state is not None and hasattr(state, "global_step"):
            step      = state.global_step
            max_steps = (
                getattr(state, "max_steps", None)
                or getattr(getattr(state, "args", None), "max_steps", None)
                or max(1000, step * 10)
            )
            start = kwargs.get("entropy_decay_start_step", 200)
            final = kwargs.get("entropy_decay_final_factor", 0.0)
            if step >= start:
                progress = min(1.0, (step - start) / max(1, max_steps - start))
                decay    = 1.0 - (1.0 - final) * progress

        normalized = normalized * decay

        # Align length to number of completions (guard against shape mismatches)
        n = len(completions)
        if len(normalized) != n:
            warnings.warn(f"Entropy reward: shape mismatch ({len(normalized)} vs {n}). Adjusting.")
            if len(normalized) > n:
                normalized = normalized[:n]
            else:
                pad = torch.zeros(n - len(normalized), device=normalized.device)
                normalized = torch.cat([normalized, pad])

        return normalized.cpu().tolist()

    except Exception as e:
        warnings.warn(f"Entropy reward: failed ({e}). Returning zeros.")
        return [0.0] * len(completions)


# ---------------------------------------------------------------------------
# Confidence analysis rewards (diagnostic / logging)
# ---------------------------------------------------------------------------

def mean_confidence_reward(completions, **kwargs):
    """Returns the model's last stated confidence value clipped to [0, 1], or 0.0 if absent."""
    out = []
    for completion in completions:
        matches = re.findall(r"<confidence>(.*?)</confidence>", completion[0]["content"], re.DOTALL)
        v = _safe_float(matches[-1]) if matches else None
        out.append(max(0.0, min(1.0, v)) if v is not None else 0.0)
    return out


def confidence_one_or_zero(completions, **kwargs):
    """Returns 1.0 if the model's last confidence is essentially 0 or 1 (within 0.01), else 0.0."""
    out = []
    for completion in completions:
        matches = re.findall(r"<confidence>(.*?)</confidence>", completion[0]["content"], re.DOTALL)
        v = _safe_float(matches[-1]) if matches else None
        if v is not None:
            v = max(0.0, min(1.0, v))
            out.append(1.0 if abs(v) < 0.01 or abs(v - 1.0) < 0.01 else 0.0)
        else:
            out.append(0.0)
    return out
