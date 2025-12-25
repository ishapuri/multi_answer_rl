
import hashlib  
import numpy as np 
import os
from concurrent.futures import ThreadPoolExecutor
import concurrent
import json 
from openai import OpenAI
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import re
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

def string_to_short_id(string):
    # Use SHA-256 hash function
    hash_object = hashlib.sha256(string.encode())
    # Convert the hash to a hexadecimal string
    hex_dig = hash_object.hexdigest()
    # Convert the hexadecimal string to an integer
    hash_int = int(hex_dig, 16)
    # Truncate to 8 digits by taking modulo 10^8
    hash_id = hash_int % 10**10
    return hash_id

def hash_dataset(example,key):
    return {"id": string_to_short_id(example[key])} 

def compute_pass_n(evals,k):
    n = len(evals[0])  
    corrects,totals = [],[] 
    for i in range(len(evals)):
        eval_list = evals[i]
        #count number of 1s in eval_list
        count = 0
        for j in range(n):
            if eval_list[j] == 1:
                count += 1
        corrects.append(count)
        totals.append(n)
    return estimate_pass_at_k(totals,corrects,k).mean()

def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
    import itertools
    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def get_brier(correctness,confidence):
    brier_score = np.mean((confidence - correctness) ** 2)
    return brier_score

def get_ece(correctness,confidence):
    # Calculate ECE using 10 bins, including 0 and 1
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    # Ensure 0 and 1 are included in their own bins
    bin_edges[0] = -np.inf  # Include 0 in first bin
    bin_edges[-1] = np.inf  # Include 1 in last bin
    bin_indices = np.digitize(confidence, bin_edges) - 1
    
    ece = 0
    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_conf = np.mean(confidence[mask])
            bin_acc = np.mean(correctness[mask])
            bin_weight = np.sum(mask) / len(confidence)
            ece += bin_weight * np.abs(bin_conf - bin_acc)
    return ece

def get_auroc(correctness,confidence):
    fpr, tpr, _ = roc_curve(correctness, confidence)
    auroc = auc(fpr, tpr)
    return auroc