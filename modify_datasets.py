# from datasets import load_dataset, Dataset, DatasetDict
# import re
# from tqdm import tqdm

# def main():
#     # Variables for substring to replace and its replacement
#     target_substring = "Your answer will be verified with exact match score. To ensure correct verification, only provide the answer within the <answer> </answer> tags. Do not put any sentences or reasoning process within the <answer> </answer> tags."
#     replacement_substring = "Your answer will be verified with exact match score. To ensure correct verification, only provide the answer within the <answer{i}> </answer{i}> tags. Do not put any sentences or reasoning process within the <answer{i}> </answer{i}> tags."

#     # Load the original dataset
#     orig_dataset = load_dataset("mehuldamani/hotpot_qa")

#     # Only process "train" and "test" splits, ignore others if present
#     processed_splits = {}
#     for split in ["train", "test"]:
#         if split in orig_dataset:
#             orig_split = orig_dataset[split]
#             # Process each example with tqdm progress bar
#             new_examples = []
#             for ex in tqdm(orig_split, desc=f"Processing {split} split"):
#                 new_ex = dict(ex)
#                 if "problem" in new_ex and isinstance(new_ex["problem"], str):
#                     # Replace target_substring with replacement_substring
#                     new_ex["problem"] = re.sub(re.escape(target_substring), replacement_substring, new_ex["problem"])
#                 new_examples.append(new_ex)
#             # Create a new Dataset for this split
#             processed_splits[split] = Dataset.from_list(new_examples)

#     if not processed_splits:
#         raise ValueError("Neither 'train' nor 'test' split found in the input dataset.")

#     # Assemble into a DatasetDict
#     new_dataset = DatasetDict(processed_splits)
#     # Push to Hugging Face Hub
#     repo_id = "mehuldamani/hotpot_qa_for_multi"
#     new_dataset.push_to_hub(repo_id)
#     print(f"Processed dataset pushed to HF hub as: {repo_id}")

# if __name__ == "__main__":
#     main()


from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm

def main():
    # Load the original dataset
    orig_dataset = load_dataset("mehuldamani/ambigQA")
    
    processed_splits = {}
    for split in orig_dataset.keys():
        orig_split = orig_dataset[split]
        new_examples = []
        for ex in tqdm(orig_split, desc=f"Processing {split} split"):
            new_ex = dict(ex)
            new_ex["source"] = "ambigQA"
            new_examples.append(new_ex)
        processed_splits[split] = Dataset.from_list(new_examples)
    
    new_dataset = DatasetDict(processed_splits)
    # Push to Hugging Face Hub
    repo_id = "mehuldamani/ambigQA"
    new_dataset.push_to_hub(repo_id)
    print(f"Processed dataset with 'source' column pushed to HF hub as: {repo_id}")

if __name__ == "__main__":
    main()

