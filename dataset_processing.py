from datasets import load_dataset
from system_prompts import get_sys_prompt, get_sys_prompt_formatted
import numpy as np 

def process_dataset(dataset, script_args):
    #sys_prompt = get_sys_prompt(script_args.sys_prompt_name) 
    # print("script_args: ", script_args)
    # print("script_args.__dict__: ", script_args.__dict__)
    # print(vars(script_args))

    num_candidates = script_args.num_candidates if "num_candidates" in script_args.__dict__.keys() else 1
    sys_prompt = get_sys_prompt_formatted(script_args.sys_prompt_name, num_candidates)


    if script_args.task_spec == "gen":
        dataset = make_generation_dataset(dataset, sys_prompt)

    return dataset

def make_generation_dataset(dataset,sys_prompt):
    def make_generation_conversation(example):
        if 'question' in example.keys():
            user_format = (
                f"\n\nPROBLEM: {example['question']}\n\n"
                )
        elif 'problem' in example.keys():
            user_format = (
                    f"\n\nPROBLEM: {example['problem']}\n\n"
                    )
        else:
            user_format = (
                f"\n\nWRITING PROMPT: {example['prompt']}\n\n"
                )
        result = {
            "prompt": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_format},
            ],
        }
        # Preserve answer/answers field if it exists
        if "answer" in example:
            result["answer"] = example["answer"]
        if "answers" in example:
            result["answers"] = example["answers"]
        return result
    
    # For DatasetDict, we need to map over each split individually
    # The mapping function preserves answer/answers fields, so columns are preserved by default
    if hasattr(dataset, 'keys'):  # It's a DatasetDict
        from datasets import DatasetDict
        mapped_splits = {}
        for split_name in dataset.keys():
            mapped_splits[split_name] = dataset[split_name].map(make_generation_conversation)
        dataset = DatasetDict(mapped_splits)
    else:  # It's a single Dataset
        dataset = dataset.map(make_generation_conversation)
    return dataset


def sft_dataset_process(dataset,script_args,sys_prompt=None): 
 
    if script_args.dataset_name == "mehuldamani/deepseek-verifier-v1": 
        def mapping(example):
            user_format = (
                f"\n\nPROBLEM: {example['problem']}\n\n"
                    )
            ans_format = (
                f"{example['output_0']}\n"
                f"{example['demo']}\n"
            )
            messages = [ {"role":"system","content":sys_prompt}, {"role":"user","content":user_format}, {"role":"assistant","content":ans_format}]
            return {"messages": messages}
            
        dataset = dataset.map(mapping,remove_columns=["prompt"])


    return dataset


