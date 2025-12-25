from dataclasses import dataclass, field
from typing import Optional, List
from typing import Dict

@dataclass
class GlobalArgs:
    dataset_name: str = field(default=None,metadata={"help": "Name of the dataset to load"})
    dataset_config: Optional[str] = field(default=None, metadata={"help": "Specific configuration for the dataset, if applicable"})
    split: str = field(default="test",metadata={"help": "Dataset split to use (e.g., train, test, validation)"})
    hash_key: str = field(default="prompt",metadata={"help": "Column key to generate a unique hash ID for each example"})
    gpu_memory_utilization: float = field(default=0.9,metadata={"help": "Fraction of GPU memory to utilize"})
    store_name: str = field(default=None,metadata={"help": "Name of the repository to push the final dataset"})
    log_path: str = field(default=None,metadata={"help": "Path to the log file"})
    sample_size: int = field(default=None,metadata={"help": "Number of samples to use for evaluation"}) 
    fresh: bool = field(default=False, metadata={"help": "Whether to overwrite existing results"})

@dataclass
class LocalConfig:
    name: str = field(metadata={"help": "Name of the local configuration, used for naming outputs"})
    model: str = field(metadata={"help": "Model identifier to use for generation"})
    tokenize_key: str = field(default="prompt",metadata={"help": "Key in the dataset to tokenize before feeding into the model"})
    n: int = field(default=1,metadata={"help": "Number of samples to generate per input"})
    temperature: float = field(default=0,metadata={"help": "Sampling temperature for generation"})
    max_tokens: int = field(default=4096,metadata={"help": "Maximum number of tokens to generate per sample"})
    seed: Optional[int] = field(default=42, metadata={"help": "Random seed for reproducibility"})
    check_fn: str = field(default=None, metadata={"help": "Function to check the dataset post-processing"})
    check_fn_args: Dict = field(default_factory=dict, metadata={"help": "Arguments to pass to the check function"})
    task_spec: str = field(default="gen", metadata={"help": "Task-specific prompt to use for generation"})
    sys_prompt_name: str = field(default="ver", metadata={"help": "System prompt name."})
    pass_k_vals: List = field(default_factory=list, metadata={"help": "List of k values to pass to the check function"}) 
    correctness_fn: str = field(default=None, metadata={"help": "Correctness function to use for the check function"})
    vllm_task: List[str] = field(default_factory=lambda: ["generate"], metadata={"help": "List of tasks to use for vLLM"})
    class_model: str = field(default=None, metadata={"help": "Model to use for classification if using a hybrid task"})
    fresh: bool = field(default=False, metadata={"help": "Whether to overwrite existing results"})
    use_hf: bool = field(default=False, metadata={"help": "Whether to use HF for classification"}) 
    split_at_confidence: bool = field(default=False, metadata={"help": "Whether to split at confidence"})
    num_candidates: int = field(default=1, metadata={"help": "Number of candidates to generate"})