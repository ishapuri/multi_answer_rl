from dataclasses import dataclass, field
import trl 
from trl import ScriptArguments 
from typing import Optional, Any
from transformers import TrainingArguments
from trl import SFTConfig

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty'.
    """

    dataset_name: str = field(metadata={"help": "Dataset name."})
    dataset_config: Optional[str] = field(
        default=None,
        metadata={
            "help": "Dataset configuration name. Corresponds to the `name` argument of the `datasets.load_dataset` "
            "function."
        },
    )
    dataset_train_split: str = field(default="train", metadata={"help": "Dataset split to use for training."})
    dataset_test_split: str = field(default="test", metadata={"help": "Dataset split to use for evaluation."})

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty'"
        },
    )

    sys_prompt_name: str = field(default="ver", metadata={"help": "System prompt name."})
    task_spec: str = field(default="gen", metadata={"help": "Task specification."})
    set_pad_token: Optional[int] = field(default=None, metadata={"help": "Set the pad token to this id"})

    format_pattern: Optional[str] = field(
        default="ta",
        metadata={"help": "The format pattern to use for the reward function."},
    )

    num_candidates: Optional[int] = field(
        default=None,
        metadata={"help": "Number of candidate answers to generate for each input."},
    )

    eval_sample_size: Optional[int] = field(default=100, metadata={"help": "Number of samples to use for evaluation."})



@dataclass
class GRPOConfig(trl.GRPOConfig):
    """
    args for callbacks, benchmarks etc
    """

    _VALID_DICT_FIELDS = TrainingArguments._VALID_DICT_FIELDS + ["model_init_kwargs"]

    run_name: Optional[str] = field(
        default=None,
        metadata={"help": "An optional descriptor for the run. Notably used for wandb, mlflow and comet logging."},
    )

    # Parameters whose default values are overridden from TrainingArguments
    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "The initial learning rate for AdamW."},
    )
    logging_steps: float = field(
        default=10,
        metadata={
            "help": "Log every X updates steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, "
            "will be interpreted as ratio of total training steps."
        },
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA "
            "architecture or Intel XPU or using CPU (use_cpu) or Ascend NPU. If not set, it defaults to `True` if "
            "`fp16` is not set."
        },
    )
    max_prompt_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left."
        },
    )
    num_generations: Optional[int] = field(
        default=8,
        metadata={
            "help": "Number of generations to sample. The effective batch size (num_processes * per_device_batch_size "
            "* gradient_accumulation_steps) must be evenly divisible by this value."
        },
    )
    max_completion_length: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum length of the generated completion."},
    )
   
    shuffle_dataset: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to shuffle the training dataset."},
    )

    # Parameters that control generation
    generation_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Batch size to use for generation. If `None`, it defaults to the effective training batch size: "
            "`per_device_train_batch_size * num_processes * steps_per_generation`."
        },
    )
    steps_per_generation: Optional[int] = field(
        default=None,
        metadata={"help": "Number of steps per generation. If `None`, it defaults to `gradient_accumulation_steps`."},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for sampling. The higher the temperature, the more random the completions."},
    )
    top_p: float = field(
        default=1.0,
        metadata={
            "help": "Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. "
            "Set to 1.0 to consider all tokens."
        },
    )
    top_k: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of highest probability vocabulary tokens to keep for top-k-filtering. If `None`, "
            "top-k-filtering is disabled and all tokens are considered."
        },
    )
    min_p: Optional[float] = field(
        default=None,
        metadata={
            "help": "Minimum token probability, which will be scaled by the probability of the most likely token. It "
            "must be a value between 0.0 and 1.0. Typical values are in the 0.01-0.2 range."
        },
    )
  
    # Parameters that control generation acceleration powered by vLLM
    use_vllm: bool = field(
        default=False,
        metadata={
            "help": "Whether to use vLLM for generating completions. If set to `True`, the trainer will use vLLM for "
            "generation instead of the default model.generate(). Requires `vllm` to be installed."
        },
    )
    vllm_mode: str = field(
        default="colocate",
        metadata={
            "help": "Mode to use for vLLM integration when `use_vllm` is set to `True`. Must be one of `'server'` or "
            "`'colocate'`. `'server'`: The trainer will send generation requests to a separate vLLM server. Make sure "
            "a TRL vLLM server is running (start with `trl vllm-serve`). `'colocate'`: vLLM will run in the same "
            "process and share the training GPUs. This avoids the need for a separate server but may cause resource "
            "contention with training."
        },
    )
    vllm_enable_sleep_mode: bool = field(
        default=True,
        metadata={
            "help": "Whether to enable sleep mode for vLLM. If `True`, vLLM will sleep during the optimization step "
            "and woken for weight sync and generation."
        },
    )
    # Parameters that control colocated vLLM execution (only used when `vllm_mode` is `"colocate"`)
    vllm_gpu_memory_utilization: float = field(
        default=0.3,
        metadata={
            "help": "Control the GPU memory utilization for vLLM. This setting only applies when `vllm_mode` is set "
            "to `'colocate'`. If you are using `vllm_mode='server'`, this parameter must be passed separately when "
            "launching the vLLM server via the `--vllm_gpu_memory_utilization` flag."
        },
    )
    vllm_tensor_parallel_size: int = field(
        default=1,
        metadata={
            "help": "Control the tensor parallel size for vLLM. This setting only applies when `vllm_mode` is set "
            "to `'colocate'`. If you are using `vllm_mode='server'`, this parameter must be passed separately when "
            "launching the vLLM server via the `--vllm_tensor_parallel_size` flag."
        },
    )

    # Parameters that control the training
    beta: float = field(
        default=0.0,
        metadata={
            "help": "KL coefficient. If `0.0` (default), the reference model is not loaded, reducing memory usage and "
            "improving training speed."
        },
    )
    num_iterations: int = field(
        default=1,
        metadata={"help": "Number of iterations per batch (denoted as μ in the algorithm)."},
    )
    epsilon: float = field(
        default=0.2,
        metadata={"help": "Epsilon value for clipping."},
    )
    delta: Optional[float] = field(
        default=None,
        metadata={
            "help": "Enables the upper clipping bound in two-sided GRPO loss when set to a float. If `None` "
            "(default), standard GRPO clipping is used. Recommended to be greater than `1 + ε` when enabled. This "
            "method is introduced in the [INTELLECT-2 tech report](https://huggingface.co/papers/2505.07291)."
        },
    )
    epsilon_high: Optional[float] = field(
        default=None,
        metadata={
            "help": "Upper-bound epsilon value for clipping. If not specified, it defaults to the same value as the "
            "lower-bound specified in argument `epsilon`. Paper DAPO recommends `0.28`."
        },
    )
    importance_sampling_level: str = field(
        default="token",
        metadata={
            "help": "Controls whether importance sampling ratios are computed at the `'token'` or `'sequence'` level. "
            "`'token'` keeps the raw per-token log-probability ratios (one weight per token).  `'sequence'` averages "
            "the log-probability ratios across valid tokens to produce a single ratio per sequence. The GSPO paper "
            "shows that sequence-level sampling often yields more stable training and better alignment with "
            "sequence-level rewards."
        },
    )
    reward_weights: Optional[list[float]] = field(
        default=None,
        metadata={
            "help": "Weights for each reward function. Must match the number of reward functions. If `None`, all "
            "rewards are weighted equally with weight `1.0`."
        },
    )
    scale_rewards: str = field(
        default="none",
        metadata={
            "help": "Specifies the scaling strategy for rewards. Supported values are: "
            "`True` or `group'` (default): rewards are scaled by the standard deviation within each group, ensuring "
            "unit variance within a group. "
            "`'batch'`: rewards are scaled by the standard deviation across the entire batch, as recommended in the "
            "PPO Lite paper. "
            "`False` or `'none'`: no scaling is applied. The Dr. GRPO paper recommends not scaling rewards, as "
            "scaling by the standard deviation introduces a question-level difficulty bias."
        },
    )

    repetition_penalty: float = field(
        default=1.0,
        metadata={
            "help": "Float that penalizes new tokens based on whether they appear in the prompt and the generated "
            "text so far. Values > 1.0 encourage the model to use new tokens, while values < 1.0 encourage the model "
            "to repeat tokens."
        },
    )
    
    loss_type: str = field(
        default="dapo",
        metadata={
            "help": "Specifies the loss formulation to use. Supported values are 'grpo', 'dapo', 'bnpo', and "
            "'dr_grpo'. "
            "'grpo': Aggregates token-level losses by normalizing over sequence length. Not recommended due to length "
            "bias—this approach tends to prefer shorter completions with positive advantages and longer ones with "
            "negative advantages. "
            "'dapo' (default): Aggregates token-level losses by normalizing with the number of active token in the "
            "global accumulated batch. This method was introduced in the DAPO paper to eliminate length bias. "
            "'dr_grpo': Aggregates token-level losses by normalizing with a global constant. This method was "
            "introduced in the Dr. GRPO paper to eliminate length bias. The value of the constant corresponds to "
            "`max_completion_length`. "
            "'bnpo': Aggregates token-level losses by normalizing with the number of active token in the local batch. "
            "Note that normalization is performed over the local batch only, so results may slightly vary depending "
            "on the local batch size, despite a constant effective batch size. When using "
            "`per_device_train_batch_size==1`, the loss is equivalent to the GRPO loss."
        },
    )
    mask_truncated_completions: bool = field(
        default=False,
        metadata={
            "help": "When enabled, truncated completions are excluded from the loss calculation, preventing them from "
            "being incorrectly penalized and introducing noise during training. According to the DAPO paper, this is "
            "a good practice for training stability."
        },
    )
    sync_ref_model: bool = field(
        default=False,
        metadata={
            "help": "Whether to synchronize the reference model with the active model every `ref_model_sync_steps` "
            "steps, using the `ref_model_mixup_alpha` parameter."
        },
    )
    ref_model_mixup_alpha: float = field(
        default=0.6,
        metadata={
            "help": "α parameter from the TR-DPO paper, which controls the mix between the current policy and the "
            "previous reference policy during updates. The reference policy is updated according to the equation: "
            "`π_ref = α * π_θ + (1 - α) * π_ref_prev`. To use this parameter, you must set `sync_ref_model=True`."
        },
    )
    ref_model_sync_steps: int = field(
        default=512,
        metadata={
            "help": "τ parameter from the TR-DPO paper, which determines how frequently the current policy is "
            "synchronized with the reference policy. To use this parameter, you must set `sync_ref_model=True`."
        },
    )
    top_entropy_quantile: float = field(
        default=1.0,
        metadata={
            "help": "ρ parameter from Beyond the 80/20 Rule. Keeps in the policy loss term only the top-ρ quantile of "
            "tokens by entropy of the probability distribution at each sequence position, improving results. Range: "
            "[0.0-1.0]. A value of `0.0` masks all but the highest entropy token; `1.0` keeps all tokens. The paper "
            "recommends a value of `0.2`. If used with `mask_truncated_completions=True`, only tokens from "
            "non-truncated completions are considered."
        },
    )
    use_liger_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use the Liger GRPO loss."},
    )
    vllm_importance_sampling_correction: bool = field(
        default=True,
        metadata={
            "help": "Whether to apply Truncated Importance Sampling (TIS) between vLLM completion logprobs and "
            "recomputed logprobs. Your Efficient RL Framework Secretly Brings You Off-Policy RL "
            "Training highlights that using a separate generation framework (such as vLLM) can introduce off-policy "
            "effects due to subtle implementation differences between generation and training backends. TIS is "
            "proposed as a remedy for this issue."
        },
    )
    vllm_importance_sampling_cap: float = field(
        default=2.0,
        metadata={
            "help": "Truncation parameter C for Truncated Importance Sampling (TIS). This sets an upper bound on the "
            "importance sampling ratio, improving training stability."
        },
    )

    # Parameters that control the logging
    log_completions: bool = field(
        default=False,
        metadata={
            "help": "Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps. If `rich` is "
            "installed, it prints the sample. If `wandb` logging is enabled, it logs it to `wandb`."
        },
    )
    num_completions_to_print: Optional[int] = field(
        default=None,
        metadata={"help": "Number of completions to print with `rich`. If `None`, all completions are logged."},
    )

    num_completions_to_log : Optional[int] = field(
        default=5,
        metadata={"help": "Number of completions to log per step. If `None`, all completions are logged."},
    )
    wandb_log_unique_prompts: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to log unique prompts in wandb. If `True`, only unique prompts are logged. If `False`, "
            "all prompts are logged."
        },
    )

    wandb_project: Optional[str] = field(
        default="internalized_inf_scaling",
        metadata={"help": ("The project to store runs under.")},
    )


    reward_model: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the reward model."},
    )


@dataclass
class ModelConfig:
    """
    Configuration class for the models.

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        model_name_or_path (`str` or `None`, *optional*, defaults to `None`):
            Model checkpoint for weights initialization.
        model_revision (`str`, *optional*, defaults to `"main"`):
            Specific model version to use. It can be a branch name, a tag name, or a commit id.
        torch_dtype (`Literal["auto", "bfloat16", "float16", "float32"]` or `None`, *optional*, defaults to `None`):
            Override the default `torch.dtype` and load the model under this dtype. Possible values are

                - `"bfloat16"`: `torch.bfloat16`
                - `"float16"`: `torch.float16`
                - `"float32"`: `torch.float32`
                - `"auto"`: Automatically derive the dtype from the model's weights.

        trust_remote_code (`bool`, *optional*, defaults to `False`):
            Whether to allow for custom models defined on the Hub in their own modeling files. This option should only
            be set to `True` for repositories you trust and in which you have read the code, as it will execute code
            present on the Hub on your local machine.
        attn_implementation (`str` or `None`, *optional*, defaults to `None`):
            Which attention implementation to use. You can run `--attn_implementation=flash_attention_2`, in which case
            you must install this manually by running `pip install flash-attn --no-build-isolation`.
        use_peft (`bool`, *optional*, defaults to `False`):
            Whether to use PEFT for training.
        lora_r (`int`, *optional*, defaults to `16`):
            LoRA R value.
        lora_alpha (`int`, *optional*, defaults to `32`):
            LoRA alpha.
        lora_dropout (`float`, *optional*, defaults to `0.05`):
            LoRA dropout.
        lora_target_modules (`Union[str, list[str]]` or `None`, *optional*, defaults to `None`):
            LoRA target modules.
        lora_modules_to_save (`list[str]` or `None`, *optional*, defaults to `None`):
            Model layers to unfreeze & train.
        lora_task_type (`str`, *optional*, defaults to `"CAUSAL_LM"`):
            Task type to pass for LoRA (use `"SEQ_CLS"` for reward modeling).
        use_rslora (`bool`, *optional*, defaults to `False`):
            Whether to use Rank-Stabilized LoRA, which sets the adapter scaling factor to `lora_alpha/√r`, instead of
            the original default value of `lora_alpha/r`.
        load_in_8bit (`bool`, *optional*, defaults to `False`):
            Whether to use 8 bit precision for the base model. Works only with LoRA.
        load_in_4bit (`bool`, *optional*, defaults to `False`):
            Whether to use 4 bit precision for the base model. Works only with LoRA.
        bnb_4bit_quant_type (`str`, *optional*, defaults to `"nf4"`):
            Quantization type (`"fp4"` or `"nf4"`).
        use_bnb_nested_quant (`bool`, *optional*, defaults to `False`):
            Whether to use nested quantization.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Model checkpoint for weights initialization."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "Specific model version to use. It can be a branch name, a tag name, or a commit id."},
    )
    torch_dtype: Optional[str] = field(
        default="auto",
        metadata={
            "help": "Override the default `torch.dtype` and load the model under this dtype.",
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={
            "help": "Whether to allow for custom models defined on the Hub in their own modeling files. This option "
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        },
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which attention implementation to use. You can run `--attn_implementation=flash_attention_2`, in "
            "which case you must install this manually by running `pip install flash-attn --no-build-isolation`."
        },
    )
    use_peft: bool = field(
        default=False,
        metadata={"help": "Whether to use PEFT for training."},
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA R value."},
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha."},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout."},
    )
    lora_target_modules: Optional[list[str]] = field(
        default=None,
        metadata={"help": "LoRA target modules."},
    )
    lora_modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={"help": "Model layers to unfreeze & train."},
    )
    lora_task_type: str = field(
        default="CAUSAL_LM",
        metadata={"help": "Task type to pass for LoRA (use 'SEQ_CLS' for reward modeling)."},
    )
    use_rslora: bool = field(
        default=False,
        metadata={
            "help": "Whether to use Rank-Stabilized LoRA, which sets the adapter scaling factor to `lora_alpha/√r`, "
            "instead of the original default value of `lora_alpha/r`."
        },
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Whether to use 8 bit precision for the base model. Works only with LoRA."},
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Whether to use 4 bit precision for the base model. Works only with LoRA."},
    )
    bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization type.", "choices": ["fp4", "nf4"]},
    )
    use_bnb_nested_quant: bool = field(
        default=False,
        metadata={"help": "Whether to use nested quantization."},
    )

    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("You can't use 8 bit and 4 bit precision at the same time")

        if hasattr(self.lora_target_modules, "__len__") and len(self.lora_target_modules) == 1:
            self.lora_target_modules = self.lora_target_modules[0]






