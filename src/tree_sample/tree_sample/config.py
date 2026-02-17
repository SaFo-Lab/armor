# adapted from https://github.com/thu-ml/STAIR

from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class TSBaseConfig:

    generate_samples_number_list: Optional[List[int]] = field(
        default=None
    )
    actor_model_dir: Optional[str] = field(
        default=None, metadata={"help": "Path to actor model dir"}
    )
    
    actor_model_dir: Optional[str] = field(
        default=None, metadata={"help": "Path to actor model dir"}
    )

    worker_num: int = field(
        default=1, metadata={"help": "Should be 1 when local, >=1 when vllm"}
    )
    worker_prompt_num: int = field(
        default=10, metadata={"help": "Prompt number for each worker"}
    )

    temperature: float = field(
        default=1.2, metadata={"help": "Control diversity. Higher for more diversity."}
    )
    top_p: float = field(
        default=0.9, metadata={"help": "Control diversity. Higher for more diversity."}
    )
    top_k: int = field(
        default=50, metadata={"help": "Control diversity. Higher for more diversity."}
    )
    max_tokens: int = field(
        default=1024, metadata={"help": "Max tokens for each step."}
    )
    seed: int = field(
        default=42, metadata={"help": "Random seed"}
    )

    stop_tokens: Optional[List[str]] = field(
        default=None, metadata={"help": "Stop tokens for a step"}
    )
    end_tokens: Optional[List[str]] = field(
        default=None, metadata={"help": "End tokens for complete response"}
    )

    train_prompt_path: Optional[str] = field(
        default=None, metadata={"help": "Path to training prompt file"}
    )
    output_path: Optional[str] = field(
        default=None, metadata={"help": "Path to output mct result dir"}
    )

    max_depth: int = field(
        default=7, metadata={"help": "Hyperparam for max_depth of tree"}
    )

    generate_samples_number: int = field(
        default=4, metadata={"help": "Numbers of expanded children nodes in expand phase"}
    )


    use_cache: bool = field(
        default=False, metadata={"help": "Whether to use a cache when generating tree"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Path to cache dir"}
    )
    log_file: Optional[str] = field(
        default=None, metadata={"help": "Path to output log"}
    )

    