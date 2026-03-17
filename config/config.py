from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class AdapterConfig:
    # Adapter配置
    adapter_dim: int = 32
    dropout: float = 0.1
    enabled: bool = True  # 是否启用Adapter

@dataclass
class PeftConfig:
    r: int = 16
    lora_alpha: int = 32
    target_modules: List = field(default_factory=lambda: [ "q_proj", "v_proj", "o_proj", "up_proj","gate_proj","down_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    lora_dropout: float = 0.05
    inference_mode: bool = False

@dataclass
class TrainConfig:
    train_strategy : str = "ddp"
    deepspeed_config: str = "./config/deepspeed.json"
    seed : int = 1234
    lr: float = 1e-4
    batch_size: int = 1
    total_train_steps: int = 100000
    grad_accumulate_step : int = 5
    eval_step: int = 10
    train_epoch: int = 20
    warmup_steps: int = 1000
    use_bfloat16: bool = True

@dataclass
class EvalConfig:
    batch_size: int = 2
    local_rank : int = 0
    peft_path: str = ""

@dataclass
class DataConfig:
    train_data_path: str = "Qwen2-Audio-finetune/data/daic_woz/train"
    eval_data_path: str = "Qwen2-Audio-finetune/data/daic_woz/val"
    train_prompt_path: str = "Qwen2-Audio-finetune/data/daic_woz/train/daic_woz_multiprompt.jsonl"
    val_prompt_path: str = "Qwen2-Audio-finetune/data/daic_woz/val/daic_woz_multiprompt.jsonl"
    train_scp_filename: str = "daic_woz.scp"
    eval_scp_filename: str = "daic_woz.scp"
    train_task_filename: str = "daic_woz_multitask.jsonl"
    eval_task_filename: str = "daic_woz_multitask.jsonl"
    wav_type: str = "wav"
    num_workers: int = 4
    prefetch_factor: int  = 4

@dataclass
class EnvConfig:
    device_type: str = "cuda" # npu gpu
    save_path: str = "Qwen2-Audio-finetune/output_model"
    model_path: str = "Qwen2-Audio-7B-Instruct"

@dataclass
class WandBConfig:
    enabled: bool = False
    project: str = "qwen2-audio-finetune"
    entity: str = ""
    run_name: str = ""
    group: str = ""
    job_type: str = "train"
    mode: str = "online"
    dir: str = ""
    log_step: int = 10
    watch_model: bool = False
    watch_log: str = "gradients"
    watch_log_freq: int = 100
    save_code: bool = False
    tags: List[str] = field(default_factory=list)

@dataclass
class SLAMLLMConfig:
    encoder_path: str = "/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/whisper/large-v3.pt"
    encoder_dim : int = 1280
    ds_rate: int = 5
    llm_path: str = "/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/Qwen2.5-7B-Instruct"
    llm_dim : int = 3584

@dataclass
class Config:
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    data: DataConfig = field(default_factory=DataConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    peft: PeftConfig = field(default_factory=PeftConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)  # 新增Adapter配置
    slam: SLAMLLMConfig = field(default_factory=SLAMLLMConfig)
