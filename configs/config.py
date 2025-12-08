from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

@dataclass
class DatasetConfig:
    """数据集配置"""
    data_path: str
    map_path: str
    batch_size: int = 128
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    if_vqvae: bool = True
    dataset_name: str = 'Beijing'
@dataclass
class ModelConfig:
    """模型配置"""
    vocab_size: int = 2048
    embedding_dim: int = 512
    commitment_cost: float = 0.25
    decay: float = 0.99
    hidden_dims: List[int] = None
    spatial_loc_nums: List[int] = None
    temporal_scale_nums: List[int] = None
    using_znorm: bool = True
    beta: float = 0.25
    out_loc_num: int = 900
    seq_len: int = 168
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 128, 256]

@dataclass
class TrainingConfig:
    """训练配置"""
    epochs: int = 100
    learning_rate: float = 0.0003
    codebook_learning_rate: float = 0.001
    checkpoint_dir: str = "checkpoints/vqvae"
    device: str = "cuda:0"
    gradient_clip_val: Optional[float] = 1.0
    early_stopping_patience: int = 10
    min_lr: float = 1.0e-6
@dataclass
class OptimizerConfig:
    """优化器配置"""
    type: str = "Adam"
    weight_decay: float = 0.0001
    beta1: float = 0.9
    beta2: float = 0.999

@dataclass  
class SavingConfig:
    """保存配置"""
    save_dir: str = "results"
    model_dir: str = "checkpoints"
    log_file: str = "training.log"
    save_interval: int = 10
    save_best: bool = True
    metric_for_best: str = "loss"


@dataclass
class LoggingConfig:
    """日志配置"""
    use_tensorboard: bool = True
    log_dir: str = "runs"
    log_interval: int = 100
    metrics: List[str] = None
    exp_name: str = "vqvae_exp"
    log_step: int = 100
    log_model_info: bool = True
    log_dataset_info: bool = True
    log_training_info: bool = True
    log_codebook_info: bool = True
    log_memory_usage: bool = False
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["loss", "recon_loss", "vq_loss", "perplexity", "usage"]

@dataclass
class Config:
    """总配置"""
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    saving: SavingConfig
    logging: LoggingConfig
    optimizer: OptimizerConfig
    @classmethod
    def from_dict(cls, config_dict: dict):
        """从字典创建配置"""
        return cls(
            dataset=DatasetConfig(**config_dict['dataset']),
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            saving=SavingConfig(**config_dict.get('saving', {})),
            logging=LoggingConfig(**config_dict.get('logging', {})),
            optimizer=OptimizerConfig(**config_dict.get('optimizer', {}))
        )

    def to_dict(self):
        """将配置转换为字典"""
        return {
            'dataset': self.dataset.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'saving': self.saving.__dict__,
            'logging': self.logging.__dict__,
            'optimizer': self.optimizer.__dict__
        }