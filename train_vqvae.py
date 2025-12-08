import sys
from pathlib import Path
root_dir = str(Path(__file__).parent)  # 指向项目根目录
sys.path.append(root_dir)

import torch
import argparse
import yaml
from trainers.vqvae_trainer import VQVAETrainer
from utils.dataset import TrajectoryDataset, DatasetConfig
from torch.utils.data import random_split
from configs.config import Config

def parse_args():
    parser = argparse.ArgumentParser(description='Train VQVAE model')
    parser.add_argument('--config', type=str, default='configs/vqvae_config.yaml',
                      help='path to config file')
    parser.add_argument('--seed', type=int, default=42,
                      help='random seed')
    return parser.parse_args()

def main():
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 加载配置文件
    with open(args.config, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # 创建数据集配置
    dataset_config = DatasetConfig(**config_dict['dataset'])
    
    # 创建完整数据集
    dataset = TrajectoryDataset(
        config=dataset_config,
    )
    
    # 划分训练集和验证集
    total_size = len(dataset)
    train_size = int(total_size * 0.8)  # 80% 用于训练
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"Dataset sizes - Total: {total_size}, Train: {train_size}, Val: {val_size}")
    
    # 创建训练器
    trainer = VQVAETrainer(
        config_path=args.config,
        train_dataset=train_dataset,
        val_dataset=val_dataset
    )
    
    # 开始训练
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error during training: {e}")
        raise e
    finally:
        if hasattr(trainer, 'writer'):
            trainer.writer.close()
        
    print(f"Training completed! Results saved in: {trainer.exp_dir}")

if __name__ == "__main__":
    main()