import sys
from pathlib import Path
root_dir = str(Path(__file__).parent)  # 指向项目根目录
sys.path.append(root_dir)
import torch
import argparse
import yaml
from trainers.generate_trainer import GenerateTrainer
from utils.dataset import TrajectoryDataset, DatasetConfig      
from torch.utils.data import random_split
import random
import numpy as np
from configs.config import Config
from models.vqvae import VQVAE
import warnings
warnings.filterwarnings('ignore') 


def parse_args():
    parser = argparse.ArgumentParser(description='Train Generate model')
    parser.add_argument('--config', type=str, default='configs/generate_config.yaml',
                      help='path to config file')
    parser.add_argument('--seed', type=int, default=42,
                      help='random seed')
    parser.add_argument('--resume', type=str, default=None,
                      help='path to checkpoint to resume from')
    return parser.parse_args()

def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def verify_model_loading(model, checkpoint_path):
    """验证模型权重加载是否正确"""
    # 确保模型在验证前处于eval模式
    model.eval()
    
    # 保存当前模型状态
    original_state = {name: param.clone() for name, param in model.named_parameters()}
    
    # 加载checkpoint前先打印checkpoint的基本信息
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print("\nCheckpoint信息:")
    print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"包含的键: {list(checkpoint.keys())}")
    
    # 加载前确保模型结构匹配
    state_dict = checkpoint['model_state_dict']
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())
    
    # 检查是否有缺失或多余的键
    missing_keys = model_keys - checkpoint_keys
    unexpected_keys = checkpoint_keys - model_keys
    
    if missing_keys:
        print("\n缺失的键:", missing_keys)
    if unexpected_keys:
        print("\n多余的键:", unexpected_keys)
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 验证方式1：检查所有参数是否存在且形状一致
    print("\n=== 验证方式1：检查参数形状 ===")
    for name, param in model.named_parameters():
        if name in original_state:
            if param.shape != original_state[name].shape:
                print(f"参数形状不匹配 - {name}: checkpoint形状 {param.shape}, 原始形状 {original_state[name].shape}")
        else:
            print(f"参数缺失 - {name}")
    
    # 验证方式2：检查参数值
    print("\n=== 验证方式2：检查参数值 ===")
    max_diff = 0
    diff_param_name = ""
    for name, param in model.named_parameters():
        if name in original_state:
            diff = (param - original_state[name]).abs().max().item()
            if diff > max_diff:
                max_diff = diff
                diff_param_name = name
            if diff > 1e-6:  # 设置一个小的阈值
                print(f"参数值有显著差异 - {name}: 最大差异 = {diff}")
    
    print(f"\n最大参数差异在 {diff_param_name}: {max_diff}")
    return max_diff < 1e-6

def main():
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载配置文件
    with open(args.config, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # 创建数据集配置  
    dataset_config = DatasetConfig(**config_dict['dataset'])
    
    # 创建完整数据集
    dataset = TrajectoryDataset(config=dataset_config)
    
    # 使用比例进行分割
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [0.5, 0.2, 0.3],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    # 创建VQVAE模型
    from models.vqvae import VQVAE
    vqvae_config = config_dict.get('vqvae', {})
    vqvae = VQVAE(**vqvae_config)
    
    # 加载预训练的VQVAE权重
    vqvae_checkpoint_path = config_dict.get('vqvae_checkpoint')
    print(vqvae_checkpoint_path)
    if vqvae_checkpoint_path:
        print(f"正在加载预训练VQVAE模型: {vqvae_checkpoint_path}")
        try:
            vqvae_checkpoint = torch.load(vqvae_checkpoint_path, map_location='cpu')
            vqvae.load_state_dict(vqvae_checkpoint['model_state_dict'])
            print(f"成功加载预训练VQVAE模型")
            # 冻结VQVAE参数
            for param in vqvae.parameters():
                param.requires_grad = False
            print("VQVAE模型参数已冻结")
        except Exception as e:
            print(f"加载VQVAE模型失败: {e}")
            raise e
    else:
        print("警告: 未指定预训练VQVAE模型路径，将使用随机初始化的VQVAE")
    
    # 创建生成模型
    from models.next_scale_generator import NextScaleGenerator
    generator_config = config_dict.get('generator', {})
    model = NextScaleGenerator(vqvae=vqvae, **generator_config)
    
    # 创建训练器
    trainer = GenerateTrainer(
        config_path=args.config,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset
    )
    
    # 如果需要从检查点恢复
    if args.resume:
        print(f"\n开始加载checkpoint - {args.resume}")
        
        # 确保在加载前设置好随机种子
        set_seed(args.seed)
        
        # 确保模型在正确的设备上
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        trainer.model = trainer.model.to(device)
        
        # 先加载checkpoint
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint and hasattr(trainer, 'scheduler'):
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'epoch' in checkpoint:
            trainer.current_epoch = checkpoint['epoch']
            print(f"从epoch {trainer.current_epoch} 继续训练")
        
        # 打印当前优化器状态
        print("\n当前优化器状态：")
        for param_group in trainer.optimizer.param_groups:
            print(f"学习率: {param_group['lr']}")
        
        # 验证加载是否正确
        print(f"\n开始验证模型加载")
        is_loaded_correctly = verify_model_loading(trainer.model, args.resume)
        if not is_loaded_correctly:
            print("\n警告：模型权重加载存在显著差异！")
            print("可能的原因：")
            print("1. checkpoint可能不是期望的训练状态")
            print("2. 模型结构可能发生了变化")
            print("3. 随机种子或其他训练设置可能不一致")
            
            user_input = input("是否继续训练？(y/n): ")
            if user_input.lower() != 'y':
                print("训练已取消")
                return
    
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

    # # 在测试集上评估
    # test_metrics = trainer.test(val_dataset)  # 这里使用验证集作为测试集
    # print("\nTest Results:")
    # for metric_name, value in test_metrics.items():
    #     print(f"{metric_name}: {value:.4f}")
    
    
if __name__ == "__main__":
    main()