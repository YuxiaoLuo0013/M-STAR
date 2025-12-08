import sys
from pathlib import Path
root_dir = str(Path(__file__).parent)  # 指向项目根目录
sys.path.append(root_dir)

import torch
import yaml
import argparse
import numpy as np
from tqdm import tqdm
from models.vqvae import VQVAE
from models.next_scale_generator import NextScaleGenerator
from utils.dataset import TrajectoryDataset, DatasetConfig
from torch.utils.data import DataLoader, Subset
from utils.metrics import evaluate,evaluate_travel_pattern
from torch.utils.data import random_split
def parse_args():
    parser = argparse.ArgumentParser(description='验证训练好的生成器模型')
    parser.add_argument('--config', type=str, default='configs/generate_config.yaml',
                      help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, 
                      default='./results/best_beijing/checkpoints/dur00087.pt',
                      help='生成器模型检查点路径')
    parser.add_argument('--num_samples', type=int, default=100,
                      help='生成样本数量')
    parser.add_argument('--temperature', type=list, default=0.7,
                      help='采样温度')
    parser.add_argument('--top_k', type=int, default=50,
                      help='top-k采样参数')
    parser.add_argument('--save_dir', type=str, default='results/model_evaluation',
                      help='评估结果保存目录')
    parser.add_argument('--dataset_name', type=str, default='Beijing',
                      help='数据集名称')
    return parser.parse_args()

def load_models(config_dict, checkpoint_path):
    """加载VQVAE和生成器模型"""
    # 创建并加载VQVAE
    vqvae_config = config_dict.get('vqvae', {})
    vqvae = VQVAE(**vqvae_config)
    vqvae_checkpoint_path = config_dict.get('vqvae_checkpoint')
    if vqvae_checkpoint_path:
        print(f"加载VQVAE模型: {vqvae_checkpoint_path}")
        vqvae_checkpoint = torch.load(vqvae_checkpoint_path, map_location='cpu')
        vqvae.load_state_dict(vqvae_checkpoint['model_state_dict'])
        vqvae.eval()
    
    # 创建并加载生成器
    generator_config = config_dict.get('generator', {})
    model = NextScaleGenerator(vqvae=vqvae, **generator_config)
    print(f"加载生成器模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return vqvae, model

def evaluate_model_on_dataset(model, dataset, device, args):
    """在数据集上评估模型"""
    dataloader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    real_trajs = []
    generated_trajs = []
    
    print("收集真实轨迹和生成轨迹...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # 获取真实轨迹和home locations
            real_traj = batch['trajectory'][:,0,:]
            real_trajs.append(real_traj)
            home_locations = batch['home_commute'].to(self.device)
            cluster = batch['trajectory_levels'].to(self.device)
            outputs = self.model.generate(
                home_locations = home_locations,
                # temperature=0.7,
                # top_k=20,
                cluster = cluster,

            generated_traj = outputs['x_hat_loc']
            generated_trajs.append(generated_traj)
    
    # 转换为numpy数组用于评估
    real_trajs = torch.cat(real_trajs, dim=0).cpu().numpy()
    generated_trajs = torch.cat(generated_trajs, dim=0).cpu().numpy()
    
    print(f"评估 {len(real_trajs)} 个真实轨迹和生成轨迹...")
    # 计算评估指标
    d_jsd, g_jsd, du_jsd, p_jsd, l_jsd, f_jsd, cpc, od_mape, pop_mape = evaluate(args.dataset_name, 168, real_trajs, generated_trajs)

    return metrics, real_trajs, generated_trajs

def save_results(metrics, real_trajs, generated_trajs, save_dir, dataset_name):
    """保存评估结果和轨迹样本"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存评估指标
    with open(save_dir / 'metrics.txt', 'w') as f:
        for name, value in metrics.items():
            f.write(f"{name}: {value}\n")
    
    # 保存轨迹样本
    np.save(save_dir / f"{dataset_name}_real_trajectories.npy", real_trajs)
    np.save(save_dir / f"{dataset_name}_generated_trajectories.npy", generated_trajs)
    
    print(f"结果已保存到: {save_dir}")

def main():
    args = parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    vqvae, model = load_models(config_dict, args.checkpoint)
    vqvae = vqvae.to(device)
    model = model.to(device)
    
    # 创建数据集
    dataset_config = DatasetConfig(**config_dict['dataset'])
    dataset_config.if_vqvae = False  # 设置为生成器模式
    dataset = TrajectoryDataset(config=dataset_config,if_test=True)
    
    # 划分训练集和验证集
    total_size = len(dataset)
    train_size = int(total_size * 0.5)  # 80% 用于训练
    val_size = int(total_size * 0.2)
    test_size = int(total_size * 0.3)
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [0.5, 0.2, 0.3],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 评估模型
    metrics, real_trajs, generated_trajs = evaluate_model_on_dataset(model, test_dataset, device, args)
    
    # 打印评估结果
    print("\n模型评估结果:")
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")
    
    # 保存结果
    save_results(metrics, real_trajs, generated_trajs, args.save_dir, args.dataset_name)
    
if __name__ == "__main__":
    main()
