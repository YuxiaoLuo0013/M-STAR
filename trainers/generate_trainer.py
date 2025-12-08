import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import logging
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from typing import Dict, Optional
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.metrics import EvalUtils,evaluate
from collections import Counter
import torch.nn.functional as F

def check_change(sequence,scale_tokens):
    """计算序列中发生变化的比例"""
    scale_tokens=torch.tensor(scale_tokens)
    scale_tokens[0]=0
    changes= (sequence[:,1:] != sequence[:,:-1])
    # changes=torch.cat(((torch.zeros((changes.shape[0],1)).to(changes.device)),changes),dim=-1)
    changes[:,(torch.cumsum(scale_tokens[:-1], dim=0))]=0
    return changes.long()
    
class MetricTracker:
    """用于跟踪和计算训练指标的类"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.metrics = {}
        self.counts = {}
        
    def update(self, metrics_dict: Dict[str, float], batch_size: int = 1):
        for key, value in metrics_dict.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0
            self.metrics[key] += value * batch_size
            self.counts[key] += batch_size
            
    def get_value(self, key: str) -> float:
        if key in self.metrics and self.counts[key] > 0:
            return self.metrics[key] / self.counts[key]
        return 0.0
    
    def average(self) -> Dict[str, float]:
        return {key: self.metrics[key] / self.counts[key] 
                for key in self.metrics}

class GenerateTrainer:
    def __init__(self, 
                 config_path: str,
                 model: Optional[nn.Module] = None,
                 train_dataset: Optional[torch.utils.data.Dataset] = None,
                 val_dataset: Optional[torch.utils.data.Dataset] = None):
        """
        生成模型训练器初始化
        """
        # 加载配置
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        self.config = config_dict
        
        self.setup_logging()
        self.setup_device()
        
        # 设置模型
        self.model = model
        if self.model is None:
            self.setup_model()
        else:
            self.model.to(self.device)
            self.logger.info("使用预定义模型")
            
        self.setup_optimizer()
        
        # 设置数据集
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        if self.train_dataset is None or self.val_dataset is None:
            self.setup_data()
        else:
            self.setup_dataloaders()
            self.logger.info(f"使用预定义数据集 - 训练集大小: {len(self.train_dataset)}, "
                           f"验证集大小: {len(self.val_dataset)}")
        
        self.best_accuracy = 0.0
        self.patience_counter = 0
        self.train_metrics = MetricTracker()
        self.val_metrics = MetricTracker()
        
        # 计算总steps数
        self.total_steps = len(self.train_loader) * self.config['training']['epochs']
        
        # 设置学习率调度器
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.total_steps,
            eta_min=self.config['training']['min_lr']
        )
        
        self.log_training_details()

    def setup_logging(self):
        """设置日志记录"""
        # 创建实验目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = Path(self.config['saving']['save_dir']) / timestamp
        self.checkpoint_dir = self.exp_dir / self.config['saving']['model_dir']
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        log_file = self.exp_dir / self.config['saving']['log_file']
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 设置TensorBoard
        self.writer = SummaryWriter(log_dir=self.exp_dir / 'tensorboard')
        
        self.logger.info(f"实验目录: {self.exp_dir}")
        self.logger.info(f"配置: {self.config}")

    def setup_device(self):
        """设置设备"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"使用设备: {self.device}")

    def setup_model(self):
        """设置模型 - 这个方法应该在子类中实现"""
        raise NotImplementedError("子类应该实现setup_model方法")

    def setup_optimizer(self):
        """设置优化器"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        self.logger.info(f"优化器: AdamW, 学习率: {self.config['training']['learning_rate']}")

    def setup_data(self):
        """设置数据 - 这个方法应该在子类中实现"""
        raise NotImplementedError("子类应该实现setup_data方法")

    def setup_dataloaders(self):
        """设置数据加载器"""
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['dataset']['batch_size'],
            shuffle=self.config['dataset'].get('shuffle', True),
            num_workers=self.config['dataset'].get('num_workers', 4),
            pin_memory=self.config['dataset'].get('pin_memory', True)
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['dataset']['batch_size'],
            shuffle=False,
            num_workers=self.config['dataset'].get('num_workers', 4),
            pin_memory=self.config['dataset'].get('pin_memory', True)
        )
        
        self.logger.info(f"数据加载器已设置 - 训练批次: {len(self.train_loader)}, "
                       f"验证批次: {len(self.val_loader)}")

    def log_training_details(self):
        """记录训练详情"""
        self.logger.info(f"开始训练 - 总批次: {len(self.train_loader)}, "
                       f"总步数: {self.total_steps}")
        self.logger.info(f"学习率调度器: CosineAnnealingLR, "
                       f"最小学习率: {self.config['training']['min_lr']}")

    def log_metrics(self, metrics: Dict[str, float], step: int, phase: str):
        """记录指标到TensorBoard"""
        for name, value in metrics.items():
            self.writer.add_scalar(f"{phase}/{name}", value, step)

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }
        
        # 保存最新检查点
        latest_path = self.checkpoint_dir / 'latest_model.pt'
        torch.save(checkpoint, latest_path)
        
        # 如果是最佳模型，也保存为best_model.pt
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"保存最佳模型到 {best_path}")
        

    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        self.model.train()
        self.train_metrics.reset()
        
        pbar = tqdm(total=len(self.train_loader), desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 计算当前的全局训练步数
            train_step = epoch * len(self.train_loader) + batch_idx
            
            # 获取输入数据
            trajectories = batch['trajectory']
            home_locations = batch['home_commute']
            cluster = batch['trajectory_levels']
            # 将数据移到设备上
            trajectories = trajectories.to(self.device)
            home_locations = home_locations.to(self.device)
            cluster = cluster.to(self.device)
            # 清零梯度
            self.optimizer.zero_grad() 
            
            # 前向传播，添加train_step参数
            outputs = self.model(trajectories, home_locations, cluster,train_step=train_step)
            
            # 计算损失
            logits = outputs['logits']           # [B, L, V]
            target_ids = outputs['target_ids']
            B, L, V = logits.shape

            # 1. CrossEntropy Loss
            logits_flat = logits.reshape(-1, V)
            target_ids_flat = target_ids.reshape(-1)
            cross_entropy_loss = nn.CrossEntropyLoss()(logits_flat, target_ids_flat)

            loss = cross_entropy_loss
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if self.config['training']['gradient_clip_val'] is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['gradient_clip_val']
                )
            
            # 更新参数
            self.optimizer.step()
            self.scheduler.step()
            
            # 计算准确率
            pred = logits_flat.argmax(dim=-1)
            correct = (pred == target_ids_flat).float().sum()
            accuracy = correct / target_ids_flat.numel()
            
            # 更新指标
            metrics = {
                'loss': loss.item(),
                'accuracy': accuracy.item(),
                'lr': self.optimizer.param_groups[0]['lr']
            }
            self.train_metrics.update(metrics, trajectories.size(0))
            
            # 更新进度条
            pbar.set_postfix({
                "Loss": f"{self.train_metrics.get_value('loss'):.4f}",
            })
            pbar.update()
            
            # 记录训练过程
            if batch_idx % self.config['logging']['log_interval'] == 0:
                step = epoch * len(self.train_loader) + batch_idx
                self.log_metrics(metrics, step, "train")
                
        pbar.close()
        return self.train_metrics.average()

    def validate(self):
        self.model.eval()
        real_trajs = []
        generated_trajs = []
        
        with torch.no_grad():
            # 收集整个验证集的轨迹
            for batch in self.val_loader:
                # 获取真实轨迹
                real_traj = batch['trajectory'][:,0,:] 
                real_trajs.append(real_traj)
                home_locations=batch['home_commute'].to(self.device)
                cluster=batch['trajectory_levels'].to(self.device)
                outputs = self.model.generate(
                home_locations=home_locations,
                # temperature=0.7,
                # top_k=20,
                cluster=cluster,
            )
                generated_traj = outputs['x_hat_loc']
                generated_trajs.append(generated_traj.cpu())

        real_trajs=torch.cat(real_trajs,dim=0).detach().cpu().numpy()
        generated_trajs=torch.cat(generated_trajs,dim=0).detach().cpu().numpy()
        # 在整个验证集上计算指标
        d_jsd, g_jsd, du_jsd, p_jsd, l_jsd, f_jsd, cpc, od_mape, pop_mape = evaluate(self.config['dataset']['dataset_name'], 168, real_trajs, generated_trajs)
        
        # 记录验证指标
        val_metrics = {
            'val/d_jsd': d_jsd,
            'val/g_jsd': g_jsd, 
            'val/du_jsd': du_jsd,
            'val/p_jsd': p_jsd,
            'val/l_jsd': l_jsd,
            'val/f_jsd': f_jsd,
            'val/cpc': cpc,
            'val/od_mape': od_mape,
            'val/pop_mape': pop_mape
        }
        
        return val_metrics


    def train(self):
        """训练模型"""
        self.logger.info("开始训练...")
        
        best_metric = float('inf')  # 对于越大越好的指标，初始化为负无穷
        metric_for_best = self.config['saving'].get('metric_for_best', 'du_jsd')  # 默认使用 cpc
        
        for epoch in range(self.config['training']['epochs']):
            self.logger.info(f"Epoch {epoch+1}/{self.config['training']['epochs']}")
            
            # 训练一个epoch
            train_metrics = self.train_epoch(epoch)
            self.logger.info(f"训练指标: {train_metrics}")
            
            # 验证
            if epoch % self.config['logging'].get('eval_interval') == 0:
                val_metrics = self.validate()
                self.logger.info(f"验证指标: {val_metrics}")
                
                # 检查是否是最佳模型
                current_metric = val_metrics['val/du_jsd']+val_metrics['val/p_jsd']
                is_best = current_metric < best_metric  # 直接比较，越大越好
                
                if is_best:
                    best_metric = current_metric
                    self.patience_counter = 0
                    self.logger.info(f"新的最佳{metric_for_best}: {best_metric:.4f}")
                else:
                    self.patience_counter += 1
                    self.logger.info(f"最佳{metric_for_best}未改善，耐心计数: {self.patience_counter}")
        
                self.save_checkpoint(epoch, is_best if 'is_best' in locals() else False)
                
            # 早停
            if self.patience_counter >= self.config['training'].get('early_stopping_patience', float('inf')):
                self.logger.info(f"早停触发，{self.patience_counter}个epoch未改善")
                break
        
        self.logger.info(f"训练完成! 最佳{metric_for_best}: {best_metric:.4f}")
        return best_metric
