# import sys
# from pathlib import Path
# root_dir = str(Path(__file__).parent.parent)  # 指向 next-scale 目录
# sys.path.append(root_dir)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import logging
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import numpy as np
from typing import Dict, Any, Optional
from models.vqvae import VQVAE
from configs.config import Config
from torch.optim.lr_scheduler import CosineAnnealingLR

class MetricTracker:
    """用于跟踪和计算训练指标的类"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.metrics = {}
        self.counts = {}
        self.total_correct = 0.0  # 改为float类型
        self.total_tokens = 0
        
    def update(self, metrics_dict: Dict[str, float], batch_size: int = 1):
        # 更新普通指标
        for key, value in metrics_dict.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0
            self.metrics[key] += value * batch_size
            self.counts[key] += batch_size
            
    def get_value(self, key: str) -> float:
        """获取指定指标的当前平均值"""
        if key in self.metrics and self.counts[key] > 0:
            return self.metrics[key] / self.counts[key]
        return 0.0

    def get_accuracy(self) -> float:
        return self.total_correct / self.total_tokens if self.total_tokens > 0 else 0.0
    
    def average(self) -> Dict[str, float]:
        return {key: self.metrics[key] / self.counts[key] 
                for key in self.metrics}

class VQVAETrainer:
    def __init__(self, 
                 config_path: str,
                 model: Optional[nn.Module] = None,
                 train_dataset: Optional[torch.utils.data.Dataset] = None,
                 val_dataset: Optional[torch.utils.data.Dataset] = None):
        """
        VQVAE训练器初始化
        Args:
            config_path: 配置文件路径
            model: 预定义的模型
            train_dataset: 
            val_dataset: 验证数据集
        """
        # 加载配置
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        self.config = Config.from_dict(config_dict)
        
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
        
        self.best_loss = float('inf')
        self.best_accuracy = 0.0  # 添加最佳准确率跟踪
        self.patience_counter = 0  # 添加早停计数器
        self.train_metrics = MetricTracker()
        self.val_metrics = MetricTracker()
        
        # 计算总steps数
        self.total_steps = len(self.train_loader) * self.config.training.epochs
        
        # 初始化优化器后添加学习率调度器
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.total_steps,  # 使用总steps
            eta_min=self.config.training.min_lr
        )
        
        # 添加codebook优化器的调度器
            # self.codebook_scheduler = CosineAnnealingLR(
            #     self.codebook_optimizer,
            #     T_max=self.total_steps,  # 使用总steps
            #     eta_min=self.config.training.min_lr
            # )
        
        self.logger.info(f"初始化学习率调度器 - 主学习率: {self.config.training.learning_rate}, "
                        f"最小学习率: {self.config.training.min_lr}")
                        # f"Codebook学习率: {self.config.training.codebook_learning_rate}")
        
        # 在设置完所有组件后，添加详细的日志信息
        self.log_training_details()
    
    def setup_model(self):
        """设置模型"""  
        self.model = VQVAE(
            vocab_size=self.config.model.vocab_size,    
            embedding_dim=self.config.model.embedding_dim,
            out_loc_num=self.config.model.out_loc_num,
            seq_len=self.config.model.seq_len,
            using_znorm=self.config.model.using_znorm,
            beta=self.config.model.beta,
            spatial_loc_nums=self.config.model.spatial_loc_nums,
            temporal_scale_nums=self.config.model.temporal_scale_nums
        ).to(self.device)
        self.logger.info("模型创建完成")
    
    def setup_logging(self):
        """设置日志记录"""
        # 创建时间戳目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_dir = Path(self.config.saving.save_dir) / timestamp
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置文件日志
        log_file = self.exp_dir / self.config.saving.log_file
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 设置tensorboard
        if self.config.logging.use_tensorboard:
            self.writer = SummaryWriter(
                log_dir=str(self.exp_dir / self.config.logging.log_dir)
            )
        
        # 保存配置文件副本
        with open(self.exp_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config.to_dict(), f)
            
        self.logger.info(f"实验目录创建于: {self.exp_dir}")
    
    def setup_device(self):
        """设置设备"""
        self.device = torch.device(self.config.training.device)
        self.logger.info(f"使用设备: {self.device}")
    
    def setup_optimizer(self):
        """设置优化器"""
        # 将模型参数分成两组：codebook和其他参数
        codebook_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'quantizer.embedding.weight' in name:  # 只针对VQ-VAE中的codebook参数
                codebook_params.append(param)
                # print(f"Codebook参数: {name}")  # 用于调试确认
            else:  # 其他所有参数（包括其他embedding层）
                other_params.append(param)
                # print(f"其他参数: {name}")  # 用于调试确认
        
        # 创建两个优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            betas=(self.config.optimizer.beta1, self.config.optimizer.beta2),
            weight_decay=self.config.optimizer.weight_decay
        )
        
        # self.codebook_optimizer = torch.optim.Adam(
        #     codebook_params,
        #      lr=self.config.training.learning_rate,
        #     betas=(self.config.optimizer.beta1, self.config.optimizer.beta2),
        #     weight_decay=self.config.optimizer.weight_decay
        # )
        
        self.logger.info(f"主优化器(Adam)创建完成，学习率: {self.config.training.learning_rate}")
        # self.logger.info(f"Codebook优化器(SGD)创建完成，学习率: {self.config.training.codebook_learning_rate}")
    
    def setup_data(self):
        """设置数据集"""
        raise NotImplementedError("请在子类中实现数据集设置")
    
    def setup_dataloaders(self):
        """设置数据加载器"""
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.dataset.batch_size,
            shuffle=self.config.dataset.shuffle,
            num_workers=self.config.dataset.num_workers,
            pin_memory=self.config.dataset.pin_memory
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.dataset.batch_size,
            shuffle=False,
            num_workers=self.config.dataset.num_workers,
            pin_memory=self.config.dataset.pin_memory
        )
    
    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        self.model.train()
        self.train_metrics.reset()
        
        # 设置进度条，只显示loss
        pbar = tqdm(total=len(self.train_loader), desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 获取当前步数
            global_step = epoch * len(self.train_loader) + batch_idx
            
            # 清零梯度
            self.optimizer.zero_grad()
            # self.codebook_optimizer.zero_grad()
            
            # 前向传播
            x = batch.to(self.device)
            output_dict = self.model(x)
            loss_dict = output_dict['loss_dict']
            loss = loss_dict['loss']
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪（如果需要）
            if self.config.training.gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.training.gradient_clip_val
                )
            
            # 更新优化器
            self.optimizer.step()
                
            # 获取当前学习率并放大记录
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 更新学习率调度器
            self.scheduler.step()

            
            # 直接计算相等的像素数量
            recon_x = output_dict['x_hat_loc']
            accuracy = (recon_x == x[:,0]).float().mean().item()
            
            metrics = {
                'loss': loss_dict['loss'].item(),
                'ce_loss': loss_dict['ce_loss'].item(),
                'vq_loss': loss_dict['mean_vq_loss'].item(),
                'accuracy': accuracy,
                'usage': output_dict['usages'],
                'lr_e3': current_lr * 1e3,  # 放大1000倍
                # 'codebook_lr_e3': current_codebook_lr * 1e3  # 放大1000倍
            }
            self.train_metrics.update(metrics, x.size(0))
            
            # 进度条只显示loss
            pbar.set_postfix_str(f"Loss: {self.train_metrics.get_value('loss'):.4f}")
            pbar.update()
            
            # 记录训练过程
            if batch_idx % self.config.logging.log_interval == 0:
                step = epoch * len(self.train_loader) + batch_idx
                self.log_metrics(metrics, step, "train")
                self
        pbar.close()
        return self.train_metrics.average()

    def validate(self, epoch: int):
        """验证模型"""
        self.model.eval()
        self.val_metrics.reset()
        
        pbar = tqdm(total=len(self.val_loader), desc='Validation')
        
        with torch.no_grad():
            for batch in self.val_loader:
                x = batch.to(self.device)
                
                # 前向传播
                output_dict = self.model(x)
                loss_dict = output_dict['loss_dict']
                
                # 直接计算相等的像素数量
                recon_x = output_dict['x_hat_loc']
                accuracy = (recon_x == x[:,0]).float().mean().item()
                
                # 获取当前学习率
                current_lr = self.optimizer.param_groups[0]['lr']
                # current_codebook_lr = self.codebook_optimizer.param_groups[0]['lr']
                
                metrics = {
                    'loss': loss_dict['loss'].item(),
                    'ce_loss': loss_dict['ce_loss'].item(),
                    'vq_loss': loss_dict['mean_vq_loss'].item(),
                    'accuracy': accuracy,
                    'usage': output_dict['usages'],
                    'lr_e3': current_lr * 1e3,  # 放大1000倍
                    # 'codebook_lr_e3': current_codebook_lr * 1e3  # 放大1000倍
                }
                self.val_metrics.update(metrics, x.size(0))
                
                # 进度条只显示loss
                pbar.set_postfix_str(f"Loss: {self.val_metrics.get_value('loss'):.4f}")
                pbar.update()
        
        pbar.close()
        
        # 记录验证结果
        avg_metrics = self.val_metrics.average()
        self.log_metrics(avg_metrics, epoch, "val")
        
        return avg_metrics
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """记录指标"""
        # 记录到tensorboard
        if self.config.logging.use_tensorboard:
            for name, value in metrics.items():
                self.writer.add_scalar(f'{prefix}/{name}', value, step)
        
        # 记录到日志文件
        metrics_str = ", ".join([f"{name}: {value:.6f}" for name, value in metrics.items()])
        self.logger.info(f"{prefix} Step {step} - {metrics_str}")
    
    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """保存检查点"""
        checkpoint_dir = self.exp_dir / self.config.saving.model_dir
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config.to_dict()
        }
        
        # 保存常规检查点
        if epoch % self.config.saving.save_interval == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"保存检查点到: {checkpoint_path}")
            
        # 保存最佳模型
        if is_best and self.config.saving.save_best:
            best_path = checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"保存最佳模型到: {best_path}")
    
    def train(self):
        """训练模型"""
        self.logger.info("开始训练...")
        self.logger.info(f"总轮次: {self.config.training.epochs}")
        
        epoch_pbar = tqdm(total=self.config.training.epochs, 
                         desc="Training Progress", 
                         position=0, 
                         leave=True)
        
        try:
            for epoch in range(self.config.training.epochs):
                train_metrics = self.train_epoch(epoch)
                val_metrics = self.validate(epoch)
                
                # 检查是否为最佳模型
                current_accuracy = val_metrics['accuracy']
                is_best = current_accuracy > self.best_accuracy
                
                if is_best:
                    self.best_accuracy = current_accuracy
                    self.patience_counter = 0
                    self.logger.info(f"新的最佳模型! Accuracy: {self.best_accuracy:.6f}")
                else:
                    self.patience_counter += 1
                    
                self.save_checkpoint(epoch, val_metrics['loss'], is_best)
                
                # 早停检查
                if self.patience_counter >= self.config.training.early_stopping_patience:
                    self.logger.info(f"早停触发! {self.config.training.early_stopping_patience} 个epoch未改善")
                    break
                
                epoch_pbar.update(1)
                epoch_pbar.set_postfix_str(
                    f"Best Accuracy: {self.best_accuracy:.6f}"
                )
                
        except KeyboardInterrupt:
            self.logger.info("训练被手动中断")
        
        finally:
            epoch_pbar.close()
            if self.config.logging.use_tensorboard:
                self.writer.close()
            
            self.logger.info("训练完成!")
            self.logger.info(f"最佳验证准确率: {self.best_accuracy:.6f}")
            self.logger.info(f"模型和日志保存在: {self.exp_dir}")
        
    def log_training_details(self):
        """记录详细的训练配置和模型信息"""
        self.logger.info("\n" + "="*50)
        self.logger.info("训练配置详情:")
        self.logger.info("-"*50)
        
        # 模型结构信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # 将参数量转换为更易读的形式（K或M）
        def format_num_params(num):
            if num >= 1e6:
                return f"{num/1e6:.2f}M"
            elif num >= 1e3:
                return f"{num/1e3:.2f}K"
            return str(num)
        
        self.logger.info(f"模型信息:")
        self.logger.info(f"- 总参数量: {format_num_params(total_params)} ({total_params:,})")
        self.logger.info(f"- 可训练参数量: {format_num_params(trainable_params)} ({trainable_params:,})")
        
        # 按模块统计参数量
        module_params = {}
        for name, param in self.model.named_parameters():
            module_name = name.split('.')[0]  # 获取顶层模块名
            if module_name not in module_params:
                module_params[module_name] = 0
            module_params[module_name] += param.numel()
        
        self.logger.info("\n模块参数分布:")
        for module_name, num_params in module_params.items():
            self.logger.info(f"- {module_name}: {format_num_params(num_params)} ({num_params:,})")
        
        self.logger.info(f"\nVQ-VAE 结构:")
        self.logger.info(f"- Codebook大小: {self.config.model.vocab_size}")
        self.logger.info(f"- 嵌入维度: {self.config.model.embedding_dim}")
        self.logger.info(f"- 序列长度: {self.config.model.seq_len}")
        self.logger.info(f"- 空间位置数: {self.config.model.spatial_loc_nums}")
        self.logger.info(f"- 时间尺度数: {self.config.model.temporal_scale_nums}")
        
        # 训练配置信息
        self.logger.info("\n优化器配置:")
        self.logger.info(f"- 主学习率: {self.config.training.learning_rate}")
        # self.logger.info(f"- Codebook学习率: {self.config.training.codebook_learning_rate}")
        self.logger.info(f"- 最小学习率: {self.config.training.min_lr}")
        self.logger.info(f"- Beta1: {self.config.optimizer.beta1}")
        self.logger.info(f"- Beta2: {self.config.optimizer.beta2}")
        self.logger.info(f"- Weight Decay: {self.config.optimizer.weight_decay}")
        
        # 数据集信息
        self.logger.info("\n数据集信息:")
        self.logger.info(f"- 训练集大小: {len(self.train_dataset):,}")
        self.logger.info(f"- 验证集大小: {len(self.val_dataset):,}")
        self.logger.info(f"- Batch Size: {self.config.dataset.batch_size}")
        self.logger.info(f"- 训练步数/epoch: {len(self.train_loader):,}")
        self.logger.info(f"- 总训练步数: {self.total_steps:,}")
        
        # VQ-VAE 特定参数
        self.logger.info("\nVQ-VAE 参数:")
        self.logger.info(f"- Beta (commitment loss): {self.config.model.beta}")
        self.logger.info(f"- 是否使用Z-Normalization: {self.config.model.using_znorm}")
        
        # 训练设置
        self.logger.info("\n训练设置:")
        self.logger.info(f"- 训练设备: {self.device}")
        self.logger.info(f"- 训练轮次: {self.config.training.epochs}")
        self.logger.info(f"- 早停耐心值: {self.config.training.early_stopping_patience}")
        if self.config.training.gradient_clip_val:
            self.logger.info(f"- 梯度裁剪阈值: {self.config.training.gradient_clip_val}")
        
        self.logger.info("="*50 + "\n")

    def test(self, test_dataset: Optional[torch.utils.data.Dataset] = None):
        """
        在测试集上评估模型
        Args:
            test_dataset: 可选的测试数据集，如果未提供则使用验证集
        """
        self.logger.info("开始测试...")
        
        # 使用提供的测试集或验证集
        if test_dataset is not None:
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config.dataset.batch_size,
                shuffle=False,
                num_workers=self.config.dataset.num_workers,
                pin_memory=self.config.dataset.pin_memory
            )
            dataset_size = len(test_dataset)
        else:
            test_loader = self.val_loader
            dataset_size = len(self.val_dataset)
        
        self.model.eval()
        test_metrics = MetricTracker()
        
        # 用于存储详细结果的列表
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(total=len(test_loader), desc='Testing')
        
        with torch.no_grad():
            for batch in test_loader:
                x = batch.to(self.device)
                
                # 前向传播
                ids_list = self.model.encode(x)
                output_dict = self.model.decode(ids_list)
                loss_dict = output_dict['loss_dict']
                
                # 获取预测结果
                recon_x = output_dict['x_hat_loc']
                
                # 计算准确率
                accuracy = (recon_x == x[:,0]).float().mean().item()
                
                # 存储预测和目标
                all_predictions.append(recon_x.cpu())
                all_targets.append(x[:,0].cpu())
                
                metrics = {
                    'loss': loss_dict['loss'].item(),
                    'ce_loss': loss_dict['ce_loss'].item(),
                    'vq_loss': loss_dict['mean_vq_loss'].item(),
                    'accuracy': accuracy,
                    'usage': output_dict['usages']
                }
                test_metrics.update(metrics, x.size(0))
                
                pbar.set_postfix_str(f"Loss: {test_metrics.get_value('loss'):.4f}")
                pbar.update()
        
        pbar.close()
        
        # 计算并记录最终结果
        avg_metrics = test_metrics.average()
        self.logger.info("\n" + "="*50)
        self.logger.info("测试结果:")
        self.logger.info("-"*50)
        for name, value in avg_metrics.items():
            self.logger.info(f"{name}: {value:.6f}")
        
        # 合并所有预测和目标
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # 计算每个位置的准确率
        position_accuracy = (all_predictions == all_targets).float().mean(dim=0)
        
        # 记录位置准确率
        self.logger.info("\n位置准确率:")
        for pos, acc in enumerate(position_accuracy):
            self.logger.info(f"Position {pos}: {acc.item():.6f}")
        
        self.logger.info("="*50)
        
        return avg_metrics

if __name__ == "__main__":
    # 使用实际的配置文件路径
    config_path = "./configs/vqvae_config.yaml"
    
    # 创建测试数据集
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=1000, num_scales=4, seq_len=168):
            self.data = torch.randint(0, 16, (num_samples, num_scales, seq_len))
            
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    # 创建训练集和验证集
    train_dataset = DummyDataset(num_samples=1000)
    val_dataset = DummyDataset(num_samples=200)
    
    # 创建训练器并开始训练
    trainer = VQVAETrainer(
        config_path=config_path,
        train_dataset=train_dataset,
        val_dataset=val_dataset
    )
    
    trainer.train()
    
    # 添加测试部分
    test_dataset = DummyDataset(num_samples=100)
    trainer.test(test_dataset)
    
    # 清理临时配置文件
    import os
    os.unlink(config_path)