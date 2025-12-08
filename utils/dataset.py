import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing.pool import ThreadPool
import editdistance
from tqdm import tqdm
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
import pickle
import multiprocessing
from Levenshtein import distance as lev_distance
import os

def resample_trajectory_levels(trajectory_levels, home_commute_locations):
    """
    根据home和commute位置是否相同将数据分为两类，然后在每类中按照各自的标签分布进行采样
    Args:
        trajectory_levels: 当前的轨迹标签
        home_commute_locations: home和commute位置
    Returns:
        重新采样后的标签
    """
    # 创建新的标签数组
    new_levels = trajectory_levels.clone()
    
    # 找到home和commute位置相同和不相同的索引
    same_location_mask = home_commute_locations[:, 0] == home_commute_locations[:, 1]
    same_location_indices = torch.where(same_location_mask)[0]
    diff_location_indices = torch.where(~same_location_mask)[0]
    
    # 计算每类中的标签分布
    same_location_dist = torch.bincount(
        trajectory_levels[same_location_indices], 
        minlength=8
    ).float()
    same_location_dist = same_location_dist / same_location_dist.sum()
    
    diff_location_dist = torch.bincount(
        trajectory_levels[diff_location_indices], 
        minlength=8
    ).float()
    diff_location_dist = diff_location_dist / diff_location_dist.sum()
    
    # 对相同位置的轨迹进行重新采样
    if len(same_location_indices) > 0:
        resampled_labels = torch.multinomial(
            same_location_dist,
            num_samples=len(same_location_indices),
            replacement=True
        )
        new_levels[same_location_indices] = resampled_labels
    
    # 对不同位置的轨迹进行重新采样
    if len(diff_location_indices) > 0:
        resampled_labels = torch.multinomial(
            diff_location_dist,
            num_samples=len(diff_location_indices),
            replacement=True
        )
        new_levels[diff_location_indices] = resampled_labels
    return new_levels

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
class TrajectoryDataset(Dataset):
    def __init__(self, config,if_test=False):
        """
        轨迹数据集初始化
        Args:
            config: 数据集配置
            if_vqvae: 是否用于VQVAE模型
            if_test: 是否用于测试
        """
        super().__init__()
        self.config = config
        self.if_vqvae = config.if_vqvae
        self.if_test = if_test
        try:
            # 使用 torch.load 替代 np.load
            if str(config.data_path).endswith('.npy'):
                data = torch.from_numpy(np.load(config.data_path))
            else:
                data = torch.load(config.data_path)
            self.process_trajectories(data)
            self.calculate_home_commute_locations()
        
            self.calculate_trajectory_levels()
        except Exception as e:
            raise RuntimeError(f"加载数据失败: {str(e)}")

    def process_trajectories(self, data: torch.Tensor) -> None:
        valid_mask = torch.tensor([len(traj) == 168 for traj in data])
        valid_trajectories = data[valid_mask]  # 限制为2个轨迹以匹配注意力掩码
            
        # 加载映射网格数据
        mapping_grids = torch.from_numpy(np.load(self.config.map_path)).int()
        
        # 进行映射操作
        mapped_trajectories = mapping_grids[valid_trajectories.long()].permute(0, 2, 1)

        self.data = mapped_trajectories.long()

        
    def calculate_trajectory_levels(self):
        if self.config.dataset_name == 'Beijing':
            load_path = './data/trajs_data/Beijing/cluster_results.pkl'
        else:
            load_path = './data/trajs_data/Shenzhen/cluster_results.pkl'
        with open(load_path, 'rb') as f:
            cluster_results = pickle.load(f)
        trajectory_levels = torch.tensor(cluster_results).long()
        
        if self.if_test:
            # 根据home和commute位置是否相同进行重新采样
            trajectory_levels = resample_trajectory_levels(
                trajectory_levels,
                self.home_commute_locations
            )
        
        self.trajectory_levels = trajectory_levels
    

    def calculate_home_commute_locations(self):
        """计算每个轨迹的home location和commute location"""
        # 重塑数据为 (batch_size, 7, 24) 并只取前6小时
        reshaped_data = self.data[:,0,:].reshape(-1, 7, 24)[:, :, :8]
        
        # 计算每个轨迹最频繁出现的位置作为home location
        self.home_locations = torch.tensor([    
            torch.bincount(chunk.flatten().long()).argmax()
            for chunk in reshaped_data
        ])
        
        # 计算commute location
        reshaped_data = self.data[:,0,:].reshape(-1, 7, 24)[:, :5, 10:18]
        self.commute_locations = torch.tensor([    
            torch.bincount(chunk.flatten().long()).argmax()
                for chunk in reshaped_data
            ])
        
        self.home_commute_locations = torch.cat([self.home_locations.unsqueeze(-1), self.commute_locations.unsqueeze(-1)], dim=-1)

        if self.if_test:
            # 加载真实的home-commute对应关系
            home_commute_pairs = torch.load('./data/trajs_data/Beijing/home_commute_pairs.pt')
            # home_commute_pairs = torch.load('./data/trajs_data/Shenzhen/home_commute_pairs.pt')
            
            # 计算原始数据中home和commute相等/不相等的人群比例
            same_location_mask = self.home_locations == self.commute_locations
            same_location_ratio = same_location_mask.float().mean().item()
            print(f"Original same location ratio: {same_location_ratio:.4f}")
            
            # 初始化新的home和commute locations，默认使用原始值
            new_home_locations = self.home_locations.clone()
            new_commute_locations = self.commute_locations.clone()
            
            # 将home_commute_pairs分为相同位置和不同位置两类
            same_pairs = [pair for pair in home_commute_pairs if pair[0] == pair[1]]
            diff_pairs = [pair for pair in home_commute_pairs if pair[0] != pair[1]]
            
            # 对same_pairs进行随机打乱
            same_pairs = [same_pairs[i] for i in torch.randperm(len(same_pairs))]
            
            # 获取相同位置的home locations并计算分布
            same_home_locations = self.home_commute_locations[same_location_mask][:, 0]
            same_home_dist = torch.histc(same_home_locations.float(), bins=900, min=0, max=900)
            # same_home_dist = torch.histc(same_home_locations.float(), bins=2236, min=0, max=2236)
            same_home_dist = same_home_dist / same_home_dist.sum()

            new_home_locations[same_location_mask] = torch.multinomial(same_home_dist, len(same_home_locations), replacement=True)
            new_commute_locations[same_location_mask]=new_home_locations[same_location_mask]
            print(f"Total pairs: {len(home_commute_pairs)}")
            print(f"Same location pairs: {len(same_pairs)}")
            print(f"Different location pairs: {len(diff_pairs)}")
            
            # 只对home和commute不相等的人群进行采样
            diff_indices = torch.where(~same_location_mask)[0]
            num_diff = len(diff_indices)
            
            if num_diff > 0 and len(diff_pairs) > 0:
                # 加载每个home location对应的commute location分布
                commute_dist_by_home_path = './data/trajs_data/Beijing/commute_dist_by_home.pt'
                # commute_dist_by_home_path = './data/trajs_data/Shenzhen/commute_dist_by_home.pt'
                try:
                    commute_dist_by_home = torch.load(commute_dist_by_home_path)
                    
                    # 对每个需要不同home和commute位置的样本进行采样
                    for i, idx in enumerate(diff_indices):
                        # 从同分布中采样home位置
                        home_loc = torch.multinomial(same_home_dist, 1).item()
                        
                        # 根据home位置获取其对应的commute位置分布
                        commute_dist = commute_dist_by_home[home_loc]
                        
                        # 如果该home位置没有对应的commute分布（全为0），则随机选择一个不同的位置
                        if commute_dist.sum() == 0:
                            commute_loc = torch.randint(0, 900, (1,)).item()
                            while commute_loc == home_loc:  # 确保home和commute不同
                                commute_loc = torch.randint(0, 900, (1,)).item()
                        else:
                            # 否则根据分布采样commute位置
                            commute_loc = torch.multinomial(commute_dist, 1).item()
                        
                        new_home_locations[idx] = home_loc
                        new_commute_locations[idx] = commute_loc
                
                except FileNotFoundError:
                    print("Warning: commute_dist_by_home.pt not found, falling back to random sampling")
                    # 如果文件不存在，回退到原来的随机采样方法
                    sampled_indices = torch.randint(0, len(diff_pairs), (num_diff,))
                    for i, idx in enumerate(sampled_indices):
                        home_loc, commute_loc = diff_pairs[idx]
                        new_home_locations[diff_indices[i]] = home_loc
                        new_commute_locations[diff_indices[i]] = commute_loc
            elif num_diff > 0:
                # 如果没有不同的对，基于相同位置的分布采样home位置，并确保commute位置不同
                for i in diff_indices:
                    # 基于相同位置的分布采样home位置
                    home_loc = torch.multinomial(same_home_dist, 1).item()
                    # 随机选择commute位置，确保与home位置不同
                    commute_loc = torch.randint(0, 900, (1,)).item()
                    while commute_loc == home_loc:  # 确保home和commute不同
                        commute_loc = torch.randint(0, 900, (1,)).item()
                    new_home_locations[i] = home_loc
                    new_commute_locations[i] = commute_loc
            
            # 更新home和commute locations
            self.home_locations = new_home_locations
            self.commute_locations = new_commute_locations
            self.home_commute_locations = torch.cat([
                self.home_locations.unsqueeze(-1),
                self.commute_locations.unsqueeze(-1)
            ], dim=-1)
            
            # 验证新的比例
            new_same_location_mask = self.home_locations == self.commute_locations
            new_same_location_ratio = new_same_location_mask.float().mean().item()
            print(f"New same location ratio: {new_same_location_ratio:.4f}")
            
            # 打印一些统计信息
            print(f"Number of samples: {len(self.home_locations)}")
            print(f"Number of same location samples: {same_location_mask.sum().item()}")
            print(f"Number of different location samples: {(~same_location_mask).sum().item()}")
            
            # 打印一些home-commute对的示例
            print("\nSample home-commute pairs:")
            for i in range(min(5, len(self.home_locations))):
                print(f"Sample {i+1}: Home={self.home_locations[i].item()}, Commute={self.commute_locations[i].item()}")

    
    def calculate_home_location_proportion(self):
        """计算每条轨迹中home location的占比"""
        if not hasattr(self, 'home_locations'):
            self.calculate_home_commute_locations()
        
        # 初始化结果张量
        self.home_location_proportions = torch.zeros(len(self.data))
        
        # 计算每条轨迹中home location的占比
        for i in range(len(self.data)):
            trajectory = self.data[i, 0]  # 获取第一条轨迹
            home_loc = self.home_locations[i]
            
            # 计算home location出现的次数
            home_count = (trajectory == home_loc).sum().item()
            
            # 计算总位置数
            total_locations = len(trajectory)
            
            # 计算占比
            self.home_location_proportions[i] = home_count / total_locations
        
        # 归一化home location占比
        self.normalize_home_location_proportions()
        
        return self.home_location_proportions
    
    def normalize_home_location_proportions(self):
        """归一化home location占比到[0,1]范围"""
        if not hasattr(self, 'home_location_proportions'):
            self.calculate_home_location_proportion()
        
        # 使用最小-最大归一化
        min_prop = self.home_location_proportions.min()
        max_prop = self.home_location_proportions.max()
        
        # 避免除以零的情况
        if max_prop > min_prop:
            self.normalized_home_location_proportions = (self.home_location_proportions - min_prop) / (max_prop - min_prop)
        else:
            # 如果所有值都相同，则全部设为0.5
            self.normalized_home_location_proportions = torch.ones_like(self.home_location_proportions) * 0.5
        
        return self.normalized_home_location_proportions

    def __getitem__(self, index: int) -> torch.Tensor:
        if self.config.if_vqvae:
            return self.data[index]
        else:
            return {
                'trajectory': self.data[index],
                'home_commute': self.home_commute_locations[index],
                'trajectory_levels': self.trajectory_levels[index],
            }


    def __len__(self) -> int:
        return len(self.data)
    

def analyze_unique_ids(data_path):
    # Load the trajectory data
    data = np.load(data_path)
    
    # Convert to tensor for easier processing
    data_tensor = torch.from_numpy(data)
    
    # Calculate number of unique IDs for each trajectory
    unique_id_counts = []
    for trajectory in data_tensor:
        unique_ids = torch.unique(trajectory)
        unique_id_counts.append(len(unique_ids))
    
    # Convert to numpy array for easier analysis
    unique_id_counts = np.array(unique_id_counts)
    
    # Calculate percentiles for 5 levels
    percentiles = np.percentile(unique_id_counts, [10,20,30, 40,50, 60,70, 80,90,100])
    
    # Categorize trajectories into 5 levels
    levels = np.zeros_like(unique_id_counts)
    levels[unique_id_counts <= percentiles[0]] = 1
    levels[(unique_id_counts > percentiles[0]) & (unique_id_counts <= percentiles[1])] = 2
    levels[(unique_id_counts > percentiles[1]) & (unique_id_counts <= percentiles[2])] = 3
    levels[(unique_id_counts > percentiles[2]) & (unique_id_counts <= percentiles[3])] = 4
    levels[unique_id_counts > percentiles[3]] = 5
    # Print thresholds for each level
    print("\nThresholds for each level:")
    print(f"Level 1: <= {percentiles[0]:.2f} unique IDs")
    print(f"Level 2: {percentiles[1]:.2f} < x <= {percentiles[2]:.2f} unique IDs")
    print(f"Level 3: {percentiles[2]:.2f} < x <= {percentiles[3]:.2f} unique IDs")
    print(f"Level 4: {percentiles[3]:.2f} < x <= {percentiles[4]:.2f} unique IDs")
    print(f"Level 5: {percentiles[4]:.2f} < x <= {percentiles[5]:.2f} unique IDs")
    print(f"Level 6: {percentiles[5]:.2f} < x <= {percentiles[6]:.2f} unique IDs")
    print(f"Level 7: {percentiles[6]:.2f} < x <= {percentiles[7]:.2f} unique IDs")
    print(f"Level 8: {percentiles[7]:.2f} < x <= {percentiles[8]:.2f} unique IDs")
    print(f"Level 9: {percentiles[8]:.2f} < x <= {percentiles[9]:.2f} unique IDs")
    print(f"Level 10: > {percentiles[9]:.2f} unique IDs")
    # Print statistics
    print(f"Total number of trajectories: {len(unique_id_counts)}")
    print(f"Average number of unique IDs: {unique_id_counts.mean():.2f}")
    print(f"Minimum number of unique IDs: {unique_id_counts.min()}")
    print(f"Maximum number of unique IDs: {unique_id_counts.max()}")
    print("\nDistribution across 5 levels:")
    for level in range(1, 6):
        count = np.sum(levels == level)
        percentage = (count / len(levels)) * 100
        print(f"Level {level}: {count} trajectories ({percentage:.2f}%)")
    

def extract_and_save_od_matrix(data_path, save_path, max_location_id=900):
    """
    从轨迹数据中提取OD矩阵并保存
    Args:
        data_path: 轨迹数据路径
        save_path: OD矩阵保存路径
        max_location_id: 最大位置ID
    """
    # 加载轨迹数据
    data = torch.from_numpy(np.load(data_path))
    print(f"Loaded data shape: {data.shape}")
    
    # 初始化OD矩阵
    od_matrix = torch.zeros((max_location_id, max_location_id))
    
    # 计算OD矩阵
    print("Computing OD matrix...")
    for i in tqdm(range(len(data))):
        # 获取轨迹
        trajectory = data[i]  # 数据格式是 (200000, 168)
        for t in range(len(trajectory)-1):
            origin = trajectory[t].item()  # 转换为Python标量
            destination = trajectory[t+1].item()
            # 排除origin和destination相同的情况
            if origin < max_location_id and destination < max_location_id and origin != destination:
                od_matrix[origin, destination] += 1
    
    # 归一化OD矩阵
    print("Normalizing OD matrix...")
    row_sums = od_matrix.sum(dim=1, keepdim=True)
    # 避免除以零
    row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
    od_matrix = od_matrix / row_sums
    od_matrix = torch.nan_to_num(od_matrix, nan=0.0)  # 处理可能的NaN值
    
    # 保存OD矩阵
    print(f"Saving OD matrix to {save_path}...")
    torch.save(od_matrix, save_path)
    
    # 打印一些统计信息
    print("\nOD Matrix Statistics:")
    print(f"Shape: {od_matrix.shape}")
    print(f"Non-zero elements: {(od_matrix > 0).sum().item()}")
    print(f"Average non-zero probability: {od_matrix[od_matrix > 0].mean().item():.4f}")
    
    return od_matrix

def extract_and_save_population_dist(data_path, save_path, max_location_id=900):
    """
    从轨迹数据中提取人口分布并保存
    Args:
        data_path: 轨迹数据路径
        save_path: 人口分布保存路径
        max_location_id: 最大位置ID
    """
    # 加载轨迹数据
    data = torch.from_numpy(np.load(data_path))
    
    # 计算home locations
    print("Computing home locations...")
    reshaped_data = data.reshape(-1, 7, 24)[:, :, :6]
    home_locations = torch.tensor([    
        torch.bincount(chunk.flatten().long()).argmax().item()  # 转换为Python标量
        for chunk in tqdm(reshaped_data)
    ])
    
    # 计算人口分布
    print("Computing population distribution...")
    population_dist = torch.bincount(home_locations, minlength=max_location_id).float()
    population_dist = population_dist / population_dist.sum()
    
    # 保存人口分布
    print(f"Saving population distribution to {save_path}...")
    torch.save(population_dist, save_path)
    
    # 打印一些统计信息
    print("\nPopulation Distribution Statistics:")
    print(f"Shape: {population_dist.shape}")
    print(f"Non-zero locations: {(population_dist > 0).sum().item()}")
    print(f"Maximum population ratio: {population_dist.max().item():.4f}")
    
    return population_dist

def extract_and_save_population_dist_by_type(data_path, save_path_same, save_path_diff, max_location_id=900):
    """
    从轨迹数据中提取两种类型人群的人口分布并保存
    Args:
        data_path: 轨迹数据路径
        save_path_same: 相同位置人群的人口分布保存路径
        save_path_diff: 不同位置人群的人口分布保存路径
        max_location_id: 最大位置ID
    """
    # 加载轨迹数据
    data = torch.from_numpy(np.load(data_path))
    
    # 计算home和commute locations
    print("Computing home and commute locations...")
    reshaped_data = data.reshape(-1, 7, 24)
    
    # 计算home locations (前6小时)
    home_locations = torch.tensor([    
        torch.bincount(chunk[:, :8].flatten().long()).argmax().item()
        for chunk in tqdm(reshaped_data)
    ])
    
    # 计算commute locations (工作日10-18点)
    commute_locations = torch.tensor([    
        torch.bincount(chunk[:5, 10:18].flatten().long()).argmax().item()
        for chunk in tqdm(reshaped_data)
    ])
    
    # 判断home和commute是否相同
    same_location_mask = home_locations == commute_locations
    
    # 计算两种类型的人口分布
    print("Computing population distributions by type...")
    population_dist_same = torch.bincount(
        home_locations[same_location_mask], 
        minlength=max_location_id
    ).float()
    population_dist_same = population_dist_same / population_dist_same.sum()
    
    population_dist_diff = torch.bincount(
        home_locations[~same_location_mask], 
        minlength=max_location_id
    ).float()
    population_dist_diff = population_dist_diff / population_dist_diff.sum()
    
    # 保存人口分布
    print(f"Saving population distributions to {save_path_same} and {save_path_diff}...")
    torch.save(population_dist_same, save_path_same)
    torch.save(population_dist_diff, save_path_diff)
    
    # 打印一些统计信息
    print("\nPopulation Distribution Statistics:")
    print(f"Same location ratio: {same_location_mask.float().mean().item():.4f}")
    print(f"Different location ratio: {(~same_location_mask).float().mean().item():.4f}")
    print(f"Same location distribution shape: {population_dist_same.shape}")
    print(f"Different location distribution shape: {population_dist_diff.shape}")
    print(f"Non-zero locations in same distribution: {(population_dist_same > 0).sum().item()}")
    print(f"Non-zero locations in different distribution: {(population_dist_diff > 0).sum().item()}")
    
    return population_dist_same, population_dist_diff

def extract_and_save_commute_dist_by_home(data_path, save_path, max_location_id=900):
    """
    从轨迹数据中提取每个home location对应的commute location分布并保存
    Args:
        data_path: 轨迹数据路径
        save_path: 保存路径
        max_location_id: 最大位置ID
    """
    # 加载轨迹数据
    data = torch.from_numpy(np.load(data_path))
    
    # 计算home和commute locations
    print("Computing home and commute locations...")
    reshaped_data = data.reshape(-1, 7, 24)
    
    # 计算home locations (前6小时)
    home_locations = torch.tensor([    
        torch.bincount(chunk[:, :6].flatten().long()).argmax().item()
        for chunk in tqdm(reshaped_data)
    ])
    
    # 计算commute locations (工作日10-18点)
    commute_locations = torch.tensor([    
        torch.bincount(chunk[:5, 10:18].flatten().long()).argmax().item()
        for chunk in tqdm(reshaped_data)
    ])
    
    # 只考虑home和commute不同的情况
    diff_location_mask = home_locations != commute_locations
    home_locations_diff = home_locations[diff_location_mask]
    commute_locations_diff = commute_locations[diff_location_mask]
    
    # 初始化commute分布矩阵
    commute_dist_by_home = torch.zeros((max_location_id, max_location_id))
    
    # 计算每个home location对应的commute location分布
    print("Computing commute distribution by home location...")
    for i in tqdm(range(len(home_locations_diff))):
        home_loc = home_locations_diff[i].item()
        commute_loc = commute_locations_diff[i].item()
        
        if home_loc < max_location_id and commute_loc < max_location_id:
            commute_dist_by_home[home_loc, commute_loc] += 1
    
    # 归一化commute分布矩阵
    print("Normalizing commute distribution matrix...")
    row_sums = commute_dist_by_home.sum(dim=1, keepdim=True)
    # 避免除以零
    row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
    commute_dist_by_home = commute_dist_by_home / row_sums
    commute_dist_by_home = torch.nan_to_num(commute_dist_by_home, nan=0.0)  # 处理可能的NaN值
    
    # 保存commute分布矩阵
    print(f"Saving commute distribution matrix to {save_path}...")
    torch.save(commute_dist_by_home, save_path)
    
    # 打印一些统计信息
    print("\nCommute Distribution Matrix Statistics:")
    print(f"Shape: {commute_dist_by_home.shape}")
    print(f"Non-zero elements: {(commute_dist_by_home > 0).sum().item()}")
    print(f"Average non-zero probability: {commute_dist_by_home[commute_dist_by_home > 0].mean().item():.4f}")
    
    return commute_dist_by_home

def extract_and_save_home_commute_pairs(data_path, save_path, max_location_id=900):
    """
    从轨迹数据中提取真实的home-commute对应关系并保存
    Args:
        data_path: 轨迹数据路径
        save_path: 保存路径
        max_location_id: 最大位置ID
    """
    # 加载轨迹数据
    data = torch.from_numpy(np.load(data_path))
    
    # 计算home和commute locations
    print("Computing home and commute locations...")
    reshaped_data = data.reshape(-1, 7, 24)
    
    # 计算home locations (前6小时)
    home_locations = torch.tensor([    
        torch.bincount(chunk[:, :6].flatten().long()).argmax().item()
        for chunk in tqdm(reshaped_data)
    ])
    
    # 计算commute locations (工作日10-18点)
    commute_locations = torch.tensor([    
        torch.bincount(chunk[:5, 10:18].flatten().long()).argmax().item()
        for chunk in tqdm(reshaped_data)
    ])
    
    # 创建home-commute对列表
    home_commute_pairs = []
    for i in range(len(home_locations)):
        home_loc = home_locations[i].item()
        commute_loc = commute_locations[i].item()
        
        if home_loc < max_location_id and commute_loc < max_location_id:
            home_commute_pairs.append((home_loc, commute_loc))
    
    # 保存home-commute对
    print(f"Saving {len(home_commute_pairs)} home-commute pairs to {save_path}...")
    torch.save(home_commute_pairs, save_path)
    
    # 打印一些统计信息
    print("\nHome-Commute Pairs Statistics:")
    print(f"Total pairs: {len(home_commute_pairs)}")
    
    # 计算相同位置和不同位置的比例
    same_location_count = sum(1 for h, c in home_commute_pairs if h == c)
    diff_location_count = len(home_commute_pairs) - same_location_count
    print(f"Same location pairs: {same_location_count} ({same_location_count/len(home_commute_pairs):.4f})")
    print(f"Different location pairs: {diff_location_count} ({diff_location_count/len(home_commute_pairs):.4f})")
    
    return home_commute_pairs

if __name__ == "__main__":
    # 设置路径
    data_path = './data/trajs_data/Shenzhen/week_trajs.npy'
    # od_matrix_path = './data/trajs_data/Beijing/od_matrix.pt'
    population_dist_path = './data/trajs_data/Shenzhen/population_dist.pt'
    population_dist_same_path = './data/trajs_data/Shenzhen/population_dist_same.pt'
    population_dist_diff_path = './data/trajs_data/Shenzhen/population_dist_diff.pt'
    commute_dist_by_home_path = './data/trajs_data/Shenzhen/commute_dist_by_home.pt'
    home_commute_pairs_path = './data/trajs_data/Shenzhen/home_commute_pairs.pt'
    
    # 提取并保存OD矩阵
    # od_matrix = extract_and_save_od_matrix(data_path, od_matrix_path)
    max_location_id=2236
    # 提取并保存人口分布
    population_dist = extract_and_save_population_dist(data_path, population_dist_path,max_location_id=max_location_id)
    
    # 提取并保存两种类型人群的人口分布
    population_dist_same, population_dist_diff = extract_and_save_population_dist_by_type(
        data_path, population_dist_same_path, population_dist_diff_path,max_location_id=max_location_id
    )
    
    # 提取并保存每个home location对应的commute location分布
    commute_dist_by_home = extract_and_save_commute_dist_by_home(
        data_path, commute_dist_by_home_path,max_location_id=max_location_id
    )
    
    # 提取并保存真实的home-commute对应关系
    home_commute_pairs = extract_and_save_home_commute_pairs(
        data_path, home_commute_pairs_path,max_location_id=max_location_id
    )

    
    
    