from typing import List, Optional, Sequence, Tuple, Union, Dict

import numpy as np
import torch
from torch import distributed as tdist, nn as nn
from torch.nn import functional as F
import math
class VectorQuantizer(nn.Module):
    # VQGAN originally use beta=1.0, never tried 0.25; SD seems using 0.25
    def __init__(
        self, vocab_size, channel_vae, using_znorm, beta: float = 0.25,
        temporal_scale_nums=None,  # share_quant_resi: args.qsr
    ):
        super().__init__()
        self.vocab_size: int = vocab_size
        self.channel_vae: int = channel_vae
        self.using_znorm: bool = using_znorm
        self.temporal_scale_nums: list[int] = temporal_scale_nums[::-1] if temporal_scale_nums is not None else None # 从大尺度到小尺度 
        self.quant_resi = Phi(channel_vae)
        
        
        self.register_buffer('ema_vocab_hit_SV', torch.full((len(self.temporal_scale_nums), self.vocab_size), fill_value=0.0))
        self.record_hit = 0
        
        self.beta: float = beta
        self.embedding = nn.Embedding(self.vocab_size, self.channel_vae)
        
        # # only used for progressive training of VAR (not supported yet, will be tested and supported in the future)
        # self.prog_si = -1   # progressive training: not supported yet, prog_si always -1


    def eini(self, eini):
        if eini > 0: nn.init.trunc_normal_(self.embedding.weight.data, std=eini)
        elif eini < 0: self.embedding.weight.data.uniform_(-abs(eini) / self.vocab_size, abs(eini) / self.vocab_size)
    
    # ===================== `forward` is only used in VAE training =====================
    def forward(self, f_BSLW: torch.Tensor, ret_usages=False) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
        dtype = f_BSLW.dtype
        if dtype != torch.float32: f_BSLW = f_BSLW.float()
        B, S, L, W = f_BSLW.shape
        f_multi = f_BSLW

        ids_queue = []
        down_f_queue = []
        f_rest_queue=[]
        with torch.cuda.amp.autocast(enabled=False):
            mean_vq_loss: torch.Tensor = 0.0
            vocab_hit_V = torch.zeros(self.vocab_size, dtype=torch.float, device=f_BSLW.device)
            
            for i, temporal_scale in enumerate(self.temporal_scale_nums): # 从大尺度到小尺度
                # find the nearest embedding
                if self.using_znorm:   
                    #downsample
                    if i != len(self.temporal_scale_nums)-1:
                        down_f = F.interpolate(f_multi[:,-i,:,:].transpose(1, 2).unsqueeze(1),
                                            size=(W,temporal_scale),
                                         mode='area').squeeze(1).transpose(1, 2)

                    else:
                        down_f=f_multi[:,0,:,:]

                    if i == 0:
                        up_h_BLW = torch.zeros_like(down_f)   
                    rest_f = down_f - up_h_BLW

                    f_rest_queue.append(rest_f)
                    rest_f = F.normalize(rest_f, dim=-1)  # 确保残差也被归一化
                    idx_N = torch.argmax(rest_f @ F.normalize(self.embedding.weight.data, dim=1).T, dim=2)
                # 统计codebook使用情况
                hit_V = idx_N.view(-1).bincount(minlength=self.vocab_size).float()
                vocab_hit_V.add_(hit_V)
                
                # 更新EMA统计
                if self.training:
                    if self.record_hit == 0:
                        self.ema_vocab_hit_SV[i].copy_(hit_V)
                    elif self.record_hit < 100:
                        self.ema_vocab_hit_SV[i].mul_(0.9).add_(hit_V.mul(0.1))
                    else:
                        self.ema_vocab_hit_SV[i].mul_(0.99).add_(hit_V.mul(0.01))
                
                idx_Bhw = idx_N.view(B, temporal_scale)
                ids_queue.append(idx_Bhw)
                down_f_queue.append(down_f)
                if i == len(self.temporal_scale_nums)-1:
                    break
                #upsample
                up_temporal_scale = self.temporal_scale_nums[i+1]
                up_h_BLW = F.interpolate(down_f.transpose(1, 2).unsqueeze(1),
                                       size=(W,up_temporal_scale),
                                       mode='bicubic').squeeze(1).transpose(1, 2)
                up_h_BLW = self.quant_resi[i](up_h_BLW)
            
                # calc loss
            commitment_loss = 0
            for i, ids in enumerate(ids_queue):
                s_rest = self.embedding(ids)
                # s_rest = down_f_queue[i] + (s_rest - down_f_queue[i]).detach()
                if i == 0:
                    f_hat = s_rest
                    # f_hat=f_rest_queue[i]
                else:
                    f_hat = F.interpolate(f_hat.transpose(1, 2).unsqueeze(1),
                                        size=(W,self.temporal_scale_nums[i]),
                                        mode='bicubic').squeeze(1).transpose(1, 2)
                    f_hat = self.quant_resi[i-1](f_hat)+s_rest

                commitment_loss =commitment_loss+F.mse_loss(down_f_queue[i], f_hat.detach())
                # mean_vq_loss+=F.mse_loss(f_hat, down_f_queue[i])
                # f_hat = down_f_queue[i] + (f_hat - down_f_queue[i]).detach()
                # norm_f_hat = F.normalize(f_hat, dim=-1)
                # norm_down_f = F.normalize(down_f_queue[i], dim=-1)
            codebook_loss = F.mse_loss(f_hat,down_f_queue[i].detach())  # [B]

            # commitment_loss = self.beta*F.mse_loss(down_f_queue[i], f_hat.detach())  # [B] 编码器输出靠近编码本向量

            mean_vq_loss = codebook_loss + (0.1* commitment_loss/len(self.temporal_scale_nums))
                
                
            # mean_vq_loss=mean_vq_loss/len(self.temporal_scale_nums)
            f_hat = f_BSLW.mean(1) + (f_hat - f_BSLW.mean(1)).detach()

                # 应用 straight-through estimator
                # f_hat = down_f_queue[i] + (f_hat - down_f_queue[i]).detach()
                # f_hat= down_f_queue[i] + (f_hat - down_f_queue[i]).detach()
            if self.training:
                self.record_hit += 1

            # 计算usages
            if ret_usages:
                # 计算每个尺度的使用率
                scale_usages = []
                for si in range(len(self.temporal_scale_nums)):
                    margin = (f_BSLW.numel() / f_BSLW.shape[3]) / self.vocab_size * 0.05
                    scale_usage = (self.ema_vocab_hit_SV[si] >= margin).float().mean().item() * 100
                    scale_usages.append(scale_usage)
                
                # 先对所有尺度的使用频率求和
                total_hit_count = torch.sum(self.ema_vocab_hit_SV, dim=0)  # [vocab_size]
                
                # 基于总和计算使用率
                # 这里使用相同的margin计算标准
                margin = (f_BSLW.numel() / f_BSLW.shape[3]) / self.vocab_size * 0.01
                overall_usage = (total_hit_count >= margin).float().mean().item() * 100
                
                usages = overall_usage
            else:
                usages = None
            # mean_vq_loss=F.mse_loss(f_hat,f_BSLW[:,0,:,:])
            return f_hat, usages, mean_vq_loss
    
    def encode_quant(self, f_BSLW: torch.Tensor) -> List[torch.Tensor]:
        """
        将输入序列编码为离散索引列表
        Args:
            f_BSLW: 输入张量，形状为 [B, S, L, W]
        Returns:
            ids_list: 各个尺度的编码索引列表
        """
        B, S, L, W = f_BSLW.shape
        f_multi = f_BSLW
        ids_list = []
        
        with torch.no_grad():
            up_h_BLW = None
            
            for i, temporal_scale in enumerate(self.temporal_scale_nums):
                # 下采样处理
                if i != len(self.temporal_scale_nums)-1:
                    down_f = F.interpolate(
                        f_multi[:,-i,:,:].transpose(1, 2).unsqueeze(1),
                        size=(W, temporal_scale),
                        mode='area'
                    ).squeeze(1).transpose(1, 2)
                else:
                    down_f = f_multi[:,0,:,:]
                
                # 初始化第一个尺度的up_h_BLW
                if i == 0:
                    up_h_BLW = torch.zeros_like(down_f)
                
                # 计算残差
                rest_f = down_f - (up_h_BLW if up_h_BLW is not None else 0)
                
                # 根据是否使用z-normalization选择不同的编码方式
                if self.using_znorm:
                    rest_f = F.normalize(rest_f, dim=-1)
                    idx_N = torch.argmax(
                        rest_f @ F.normalize(self.embedding.weight.data, dim=1).T, 
                        dim=2
                    )
                else:
                    d_no_grad = torch.sum(rest_f.square(), dim=-1, keepdim=True)
                    d_no_grad = d_no_grad + torch.sum(self.embedding.weight.data.square(), dim=1)
                    d_no_grad.addmm_(
                        rest_f, 
                        self.embedding.weight.data.T, 
                        alpha=-2, 
                        beta=1
                    )
                    idx_N = torch.argmin(d_no_grad, dim=-1)
                
                # 存储当前尺度的编码索引
                idx_Bhw = idx_N.view(B, temporal_scale)
                ids_list.append(idx_Bhw)
                
                # 如果不是最后一个尺度，准备下一个尺度的up_h_BLW
                if i < len(self.temporal_scale_nums)-1:
                    up_temporal_scale = self.temporal_scale_nums[i+1]
                    up_h_BLW = F.interpolate(
                        down_f.transpose(1, 2).unsqueeze(1),
                        size=(W, up_temporal_scale),
                        mode='bicubic'
                    ).squeeze(1).transpose(1, 2)
                    up_h_BLW = self.quant_resi[i](up_h_BLW)
        return ids_list

    def decode_quant(self, ids_list: List[torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            for i, ids in enumerate(ids_list):
                # 1. 从codebook中获取重建向量
                s_rest = self.embedding(ids)  # [B, temporal_scale, channel_vae]
                
                if i == 0:
                    # 第一个尺度直接使用重建向量
                    f_down = s_rest
                else:
                    # 后续尺度需要:
                    # 1. 将之前的结果上采样
                    f_hat = F.interpolate(
                        f_down.transpose(1, 2).unsqueeze(1),
                        size=(self.channel_vae, self.temporal_scale_nums[i]),
                        mode='bicubic'
                    ).squeeze(1).transpose(1, 2)
                    f_hat=self.quant_resi[i-1](f_hat)
                    # 2. 通过残差网络处理并加上当前尺度的重建向量
                    f_down = f_hat + s_rest
                    
            return f_down

    def lds_to_generator_input(self, ids_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将离散索引列表转换为生成器输入 for getting teacher-forcing input without ids_0
        Args:
            ids_list: 各个尺度的编码索引列表
        Returns:
            next_scale_input: 生成器输入
            target_ids: 目标索引，形状为 [B, num_ids]
        """
        next_scale_input = []

        #sos token
        s_rest = self.embedding(ids_list[:,0]).unsqueeze(1)
        next_scale = torch.zeros_like(s_rest)
        next_scale_input.append(next_scale)

        for i in range(len(self.temporal_scale_nums)-1):
            f_down=s_rest+next_scale
            f_hat= F.interpolate(
                f_down.transpose(1, 2).unsqueeze(1),
                size=(self.channel_vae, self.temporal_scale_nums[i+1]),
                mode='bicubic'
            ).squeeze(1).transpose(1, 2)
            f_hat=self.quant_resi[i](f_hat)
            next_scale=f_hat
            s_rest=self.embedding(ids_list[:,sum(self.temporal_scale_nums[:i+1]):self.temporal_scale_nums[i+1]+sum(self.temporal_scale_nums[:i+1])])
            next_scale_input.append(next_scale)
        # 将所有尺度的target_ids在第二个维度上拼接
        # target_ids = torch.cat(target_ids, dim=1)  # [B, sum(temporal_scales)]
        next_scale_input = torch.cat(next_scale_input, dim=1)
        return next_scale_input

    def get_next_autoregressive_input(self,f_hat:torch.Tensor,last_ids:torch.Tensor,scale_idx:int)->Tuple[torch.Tensor,torch.Tensor]:   
        f_down_next=f_hat+self.embedding(last_ids)
        f_down_next=F.interpolate(f_down_next.transpose(1, 2).unsqueeze(1),
                                 size=(self.channel_vae, self.temporal_scale_nums[scale_idx]),
                                 mode='bicubic').squeeze(1).transpose(1, 2)
        f_hat_next=self.quant_resi[scale_idx-1](f_down_next)
        next_scale_input=f_hat_next
        return next_scale_input,f_hat_next

    def get_next_autoregressive_input_train(self,f_hat:torch.Tensor,last_avg_embed:torch.Tensor,scale_idx:int)->Tuple[torch.Tensor,torch.Tensor]:   
        f_down_next=f_hat+last_avg_embed
        f_down_next=F.interpolate(f_down_next.transpose(1, 2).unsqueeze(1),
                                 size=(self.channel_vae, self.temporal_scale_nums[scale_idx]),
                                 mode='bicubic').squeeze(1).transpose(1, 2)
        f_hat_next=self.quant_resi[scale_idx-1](f_down_next)
        next_scale_input=f_hat_next
        return next_scale_input,f_hat_next


class Phi(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        # 定义3个不同的1D卷积层和对应的MLP
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
        ])
        
        # 3个不同的MLP层
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            ) for _ in range(7)
        ])
        
        self.resi_ratio = 0.5
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(7)
        ])

    def forward(self, h_BLW, idx):  # [B, L, W]
        # 1. 使用指定的卷积层
        h_conv = h_BLW.transpose(1, 2)  # [B, W, L]
        h_conv = self.convs[idx](h_conv)
        h_conv = h_conv.transpose(1, 2)  # [B, L, W]
        
        # 2. 第一个残差连接
        h_mid = h_BLW + h_conv * self.resi_ratio
        
        # 3. 使用对应的LayerNorm和MLP
        # h_norm = self.layer_norms[idx](h_mid)
        h_mlp = self.mlps[idx](h_mid)
        
        # 4. 第二个残差连接
        out = h_mid + h_mlp 

        return out

    def __getitem__(self, idx):
        # 返回一个使用特定卷积层和MLP的函数
        def forward_with_idx(h_BLW):
            return self.forward(h_BLW, idx)
        return forward_with_idx


