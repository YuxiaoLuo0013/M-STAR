import torch
import torch.nn as nn
import math
from functools import partial
from typing import Union, Tuple, Optional, List
from .vqvae import VQVAE
from .quant import VectorQuantizer
import numpy as np
import torch.nn.functional as F
import bisect
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class AdaLN(nn.Module):
    """Adaptive Layer Normalization"""
    def __init__(self, hidden_size, cond_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.hidden_size = hidden_size
        # 生成gamma, scale和shift参数
        self.ada_lin = nn.Sequential(
            nn.Linear(cond_dim, hidden_size * 3),  # 现在生成3个参数：gamma, scale, shift
            nn.SiLU(),
            nn.Linear(hidden_size * 3, hidden_size * 3)
        )

    def forward(self, x, cond):
        # x: (B, L, C), cond: (B, D)
        params = self.ada_lin(cond)  # (B, 3*C)
        gamma, scale, shift = params.chunk(3, dim=-1)  # 各自 (B, C)
        
        # 计算均值和方差
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # 归一化并应用scale和shift
        norm_x = (x - mean) / (var + self.eps).sqrt()
        norm_x = norm_x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        
        # 返回归一化后的x和gamma，让调用者决定如何使用gamma
        return norm_x, gamma

class NextScaleGenerator(nn.Module):
    def __init__(
        self, 
        vqvae: VQVAE,
        hidden_size=512, 
        n_heads=8, 
        n_layers=4,
        mlp_ratio=4.,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path_rate=0.1,
        cond_drop_rate=0.1,
        num_scales=4,
        max_seq_len=168,
        temperature=0.8,
        top_k=0,
    ):
        super().__init__()
        assert hidden_size % n_heads == 0
        self.moving_avg = moving_avg(kernel_size=3,stride=1)
        # 基础参数
        self.Cvae = vqvae.embedding_dim
        self.vocab_size = vqvae.vocab_size
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.num_scales = num_scales
        self.cond_drop_rate = cond_drop_rate
        self.temporal_scale_nums = vqvae.quantizer.temporal_scale_nums
        
        # 预计算训练步数间隔 - 从后向前计算
        reversed_scale_nums = list(reversed(self.temporal_scale_nums[:-1]))  # 排除最后一个尺度
        self.scale_steps = [max(500,int(scale_len * 50)) for scale_len in reversed_scale_nums]
        self.cumulative_steps = [0] + [sum(self.scale_steps[:i+1]) for i in range(len(self.scale_steps))]

        # VQVAE代理
        
        # 1. 输入嵌入
        self.token_embed = nn.Linear(self.Cvae, hidden_size)
        
        # 2. home location嵌入
        # self.home_embed = nn.Embedding(2236, int(hidden_size/2))
        self.home_embed = nn.Embedding(900, int(hidden_size/2))
        # 添加home token的位置嵌入
        self.home_pos_embed = nn.Parameter(torch.zeros(1, hidden_size))
 
        
        # 3. 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, np.sum(self.temporal_scale_nums), hidden_size))

        
        # 4. 尺度编码
        self.scale_embed = nn.Embedding(len(self.temporal_scale_nums), hidden_size)
        
        # 预计算scale_ids
        scale_ids = []
        for scale_idx, scale_len in enumerate(self.temporal_scale_nums):
            # 为每个尺度创建重复的索引
            scale_ids.extend([scale_idx] * scale_len)
        self.register_buffer('scale_ids', torch.tensor(scale_ids))

        # 5. 共享的AdaLN
        self.shared_attn_norm = AdaLN(hidden_size, hidden_size)
        self.shared_ffn_norm = AdaLN(hidden_size, hidden_size)
        
        # 6. Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'attn': nn.MultiheadAttention(
                    embed_dim=hidden_size,
                    num_heads=n_heads,
                    dropout=attn_drop_rate,
                    batch_first=True
                ),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
                    nn.GELU(),
                    nn.Dropout(drop_rate),
                    nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
                    nn.Dropout(drop_rate)
                ),
                'drop_path': nn.Dropout(dpr[i])
            }) for i in range(n_layers)
        ])
        # 7. 输出头
        self.final_norm = AdaLN(hidden_size, hidden_size)
        self.head = nn.Linear(hidden_size, self.vocab_size)
        # self.head2 = nn.Linear(hidden_size,2)
        # 预计算多尺度注意力掩码
        self._create_multiscale_attention_masks()
        
        # 初始化权重
        self.apply(self._init_weights)

        self.vae_proxy = vqvae  # 直接存储引用，不使用元组
        quant: VectorQuantizer = vqvae.quantizer
        self.vae_quant_proxy = quant  # 直接存储引用，不使用元组
        self.temperature=temperature
        self.top_k=10
        self.unique_id_count_embed = nn.Embedding(8, hidden_size)


        # 添加KV Cache相关的属性
        self.kv_cache = None
        self.cache_len = 0
        self.batch_size = 0

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Parameter):
            nn.init.normal_(m, std=0.02)

    def _create_multiscale_attention_masks(self):
        """预计算所有可能的多尺度注意力掩码"""
        device = next(self.parameters()).device
        
        # 创建尺度索引张量
        scale_indices = []

        
        for scale_idx, scale_len in enumerate(self.temporal_scale_nums):
            # 为每个尺度创建对应的索引张量
            scale_indices.append(torch.full((scale_len,), scale_idx + 1, device=device))
        
        d = torch.cat(scale_indices).view(-1, 1)  # shape: (total_len, 1)
        dT = d.transpose(0, 1)  # shape: (1, total_len)
        
        scale_mask = (d >= dT).float()  # shape: (total_len, total_len)
        
        # 将0/1掩码转换为0/-inf掩码
        attn_mask = torch.where(scale_mask > 0, 0., float('-inf'))  # shape: (total_len, total_len)
        
        # 为每个尺度创建对应的掩码
        for scale_idx in range(len(self.temporal_scale_nums)):
            # 计算当前尺度的总长度
            prefix_len = 0  
            for i in range(scale_idx + 1):
                prefix_len += self.temporal_scale_nums[i]
            
            # 提取当前尺度的掩码
            curr_mask = attn_mask[:prefix_len, :prefix_len]
            
            # 注册为buffer
            self.register_buffer(f'attention_mask_scale_{scale_idx}', curr_mask.contiguous())

    def _init_kv_cache(self, batch_size, device):
        """初始化KV Cache"""
        self.kv_cache = [{
            'key': torch.zeros(batch_size, self.n_heads, 0, self.hidden_size // self.n_heads, device=device),
            'value': torch.zeros(batch_size, self.n_heads, 0, self.hidden_size // self.n_heads, device=device)
        } for _ in range(len(self.blocks))]
        self.cache_len = 0
        self.batch_size = batch_size

    def _update_kv_cache(self, key, value, layer_idx, scale_idx):
        """更新KV Cache"""
        # 直接更新cache，保留所有token
        # 确保 key 和 value 的维度正确
        if self.cache_len == 0:
            # 如果是第一次更新，直接赋值
            self.kv_cache[layer_idx]['key'] = key
            self.kv_cache[layer_idx]['value'] = value
        else:
            # 否则进行拼接
            self.kv_cache[layer_idx]['key'] = torch.cat([self.kv_cache[layer_idx]['key'], key], dim=2)
            self.kv_cache[layer_idx]['value'] = torch.cat([self.kv_cache[layer_idx]['value'], value], dim=2)

    def _clear_kv_cache(self):
        """清除KV Cache"""
        self.kv_cache = None
        self.cache_len = 0
    def tf_generate(self, locs: list, home_locations: torch.Tensor,cluster:torch.Tensor):
        # temperature=[1,0.9,0.75,0.5,0.25,0.1,0.1,0.1]
        with torch.no_grad():
            ids_list = self.vae_proxy.encode_vqvae(locs)
            ids_list=torch.cat(ids_list,dim=-1)
            B = ids_list.size(0)
            
            
            # 1. 获取条件信息
            home_emb = self.home_embed(home_locations).reshape(B,-1)  # (B, C)
            
            # 2. 处理所有尺度的输入序列
            x_wo_ids_0= self.vae_quant_proxy.lds_to_generator_input(ids_list)
            cluster_embed = self.unique_id_count_embed(cluster)
            home_emb=home_emb+cluster_embed
            x_wo_ids_0=self.token_embed(x_wo_ids_0)
            # 使用预计算的scale_ids添加尺度嵌入
            batch_scale_ids = self.scale_ids.expand(B, -1)  # 扩展到批次大小
            x_wo_ids_0 = x_wo_ids_0 + self.scale_embed(batch_scale_ids)
            
            # 3. 拼接所有尺度的tokens
            x = x_wo_ids_0  # 使用处理后的x_wo_ids_0作为x
            
            # 4. 添加位置编码
            x = x + self.pos_embed[:, :x.size(1)]
            
            # 5. 添加home location条件
            home_emb_with_pos = home_emb + self.home_pos_embed  # 添加位置嵌入
            x[:,0,:] = home_emb_with_pos+self.pos_embed[:,0,:]+self.scale_embed(torch.full((B,), 0, device=x.device))

            attn_mask = getattr(self, f'attention_mask_scale_{len(self.temporal_scale_nums)-1}')

            # 7. Transformer处理
            for block in self.blocks:
                # Self-attention with shared AdaLN
                normed_x, gamma1 = self.shared_attn_norm(x, home_emb)
                attn_out, _ = block['attn'](
                    query=normed_x,
                    key=normed_x,
                    value=normed_x,
                    attn_mask=attn_mask
                )
                # 应用gamma到attention输出
                x = x + block['drop_path'](attn_out * gamma1.unsqueeze(1))
                
                # FFN with shared AdaLN
                normed_x, gamma2 = self.shared_ffn_norm(x, home_emb)
                ffn_out = block['ffn'](normed_x)
                # 应用gamma到FFN输出
                x = x + block['drop_path'](ffn_out * gamma2.unsqueeze(1))
            
            # 8. 输出头
            x, _ = self.final_norm(x, home_emb)
            logits = self.head(x)
            
            if self.top_k > 0:
                values, _ = torch.topk(logits, self.top_k, dim=-1)  
                min_values = values[:, :, -1].unsqueeze(-1)
                logits = torch.where(logits < min_values, 
                                torch.full_like(logits, -torch.inf), 
                                logits)
                probs = F.softmax(logits, dim=-1)
            
            # 为每个位置采样10个不同的token
            flat_probs = probs.reshape(-1, probs.size(-1))
            next_tokens_flat = torch.multinomial(flat_probs, 1, replacement=True)
            next_tokens = next_tokens_flat.view(B, -1)
            
            return next_tokens,ids_list

    def forward(self, locs: list, home_locations: torch.Tensor, cluster: torch.Tensor, train_step:int=0):
        ids_list = self.vae_proxy.encode_vqvae(locs)
        ids_list=torch.cat(ids_list,dim=-1)
        B = ids_list.size(0)
        x_wo_ids_0= self.vae_quant_proxy.lds_to_generator_input(ids_list)
        cluster_embed = self.unique_id_count_embed(cluster)

        home_emb = self.home_embed(home_locations).reshape(B,-1)+cluster_embed
        x_wo_ids_0=self.token_embed(x_wo_ids_0)
        # 使用预计算的scale_ids添加尺度嵌入
        batch_scale_ids = self.scale_ids.expand(B, -1)  # 扩展到批次大小
        x_wo_ids_0 = x_wo_ids_0 + self.scale_embed(batch_scale_ids)
        
        # 3. 拼接所有尺度的tokens
        x = x_wo_ids_0  # 使用处理后的x_wo_ids_0作为x
        
        # 4. 添加位置编码
        x = x + self.pos_embed[:, :x.size(1)]
        
        # 5. 添加home location条件
        home_emb_with_pos = home_emb + self.home_pos_embed  # 添加位置嵌入
        x[:,0,:] = home_emb_with_pos+self.pos_embed[:,0,:]+self.scale_embed(torch.full((B,), 0, device=x.device))
        attn_mask = getattr(self, f'attention_mask_scale_{len(self.temporal_scale_nums)-1}')

        # 7. Transformer处理
        for block in self.blocks:
            # Self-attention with shared AdaLN
            normed_x, gamma1 = self.shared_attn_norm(x, home_emb)
            attn_out, _ = block['attn'](
                query=normed_x,
                key=normed_x,
                value=normed_x,
                attn_mask=attn_mask
            )
            # 应用gamma到attention输出
            x = x + block['drop_path'](attn_out * gamma1.unsqueeze(1))
            
            # FFN with shared AdaLN
            normed_x, gamma2 = self.shared_ffn_norm(x, home_emb)
            ffn_out = block['ffn'](normed_x)
            # 应用gamma到FFN输出
            x = x + block['drop_path'](ffn_out * gamma2.unsqueeze(1))
        
        # 8. 输出头
        x, _ = self.final_norm(x, home_emb)
        logits = self.head(x)
        # logits_transtion=self.head2(x)
        output={
            'logits':logits,
            # 'logits_transtion':logits_transtion,
            'target_ids':ids_list,

        }
        return output

    def generate(self, home_locations: torch.Tensor, cluster: torch.Tensor = None):
        with torch.no_grad():
            B = home_locations.size(0)
            device = home_locations.device
            
            # 初始化KV Cache
            self._init_kv_cache(B, device)

            
            home_emb = self.home_embed(home_locations).reshape(B,-1)  # (B, C)
            cluster_embed = self.unique_id_count_embed(cluster)

            home_emb = home_emb+cluster_embed
            
            # 初始化
            x = home_emb.unsqueeze(1)  # 从home token开始
            next_tokens = []
            f_hat_next = 0  # 初始化为None更合适
            # 逐尺度生成
            for i in range(len(self.temporal_scale_nums)):
                # 准备当前尺度的输入
                
                if i == 0:
                    x=x+self.pos_embed[:,0,:]+self.scale_embed(torch.full((B, 1), i, device=x.device))
                else:
                    # 后续尺度，使用前一尺度的信息
                    x_next, f_hat_next = self.vae_quant_proxy.get_next_autoregressive_input(
                        f_hat_next, 
                        next_tokens[i-1], 
                        i
                    )
                    scale_emb = self.scale_embed(torch.full((B, x_next.size(1)), i, device=x.device))
                    x_next=self.token_embed(x_next)
                    x_next = x_next + scale_emb
                    x_next=x_next+self.pos_embed[:,sum(self.temporal_scale_nums[:i]):sum(self.temporal_scale_nums[:i])+self.temporal_scale_nums[i],:]
                    x = x_next
                # 获取当前尺度的注意力掩码
                # attn_mask = getattr(self, f'attention_mask_scale_{i}')

                # Transformer blocks 处理
                
                # 使用KV Cache的Transformer处理
                for block_idx, block in enumerate(self.blocks):
                    normed_x, gamma1 = self.shared_attn_norm(x, home_emb)
                    

                    # Self-attention with shared AdaLN
                                        
                    # 手动计算 QKV
                    qkv = F.linear(normed_x, block['attn'].in_proj_weight, block['attn'].in_proj_bias)
                    q, k, v = qkv.chunk(3, dim=-1)
                    
                    # 重塑为多头形式
                    B, L, C = q.shape
                    q = q.view(B, L, self.n_heads, C // self.n_heads).transpose(1, 2)
                    k = k.view(B, L, self.n_heads, C // self.n_heads).transpose(1, 2)
                    v = v.view(B, L, self.n_heads, C // self.n_heads).transpose(1, 2)
                    
                    # 使用KV Cache
                    if self.cache_len > 0:
                        key = torch.cat([self.kv_cache[block_idx]['key'], k], dim=2)
                        value = torch.cat([self.kv_cache[block_idx]['value'], v], dim=2)
                    else:
                        key = k
                        value = v
                    
                    # 更新KV Cache
                    self._update_kv_cache(k, v, block_idx, self.temporal_scale_nums[i])
                    
                    # 使用 scaled_dot_product_attention
                    attn_out = F.scaled_dot_product_attention(
                        q, key, value,
                        dropout_p=0.0  # 生成时不使用dropout
                    )
                    
                    # 重塑回原始形状
                    attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, C)
                    attn_out = block['attn'].out_proj(attn_out)
                    x = x + attn_out * gamma1.unsqueeze(1)
                    
                    # FFN with shared AdaLN
                    normed_x, gamma2 = self.shared_ffn_norm(x, home_emb)
                    ffn_out = block['ffn'](normed_x)
                    x = x + ffn_out * gamma2.unsqueeze(1)



                
                # 最终的归一化和预测
                self.cache_len += key.size(2)  # 更新cache长度，加上当前scale的token数量

                x, _ = self.final_norm(x, home_emb)
                logits = self.head(x)
                
                # 对每个位置的每个vocabulary索引使用不同的温度
                # 首先获取每个位置的top1 logits值和索引
                top1_values, top1_indices = torch.max(logits, dim=-1)  # 每个都是(B, L)
                
                # 在序列维度上归一化top1值，即每个batch内的序列归一化
                # 每个位置都会基于自身top1 logits的置信度获得一个温度
                batch_max = top1_values.max(dim=1, keepdim=True)[0]  # (B, 1)
                batch_min = top1_values.min(dim=1, keepdim=True)[0]  # (B, 1)
                
                # 防止除零
                denominator = batch_max - batch_min
                denominator = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
                
                # 归一化top1值，得到每个位置的置信度评分 (B, L)
                token_confidence = (top1_values - batch_min) / denominator
                
                # token_temp_factors = 1.4 - 1.0 * token_confidence  # 深圳

                token_temp_factors = 1.5 - 1 * token_confidence  # 北京
                # Top-k sampling - 先进行top-k过滤
                if self.top_k > 0:
                    # 先执行top-k过滤
                    values, _ = torch.topk(logits, self.top_k, dim=-1)  
                    min_values = values[:, :, -1].unsqueeze(-1)
                    
                    # 创建掩码，将top-k以外的logits设为-inf
                    mask = logits < min_values
                    logits = logits.masked_fill(mask, float('-inf'))
                
                # 温度缩放 - 对每个位置应用独立的温度
                # 扩展维度以便广播
                token_temp = token_temp_factors.unsqueeze(-1)  # (B, L, 1)
                
                # 应用温度缩放
                scaled_logits = logits / token_temp

                probs=F.softmax(scaled_logits,dim=-1)
                
                # 采样
                flat_probs = probs.reshape(-1, probs.size(-1))
                next_tokens_flat = torch.multinomial(flat_probs, 1, replacement=True)
                next_token = next_tokens_flat.view(B, -1)
                next_tokens.append(next_token)
                
            # 处理最终输出
            f_last = f_hat_next + self.vae_quant_proxy.embedding(next_tokens[-1])
            f_last = self.moving_avg(f_last)
 

            
            f_last=f_last+self.vae_proxy.pos_embedding2
            f_last = self.vae_proxy.transformer_decoder(f_last)
            x_hat_loc = self.vae_proxy.final_proj(f_last).argmax(dim=-1)

            return { 
                'x_hat_loc': x_hat_loc,
                'output_ids': next_tokens
            }
