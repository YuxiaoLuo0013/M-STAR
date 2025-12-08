import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import numpy as np
from .quant import Phi, VectorQuantizer
from typing import List, Optional, Tuple  


class PerceiverResampler(nn.Module):
    def __init__(self, d_model, dim_feedforward, num_latents, num_layers):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, d_model))
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=d_model , num_heads=8, dropout=0.1, batch_first=True)
            for _ in range(num_layers)
        ])
        self.feedforward_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model,dim_feedforward),
                nn.ReLU(),  
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(0.1)
            )
            for _ in range(num_layers)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        # 将输入与latents结合
        q = self.latents.unsqueeze(0).expand(x.size(0), -1, -1)  # 扩展到批次大小
        for attn_layer, ff_layer, norm_layer in zip(self.attention_layers, self.feedforward_layers, self.norm_layers):
            # x=torch.cat([q,x],dim=1)
            attn_output, _ = attn_layer(q, x, x)
            q = norm_layer(q + attn_output)
            q = norm_layer(q + ff_layer(q))
        return q


class PatchSelfAttention(nn.Module):
    """在patch内部执行self-attention的模块"""
    def __init__(self, patch_size, feature_dim, hidden_dim=None, return_all=False):
        super().__init__()
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim or feature_dim
        
        self.norm = nn.LayerNorm(feature_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, feature_dim),
            nn.Dropout(0.1)
        )
        self.final_norm = nn.LayerNorm(feature_dim)
        self.return_all = return_all
    def forward(self, x):
        # x形状: [B*S, num_patches, patch_size, feature_dim]
        batch_size, num_patches, patch_size, feature_dim = x.shape
        
        # 重塑以处理每个patch
        x = x.view(-1, patch_size, feature_dim)  # [B*S*num_patches, patch_size, feature_dim]
        
        # 应用self-attention
        residual = x
        x = self.norm(x)
        attn_output, _ = self.attention(x, x, x)
        x = residual + attn_output
        
        # 前馈网络
        residual = x
        x = self.norm(x)
        x = residual + self.ff(x)
        x = self.final_norm(x)
        
        # 只取每个patch的最后一个token作为patch的表示
        # x = x[:, -1, :]  # [B*S*num_patches, feature_dim]
        
        # 重塑回原始批次和patch数量
        
        if self.return_all:
            x = x.view(batch_size, num_patches*patch_size, feature_dim)
            return x
        else:
            x= x[:, -1, :]
            x= x.view(batch_size, num_patches, feature_dim)
            return x



class VQVAE(nn.Module):
    def __init__(self, vocab_size, embedding_dim, out_loc_num, seq_len, using_znorm, beta, spatial_loc_nums=None, temporal_scale_nums=None):
        super(VQVAE, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.spatial_loc_nums = spatial_loc_nums
        self.temporal_scale_nums = temporal_scale_nums
        self.patch_size = 3  # 每个patch包含8个token
        self.num_patches = seq_len // self.patch_size  # 56个patch

        # 为每个空间位置创建embedding层
        self.spatial_embeddings = nn.ModuleList()
        if spatial_loc_nums is not None:
            # for loc_num in [2236, 583,171,54]:
            for loc_num in np.array(self.spatial_loc_nums)[[0,2,4,6]]:
                self.spatial_embeddings.append(
                    nn.Embedding(loc_num, embedding_dim
                    )
                )

        # 创建正弦余弦位置编码
        self.pos_embedding1 = self._create_sincos_position_embedding(seq_len,embedding_dim)

        self.pos_embedding2 = self._create_sincos_position_embedding(seq_len,embedding_dim)
        


        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,  # 直接使用feature_dim，不需要patch_embedding
            nhead=8,
            dim_feedforward=4*embedding_dim,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 添加投影层，将特征维度从64映射到embedding_dim
        self.proj_to_embedding = nn.Linear(64, embedding_dim)

        self.quantizer = VectorQuantizer(vocab_size=vocab_size, channel_vae=embedding_dim, using_znorm=using_znorm, beta=beta, temporal_scale_nums=temporal_scale_nums)

        # 将简单的Linear层替换为Transformer Decoder
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=4*embedding_dim,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=2)
        self.final_proj = nn.Linear(embedding_dim, out_loc_num)

        
    def forward(self, x):
        ids = x
        B, S, L = x.shape

        # ids_0=x[:,0]
        # ids_1=ids_0[:,1:]
        # move_ratio=torch.sum(ids_1!=ids_0[:,:-1],dim=-1)
        # move_ratio=move_ratio/(L-1)

        embeddings = []
        for scale_idx in range(8):
            scale_embeddings = self.spatial_embeddings[int(scale_idx/2)](x[:, scale_idx])  # [B, L, D]
            embeddings.append(scale_embeddings)
        
        x = torch.stack(embeddings, dim=1)  # [B, S, L, D]
        x = x.view(B*S, L, -1)
        x = x + self.pos_embedding1




        x = self.transformer_encoder(x)

        x = x.view(B, S, L, -1)
        f_hat, usages, mean_vq_loss = self.quantizer(x, ret_usages=True)

      
        f_hat=f_hat+ self.pos_embedding2
        # f_hat=self.resampler_up(f_hat)
        dec_output = self.transformer_decoder(f_hat)
        x_hat = self.final_proj(dec_output)

        # 计算每个batch的cross entropy loss并乘以对应的move_ratio
        ce_loss = F.cross_entropy(
            x_hat.view(-1, x_hat.size(-1)),
            ids[:,0].reshape(-1) # 不要立即求平均
        ) # 先对每个batch内的序列求平均s

        loss = ce_loss + mean_vq_loss

        x_hat_loc = x_hat.argmax(dim=-1)
        loss_dict = {
            'ce_loss': ce_loss,
            'mean_vq_loss': mean_vq_loss,
            'loss': loss
        }
        output_dict = {
            'x_hat_loc': x_hat_loc,
            'loss_dict': loss_dict,
            'usages': usages,
        }
        return output_dict
    
    def encode_vqvae(self, x):
        B, S, L = x.shape
        
        embeddings = []
        for scale_idx in range(8):
            scale_embeddings = self.spatial_embeddings[int(scale_idx/2)](x[:, scale_idx])  # [B, L, D]
            embeddings.append(scale_embeddings)
        
        x = torch.stack(embeddings, dim=1)  # [B, S, L, D]
        x = x.view(B*S, L, -1)
        x = x + self.pos_embedding1


        x = self.transformer_encoder(x)
        # # 投影到embedding_dim
        # x = self.proj_to_embedding(x)
     
        x = x.view(B, S, L, -1)
        ids_list = self.quantizer.encode_quant(x)
        return ids_list

    def decode_vqvae(self, ids_list:List[torch.Tensor]) -> torch.Tensor:
        f_hat = self.quantizer.decode_quant(ids_list)
        B = f_hat.size(0)

        
        # 最终投影并获取位置
        x_hat = self.final_proj(f_hat)
        x_hat_loc = x_hat.argmax(dim=-1)
        
        return x_hat_loc

    

    def encode(self, x):
        pass
    
    def _create_sincos_position_embedding(self, seq_len: int, d_model: int) -> torch.Tensor:
        """
        创建正弦余弦位置编码
        Args:
            seq_len: 序列长度
            d_model: 编码维度
        Returns:
            position_embedding: [1, seq_len, d_model]
        """
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, seq_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pe, requires_grad=False)

if __name__ == '__main__':
    spatial_loc_nums=[900,225,64,16]
    temporal_scale_nums=[168,56,7,1]
    vqvae = VQVAE(num_embeddings=10, embedding_dim=128, out_loc_num=spatial_loc_nums[0], seq_len=168, using_znorm=True, beta=0.25, spatial_loc_nums=spatial_loc_nums, temporal_scale_nums=temporal_scale_nums)
    
    # 生成范围在0-15的随机整数张量
    x = torch.randint(0, 16, size=(32, 4, 168))  # [batch_size, num_scales, seq_len]

    output_dict = vqvae(x)
    print(output_dict)