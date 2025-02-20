import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import pytorch_lightning as pl
import math
from torch.optim.lr_scheduler import CosineAnnealingLR

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)).to(torch.float))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.1,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q_in: torch.Tensor, k_in: torch.Tensor, v_in:torch.Tensor, \
            attn_mask:Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None) -> torch.Tensor:  
        assert attn_mask is None, "attn_mask has not been implemented yet"
        if key_padding_mask is not None:
            assert key_padding_mask.shape[0] == q_in.shape[1] and key_padding_mask.shape[1] == q_in.shape[0]
            mask = ~(~key_padding_mask.unsqueeze(1) * ~key_padding_mask.unsqueeze(2)).unsqueeze(1)
            assert mask.dtype == torch.bool
        else:
            mask = None

        assert q_in.shape == k_in.shape == v_in.shape
        N, B, C = q_in.shape
        
        q = self.q_proj(q_in).reshape(N, B, self.num_heads, self.head_dim).permute(1, 2, 0, 3) # B, num_heads, N, head_dim
        k = self.k_proj(k_in).reshape(N, B, self.num_heads, self.head_dim).permute(1, 2, 0, 3) # B, num_heads, N, head_dim
        v = self.v_proj(v_in).reshape(N, B, self.num_heads, self.head_dim).permute(1, 2, 0, 3) # B, num_heads, N, head_dim
        
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if mask is not None:
            attn = attn.masked_fill(mask, float('-inf'))
        attn = attn.softmax(dim=-1, dtype=torch.float32).to(q.dtype)
        if mask is not None:
            attn = attn.masked_fill(mask, 0.0)
        
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C).permute(1, 0, 2) # N, B, C
        x = self.proj(x)
        x = self.proj_drop(x)
        return (x, attn)

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = Attention(d_model, nhead, attn_drop=dropout)
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        
        self.pos_encoder = PositionalEncoding(d_model, dropout, 50)

    def with_pos_embed(self, tensor):
        return self.pos_encoder(tensor)

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src)
        src2 = self.self_attn(q, k, v_in=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2)
        src2 = self.self_attn(q, k, v_in=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        # src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
        #                       key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn = Attention(d_model, nhead, attn_drop=dropout)
        # self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = Attention(d_model, nhead, attn_drop=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, v_in=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(q_in=self.with_pos_embed(tgt, query_pos),
                                   k_in=self.with_pos_embed(memory, pos),
                                   v_in=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, v_in=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        # tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask)[0]

        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(q_in=self.with_pos_embed(tgt2, query_pos),
                                   k_in=self.with_pos_embed(memory, pos),
                                   v_in=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        # tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
        #                            key=self.with_pos_embed(memory, pos),
        #                            value=memory, attn_mask=memory_mask,
        #                            key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)

class RoadProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config['map_input_size'], config['map_hidden_size'])
        self.relu = nn.ReLU()
        
    def forward(self, input_data):
        src, valid_mask = input_data
        src = self.linear(src)
        src = self.relu(src)
        return src

class AgentProjector(nn.Module):
    def __init__(self, config, agent_his_input_dim=5):
        super().__init__()
        self.linear = nn.Linear(agent_his_input_dim, config['agent_his_hidden_size'])
        self.relu = nn.ReLU()
        
    def forward(self, input_data):
        src, valid_mask = input_data
        src = self.linear(src)
        src = self.relu(src)
        return src, valid_mask

class Temponet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=config['agent_his_hidden_size'], nhead=8),
            num_layers=config['num_fusion_layer']
        )

        self.config = config
        
        if config['aux_loss_temponet']:
            self.aux_head = nn.Sequential(
                nn.Linear(config['agent_his_hidden_size'], config['head_hidden_dim']),
                nn.ReLU(),
                nn.Linear(config['head_hidden_dim'], config['out_traj_dim']),
            )

    def forward(self, input_data):
        src, valid_mask = input_data
        bs, agent_count, sequence_length, dim = src.shape
        src = src.flatten(0, 1)
        valid_mask = valid_mask.flatten(0, 1)
        non_padding_agent = valid_mask.sum(-1) > 0

        src = src[non_padding_agent].permute(1, 0, 2)  # NxSxC -> SxNxC
        attn_mask = ~valid_mask[non_padding_agent]  # NxS

        memory = self.encoder(src, src_key_padding_mask=attn_mask)

        memory = memory.permute(1, 0, 2)  # SxNxC -> NxSxC
        memory[attn_mask] = -10000.0  # padding mask
        
        max_memory, _ = torch.max(memory, dim=1)  # NxC

        output_data = max_memory.new_zeros(bs * agent_count, dim)
        output_data[non_padding_agent] = max_memory
        output_data = output_data.view(bs, agent_count, dim)
        output_data = output_data.contiguous()

        if self.config['aux_loss_temponet']:
            aux_temponet_pred = self.aux_head(output_data).reshape(bs, agent_count, -1, 5)
            return output_data, aux_temponet_pred

        return output_data, None

class SpaNet_only_encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=config['fusion_hidden_size'], nhead=8),
            num_layers=config['num_fusion_layer']
        )

        self.type_embedding = nn.Embedding(2, config['fusion_hidden_size'])

        self.config = config

        if config['aux_loss_spanet']:
            self.aux_head = nn.Sequential(
                nn.Linear(config['fusion_hidden_size'], config['head_hidden_dim']),
                nn.ReLU(),
                nn.Linear(config['head_hidden_dim'], config['out_traj_dim']),
            )

    def forward(self, input_data):
        agent_his_features, non_padding_agent, map_features, non_padding_map = input_data
        agent_count = agent_his_features.shape[1]
        
        agent_his_features = agent_his_features + self.type_embedding.weight[0]
        map_features = map_features + self.type_embedding.weight[1]
        
        # focal_agent_mask = all_agent_info_clear[:,:,0,-1].bool()
        # agent_his_features[focal_agent_mask] = agent_his_features[focal_agent_mask] + self.agent_focal_or_not.weight[0]
        # agent_his_features[~focal_agent_mask] = agent_his_features[~focal_agent_mask] + self.agent_focal_or_not.weight[1]
        
        src = torch.cat([agent_his_features, map_features], dim=1) # NxSxC
        valid_mask = torch.cat([non_padding_agent, non_padding_map], dim=1) # NxS
        
        bs,node_count,dim = src.shape
                
        src = src.permute(1, 0, 2) # NxSxC -> SxNxC
        attn_mask = ~valid_mask # NxS

        memory = self.encoder(src, src_key_padding_mask=attn_mask)

        memory = memory.permute(1, 0, 2) # SxNxC -> NxSxC
        
        memory[attn_mask] = -10000.0 # padding mask
        
        output_data = memory.contiguous()

        if self.config['aux_loss_spanet']:
            aux_spanet_pred = self.aux_head(output_data[:,:agent_count]).reshape(bs, agent_count, -1, 2)
            return output_data, valid_mask, aux_spanet_pred


        return output_data, valid_mask, None
    
class TrajDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=config['in_hidden_dim'], nhead=8),
            num_layers=config['num_decoder_layer']
        )

        # self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=2)
        

        self.num_queries = config['num_queries']
        self.query_embed = nn.Embedding(config['num_queries'], config['in_hidden_dim'])

    def forward(self, input_data):
        fusion_features, fusion_non_padding = input_data
        bs,node_count,dim = fusion_features.shape

        fusion_features = fusion_features.permute(1, 0, 2) # NxSxC -> SxNxC
        # attn_mask = ~fusion_non_padding

        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1) # num_queries x N x C

        tgt = query_embed
        attn_mask = ~fusion_non_padding.bool()
        # attn_mask = attn_mask.int().bool()
        # attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_queries, -1)  # (N, num_queries, S)
        
        # hs, _ = self.lstm(fusion_features)
        # tgt = torch.zeros_like(query_embed)
        hs = self.decoder(tgt, fusion_features, memory_key_padding_mask=attn_mask)
        return hs.permute(1,0,2) # Nxnum_queriesxC


class PredHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pos_head = nn.Sequential(
            nn.Linear(config['in_hidden_dim'], config['head_hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['head_hidden_dim'], 2)
        )
        self.heading_head = nn.Sequential(
            nn.Linear(config['in_hidden_dim'], config['head_hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['head_hidden_dim'], 1)
        )
        self.vel_head = nn.Sequential(
            nn.Linear(config['in_hidden_dim'], config['head_hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['head_hidden_dim'], 2)
        )

    def forward(self, hs):
        # Ensure hs has the shape [batch_size, num_queries, hidden_dim]
        # if hs.dim() == 2:
        #     hs = hs.unsqueeze(0)
        # elif hs.dim() == 4:
        #     hs = hs.squeeze(0)

        pos = self.pos_head(hs)
        heading = self.heading_head(hs)
        vel = self.vel_head(hs)

        output = {
            "pos": pos,
            "heading": heading,
            "vel": vel
        }
        return output

class AgentPredictor(pl.LightningModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        
        self.road_projector = RoadProjector(config)
        self.ego_projector = AgentProjector(config, 13)
        self.agent_projector = AgentProjector(config)
        self.label_projector = AgentProjector(config, 1)
        self.temponet = Temponet(config)
        self.fusion_encoder = SpaNet_only_encoder(config)
        self.traj_decoder = TrajDecoder(config)
        self.pred_head = PredHead(config)
        
    def forward(self, input_data):
        ego_stat = input_data['ego_stat'][...,:13] # 10*11
        ego_map = input_data['ego_map'] # 50*3
        # ego_fut = input_data['ego_fut']
        ahead_stat = input_data['ahead_stat'][...,:5]
        # ahead_fut = input_data['ahead_fut']
        right_stat = input_data['right_stat']
        # right_fut = input_data['right_fut']
        ego_map_mask = input_data['ego_map_mask']
        ego_label = input_data['ego_stat'][...,11:12]

        ego_map_feature = self.road_projector((ego_map, ego_map_mask)) # 32*50*256

        ego_features = self.ego_projector((ego_stat, torch.ones((ego_stat.shape[0], ego_stat.shape[1])).bool()))[0] # 32*10*256
        ego_features = torch.unsqueeze(ego_features, dim=1) # 32*1*10*256

        ego_label_features = self.label_projector((ego_label, torch.ones((ego_stat.shape[0], ego_stat.shape[1])).bool()))[0] # 32*10*256
        ego_label_features = torch.unsqueeze(ego_label_features, dim=1) # 32*1*10*256

        # 使用 torch.unsqueeze 在第二个维度上增加一个维度为 1
        ahead_stat = torch.unsqueeze(ahead_stat, dim=1)
        right_stat = torch.unsqueeze(right_stat, dim=1)
        
        agent_his_features = torch.cat([ahead_stat, right_stat], dim=1)

        agent_his_features, non_padding_agent = self.agent_projector((agent_his_features, torch.ones_like(agent_his_features)[..., 0].bool()))
        
        agent_his_features = torch.cat([ego_features, agent_his_features, ego_label_features], dim=1) # 32*4*10*256

        non_padding_agent = torch.ones_like(agent_his_features)[..., 0].bool()

        temponet_output, aux_temponet_pred = self.temponet((agent_his_features, non_padding_agent))

        non_padding_agent = torch.ones_like(temponet_output)[..., 0].bool()

        fusion_features, fusion_non_padding, _ = self.fusion_encoder((temponet_output, non_padding_agent, ego_map_feature, ego_map_mask.bool()))
        
        # predictor
        hs = self.traj_decoder((fusion_features, fusion_non_padding))
        output = self.pred_head(hs)
      
        output['agent'] = aux_temponet_pred
        
        return output, aux_temponet_pred

    def training_step(self, batch, batch_idx):
        output, aux_temponet_pred = self(batch)
        loss_dict = self.compute_loss(output, batch, aux_temponet_pred)
        self.log_dict({f"train_{k}": v for k, v in loss_dict.items()})
        return loss_dict['loss']
    
    def validation_step(self, batch, batch_idx):
        output, aux_temponet_pred = self(batch)
        loss_dict = self.compute_loss(output, batch, aux_temponet_pred)
        self.log_dict({f"val_{k}": v for k, v in loss_dict.items()})
        return loss_dict['loss']
    
    def compute_loss(self, output, batch, aux_temponet_pred):
        ego_fut = batch['ego_fut']
        right_fut = batch['right_fut']
        ahead_fut = batch['ahead_fut']

        # Extract individual components from ground truth
        gt_pos_ego = ego_fut[:, :, :2]
        gt_heading_ego = ego_fut[:, :, 2:3]
        gt_vel_ego = ego_fut[:, :, 3:5]

        gt_pos_right = right_fut[:, :, :2]
        gt_heading_right = right_fut[:, :, 2:3]
        gt_vel_right = right_fut[:, :, 3:5]

        gt_pos_ahead = ahead_fut[:, :, :2]
        gt_heading_ahead = ahead_fut[:, :, 2:3]
        gt_vel_ahead = ahead_fut[:, :, 3:5]

        pred_pos_right = output['agent'][:, 2,:,:2]
        pred_heading_right = output['agent'][:, 2,:,2:3]
        pred_vel_right = output['agent'][:, 2,:,3:5]
        pred_pos_ahead = output['agent'][:, 1,:,:2]
        pred_heading_ahead = output['agent'][:, 1,:,2:3]
        pred_vel_ahead = output['agent'][:, 1,:,3:5]
        pred_pos_ego = output['pos']
        pred_heading_ego = output['heading']
        pred_vel_ego = output['vel']

        linear_sequence = torch.linspace(0, 9, steps=10, device=output['pos'].device)
        exponential_sequence = torch.exp(-linear_sequence) + 1
        # weight = torch.ones(pred_pos_ego.shape[0], device=output['pos'].device)
        # weight[:10] = exponential_sequence


        pos_loss_ego = ((((pred_pos_ego-gt_pos_ego)**2).sum(dim=-1).mean(dim=0).mean(dim=0))*exponential_sequence.unsqueeze(-1)).mean()
        pos_loss_right = ((((pred_pos_right-gt_pos_right)**2).sum(dim=-1).mean(dim=0).mean(dim=0))*exponential_sequence.unsqueeze(-1)).mean()
        pos_loss_ahead = ((((pred_pos_ahead-gt_pos_ahead)**2).sum(dim=-1).mean(dim=0).mean(dim=0))*exponential_sequence.unsqueeze(-1)).mean()

        # pos_loss_ego = F.mse_loss(pred_pos_ego, gt_pos_ego)
        heading_loss_ego = F.mse_loss(pred_heading_ego, gt_heading_ego)
        vel_loss_ego = F.mse_loss(pred_vel_ego, gt_vel_ego)

        # pos_loss_right = F.mse_loss(pred_pos_right, gt_pos_right)
        heading_loss_right = F.mse_loss(pred_heading_right, gt_heading_right)
        vel_loss_right = F.mse_loss(pred_vel_right, gt_vel_right)

        # pos_loss_ahead = F.mse_loss(pred_pos_ahead, gt_pos_ahead)
        heading_loss_ahead = F.mse_loss(pred_heading_ahead, gt_heading_ahead)
        vel_loss_ahead = F.mse_loss(pred_vel_ahead, gt_vel_ahead)

        pos_loss = pos_loss_ego + pos_loss_right + pos_loss_ahead
        heading_loss = heading_loss_ego + heading_loss_right + heading_loss_ahead
        vel_loss = vel_loss_ego + vel_loss_right + vel_loss_ahead
        # pos_loss = pos_loss_ego
        # heading_loss = heading_loss_ego
        # vel_loss = vel_loss_ego

        loss = pos_loss + heading_loss + vel_loss

        # Example FDE and ADE calculation (assuming final and average displacement error for position)
        fde = torch.norm(pred_pos_ego[:,-1] - gt_pos_ego[:,-1], dim=-1).mean()
        ade = torch.norm(pred_pos_ego - gt_pos_ego, dim=-1).mean()

        loss_dict = {
            'ego_pos_loss': pos_loss_ego,
            'ego_heading_loss': heading_loss_ego,
            'ego_vel_loss': vel_loss_ego,
            'right_pos_loss': pos_loss_right,
            'ahead_pos_loss': pos_loss_ahead,
            'loss': loss,
            'fde': fde,
            'ade': ade
        }
        return loss_dict
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
        scheduler = CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]
    
def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
