import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

import math



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
            
        # #### tril mask
        # tril_valid = torch.tril(torch.ones(key_padding_mask.shape[1], key_padding_mask.shape[1])).bool()
        # tril_valid = tril_valid.unsqueeze(0).unsqueeze(0).to(mask.device)
        # mask = ~(~mask * tril_valid)
        # #### tril mask

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
        self.self_attn = Attention(d_model, nhead, attn_drop=dropout, selfattn=True)
        # self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = Attention(d_model, nhead, attn_drop=dropout, selfattn=False)

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
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(config.map_input_size, config.map_hidden_size)
        self.relu = nn.ReLU()
        
    def forward(self, input_data):
        src, valid_mask = input_data
        src = self.linear(src)
        src = self.relu(src)
        return src


class AgentProjector(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(config.agent_his_input_dim, config.agent_his_hidden_size)
        self.relu = nn.ReLU()
        
    def forward(self, input_data):
        src, valid_mask = input_data
        src = self.linear(src)
        src = self.relu(src)
        return src, valid_mask

class Temponet(nn.Module):
    def __init__(self, config, d_model=256, nhead=8, num_encoder_layers=3,
                 dim_feedforward=1024, dropout=0.,
                 activation="relu", normalize_before=True):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        
        self.config = config
        
        if config.aux_loss_temponet:
            self.aux_head = nn.Sequential(*[
                nn.Linear(self.d_model, config.head_hidden_dim),
                nn.ReLU(),
                nn.Linear(config.head_hidden_dim, config.out_traj_dim),
            ])
        

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_data):
        src, valid_mask = input_data
        bs,agent_count,sequence_length,dim = src.shape
        src = src.flatten(0,1)
        valid_mask = valid_mask.flatten(0,1)
        non_padding_agent = valid_mask.sum(-1) > 0
        
        src = src[non_padding_agent].permute(1, 0, 2) # NxSxC -> SxNxC
        attn_mask = ~valid_mask[non_padding_agent] # NxS

        memory = self.encoder(src, src_key_padding_mask=attn_mask)

        memory = memory.permute(1, 0, 2) # SxNxC -> NxSxC
        memory[attn_mask] = -10000.0 # padding mask
        
        # reduction on T
        ### amx
        max_memory,_ = torch.max(memory, dim=1) # NxC
        ### mean
        # memory[attn_mask] = 0.
        # mean_memory = memory.sum(1) / (~attn_mask).sum(1).unsqueeze(1)
        # max_memory = mean_memory

        # output
        output_data = max_memory.new_zeros(bs*agent_count, dim)
        output_data[non_padding_agent] = max_memory
        output_data = output_data.view(bs, agent_count, dim)

        output_data = output_data.contiguous()

        if self.config.aux_loss_temponet:
            aux_temponet_pred = self.aux_head(output_data).reshape(bs, agent_count, -1, 2)
            return output_data, aux_temponet_pred

        return output_data, None



class SpaNet_only_encoder(nn.Module):
    def __init__(self, config, d_model=256, nhead=8, num_encoder_layers=2,
                 dim_feedforward=1024, dropout=0.,
                 activation="relu", normalize_before=True):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self.type_embedding = nn.Embedding(2, d_model) # agent, map
        # self.agent_focal_or_not = nn.Embedding(2, d_model)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        
        self.config = config

        if config.aux_loss_spanet:
            self.aux_head = nn.Sequential(*[
                nn.Linear(self.d_model, config.head_hidden_dim),
                nn.ReLU(),
                nn.Linear(config.head_hidden_dim, config.out_traj_dim),
            ])

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_data):
        agent_his_features, non_padding_agent, map_features, non_padding_map, all_agent_info_clear = input_data
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

        if self.config.aux_loss_spanet:
            aux_spanet_pred = self.aux_head(output_data[:,:agent_count]).reshape(bs, agent_count, -1, 2)
            return output_data, valid_mask, aux_spanet_pred


        return output_data, valid_mask, None



class TrajDecoder(nn.Module):
    def __init__(self, config, d_model=256, nhead=8, num_decoder_layers=3,
                 dim_feedforward=1024, dropout=0., num_queries=6,
                 activation="relu", normalize_before=True,
                 return_intermediate_dec=False):
        super().__init__()
        
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, d_model)
        
        self.config = config
        

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_data):
        fusion_features, fusion_non_padding = input_data
        bs,node_count,dim = fusion_features.shape

        fusion_features = fusion_features.permute(1, 0, 2) # NxSxC -> SxNxC
        attn_mask = ~fusion_non_padding

        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1) # num_queries x N x C

        tgt = torch.zeros_like(query_embed)
        hs = self.decoder(tgt, fusion_features, memory_key_padding_mask=attn_mask,
                          pos=None, query_pos=query_embed)
        return hs.transpose(1, 2).squeeze() # Nxnum_queriesxC

class PredHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.traj_head = nn.Sequential(*[
            nn.Linear(config.in_hidden_dim, config.head_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.head_hidden_dim, config.out_traj_dim),
        ])
        self.prob_head = nn.Sequential(*[
            nn.Linear(config.in_hidden_dim, config.head_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.head_hidden_dim, 1),
        ])

    def forward(self, hs):
        output = {}
        
        bs, mode_count, dim = hs.shape
        
        pred_traj = self.traj_head(hs).reshape(bs, mode_count, -1, 2)
        pred_prob = self.prob_head(hs).squeeze()

        output["pred_traj"] = pred_traj
        output["pred_prob"] = pred_prob

        return output


class AgentPredictor(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        
        # road
        self.road_projector = RoadProjector(config)
        
        # agent
        self.agent_projector = AgentProjector(config)
        self.temponet = Temponet(config)

        # fusion
        self.fusion_encoder = SpaNet_only_encoder(config, d_model=config.fusion_hidden_size, \
            num_encoder_layers=config.num_fusion_layer, dropout=config.dropout_rate, )

        # predictor
        self.traj_decoder = TrajDecoder(config, d_model=config.in_hidden_dim, num_queries=config.num_queries, \
            num_decoder_layers=config.num_decoder_layer, dropout=config.dropout_rate)
        self.pred_head = PredHead(config)
        
    def forward(self, input_data):
        road_features, agent_his_features, map_features, all_agent_info_clear = input_data
        
        # road
        road_features = self.road_projector(road_features)
        
        # agent
        agent_his_features, non_padding_agent = self.agent_projector(agent_his_features)
        
        temponet_output, aux_temponet_pred = self.temponet((agent_his_features, non_padding_agent))

        # fusion
        fusion_features, fusion_non_padding, _ = self.fusion_encoder((temponet_output, non_padding_agent, road_features, torch.ones_like(road_features).bool(), all_agent_info_clear))
        
        # predictor
        hs = self.traj_decoder((fusion_features, fusion_non_padding))
        output = self.pred_head(hs)
        
        return output, aux_temponet_pred
        

