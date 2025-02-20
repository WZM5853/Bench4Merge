import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR


class TransformerEncoderNN(nn.Module):
    def __init__(self, d_model, n_head, dropout, num_layers=6):
        super(TransformerEncoder, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dropout=dropout),
            num_layers=num_layers
        )

    def forward(self, src, mask, tracks_emb):
        src[:,:tracks_emb.shape[1]] += tracks_emb
        src = src.permute(1, 0, 2)
        output = self.transformer_encoder(src, src_key_padding_mask=mask)
        output = output.permute(1, 0, 2)
        return output


class SelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super(SelfAttention, self).__init__()
        assert d_model % n_head == 0, 'd_model should be divided by nhead'
        self.n_head = n_head
        self.head_dim = d_model//n_head
        self.k_linear = nn.Linear(d_model, d_model)
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax()

        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.flash = hasattr(F, 'scaled_dot_product_attention')

    def forward(self, k, q, v, key_padding_mask):
        B, T, C = k.size() # batch-size, sequence-length, embedding-dimensionality (n_embd)
        
        k = self.k_linear(k).contiguous().view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.q_linear(q).contiguous().view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.v_linear(v).contiguous().view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(-1).float() # (B, T, 1)
            attn_mask = torch.bmm(key_padding_mask, key_padding_mask.transpose(1, 2)) # (B, T, T)
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1) # (nh, B, T, T)
        else:
            attn_mask = torch.ones((B, self.n_head, T, T)).to(k.device)

        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, 
                                               dropout_p=self.dropout if self.training else 0, is_causal=False)
        else:
            # manual implementation of attention
            attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            attn = attn.masked_fill(attn_mask == 0, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            y = attn @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        output = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        return output


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward, dropout):
        super(TransformerEncoder, self).__init__()
        self.self_attn = SelfAttention(d_model, n_head)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.SiLU()

    def forward(self, src, tracks_emb, mask=None):
        src[:,:tracks_emb.shape[1]] += tracks_emb
        q = k = src
        src2 = self.self_attn(q, k, src, mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class PredModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.pred_length = args.pred_length
        self.hist_lenght = args.hist_length
        self.args = args
        self.agent_encoder = nn.Sequential(
            nn.Linear(7*self.hist_lenght, 128),
            nn.SiLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 256),
            nn.SiLU()
        )
        # self.map_encoder = nn.Sequential(
        #     nn.Linear(2, 256),
        #     nn.SiLU(),
        #     nn.LayerNorm(256),
        # )

        self.embedding = nn.Embedding(15, 256)

        self.layers = nn.ModuleList([
            TransformerEncoder(256, 8, 256, 0.1) for _ in range(args.num_layers)
        ])
        # self.layer = TransformerEncoderNN(256, 8, 0.1, num_layers=args.num_layers)

        self.pos_output = nn.Linear(256, 2*self.pred_length)
        self.vel_output = nn.Linear(256, 2*self.pred_length)
        self.heading_output = nn.Linear(256, 1*self.pred_length)

        self.mse_loss = nn.MSELoss()

    def pos_embedding(self, pos, dim):
        batch_size, num_tracks, _ = pos.shape
        p = torch.zeros((1, num_tracks, dim), device=pos.device)
        X = torch.arange(num_tracks, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, dim, 2, dtype=torch.float32) / dim)
        p[:, :, 0::2] = torch.sin(X)
        p[:, :, 1::2] = torch.cos(X)
        p = p.repeat(batch_size, 1, 1)
        return p

    def forward(self, batch):
        tracks_info = batch['tracks_info']
        # tracks_mask = batch['tracks_mask']
        # tracks_id = batch['tracks_id']
        # centerlines = batch['centerlines']
        # centerline_id = batch['centerlines_id']
        # centerline_mask = batch['centerlines_mask']

        # 车辆编码
        batch_size, num_tracks, num_frames, N = tracks_info.shape
        tracks_info = tracks_info.reshape(batch_size, num_tracks, -1)
        # tracks_emb = self.embedding(tracks_id)
        tracks_emb = self.pos_embedding(tracks_info, 256)
        tracks_feature = self.agent_encoder(tracks_info)
        tracks_feature = tracks_feature+tracks_emb

        # 地图编码
        # batch_size, num_centerlines,  num_points_each_centerline, C = centerlines.shape

        # centerlines_feature_valid = self.map_encoder(centerlines[centerline_mask])
        # centerlines_feature = centerlines.new_zeros(batch_size, num_centerlines,  num_points_each_centerline, centerlines_feature_valid.shape[-1])
        # centerlines_feature[centerline_mask] = centerlines_feature_valid
        # pooled_feature = centerlines_feature.max(dim=2)[0]

        # src = torch.cat([tracks_feature, pooled_feature], dim=-2)
        # centerline_valid_mask = (centerline_mask.sum(dim=-1) > 0)
        # mask = torch.cat([tracks_mask, centerline_valid_mask], dim=-1)

        src = tracks_feature
        # mask = tracks_mask
        # # transformer 信息融合
        for layer in self.layers:
            src = layer(src, tracks_emb)
        # src = self.layer(src, ~mask, tracks_emb)
        src = src[:, :num_tracks, :]

        pos = self.pos_output(src).reshape(batch_size, num_tracks, self.pred_length, 2)
        vel = self.vel_output(src).reshape(batch_size, num_tracks, self.pred_length, 2)
        heading = self.heading_output(src).reshape(batch_size, num_tracks, self.pred_length)

        pred = {
            'pos': pos,
            'vel': vel,
            'heading': heading
        }

        return pred

    def training_step(self, batch, batch_idx):
        pred = self.forward(batch)
        loss_dict = self.get_loss(pred, batch['tracks_gt'])
        bs = batch['tracks_gt'].shape[0]
        self.log_dict({f'train_{k}' : v for k, v in loss_dict.items()},
                      on_step=True, on_epoch=False, sync_dist=True, batch_size=bs)
        return loss_dict['loss']
    
    def validation_step(self, batch, batch_idx):
        pred = self.forward(batch)
        bs = batch['tracks_gt'].shape[0]
        loss_dict = self.get_loss(pred, batch['tracks_gt'])
        self.log_dict({f'val_{k}' : v for k, v in loss_dict.items()},
                      on_step=False, on_epoch=True, sync_dist=True, batch_size=bs)
        return loss_dict['loss']

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        # scheduler = MultiStepLR(optimizer, milestones=[40, 60], gamma=0.2, verbose=True)
        scheduler = CosineAnnealingLR(optimizer, T_max=200, eta_min=0, verbose=True)
        return [optimizer], [scheduler]
    
    def get_loss(self, pred, target):
        # pos_loss = self.mse_loss(pred['pos'], target[:,:,:,:2])
        linear_sequence = torch.linspace(0, 9, steps=10, device=pred['pos'].device)
        exponential_sequence = torch.exp(-linear_sequence) + 1
        weight = torch.ones(self.pred_length, device=pred['pos'].device)
        weight[:10] = exponential_sequence
        pos_loss = ((((pred['pos']-target[:,:,:,:2])**2).sum(dim=-1).mean(dim=0).mean(dim=0))*exponential_sequence.unsqueeze(-1)).mean()
        vel_loss = self.mse_loss(pred['vel'], target[:,:,:,2:4])
        heading_loss = self.mse_loss(pred['heading'], target[:,:,:,4])

        fde = torch.norm(pred['pos'][:,:,-1] - target[:,:,-1,:2], dim=-1).mean()
        self_fde = torch.norm(pred['pos'][:,:,-1] - target[:,:,-1,:2], dim=-1)[:,0].mean()
        ade = torch.norm(pred['pos'] - target[:,:,:,:2], dim=-1).mean()
        self_ade = torch.norm(pred['pos'] - target[:,:,:,:2], dim=-1)[:,0].mean()

        loss_dict = {
            'pos_loss': pos_loss,
            'vel_loss': vel_loss,
            'heading_loss': heading_loss,
            'loss': pos_loss+vel_loss+heading_loss,
            'fde': fde,
            'ade': ade,
            'self_fde': self_fde,
            'self_ade': self_ade
        }
        return loss_dict