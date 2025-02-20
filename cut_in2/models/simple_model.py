import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout):
        super(TransformerEncoder, self).__init__()
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout), num_layers)

    def forward(self, src, mask):
        return self.encoder(src=src, src_key_padding_mask=mask)


class PredModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.pred_length = args.pred_length
        self.hist_lenght = args.hist_length
        self.args = args
        self.agent_encoder = nn.Sequential(
            nn.Linear(10*self.hist_lenght, 128),
            nn.SiLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 256),
            nn.SiLU()
        )
        self.map_encoder = nn.Sequential(
            nn.Linear(220, 128),
            nn.SiLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 256),
            nn.SiLU()
        )

        self.embedding = nn.Embedding(15, 256)

        self.layer = TransformerEncoder(256, 8, 2, 256, 0.1)

        self.pos_output = nn.Linear(256, 2*self.pred_length)
        self.vel_output = nn.Linear(256, 2*self.pred_length)
        self.heading_output = nn.Linear(256, 1*self.pred_length)

        self.mse_loss = nn.MSELoss()

    def forward(self, batch):
        tracks_info = batch['tracks_info']
        tracks_mask = batch['tracks_mask']
        tracks_id = batch['tracks_id']
        centerlines = batch['centerlines']
        centerline_id = batch['centerlines_id']
        centerline_mask = batch['centerlines_mask']

        # 车辆编码
        B, N, N_h, _ = tracks_info.shape
        tracks_info = tracks_info.reshape(B, N, -1)
        tracks_emb = self.embedding(tracks_id)
        tracks_enc = self.agent_encoder(tracks_info)
        tracks_enc = tracks_enc+tracks_emb

        # 地图编码
        _, M, _, _ = centerlines.shape
        centerlines = centerlines.reshape(B, M, -1)
        map_emb = self.embedding(centerline_id)
        map_enc = self.map_encoder(centerlines)
        map_enc = map_enc+map_emb

        mask = torch.cat([tracks_mask, centerline_mask], dim=-1).transpose(0, 1)
        src = torch.cat([tracks_enc, map_enc], dim=-2)

        # 信息融合
        src = self.layer(src, ~mask)
        src = src[:, :N, :]

        pos = self.pos_output(src)
        vel = self.vel_output(src)
        heading = self.heading_output(src)

        pred = {
            'pos': pos.reshape(B, N, self.pred_length, 2),
            'vel': vel.reshape(B, N, self.pred_length, 2),
            'heading': heading.reshape(B, N, self.pred_length)
        }

        return pred

    def training_step(self, batch, batch_idx):
        pred = self.forward(batch)
        loss_dict = self.get_loss(pred, batch['tracks_gt'])
        self.log_dict({f'train_{k}' : v for k, v in loss_dict.items()},
                      on_step=True, on_epoch=False, sync_dist=True)
        return loss_dict['loss']
    
    def validation_step(self, batch, batch_idx):
        pred = self.forward(batch)
        loss_dict = self.get_loss(pred, batch['tracks_gt'])
        self.log_dict({f'val_{k}' : v for k, v in loss_dict.items()},
                      on_step=False, on_epoch=True, sync_dist=True)
        return loss_dict['loss']

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        # scheduler = MultiStepLR(optimizer, milestones=[40, 60], gamma=0.2, verbose=True)
        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0, verbose=True)
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
        ade = torch.norm(pred['pos'] - target[:,:,:,:2], dim=-1).mean()

        loss_dict = {
            'pos_loss': pos_loss,
            'vel_loss': vel_loss,
            'heading_loss': heading_loss,
            'loss': pos_loss+vel_loss+heading_loss,
            'fde': fde,
            'ade': ade
        }
        return loss_dict