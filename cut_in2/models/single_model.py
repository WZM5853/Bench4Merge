import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
# from transformers import Transformer
import torch.nn.functional as F
from torch.nn import MultiheadAttention

class PredModel(pl.LightningModule):
    def __init__(self, args) -> None:
        super(PredModel, self).__init__()
        self.args = args
        self.pred_length = args.pred_length
        self.hist_length = args.hist_length
        self.pre_mlp = nn.Sequential(
            nn.Linear(7, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.SiLU()
        )
        self.map_mlp = nn.Sequential(
            nn.Linear(2, 256),  # 假设map信息有10维
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.SiLU()
        )
        self.inter_mlp = nn.Sequential(
            nn.Linear(5, 256),  # inter信息有5维
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.SiLU()
        )
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=args.num_layers, batch_first=True)
        # self.transformer = Transformer(d_model=256, nhead=8, num_encoder_layers=3, num_decoder_layers=3)
        # self.attention = MultiheadAttention(embed_dim=256, num_heads=8)

        self.pos_output = nn.Linear(256, 2)
        self.heading_output = nn.Linear(256, 1)
        self.vel_output = nn.Linear(256,2)

        self.mse_loss = nn.MSELoss()

    def forward(self, x):
        tracks_info = x['ego_stat']
        tracks_fea = self.pre_mlp(tracks_info)

        # 使用 LSTM 预测 future features
        # predictions = []
        # lstm_out, hidden = self.lstm(tracks_fea)
        # for _ in range(self.pred_length):
        #     lstm_out, hidden = self.lstm(lstm_out, hidden)
        #     predictions.append(lstm_out[:, -1:, :])
        
        # lstm_out = torch.cat(predictions, dim=1)  # 将所有时间步的输出连接起来
        
        # attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        for i in range(self.pred_length):
            lstm_out, _ = self.lstm(tracks_fea)
            tracks_fea = torch.cat([tracks_fea, lstm_out[:,-1:,]], dim=1)

        tracks_fea = tracks_fea[:,self.hist_length:]
        
        # 处理inter_vehicle信息
        inter_info = x['inter_stat']
        inter_fea = self.inter_mlp(inter_info)

        inter_fea = F.interpolate(inter_fea.permute(0, 2, 1), size=tracks_fea.size(1), mode='linear').permute(0, 2, 1)

        # 处理map信息
        map_info = x['ego_map']
        map_fea = self.map_mlp(map_info)
        
        map_fea = F.interpolate(map_fea.permute(0, 2, 1), size=tracks_fea.size(1), mode='linear').permute(0, 2, 1)
        
        combined_fea = tracks_fea + map_fea + inter_fea
        
        pos = self.pos_output(combined_fea)
        heading = self.heading_output(combined_fea)
        vel = self.vel_output(combined_fea)
        
        output = {
            'pos': pos,
            'heading': heading,
            'vel': vel
        }
        return output
    
    def training_step(self, batch, batch_idx):
        traj_gt = batch['ego_fut']
        output = self.forward(batch)
        loss = self.get_loss(output, traj_gt)['loss']
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        pred = self.forward(batch)
        loss_dict = self.get_loss(pred, batch['ego_fut'])
        self.log_dict({f'val_{k}' : v for k, v in loss_dict.items()},
                      on_step=False, on_epoch=True, sync_dist=True)
        return loss_dict['loss']
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]
    
    def get_loss(self, pred, target):
        linear_sequence = torch.linspace(0, 9, steps=10, device=pred['pos'].device)
        exponential_sequence = torch.exp(-linear_sequence) + 1
        weight = torch.ones(self.pred_length, device=pred['pos'].device)
        weight[:10] = exponential_sequence
        pos_loss = ((((pred['pos']-target[...,:2])**2).sum(dim=-1).mean(dim=0).mean(dim=0))*exponential_sequence.unsqueeze(-1)).mean()
        heading_loss = self.mse_loss(pred['heading'], target[...,2:3])
        vel_loss = self.mse_loss(pred['vel'], target[...,3:5])

        fde = torch.norm(pred['pos'][:,-1] - target[:,-1,:2], dim=-1).mean()
        ade = torch.norm(pred['pos'] - target[...,:2], dim=-1).mean()

        loss_dict = {
            'pos_loss': pos_loss,
            'heading_loss': heading_loss,
            'vel_loss':vel_loss,
            'loss': pos_loss + heading_loss + vel_loss,
            'fde': fde,
            'ade': ade
        }
        return loss_dict

    
