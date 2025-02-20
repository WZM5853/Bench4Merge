import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
import torch.utils.data as data
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torchsummary import summary

# Assuming the dataset and model files are in the same directory as this script
from models.Transformer_model import AgentPredictor
from dataset.single_traj_dataset2 import PredDataset

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/data/wangzm/merge/final_data/DJI24_with+HD40.pkl')
    parser.add_argument('--exp_name', type=str, default='HD40+DJI24_with')
    parser.add_argument('--pred_length', type=int, choices=[30, 50], default=40)
    parser.add_argument('--hist_length', type=int, choices=[10, 1], default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_epochs', type=int, default=4000)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/output')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_layers', type=int, default=2)
    return parser.parse_args()

def main():
    torch.manual_seed(42)
    np.random.seed(0)
    args = parse_args()
    seed = torch.Generator().manual_seed(42)

    wandb_logger = WandbLogger(
        project="Transformer-three-to-three",
        save_dir=args.save_dir, 
        name=args.exp_name
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        monitor='val_fde',
        filename='{epoch}-{val_fde:.2f}',
        save_top_k=1,
        mode='min',
        save_last=True
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=wandb_logger,
        accelerator='gpu',
        gradient_clip_val=0.5,
        callbacks=[lr_monitor, checkpoint_callback]
    )

    dataset = PredDataset(args)
    train_set = int(len(dataset) * 0.8)  # 80% of the data
    val_set = len(dataset) - train_set
    train_dataset, val_dataset = data.random_split(dataset, [train_set, val_set], generator=seed)

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                   num_workers=args.num_workers, drop_last=False, collate_fn=dataset.collate_fn)
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                                 num_workers=args.num_workers, drop_last=False, collate_fn=dataset.collate_fn)
    
    # single_sample = [dataset[0]]  # Take only one sample

    # train_loader = data.DataLoader(single_sample, batch_size=1, shuffle=True, 
    #                                num_workers=args.num_workers, drop_last=False, collate_fn=dataset.collate_fn)
    # val_loader = data.DataLoader(single_sample, batch_size=1, shuffle=False, 
    #                              num_workers=args.num_workers, drop_last=False, collate_fn=dataset.collate_fn)


    config = {
        'map_input_size': 3,  # assuming map_info has 2 dimensions
        'map_hidden_size': 256,  # example hidden size
        'agent_his_input_dim': 5,  # assuming each agent's historical data has 5 dimensions
        'agent_his_hidden_size': 256,  # example hidden size
        'fusion_hidden_size': 256,  # example hidden size
        'num_fusion_layer': args.num_layers,  # number of layers in the fusion encoder
        'dropout_rate': 0.1,  # example dropout rate
        'in_hidden_dim': 256,  # example input hidden dimension for the decoder
        'head_hidden_dim': 256,  # example hidden dimension for the prediction head
        'out_traj_dim': args.pred_length * 5,  # output trajectory dimension
        'num_queries': 40,  # example number of queries for the decoder
        'num_decoder_layer': args.num_layers,  # number of layers in the decoder
        'aux_loss_temponet': True,  # whether to use auxiliary loss in temponet
        'aux_loss_spanet': True,  # whether to use auxiliary loss in spanet
        'lr': args.lr  # learning rate
    }

    model = AgentPredictor(config)
    if args.ckpt:
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt['state_dict'])
    
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == '__main__':
    main()
