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

from models.single_model2 import PredModel
from dataset.single_traj_dataset2 import PredDataset

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/data/wangzm/merge/DJI_select/merged_data_54+50.pkl')
    parser.add_argument('--exp_name', type=str, default='lstm-2-hist-1')
    parser.add_argument('--pred_length', type=int, choices=[30, 50], default=40)
    parser.add_argument('--hist_length', type=int, choices=[10, 1], default=10)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
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

    dataset = PredDataset(args)
    model = PredModel(args)

    if args.ckpt:
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt['state_dict'])

    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'Total parameters: {total_params}')
    print(f'Trainable parameters: {trainable_params}')

    # Optionally, use torchsummary to print a summary of the model
    summary(model, input_size=(args.batch_size, args.hist_length, 7))

if __name__ == '__main__':
    main()
