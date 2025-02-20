import os
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset


class PredDataset(Dataset):
    def __init__(self, args) -> None:
        self.args = args
        # self.data_dir = args.data_dir

        # files = os.listdir(self.data_dir)
        # files.sort()
        self.data_path = args.data_path
        self.all_data = pickle.load(open(self.data_path, "rb")) # []
        self.ids = list(self.all_data.keys())
        # print('Start loading data')
        # self.all_data = self.load_data(files)
        # print('Number of data: ', len(self.all_data))

    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, idx):
        data = self.process_data(self.all_data[self.ids[idx]])
        return data  # 40*7
    
    def process_data(self, data):
        # tracks = torch.tensor(data['tracks'])
        # map_data = data['map']
        ego_traj = torch.tensor(data['ego_traj']) # 66*3
        ego_vel = torch.tensor(data['ego_vel']) # 66*2
        ego_acc = torch.tensor(data['ego_accel']) # 66*2
        # ful_traj = torch.tensor(data['ego_fut_traj'])[:40] # 51*3
        map_data = torch.tensor(data['ego_map']) # 125*2
        inter_his_stat = torch.tensor(data['inter_his_stat']) # 10*5

        his_traj = ego_traj[:16]
        his_vel = ego_vel[:16]
        his_acc = ego_acc[:16]
        ego_stat = torch.concat((his_traj,his_vel,his_acc),dim=-1)[-10:]

        ful_traj = ego_traj[16:]
        ful_vel = ego_vel[16:]
        ful_acc = ego_acc[16:]
        ful_stat = torch.concat((ful_traj,ful_vel),dim=-1)[:40]

        # tracks = tracks[tracks[:,0, 10] <= 4.0]  # 40*7

        # tracks_info = tracks[:, :self.args.hist_length]  # 10*7
        # tracks_gt = tracks[:, self.args.hist_length:30+self.args.hist_length, [2,3,6,7,12]]


        result = {
            # 'ego_his_traj': his_traj,
            # 'ego_his_vel': his_vel,
            # 'ego_his_accel': his_acc,
            'ego_stat':ego_stat,
            'ego_map': map_data,
            'ego_fut':ful_stat,
            'inter_stat':inter_his_stat
            # 'curr_point': curr_point,
            # 'tracks_id': tracks_id,
            # 'centerlines': centerlines,
            # 'centerlines_mask': centerlines_mask,
            # 'centerlines_id': centerlines_id
        }

        return result
    
    def collate_fn(self, batch):
        max_map_length = max(data['ego_map'].shape[0] for data in batch)
        
        batch_ego_stat = []
        batch_ego_fut = []
        batch_inter_stat = []
        batch_ego_map = []
        batch_ego_map_mask = []

        for data in batch:
            batch_ego_stat.append(data['ego_stat'])
            batch_ego_fut.append(data['ego_fut'])
            batch_inter_stat.append(data['inter_stat'])
            
            map_data = data['ego_map']
            padding = torch.zeros(max_map_length - map_data.shape[0], map_data.shape[1])
            padded_map = torch.cat([map_data, padding], dim=0)
            mask = torch.cat([torch.ones(map_data.shape[0]), torch.zeros(padding.shape[0])], dim=0)

            batch_ego_map.append(padded_map)
            batch_ego_map_mask.append(mask)
        
        result = {
            'ego_stat': torch.stack(batch_ego_stat, dim=0).to(dtype=torch.float32),
            'ego_fut': torch.stack(batch_ego_fut, dim=0).to(dtype=torch.float32),
            'inter_stat': torch.stack(batch_inter_stat, dim=0).to(dtype=torch.float32),
            'ego_map': torch.stack(batch_ego_map, dim=0).to(dtype=torch.float32),
            'ego_map_mask': torch.stack(batch_ego_map_mask, dim=0).to(dtype=torch.float32)
        }

        return result