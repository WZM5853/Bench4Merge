import os
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset


class PredDataset(Dataset):
    def __init__(self, args) -> None:
        self.args = args
        self.data_path = args.data_path
        self.all_data = pickle.load(open(self.data_path, "rb"))
        self.ids = list(self.all_data.keys())

    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, idx):
        data = self.process_data(self.all_data[self.ids[idx]])
        return data  # 40*8
    
    def process_data(self, data):
        ego_stat = torch.tensor(data['ego_pre_info'])  # 10*12
        ego_fut = torch.tensor(data['ego_post_info'])  # 40*12
        ahead_stat = torch.tensor(data['ahead_pre_info'])[...,:5]  # 40*5 
        ahead_fut = torch.tensor(data['ahead_post_info'])[...,:5]  # 40*5 
        map_data = torch.tensor(data['map_info'])  # x*3
        right_stat = torch.tensor(data['right_pre_info'])[...,:5]  # 10*5
        right_fut = torch.tensor(data['right_post_info'])[...,:5]  # 40*5 

        result = {
            'ego_stat': ego_stat,
            'ego_map': map_data,
            'ego_fut': ego_fut,
            'ahead_stat': ahead_stat,
            'ahead_fut': ahead_fut,
            'right_stat': right_stat,
            'right_fut': right_fut
        }

        return result
    
    def collate_fn(self, batch):
        max_map_length = max(data['ego_map'].shape[0] for data in batch)
        
        batch_ego_stat = []
        batch_ego_fut = []
        batch_ahead_stat = []
        batch_ahead_fut = []
        batch_right_stat = []
        batch_right_fut = []
        batch_ego_map = []
        batch_ego_map_mask = []

        for data in batch:
            batch_ego_stat.append(data['ego_stat'])
            batch_ego_fut.append(data['ego_fut'])
            batch_ahead_stat.append(data['ahead_stat'])
            batch_ahead_fut.append(data['ahead_fut'])
            batch_right_stat.append(data['right_stat'])
            batch_right_fut.append(data['right_fut'])
            
            map_data = data['ego_map']
            padding = torch.zeros(max_map_length - map_data.shape[0], map_data.shape[1])
            padded_map = torch.cat([map_data, padding], dim=0)
            mask = torch.cat([torch.ones(map_data.shape[0]), torch.zeros(padding.shape[0])], dim=0)

            batch_ego_map.append(padded_map)
            batch_ego_map_mask.append(mask)
        
        result = {
            'ego_stat': torch.stack(batch_ego_stat, dim=0).to(dtype=torch.float32),
            'ego_fut': torch.stack(batch_ego_fut, dim=0).to(dtype=torch.float32),
            'ahead_stat': torch.stack(batch_ahead_stat, dim=0).to(dtype=torch.float32),
            'ahead_fut': torch.stack(batch_ahead_fut, dim=0).to(dtype=torch.float32),
            'right_stat': torch.stack(batch_right_stat, dim=0).to(dtype=torch.float32),
            'right_fut': torch.stack(batch_right_fut, dim=0).to(dtype=torch.float32),
            'ego_map': torch.stack(batch_ego_map, dim=0).to(dtype=torch.float32),
            'ego_map_mask': torch.stack(batch_ego_map_mask, dim=0).to(dtype=torch.float32)
        }

        return result

    
    # def input_dim()