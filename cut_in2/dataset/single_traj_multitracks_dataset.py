import os
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset


class PredDataset(Dataset):
    def __init__(self, args) -> None:
        self.args = args
        self.data_dir = args.data_dir

        files = os.listdir(self.data_dir)
        files.sort()

        print('Start loading data')
        self.all_data = self.load_data(files)
        print('Number of data: ', len(self.all_data))

    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, idx):
        return self.all_data[idx]
    
    def load_data(self, files):  
        data_dict = {}
        cnt = 0
        for file in files:
            with open(os.path.join(self.data_dir, file), 'rb') as f:
                data = pickle.load(f)
            data = self.process_data(data)
            data_dict[cnt] = data
            cnt += 1
        return data_dict
    
    def process_data(self, data):
        tracks = torch.tensor(data['tracks'])
        map_data = data['map']

        valid = tracks[:, :, -1]
        valid = ((valid.sum(axis=-1))/valid.shape[-1]) == 1

        tracks = (tracks[valid]).numpy().astype(np.float32)

        tracks_info = []
        for land_id in [1,2,3,4]:
            trasks = tracks[tracks[:,0, 10] == land_id]
            num_tracks = tracks.shape[0]
            if num_tracks < 10:
                continue
            track_info = np.zeros((10,40,15))
            for i in range(num_tracks):
                self_track = tracks[i]
                curr_point = np.copy(self_track[0, [2,3, 12]])
                tracks_copy = np.copy(tracks)
                tracks_copy = np.delete(tracks_copy, i, axis=0)

                np.random.shuffle(tracks_copy)
                track_info[0] = self_track
                track_info[1:] = tracks_copy[:9]

                track_info[:,:, [2,3,12]] -= curr_point[None, ]
                cos, sin = np.cos(curr_point[-1]), np.sin(curr_point[-1])
                rotation_matrix = np.array([[cos, -sin], [sin, cos]])
                track_info[:,:, [2,3]] = track_info[:,:, [2,3]] @ rotation_matrix

                tracks_info.append(track_info)

        tracks_info = np.array(tracks_info)

        final_tracks_info = tracks_info[:, :, :self.args.hist_length, [2,3,4,5,6,7,12]]
        tracks_gt = tracks_info[:, :, self.args.hist_length:30+self.args.hist_length, [2,3,6,7,12]]

        result = {
            'tracks_info': final_tracks_info,
            'tracks_gt': tracks_gt,
            'curr_point': curr_point,
            # 'tracks_id': tracks_id,
            # 'centerlines': centerlines,
            # 'centerlines_mask': centerlines_mask,
            # 'centerlines_id': centerlines_id
        }

        return result
    
    def collate_fn(self, batch):
        result = {}
        for data in batch:
            for key, value in data.items():
                if key in result:
                    result[key].append(value)
                else:
                    result[key] = [value]
        
        for key, value in result.items():
            if key in ['tracks_info', 'tracks_gt', 'tracks_id']:
                result[key] = torch.tensor(np.concatenate(value, axis=0), dtype=torch.float32)
            elif key in ['centerlines', 'centerlines_mask', 'centerlines_id']:
                result[key] = torch.tensor(np.concatenate([i[None] for i in value], axis=0), dtype=torch.float32)

        return result