import os
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset


class PredDataset(Dataset):
    def __init__(self, args):
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
        # frame, id, x, y, width, height, xVelocity, yVelocity, xAcceleration, yAcceleration, laneId, angle, orientation, yaw_rate, valid
        tracks = torch.tensor(data['tracks'])

        valid = tracks[:, :, -1]
        valid = ((valid.sum(axis=-1))/valid.shape[-1]) == 1  # 所有具有完整轨迹的车

        # [x, y, width, height, xVelocity, yVelocity, xAcceleration, yAcceleration, laneId, orientation]
        valid_tracks = (tracks[valid]).numpy().astype(np.float32)
        
        # 最多预测15辆车
        if valid_tracks.shape[0] > 15:
            selected_elements = np.random.choice(list(range(valid_tracks.shape[0])), size=15, replace=False)
            valid_tracks = valid_tracks[selected_elements]
            tracks_mask = np.ones(15)
        elif valid_tracks.shape[0] < 15:
            padding = np.zeros_like(valid_tracks)[:15-valid_tracks.shape[0]]
            valid_tracks = np.concatenate([valid_tracks, padding], axis=0)
            tracks_mask = valid_tracks[:,0,-1]
        else:
            tracks_mask = np.ones(15)

        tracks_info = valid_tracks[:, (10-self.args.hist_length):10, [2,3,4,5,6,7,8,9,10,12]]
        # [x, y, xVelocity, yVelocity, orientation]
        tracks_gt = valid_tracks[:, 10:, [2,3,6,7,12]]
        tracks_id = np.array(list(range(tracks_info.shape[0])))

        map_data = data['map']

        centerlines = []
        max_len = 110
        for key, value in map_data.items():
            if value.shape[0] < max_len:
                print('error')
            elif value.shape[0] > max_len:
                selected_elements = np.random.choice(list(range(value.shape[0])), size=max_len, replace=False)
                selected_elements.sort()
                value = value[selected_elements]

            centerlines.append(value)
        centerlines = np.array(centerlines, dtype=np.float32)
        centerlines_id = np.array(list(map_data.keys()))
        centerlines_mask = np.ones(centerlines.shape[0])

        result = {
            'tracks_info': tracks_info, 
            'tracks_mask': tracks_mask.astype(bool),
            'tracks_id': tracks_id,
            'tracks_gt': tracks_gt, 
            'centerlines': centerlines,
            'centerlines_id': centerlines_id,
            'centerlines_mask': centerlines_mask.astype(bool)
            }

        return result