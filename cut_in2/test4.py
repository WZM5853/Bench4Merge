import os
import sys
sys.path.append('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2')
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

# from models.joint_model import PredModel
# from dataset.multi_trajs_dataset import PredDataset
# from models.single_model import PredModel
# from dataset.single_traj_dataset import PredDataset
from models.single_traj_multitracks_model import PredModel
from dataset.single_traj_multitracks_dataset import PredDataset

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/data/wangjl/Air/dense-merge/cut-in/data/scenario_40')
    parser.add_argument('--pred_length', type=int, choices=[30, 50], default=30)
    parser.add_argument('--hist_length', type=int, choices=[10, 1], default=10)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--num_layers', type=int, default=2)
    return parser.parse_args()


def main(args):
    torch.manual_seed(42)
    np.random.seed(0)

    seed = torch.Generator().manual_seed(42)
    dataset = PredDataset(args)
    train_set = int(len(dataset) * 0.8)  # 80% of the data
    val_set = len(dataset)-train_set
    train_dataset, val_dataset = data.random_split(dataset, [train_set, val_set], generator=seed)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, drop_last=False, collate_fn=dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, drop_last=False, collate_fn=dataset.collate_fn)

    model = PredModel(args)
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt['state_dict'])
    model.eval()

    ade = []
    fde = []
    with torch.no_grad():
        for batch in tqdm(val_loader):
            pred = model.forward(batch)
            loss_dict = model.get_loss(pred, batch['tracks_gt'])
            # draw_pred_trajs_joint(batch, pred)
            draw_pred_trajs_single(batch, pred)
            ade.append(loss_dict['ade'].item())
            fde.append(loss_dict['fde'].item())
        print('ade:', np.mean(ade))
        print('fde:', np.mean(fde))


def draw_pred_trajs_joint(data, pred, gt=True):
    index = 0
    input_traj = data['tracks_info'][index][:,:,:2].detach().cpu().numpy()
    # bbx = data['tracks_info'][index][:,0,[0,1,2,3,9]].detach().cpu().numpy()
    agent_mask = data['tracks_mask'][index].detach().cpu().numpy()
    pred_pos = pred['pos'][index].detach().cpu().numpy()
    if gt:
        gt_pos = data['tracks_gt'][index].detach().cpu().numpy()
    
    centerlines = data['centerlines'][index].detach().cpu().numpy()
    centerlines_id = data['centerlines_id'][index].detach().cpu().numpy()
    centerlines_mask = data['centerlines_mask'][index].detach().cpu().numpy()
    centerlines_mask = centerlines_mask.sum(axis=1)
    plt.figure(figsize=(50, 10))

    # num_centerlines = sum(centerlines_mask)
    for i in range(centerlines.shape[0]):
        plt.plot(centerlines[i,:centerlines_mask[i], 0], centerlines[i, :centerlines_mask[i], 1], c='g')
        plt.text(centerlines[i, 0, 0], centerlines[i, 0, 1], f'lane_id:{centerlines_id[i]}', fontsize=20)

    num_tracks = sum(agent_mask)
    num_frames = input_traj.shape[1]

    for k, i in enumerate(range(num_tracks)):
        if num_frames == 1:
            plt.scatter(input_traj[i, :, 0], input_traj[i, :, 1], c='k', s=100, marker='s')
        else:
            plt.plot(input_traj[i, :, 0], input_traj[i, :, 1], 'k', linewidth=14)
        plt.plot(pred_pos[i, :, 0], pred_pos[i, :, 1], c='r', linewidth=4)
        plt.text(pred_pos[i, 0, 0], pred_pos[i, 0, 1], f'car-{k}', fontsize=15)
        if gt:
            plt.plot(gt_pos[i, :, 0], gt_pos[i, :, 1], c='b', linewidth=4)
            
        # rect=mpatches.Rectangle((bbx[i,0]-bbx[i,2]/2,bbx[i,1]-bbx[i,3]/2),bbx[i,2],bbx[i,3], 
        #                         fill=False,color="purple",linewidth=2, angle=bbx[i,4])
        # plt.gca().add_patch(rect)

    plt.savefig('/data/wangjl/wjl/cut-in/output/pred_traj.png')
    plt.close()

    return


def draw_pred_trajs_single(batch, pred, gt=True):
    curr_points = batch['curr_point']
    pred_pos = pred['pos'].detach().cpu().numpy()
    hist_pos = batch['tracks_info'][:,:,:2].detach().cpu().numpy()
    if gt:
        gt_pos = batch['tracks_gt'].detach().cpu().numpy()
    else:
        map = batch['centerlines']

    num = 0
    for scen_index in range(len(curr_points)):
        num_track = curr_points[scen_index].shape[0]
        plt.figure(figsize=(50, 10))
        for i in range(num_track):
            theta = curr_points[scen_index][i, 2]
            roation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            pred_traj = pred_pos[num] @ roation_matrix.T + curr_points[scen_index][i, :2]
            if gt:
                gt_traj = gt_pos[num,:,:2] @ roation_matrix.T + curr_points[scen_index][i, :2]
            hist_traj = hist_pos[num] @ roation_matrix.T + curr_points[scen_index][i, :2]
            # hist_traj = hist_pos[num]
            # pred_traj = pred_pos[num]
            num += 1
            if hist_traj.shape[0] == 1:
                plt.scatter(hist_traj[:, 0], hist_traj[:, 1], c='k', s=100, marker='s')
            else:
                plt.plot(hist_traj[:, 0], hist_traj[:, 1], c='k', linewidth=8)
            plt.plot(pred_traj[:, 0], pred_traj[:, 1], c='r', linewidth=8, alpha=0.9)
            if gt:
                plt.plot(gt_traj[:, 0], gt_traj[:, 1], c='b', linewidth=4)
        plt.savefig(f'/data/wangzm/merge/dense-merge/cut-in/output/pred_traj_{hist_traj.shape[0]}.png')
        print(f'/data/wangzm/merge/dense-merge/cut-in/output/pred_traj_{hist_traj.shape[0]}.png')
        plt.close()


def run_multi(args, tracks_df, map_df):
    torch.manual_seed(42)
    np.random.seed(0)
    ckpt = torch.load(args.ckpt)

    model = PredModel(args)
    model.load_state_dict(ckpt['state_dict'])
    model = model.cuda()
    model.eval()

    tracks_info = np.zeros((15, 7))  # (1,15,1,10)
    tracks_mask = np.zeros((15))      # (1,15)
    tracks_id = np.zeros((15))        # (1,15)
    centerlines = np.zeros((2, 256, 2))  # (1,15,1,10)
    centerlines_id = np.zeros((2))        # (1,15)
    centerlines_mask = np.zeros((2, 256))      # (1,15)
    # 车辆
    track_info = tracks_df.values
    track_info[:, 2:4]
    tracks_num, _ = track_info.shape
    tracks_info[:tracks_num] = track_info[:, [2,3,4,5,6,7,11]]
    tracks_mask[:tracks_num] = 1
    tracks_id[:tracks_num] = track_info[:, 1]
    
    # 地图
    centerline_id = map_df['Road-ID'].unique()
    for i, id in enumerate(centerline_id):
        centerline = map_df[map_df['Road-ID']==id]
        centerline = centerline[centerline['x']>50]
        centerline = centerline[centerline['x']<200].values[:, 1:]
        # 对centerlines排序
        centerline = centerline[np.argsort(centerline[:, 0])]
        # centerline -= np.array([50, 0])
        if centerline.shape[0] > 256:
            selected_elements = np.random.choice(list(range(centerline.shape[0])), size=110, replace=False)
            selected_elements.sort()
            centerlines[i] = centerline[selected_elements]
        else:
            centerlines[i, :centerline.shape[0]] = centerline
            centerlines_mask[i, :centerline.shape[0]] = 1
        centerlines_id[i] = id
    # 设置 batch_size=1
    ori_pos = np.copy(centerlines[1,0,:])
    centerlines[:,:] = centerlines[:,:]-ori_pos
    tracks_info[:,:2] = tracks_info[:,:2]-ori_pos
    tracks_info = torch.tensor(tracks_info[None,:,None,:], dtype=torch.float32, device=model.device)
    tracks_mask = torch.tensor(tracks_mask[None,...].astype(bool), device=model.device)
    tracks_id = torch.tensor(tracks_id[None,:], dtype=torch.int, device=model.device)
    centerlines = torch.tensor(centerlines[None,...], dtype=torch.float32, device=model.device)
    centerlines_id = torch.tensor(centerlines_id[None,...], dtype=torch.int, device=model.device)
    centerlines_mask = torch.tensor(centerlines_mask[None,...].astype(bool), device=model.device)

    input_data = {
        'tracks_info': tracks_info,
        'tracks_mask': tracks_mask,
        'tracks_id': tracks_id,
        'centerlines': centerlines,
        'centerlines_id': centerlines_id,
        'centerlines_mask': centerlines_mask
    }

    pred = model.forward(input_data)

    draw_pred_trajs_joint(input_data, pred, gt=False)

    return pred


def run_single(args, tracks_df, map_df):
    torch.manual_seed(42)
    np.random.seed(0)

    # 车辆
    tracks_info = []
    tracks_id = []
    curr_point = []

    vehicles_id = tracks_df['vehicle-ID'].unique()
    for vehicle_id in vehicles_id:
        track_info = tracks_df[tracks_df['vehicle-ID']==vehicle_id].values
        track_info = track_info[track_info[:, 0].argsort()]
        theta = np.copy(track_info[0, -1])
        curr_pos = np.copy(track_info[0, 2:4])
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        track_info[:, 2:4] = (track_info[:, 2:4]-curr_pos) @ rotation_matrix
        tracks_info.append(track_info[None, :, [2,3,4,5,6,7,11]])
        tracks_id.append(vehicle_id)
        curr_point.append(np.array([curr_pos[0], curr_pos[1], theta]).reshape(1,3))
    
    tracks_info = np.concatenate(tracks_info, axis=0)
    curr_point = np.concatenate(curr_point, axis=0)
    
    # # 地图
    centerlines = []
    centerlines_id = map_df['Road-ID'].unique().tolist()
    for centerline_id in centerlines_id:
        centerline = map_df[map_df['Road-ID']==centerline_id].values
        centerlines.append(centerline)

    ckpt = torch.load(args.ckpt)
    model = PredModel(args)
    model.load_state_dict(ckpt['state_dict'])
    model = model.cuda()
    model.eval()

    input_data = {
        'tracks_info': torch.tensor(tracks_info, dtype=torch.float32, device=model.device),
        'tracks_id': tracks_id,
        'centerlines': centerlines,
        'curr_point': [curr_point,]
    }
    pred = model.forward(input_data)

    draw_pred_trajs_single(input_data, pred, gt=False)

    return pred


def run_single_multitracks(args, tracks_df, map_df):
    torch.manual_seed(42)
    np.random.seed(0)

    # 车辆
    tracks_df = tracks_df[tracks_df['road_ID']==2]  # 选择主干道上的车辆
    vehicles_id = tracks_df['vehicle-ID'].unique()

    tracks = np.zeros((len(vehicles_id), 10, 12))

    for i, vehicle_id in enumerate(vehicles_id):
        track_info = tracks_df[tracks_df['vehicle-ID']==vehicle_id].values
        track_info = track_info[track_info[:, 0].argsort()]
        tracks[i] = track_info
    
    tracks_info = []
    curr_points =  []
    for i in range(tracks.shape[0]):
        track_info = np.zeros((10,10,12))  # num_tracks, hist_length, num_features
        self_track = tracks[i]
        curr_point = np.copy(self_track[0, [2,3, 11]])
        tracks_copy = np.copy(tracks)
        tracks_copy = np.delete(tracks_copy, i, axis=0)

        np.random.shuffle(tracks_copy)
        track_info[0] = self_track
        track_info[1:] = tracks_copy[:9]

        track_info[:,:, [2,3,11]] -= curr_point[None, ]
        cos, sin = np.cos(curr_point[-1]), np.sin(curr_point[-1])
        rotation_matrix = np.array([[cos, -sin], [sin, cos]])
        track_info[:,:, [2,3]] = track_info[:,:, [2,3]] @ rotation_matrix

        tracks_info.append(track_info[...,[2,3,4,5,6,7,11]])
        curr_points.append(curr_point)
    
    tracks = np.array(tracks_info)
    curr_points = np.array(curr_points)
    # curr_point = np.concatenate(tracks_info, axis=0)
    
    # # 地图
    centerlines = []
    centerlines_id = map_df['Road-ID'].unique().tolist()
    for centerline_id in centerlines_id:
        centerline = map_df[map_df['Road-ID']==centerline_id].values
        centerlines.append(centerline)

    ckpt = torch.load(args.ckpt)
    model = PredModel(args)
    model.load_state_dict(ckpt['state_dict'])
    model = model.cuda()
    model.eval()

    input_data = {
        'tracks_info': torch.tensor(tracks, dtype=torch.float32, device=model.device),
        # 'tracks_id': tracks_id,
        'centerlines': centerlines,
        'curr_point': [curr_points,]
    }
    pred = model.forward(input_data)
    pred['pos'] = pred['pos'][:,0]
    pred['vel'] = pred['vel'][:,0]
    pred['heading'] = pred['heading'][:,0]
    input_data['tracks_info'] = input_data['tracks_info'][:,0]
    draw_pred_trajs_single(input_data, pred, gt=False)

    return pred

if __name__ == '__main__':
    args = parse_args()
    args.ckpt = '/data/wangzm/merge/dense-merge/cut-in/output/cut-in/8ik6nuly/checkpoints/epoch=3819-val_fde=0.45.ckpt'
    
    # main(args)

    tracks_df = pd.read_csv('/data/wangzm/merge/Merge-HighwayEnv-RL16/cut_in2/data/trajectory_10_3.csv')
    map_df = pd.read_csv('/data/wangzm/merge/Merge-HighwayEnv-RL16/cut_in2/data/map2.csv')
    pred = run_single_multitracks(args, tracks_df, map_df)