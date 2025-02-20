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
from torchsummary import summary

from models.LSTM_model import PredModel

import csv
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '7'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/data/wangzm/merge/cut-in3/data/scenario_40')
    parser.add_argument('--pred_length', type=int, choices=[30, 50], default=40)
    parser.add_argument('--hist_length', type=int, choices=[10, 1], default=10)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--num_layers', type=int, default=2)
    return parser.parse_args()



def draw_pred_trajs_single(batch, pred, gt=True):
    # curr_points = batch['curr_point']
    # print(curr_points)
    pred_pos = pred['pos'].detach().cpu().numpy()
    pred_heading = pred['heading'].detach().cpu().numpy()
    pred_vel = pred['vel'].detach().cpu().numpy()
    # print(pred_pos)
    # print(pred_heading)
    # 获取历史位置
    hist_pos = batch['ego_stat'][:, :, :2].detach().cpu().numpy()

    # 如果需要，获取 ground truth 位置
    if gt:
        gt_pos = batch['tracks_gt'].detach().cpu().numpy()
    else:
        ego_map = batch['ego_map'].detach().cpu().numpy()

    # 开始绘制
    fig, ax = plt.subplots(figsize=(30, 10))
    
    # 绘制每辆车的历史轨迹和预测轨迹
    for i in range(14):
        # 绘制历史轨迹
        ax.plot(hist_pos[i, :, 0], hist_pos[i, :, 1], label=f'History Car {i+1}', linestyle='--',linewidth=3)
        # 绘制预测轨迹
        ax.plot(pred_pos[i, :, 0], pred_pos[i, :, 1], label=f'Prediction Car {i+1}',linewidth=3)
        
        # 如果有 ground truth 位置，也绘制出来
        if gt:
            ax.plot(gt_pos[i, :, 0], gt_pos[i, :, 1], label=f'GT Car {i+1}', linestyle=':',linewidth=3)
    
    ax.set_title('Predicted and Historical Trajectories')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.legend()
    ax.grid(True)
    
    plt.savefig('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/output/output_7_1_1.png')
    plt.close(fig)

def run_single(args, tracks_df, map_df):
    torch.manual_seed(42)
    np.random.seed(0)

    # 车辆
    tracks_info = []
    tracks_id = []
    ego_stat = []
    ego_map = []
    current_positions = []
    inter_stat = []
    leading_distances = []
    vehicle_char = []
    # curr_point = []
    # ego_x = []

    vehicles_id = tracks_df['vehicle-ID'].unique()
    ego_vehicle_x = None  # 用于存储 vehicle-ID 为 0 的车辆的 x 坐标

    for vehicle_id in vehicles_id:
        track_info = tracks_df[tracks_df['vehicle-ID'] == vehicle_id].values
        track_info = track_info[track_info[:, 0].argsort()]

        current_x = track_info[9, 2]  # 当前车辆的 x 坐标
        current_y = track_info[9, 3]  # 当前车辆的 y 坐标
        current_vx = track_info[9, 6]
        current_vy = track_info[9, 7]
        current_h = track_info[9, 11]

        current_positions.append((current_x, current_y,current_vx,current_vy,current_h))

        if vehicle_id == 0:
            ego_vehicle_x = current_x  # 记录 vehicle-ID 为 0 的车辆的 x 坐标

        # 减去 current_x 和 current_y 以当前位置为原点
        adjusted_track_info = track_info.copy()
        adjusted_track_info[:, 2] -= current_x
        adjusted_track_info[:, 3] -= current_y

        tracks_info.append(adjusted_track_info[None, :, [2, 3, 11, 6, 7, 8, 9]])
        tracks_id.append(vehicle_id)

        if vehicle_id > 0:
            y_values = 3.5  # y 坐标与当前车辆的 y 坐标保持一致
        else:
            y_values = 7

        # 生成 ego_map 的 x 坐标
        x_values = np.linspace(current_x - current_x - 5, current_x - current_x + 20, 125)
        y_values = np.full_like(x_values, y_values - current_y)

        x_values_real = x_values + current_x

        # 初始化y_values_2
        y_values_2 = np.zeros_like(x_values_real)
        # 根据x_values_real的值计算y_values_2
        for i, x_real in enumerate(x_values_real):
            if x_real <= 115:
                y_values_2[i] = 7
            elif 115 < x_real <= 135:
                y_values_2[i] = 7 - 0.175 * (x_real - 115)
            else:
                y_values_2[i] = 3.5

        # 将y_values_2转换为相对y值
        y_values_2 = y_values_2 - current_y
        
        # y_values_2 = np.full_like(x_values_real, y_values_2)

        ego_map.append(np.stack((x_values, y_values, y_values_2), axis=1))  # 形状为 (125, 2)

    ego_stat = np.concatenate(tracks_info, axis=0)
    ego_map = np.stack(ego_map, axis=0)  # 将所有车辆的 ego_map 堆叠，形状为 (14, 125, 2)

    # 找到每辆车的前车并计算距离
    for i, (current_x, current_y, current_vx, current_vy, current_h) in enumerate(current_positions):
        vehicle_id = tracks_id[i]
        vehicle_track_info = tracks_df[tracks_df['vehicle-ID'] == vehicle_id].values
        leading_distances_vehicle = []

        for frame in range(10):  # 对于每一帧，计算前车距离
            frame_x = vehicle_track_info[frame, 2]
            frame_y = vehicle_track_info[frame, 3]
            leading_distance = 20  # 默认距离为20
            for j, (other_x, other_y, _, _, _) in enumerate(current_positions):
                if tracks_id[j] != vehicle_id:
                    other_vehicle_track_info = tracks_df[tracks_df['vehicle-ID'] == tracks_id[j]].values
                    other_frame_x = other_vehicle_track_info[frame, 2]
                    other_frame_y = other_vehicle_track_info[frame, 3]

                    # 检查是否在指定的矩形区域内
                    if (frame_x < other_frame_x <= frame_x + 20) and (frame_y - 1 <= other_frame_y <= frame_y + 1):
                        leading_distance = other_frame_x - frame_x -5
                        break

            leading_distances_vehicle.append(leading_distance)

        leading_distances.append(leading_distances_vehicle)
    




    # 计算 inter_stat
    ego_track_info = tracks_df[tracks_df['vehicle-ID'] == 0].values
    ego_past_info = ego_track_info[:10, [2, 3, 11, 6, 7]]  # vehicle-ID 为0的车的前10帧状态

    for i, (current_x, current_y, current_vx, current_vy,current_h) in enumerate(current_positions):
        vehicle_id = tracks_id[i]
        if vehicle_id == 0:
            inter_stat.append(np.zeros((10, 5)))
            vehicle_char.append(np.zeros((10, 1)))
        elif vehicle_id in range(1, 14) and ego_vehicle_x - 5 <= current_x <= ego_vehicle_x + 5:
            # 计算相对坐标和速度
            relative_info = ego_past_info.copy()
            relative_info[:, 0] -= current_x  # 相对横坐标
            relative_info[:, 1] -= current_y  # 相对纵坐标
            relative_info[:, 2] -= current_h  # 相对航向角
            relative_info[:, 3] -= current_vx  # 相对横向速度
            relative_info[:, 4] -= current_vy  # 相对纵向速度
            inter_stat.append(relative_info)
            with open('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/vehicle_char.csv', 'r', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                next(csv_reader)  # Skip the header
                for row in csv_reader:
                    if int(row[0]) == vehicle_id:
                        vehicle_char_inter = float(row[1])
                        vehicle_char.append(np.full((10, 1), vehicle_char_inter))
        else:
            inter_stat.append(np.zeros((10, 5)))
            vehicle_char.append(np.zeros((10, 1)))
    
    vehicle_char = np.stack(vehicle_char, axis=0)

    inter_stat = np.stack(inter_stat, axis=0)  # 形状为 (14, 10, 5)

    # 增加加速度和转向角
    time = tracks_df.iloc[:,0]
    vehicle_id = tracks_df.iloc[:,1]
    acc_x = tracks_df.iloc[:,8]
    acc_y = tracks_df.iloc[:,9]
    heading = tracks_df.iloc[:,11]

    unique_vehicles = vehicle_id.unique()
    num_vehicles = len(unique_vehicles)
    num_frames = time.nunique()

    acceleration = np.zeros((num_vehicles , num_frames, 1))
    steering = np.zeros((num_vehicles , num_frames, 1))

    for i, vehicle in enumerate(unique_vehicles):
        vehicle_data = tracks_df[tracks_df.iloc[:,1]==vehicle]
        acc_x_vehicle = vehicle_data.iloc[:, 8].values
        acc_y_vehicle = vehicle_data.iloc[:, 9].values
        heading_vehicle = vehicle_data.iloc[:, 11].values

        for j in range(num_frames - 1):
            acceleration[i, j, 0] = np.sqrt(acc_x_vehicle[j]**2 + acc_y_vehicle[j]**2)
            steering[i, j, 0] = heading_vehicle[j+1] - heading_vehicle[j]

        acceleration[i, -1 , 0] = acceleration[i, -2 ,0]
        steering[i, -1, 0] = steering[i, -2, 0]
    
    acceleration_list = acceleration.tolist()
    steering_list = steering.tolist()

    # 更新 ego_stat，增加前车距离信息，增加vehicle_char
    leading_distances = np.array(leading_distances).reshape(-1, 10, 1)
    ego_stat = np.concatenate((ego_stat, acceleration_list, steering_list, leading_distances, vehicle_char), axis=2)

    
    
    
    
    ckpt = torch.load(args.ckpt)
    model = PredModel(args)
    model.load_state_dict(ckpt['state_dict'])
    model = model.cuda()
    model.eval()

    # ego_stat[:,:,1] += 5
    # ego_stat[:,:,0] += 5

    input_data = {
        'ego_stat': torch.tensor(ego_stat, dtype=torch.float32, device=model.device),
        'ego_map': torch.tensor(ego_map, dtype=torch.float32, device=model.device),
        'right_stat': torch.tensor(inter_stat, dtype=torch.float32, device=model.device),
    }
    # t_start = time.time()
    pred = model.forward(input_data)
    # t_end = time.time()
    # print('time =', t_end - t_start)

    # 将 pred 中的坐标加回 current_x 和 current_y
    pred_pos = pred['pos'].detach().cpu().numpy()
    for i, (current_x, current_y, current_vx, current_vy,current_h) in enumerate(current_positions):
        pred_pos[i, :, 0] += current_x
        pred_pos[i, :, 1] += current_y

    # 更新 pred 中的坐标
    pred['pos'] = torch.tensor(pred_pos, dtype=torch.float32, device=model.device)

    # 将 input_data 中的 ego_stat 恢复到全局坐标
    ego_stat_global = ego_stat.copy()
    for i, (current_x, current_y, current_vx, current_vy,current_h) in enumerate(current_positions):
        ego_stat_global[i, :, 0] += current_x
        ego_stat_global[i, :, 1] += current_y

    input_data['ego_stat'] = torch.tensor(ego_stat_global, dtype=torch.float32, device=model.device)

    # draw_pred_trajs_single(input_data, pred, gt=False)

    return pred

if __name__ == '__main__':
    args = parse_args()
    # args.ckpt = '/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/output/cut-in/single-10/checkpoints/epoch=331-val_fde=0.29.ckpt'
    args.ckpt = '/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/output/cut-in-10/ghtoblry/checkpoints/last.ckpt'
    # main(args)

    tracks_df = pd.read_csv('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/trajectory_10_3.csv')
    map_df = pd.read_csv('/data/wangzm/merge/dense-merge/cut-in/data/map2.csv')
    pred = run_single(args, tracks_df, map_df)

    
    pred_pos = pred['pos'].detach().cpu().numpy()
    pred_heading = pred['heading'].detach().cpu().numpy()
    pred_vel = pred['vel'].detach().cpu().numpy()
    pred_acc = pred['acc'].detach().cpu().numpy()
    pred_steering = pred['steering'].detach().cpu().numpy()
    ## print(pred_vel)
    f6 = open('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/prediction_1.csv', 'w', encoding='utf-8', newline="")
    csv_write = csv.writer(f6)
    csv_write.writerow(['time','ID', 'v_x', 'v_y', 'heading', 'x', 'y', 'acceleration', 'steering'])

    for i in range(len(pred_pos)):

        pred_h = pred_heading[i][0]

        pred_v = pred_vel[i][0]
        pred_vx = pred_v[0]
        pred_vy = pred_v[1]

        pred_p = pred_pos[i][0]
        pred_x = pred_p[0]
        pred_y = pred_p[1]

        pred_h = float(pred_h)

        pred_a = pred_acc[i][0]
        pred_a = float(pred_a)

        pred_s = pred_steering[i][0]
        pred_s = float(pred_s)

        csv_write = csv.writer(f6)
        csv_write.writerow([2.0, i, pred_vx, pred_vy, pred_h, pred_x, pred_y, pred_a, pred_s])
        print (pred_vx, pred_vy, pred_h, pred_x, pred_y, pred_a, pred_s)

    # # 加载训练好的模型权重
    # model = PredModel.load_from_checkpoint(args.ckpt, args=args)

    # # 打印模型的参数信息
    # print("Model Summary:")
    # summary(model, input_size=[(args.hist_length, 7), (args.hist_length,5), (50,2)])
