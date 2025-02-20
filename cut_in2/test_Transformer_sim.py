
import os
import sys
sys.path.append('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2')
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.Transformer_model import AgentPredictor
from dataset.single_traj_dataset2 import PredDataset
import pandas as pd
import csv

# Set which GPU to use
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

# Define paths and sample index
# data_path = '/data/wangzm/merge/final_data/IDM_71.pkl'
model_path = '/data/wangzm/merge/Bench4Merge/cut_in2/output/Transformer-three-to-three/onhwgkdh/checkpoints/last.ckpt'
output_path = '/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/output/Transformer-sim-test'

class Args:
    # data_path = data_path
    pred_length = 40  # Set the appropriate prediction length
    hist_length = 10  # Set the appropriate history length
    num_layers = 2  # Set the number of LSTM layers
    lr = 1e-3  # Learning rate
    batch_size = 32  # Batch size
    max_epochs = 4000  # Max epochs
    num_workers = 1  # Number of workers

def draw_traj(output,input_data):
    # curr_points = batch['curr_point']
    # print(curr_points)
    pred_pos = output['pos'].detach().cpu().numpy()
    
    # 获取历史位置
    hist_pos = input_data['ego_stat'][:, :, :2].detach().cpu().numpy()

    # 开始绘制
    fig, ax = plt.subplots(figsize=(30, 5))
    
    # 绘制每辆车的历史轨迹和预测轨迹
    for i in range(14):
        # 绘制历史轨迹
        ax.plot(hist_pos[i, :, 0], hist_pos[i, :, 1], label=f'History Car {i+1}', linestyle='--',linewidth=3)
        # 绘制预测轨迹
        ax.plot(pred_pos[i, :, 0], pred_pos[i, :, 1], label=f'Prediction Car {i+1}',linewidth=3)
    
    ax.set_title('Predicted and Historical Trajectories')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.legend()

    # ax.set_xlim(110, 130)
    ax.grid(True)
    
    plt.savefig('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/output/Transformer-sim-test/HD40+DJI24w-2.png')
    plt.close(fig)

def run_Transformer(args, tracks_df):

    tracks_info = []
    tracks_id = []
    ego_stat = []
    ego_map = []
    current_positions = []
    inter_stat = []
    leading_distances = []
    vehicle_char = []
    vehicle_length = []
    ahead_info = []

    #构造map、ego、ahead
    vehicles_id = tracks_df['vehicle-ID'].unique()
    ego_vehicle_x = None  # 用于存储 vehicle-ID 为 0 的车辆的 x 坐标

    # 读取 char_df 并将其转为字典
    char_dict = tracks_df.set_index('vehicle-ID').to_dict()['label']

    # 读取 length_df 并将其转为字典
    length_dict = tracks_df.set_index('vehicle-ID').to_dict()['length']

    for vehicle_id in vehicles_id:
        track_info = tracks_df[tracks_df['vehicle-ID'] == vehicle_id].values
        track_info = track_info[track_info[:, 0].argsort()]

        current_x = track_info[9, 2]  # 当前车辆的 x 坐标
        current_y = track_info[9, 3]  # 当前车辆的 y 坐标
        current_h = track_info[9, 4]
        current_vx = track_info[9, 5]
        current_vy = track_info[9, 6]
        

        current_positions.append((current_x, current_y,current_vx,current_vy,current_h))

        if vehicle_id == 0:
            ego_vehicle_x = current_x  # 记录 vehicle-ID 为 0 的车辆的 x 坐标

        # 减去 current_x 和 current_y 以当前位置为原点
        adjusted_track_info = track_info.copy()
        adjusted_track_info[:, 2] -= current_x
        adjusted_track_info[:, 3] -= current_y

        tracks_info.append(adjusted_track_info[None, :, [2, 3, 4, 5, 6, 7, 8]])
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

        # 获取 vehicle_id 对应的 char 值并存储到 vehicle_char 列表中
        vehicle_char_value = char_dict.get(vehicle_id, 0)  # 默认值为0，如果没有找到对应的char
        vehicle_char.append(vehicle_char_value)
        # 获取 vehicle_id 对应的 length 值并存储到 vehicle_length 列表中
        vehicle_length_value = length_dict.get(vehicle_id, 0)  # 默认值为0，如果没有找到对应的char
        vehicle_length.append(vehicle_length_value)

    ego_stat = np.concatenate(tracks_info, axis=0)
    ego_map = np.stack(ego_map, axis=0)  # 将所有车辆的 ego_map 堆叠，形状为 (14, 125, 2)

    # 将 vehicle_char 转为 numpy 数组并扩展维度以便与 ego_stat 合并
    vehicle_char = np.array(vehicle_char).reshape(-1, 1, 1)
    vehicle_char = np.tile(vehicle_char, (1, ego_stat.shape[1], 1))  # 扩展到 (num_vehicles, num_frames, 1)

    # 将 vehicle_char 转为 numpy 数组并扩展维度以便与 ego_stat 合并
    vehicle_length = np.array(vehicle_length).reshape(-1, 1, 1)
    vehicle_length = np.tile(vehicle_length, (1, ego_stat.shape[1], 1))  # 扩展到 (num_vehicles, num_frames, 1)

    # 找到每辆车的前车并计算距离
    for i, (current_x, current_y, current_vx, current_vy, current_h) in enumerate(current_positions):
        vehicle_id = tracks_id[i]
        vehicle_track_info = tracks_df[tracks_df['vehicle-ID'] == vehicle_id].values
        leading_distances_vehicle = []
        ahead_vehicles_info = []
        
        for frame in range(10):  # 对于每一帧，计算前车距离
            frame_x = vehicle_track_info[frame, 2]
            frame_y = vehicle_track_info[frame, 3]
            leading_distance = 20  # 默认距离为20
            ahead_vehicle_info = [20,0,0,6,0]
            for j, (other_x, other_y, _, _, _) in enumerate(current_positions):
                if tracks_id[j] != vehicle_id:
                    other_vehicle_track_info = tracks_df[tracks_df['vehicle-ID'] == tracks_id[j]].values
                    other_frame_x = other_vehicle_track_info[frame, 2]
                    other_frame_y = other_vehicle_track_info[frame, 3]
                    other_frame_heading = other_vehicle_track_info[frame,11]
                    other_frame_v_x = other_vehicle_track_info[frame,6]
                    other_frame_v_y = other_vehicle_track_info[frame,7]

                    relative_ahead_x = other_frame_x - current_x
                    relative_ahead_y = other_frame_y - current_y
                    relative_ahead_h = other_frame_heading - current_h

                    # 检查是否在指定的矩形区域内
                    if (frame_x < other_frame_x <= frame_x + 20) and (frame_y - 1 <= other_frame_y <= frame_y + 1):
                        leading_distance = other_frame_x - frame_x -5
                        ahead_vehicle_info = [relative_ahead_x, relative_ahead_y, relative_ahead_h, other_frame_v_x, other_frame_v_y]
                        break

            leading_distances_vehicle.append(leading_distance)
            ahead_vehicles_info.append(ahead_vehicle_info)

        leading_distances.append(leading_distances_vehicle)
        ahead_info.append(ahead_vehicles_info)
    
    # 计算 right_info
    ego_track_info = tracks_df[tracks_df['vehicle-ID'] == 0].values
    ego_past_info = ego_track_info[:10, [2, 3, 4, 5, 6]]  # vehicle-ID 为0的车的前10帧状态

    for i, (current_x, current_y, current_vx, current_vy,current_h) in enumerate(current_positions):
        vehicle_id = tracks_id[i]
        if vehicle_id == 0:
            inter_stat.append(np.zeros((10, 5)))

        elif vehicle_id in range(1, 14) and ego_vehicle_x - 5 <= current_x <= ego_vehicle_x + 5:
            # 计算相对坐标和速度
            relative_info = ego_past_info.copy()
            relative_info[:, 0] -= current_x  # 相对横坐标
            relative_info[:, 1] -= current_y  # 相对纵坐标
            relative_info[:, 2] -= current_h  # 相对航向角
            # relative_info[:, 3] -= current_vx  # 相对横向速度
            # relative_info[:, 4] -= current_vy  # 相对纵向速度
            inter_stat.append(relative_info)
        else:
            inter_stat.append(np.zeros((10, 5)))

    right_info = np.stack(inter_stat, axis=0)  # 形状为 (14, 10, 5)

    # 提取数据列
    time = tracks_df.iloc[:,0]
    vehicle_id = tracks_df.iloc[:,1]

    unique_vehicles = vehicle_id.unique()
    num_vehicles = len(unique_vehicles)
    num_frames = time.nunique()

    acceleration = np.zeros((num_vehicles, num_frames, 1))
    steering = np.zeros((num_vehicles, num_frames, 1))
    y_values = np.zeros((num_vehicles, num_frames, 1))

    for i, vehicle in enumerate(unique_vehicles):
        vehicle_data = tracks_df[tracks_df.iloc[:,1] == vehicle]
        acc_x_vehicle = vehicle_data.iloc[:, 7].values
        acc_y_vehicle = vehicle_data.iloc[:, 8].values
        heading_vehicle = vehicle_data.iloc[:, 4].values
        y_vehicle = vehicle_data.iloc[:, 3].values - 3.5

        for j in range(num_frames - 1):
            acceleration[i, j, 0] = np.sqrt(acc_x_vehicle[j]**2 + acc_y_vehicle[j]**2)
            steering[i, j, 0] = heading_vehicle[j + 1] - heading_vehicle[j]
            y_values[i, j, 0] = y_vehicle[j]

        acceleration[i, -1, 0] = acceleration[i, -2, 0]
        steering[i, -1, 0] = steering[i, -2, 0]
        y_values[i, -1, 0] = y_vehicle[-1]

    acceleration_list = acceleration.tolist()
    steering_list = steering.tolist()
    y_list = y_values.tolist()

    # 更新 ego_stat，增加前车距离信息，增加vehicle_char
    leading_distances = np.array(leading_distances).reshape(-1, 10, 1)
    ego_stat = np.concatenate((ego_stat, acceleration_list, steering_list, y_list, leading_distances, vehicle_char, vehicle_length), axis=2)
    ahead_info = np.array(ahead_info)
    ego_map_tensor = torch.tensor(ego_map, dtype=torch.float32)

    # Load model
    config = {
        'map_input_size': 3,
        'map_hidden_size': 256,
        'agent_his_input_dim': 5,
        'agent_his_hidden_size': 256,
        'fusion_hidden_size': 256,
        'num_fusion_layer': args.num_layers,
        'dropout_rate': 0.1,
        'in_hidden_dim': 256,
        'head_hidden_dim': 256,
        'out_traj_dim': args.pred_length * 5,
        'num_queries': 40,
        'num_decoder_layer': args.num_layers,
        'aux_loss_temponet': True,
        'aux_loss_spanet': True,
        'lr': args.lr
    }

    model = AgentPredictor(config)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()  # Move model to GPU
    model.eval()

    input_data = {
        'ego_stat': torch.tensor(ego_stat, dtype=torch.float32, device=model.device),
        'ego_map': torch.tensor(ego_map, dtype=torch.float32, device=model.device),
        'right_stat': torch.tensor(right_info, dtype=torch.float32, device=model.device),
        'ahead_stat': torch.tensor(ahead_info, dtype=torch.float32, device=model.device),
        'ego_map_mask': torch.ones_like(ego_map_tensor[:, :, 0], dtype=torch.float32, device=model.device)
    }
    output, _ = model(input_data)

    pred_pos = output['pos'].detach().cpu().numpy()
    for i, (current_x, current_y, current_vx, current_vy,current_h) in enumerate(current_positions):
        pred_pos[i, :, 0] += current_x
        pred_pos[i, :, 1] += current_y

    # 更新 pred 中的坐标
    output['pos'] = torch.tensor(pred_pos, dtype=torch.float32, device=model.device)

    # 将 input_data 中的 ego_stat 恢复到全局坐标
    ego_stat_global = ego_stat.copy()
    for i, (current_x, current_y, current_vx, current_vy,current_h) in enumerate(current_positions):
        ego_stat_global[i, :, 0] += current_x
        ego_stat_global[i, :, 1] += current_y

    input_data['ego_stat'] = torch.tensor(ego_stat_global, dtype=torch.float32, device=model.device)

    # draw_traj(output,input_data)
    
    return output


if __name__ == '__main__':
    args = Args()
    # args.ckpt = '/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/output/Transformer-three-to-three/8txgxdri/checkpoints/last.ckpt'
    tracks_df = pd.read_csv('/data/wangzm/merge/Bench4Merge/cut_in2/data/trajectory_10.csv')

    pred = run_Transformer(args, tracks_df)
    pred_pos = pred['pos'].detach().cpu().numpy()
    pred_heading = pred['heading'].detach().cpu().numpy()
    pred_vel = pred['vel'].detach().cpu().numpy()
    for i in range(len(pred_pos)):

        pred_h = pred_heading[i][0]

        pred_v = pred_vel[i][0]
        pred_vx = pred_v[0]
        pred_vy = pred_v[1]

        pred_p = pred_pos[i][0]
        pred_x = pred_p[0]
        pred_y = pred_p[1]

        pred_h = float(pred_h)
        print(pred_x,pred_y,pred_h,pred_vx,pred_vy)

    f3 = open('/data/wangzm/merge/Bench4Merge/cut_in2/data/prediction_1.csv', 'w', encoding='utf-8', newline="")
    csv_write_3 = csv.writer(f3)
    csv_write_3.writerow(['time','ID', 'v_x', 'v_y', 'heading', 'x', 'y'])

    for i in range(len(pred_pos)):

        pred_h = pred_heading[i][0]

        pred_v = pred_vel[i][0]
        pred_vx = pred_v[0]
        pred_vy = pred_v[1]

        pred_p = pred_pos[i][0]
        pred_x = pred_p[0]
        pred_y = pred_p[1]

        pred_h = float(pred_h)

        csv_write_3 = csv.writer(f3)
        csv_write_3.writerow([1.0, i, pred_vx, pred_vy, pred_h, pred_x, pred_y])