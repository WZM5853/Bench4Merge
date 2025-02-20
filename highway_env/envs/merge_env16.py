import numpy as np
import random
import csv
import pandas as pd
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle, MDPVehicle
from highway_env.road.objects import Obstacle
from highway_env.vehicle.behavior import IDMVehicle, IDMVehicle1, IDMVehicle2,IDMVehicle3,IDMVehicle4,IDMVehicle5,IDMVehicle6,IDMVehicle7,IDMVehicle8,IDMVehicle9,IDMVehicle10,IDMVehicle11,IDMVehicle12,IDMVehicle13,IDMVehicle_big
from highway_env.vehicle.kinematics import Vehicle, Vehicle2
from highway_env.envs.logdata import CircularCSVLogger, CircularCSVLogger2,CircularCSVLogger3

from cut_in2.test_Transformer_sim import run_Transformer,Args
from cut_in2.test_LSTM_sim import run_LSTM, Args_LSTM
from cut_in2.test_wzm_nochar import run_single_nochar
from vehicles_model.vehicle_init_data import simulate_idm, idm_acceleration
import time

import csv
# from os.path import exists



class MergeEnv16(AbstractEnv):

    """
    A highway merge negotiation environment.
    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """

    COLLISION_REWARD: float = -80
    RIGHT_LANE_REWARD: float = 0.02
    HIGH_SPEED_REWARD: float = 0.15
    MERGING_SPEED_REWARD: float = -1
    LANE_CHANGE_REWARD: float = -0.05
    OVER_SPOT_REWARD: float = 10
    STOP_REWARD: float = -5

    onramp = True

    # 训练时请注释
    # print('wzmwzmwzmwmzmwmzmwmzmwmzmwmmwmzmwmzmwm')

    # # 定义IDM模型参数
    # a_max = 6.0  # 最大加速度 (m/s^2)
    # b = 1.5  # 舒适减速度 (m/s^2)
    # T = 1.0  # 安全时距 (s)
    # v0 = 5  # 期望速度 (m/s)
    # s0 = 2.0  # 最小间距 (m)

    # # 初始条件
    # num_cars = 13
    # char = []
    # length = []
    # y_pos = []
    # for i in range(num_cars):
    #     char_value = random.choices([1, 2], weights=[0.2, 0.8])[0]
    #     # char_value = 2
    #     char.append(char_value)
    # for i in range(num_cars):
    #     length_value = random.choices([5, 11], weights=[0.8, 0.2])[0]
    #     # length_value = 5
    #     length.append(length_value)
    # for i in range(num_cars):
    #     y_value = random_number = random.uniform(3.0, 4.0)
    #     # length_value = 5
    #     y_pos.append(y_value)

    # x1 = random.randint(10,30)
    # x2 = x1 + (1/2)*length[0] + (1/2)*length[1] + random.randint(2,7)
    # x3 = x2 + (1/2)*length[1] + (1/2)*length[2] + random.randint(2,7)
    # x4 = x3 + (1/2)*length[2] + (1/2)*length[3] + random.randint(2,7)
    # x5 = x4 + (1/2)*length[3] + (1/2)*length[4] + random.randint(2,7)
    # x6 = x5 + (1/2)*length[4] + (1/2)*length[5] + random.randint(2,7)
    # x7 = x6 + (1/2)*length[5] + (1/2)*length[6] + random.randint(2,7)
    # x8 = x7 + (1/2)*length[6] + (1/2)*length[7] + random.randint(2,7)
    # x9 = x8 + (1/2)*length[7] + (1/2)*length[8] + random.randint(2,7)
    # x10 = x9 + (1/2)*length[8] + (1/2)*length[9] + random.randint(2,7)
    # x11 = x10 + (1/2)*length[9] + (1/2)*length[10] + random.randint(2,7)
    # x12 = x11 + (1/2)*length[10] + (1/2)*length[11] + random.randint(2,7)
    # x13 = x12 + (1/2)*length[11] + (1/2)*length[12] + random.randint(2,7) + 2

    # initial_positions = np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13])

    # v0 = random.randint(3,6)
    # v1 = v0
    # v2 = v0
    # v3 = v0
    # v4 = v0
    # v5 = v0
    # v6 = v0
    # v7 = v0
    # v8 = v0
    # v9 = v0
    # v10 = v0
    # v11 = v0
    # v12 = v0
    # v13 = random.randint(3,5)

    # # All cars start with 5 m/s except vehicle 0 which starts with 7 m/s
    # initial_velocities = np.array([v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13])

    # # 记录频率和时间
    # frequency = 10  # 10Hz
    # dt = 1.0 / frequency  # 时间步长
    # total_time = 1.0  # 总时间1秒
    # timesteps = int(total_time * frequency)

    # # 初始化存储数组
    # positions = np.zeros((timesteps, num_cars))
    # velocities = np.zeros((timesteps, num_cars))
    # accelerations = np.zeros((timesteps, num_cars))

    # # 初始状态
    # positions[0, :] = initial_positions
    # velocities[0, :] = initial_velocities



    # # 模拟每个时间步长
    # for t in range(1, timesteps):
    #     for i in range(num_cars):
    #         if i == num_cars - 1:
    #             # 最前面的车恒速行驶
    #             velocities[t, i] = velocities[t-1, i]
    #             positions[t, i] = positions[t-1, i] + velocities[t, i] * dt
    #             accelerations[t, i] = 0
    #         else:
    #             # 其他车根据IDM模型行驶
    #             # delta_v = velocities[t-1, i] - velocities[t-1, i+1]
    #             # s = positions[t-1, i+1] - positions[t-1, i]
    #             # accelerations[t, i] = idm_acceleration(velocities[t-1, i], s, delta_v)
    #             # velocities[t, i] = velocities[t-1, i] + accelerations[t, i] * dt
    #             # positions[t, i] = positions[t-1, i] + velocities[t, i] * dt

    #             velocities[t, i] = velocities[t-1, i]
    #             positions[t, i] = positions[t-1, i] + velocities[t, i] * dt
    #             accelerations[t, i] = 0

    # # car_id为0的车的初始条件
    # car_0_initial_velocity = random.randint(4,6)  # 匀速7 m/s
    # car_0_initial_position = random.randint(90,95)  # 初始位置为0
    # car_0_position_y = 7  # 车道位置为3.5

    # # 计算car_id为0的车在每个时间步长的位置
    # car_0_positions = np.zeros(timesteps)
    # car_0_velocities = np.full(timesteps, car_0_initial_velocity)
    # car_0_accelerations = np.zeros(timesteps)

    # for t in range(1, timesteps):
    #     car_0_positions[t] = car_0_positions[t-1] + car_0_velocities[t-1] * dt 

    # # 创建包含所有车的数据
    # all_data = []

    # for t in range(timesteps):
    #     for i in range(num_cars):
    #         if length[i] == 5:
    #             width = 2
    #         else:
    #             width = 2.5
    #         all_data.append({
    #             'time': t * dt,
    #             'vehicle-ID': i + 1,
    #             'x': positions[t, i],
    #             'y': y_pos[i],
    #             'heading': 0.0,
    #             'v_x': velocities[t, i],
    #             'v_y': 0.0,
    #             'acc_x': accelerations[t, i],
    #             'acc_y': 0.0,
    #             'label': char[i],
    #             'length': length[i],
    #             'width': width         
    #         })
    #     all_data.append({
    #         'time': t * dt,
    #         'vehicle-ID': 0,
    #         'x': car_0_positions[t] + car_0_initial_position,
    #         'y': car_0_position_y,
    #         'heading': 0.0,
    #         'v_x': car_0_velocities[t],
    #         'v_y': 0.0,
    #         'acc_x': car_0_accelerations[t],
    #         'acc_y': 0.0,
    #         'label': 2,
    #         'length': 5.0,
    #         'width': 2
    #     })

    # # 创建DataFrame
    # df_all = pd.DataFrame(all_data)

    # # 保存为CSV文件
    # csv_file_path_all = '/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/trajectory_10.csv'
    # df_all.to_csv(csv_file_path_all, index=False)

    # # Extract the last frame
    # last_frame = df_all[df_all['time'] == df_all['time'].max()]

    # # Save the last frame to a separate CSV
    # last_frame.to_csv('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/trajectory_1.csv', index=False)
    
    # print('wzmwzmwzmwmzmwmzmwmzmwmzmwmmwmzmwmzmwm')  
    
    t=1.0
    f6 = open('/data/wangzm/merge/Bench4Merge/cut_in2/data/trajectory.csv','w',encoding='utf-8', newline="")
    csv_write_6 = csv.writer(f6)
    csv_write_6.writerow(['time','vehicle-ID', 'x', 'y', 'heading','v_x','v_y','acc_x', 'acc_y', 'label','length','width'])


    def default_config(self) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "ContinuousAction",
            },
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "other_vehicles_type1": "highway_env.vehicle.behavior.IDMVehicle1",
            "other_vehicles_type_big": "highway_env.vehicle.behavior.IDMVehicle_big",
            "reward_speed_range": [1, 7],
            "offroad_terminal": False
        })

        return config

    def _reward(self, action: int) -> float:
        """
        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions
        But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.
        :param action: the action performed
        :return: the reward of the state-action transition
        """


        reward = self.COLLISION_REWARD * self.vehicle.crashed \
                 + self.HIGH_SPEED_REWARD * self.vehicle.speed

        if self.vehicle.speed < 2:
            reward += (self.vehicle.speed - 2)
        if self.vehicle.speed >= 2 and self.vehicle.speed < 11:
            reward += 1.5 * (np.cos(((np.pi)/9)*(self.vehicle.speed - 2) - np.pi) + 1)

        if self.vehicle.position[1] > 8.0:
            reward -= 10
        if self.vehicle.position[1] < 3.0:
            reward -= 2
        if self.vehicle.position[0] > 140:
            reward += 1
        
        #merging reward
        distance = self.vehicle.position[0] - 100
        merging_reward = 1 - np.exp(-((distance - 40) * (distance - 40))/(100))

        if self.vehicle.position[1] < 6:
            if self.vehicle.position[0] < 140:
                reward += 2 * merging_reward
        
        # if self.vehicle.position[1] >= 6:
        #     reward -=  (1/5000)*(distance**3)
        
        reward -= (1/3.5) * abs(self.vehicle.position[1]-3.5) * (1/4000) * (distance**3)

        v_list = []
        print(self.vehicle.position)
        
        for i in range(len(self.road.vehicles)):
            # logger = CircularCSVLogger('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/trajectory_10_2.csv')
            logger = CircularCSVLogger('/data/wangzm/merge/Bench4Merge/cut_in2/data/trajectory_10.csv')

            delta_f = self.road.vehicles[i].action['steering']
            beta = np.arctan(1 / 2 * np.tan(delta_f))
            v_x = self.road.vehicles[i].speed * np.cos(self.road.vehicles[i].heading + beta)
            v_y = self.road.vehicles[i].speed * np.sin(self.road.vehicles[i].heading + beta)
            a_x = self.road.vehicles[i].action['acceleration'] * np.cos(self.road.vehicles[i].heading + beta)
            a_y = self.road.vehicles[i].action['acceleration'] * np.sin(self.road.vehicles[i].heading + beta)

            ID = self.road.vehicles[i].ID
            
            df_7 = pd.read_csv('/data/wangzm/merge/Bench4Merge/cut_in2/data/trajectory_1.csv')
            vehicle_row = df_7[df_7['vehicle-ID'] == ID]
            char_value = vehicle_row['label'].values[0]

            logger.add_row([self.t,                            
                            self.road.vehicles[i].ID, 
                            self.road.vehicles[i].position[0], 
                            self.road.vehicles[i].position[1],
                            self.road.vehicles[i].heading, 
                            v_x,
                            v_y,
                            a_x,
                            a_y,
                            char_value,
                            self.road.vehicles[i].LENGTH,
                            self.road.vehicles[i].WIDTH
                            ])
            
            # df_8 = pd.read_csv('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/vehicle_length.csv')
            # length_row = df_8[df_8['vehicle-ID'] == ID]
            # vehicle_length = length_row['length'].values[0]
            
            self.csv_write_6 = csv.writer(self.f6)
            self.csv_write_6.writerow([self.t,                            
                            self.road.vehicles[i].ID, 
                            self.road.vehicles[i].position[0], 
                            self.road.vehicles[i].position[1],
                            self.road.vehicles[i].heading,
                            v_x,
                            v_y,
                            a_x,
                            a_y,
                            char_value,
                            self.road.vehicles[i].LENGTH,
                            self.road.vehicles[i].WIDTH
                            ])
            
        for i in range(len(self.road.vehicles)):
            # logger = CircularCSVLogger2('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/trajectory_1_2.csv')
            logger = CircularCSVLogger2('/data/wangzm/merge/Bench4Merge/cut_in2/data/trajectory_1.csv')
            delta_f = self.road.vehicles[i].action['steering']
            beta = np.arctan(1 / 2 * np.tan(delta_f))
            v_x = self.road.vehicles[i].speed * np.cos(self.road.vehicles[i].heading + beta)
            v_y = self.road.vehicles[i].speed * np.sin(self.road.vehicles[i].heading + beta)
            a_x = self.road.vehicles[i].action['acceleration'] * np.cos(self.road.vehicles[i].heading + beta)
            a_y = self.road.vehicles[i].action['acceleration'] * np.sin(self.road.vehicles[i].heading + beta)
            ID = self.road.vehicles[i].ID
            
            df_7 = pd.read_csv('/data/wangzm/merge/Bench4Merge/cut_in2/data/trajectory_1.csv')
            vehicle_row = df_7[df_7['vehicle-ID'] == ID]
            char_value = vehicle_row['label'].values[0]

            logger.add_row([self.t,                            
                            self.road.vehicles[i].ID, 
                            self.road.vehicles[i].position[0], 
                            self.road.vehicles[i].position[1],
                            self.road.vehicles[i].heading, 
                            v_x,
                            v_y,
                            a_x,
                            a_y,
                            char_value,
                            self.road.vehicles[i].LENGTH,
                            self.road.vehicles[i].WIDTH
                            ])

        # 运行预测模型
        args = Args()
        # args.ckpt = '/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/output/Transformer-three-to-three/anmn8kdk/checkpoints/last.ckpt'   
        # tracks_df = pd.read_csv('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/trajectory_10_2.csv')
        tracks_df = pd.read_csv('/data/wangzm/merge/Bench4Merge/cut_in2/data/trajectory_10.csv')

        t_start = time.time()
        pred = run_Transformer(args, tracks_df)
        t_end = time.time()
        print('time total =', t_end - t_start)

        pred_pos = pred['pos'].detach().cpu().numpy()
        pred_heading = pred['heading'].detach().cpu().numpy()
        pred_vel = pred['vel'].detach().cpu().numpy()

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
            
        self.t=0.1+self.t

        # Altruistic penalty
        for vehicle in self.road.vehicles:
            if vehicle.lane_index == ("b", "c", 1)and isinstance(vehicle, ControlledVehicle):
                reward += self.MERGING_SPEED_REWARD * (1 - (vehicle.speed)/10)
                reward -= 1
                if self.vehicle.speed < 4:
                    reward += 1

        for vehicle in self.road.vehicles:
            if vehicle.lane_index == ("b", "c", 2)and isinstance(vehicle, ControlledVehicle):
                reward += 2

        with open('/data/wangzm/merge/Bench4Merge/last_action.csv', "r") as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                last_action = row[0]
        current_action = self.vehicle.action['steering']
        difference_steering = float(current_action)-float(last_action)
        # print(difference_steering)
        reward -= ((6/np.pi)*np.square(difference_steering)) * 5

        with open('/data/wangzm/merge/Bench4Merge/last_speed.csv', 'r') as csvfile2:
            csvreader2 = csv.reader(csvfile2)
            for row in csvreader2:
                last_speed = row[0]
        current_speed = self.vehicle.speed
        difference_speed = float(current_speed)-float(last_speed)
        # print(abs(difference_speed))
        reward -= abs(difference_speed)


        print(self.vehicle.speed, reward)

        current_action = self.vehicle.action['steering']
        f = open('/data/wangzm/merge/Bench4Merge/last_action.csv','w',encoding='utf-8')
        csv_write_action = csv.writer(f)
        csv_write_action.writerow([current_action])
        f.close

        current_speed = self.vehicle.speed
        f2 = open('/data/wangzm/merge/Bench4Merge/last_speed.csv', 'w', encoding='utf-8')
        csv_write_speed = csv.writer(f2)
        csv_write_speed.writerow([current_speed])
        f2.close


        return reward

    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        # return self.vehicle.crashed or self.vehicle.position[0] > 150 or self.vehicle.position[1] > 15 or self.vehicle.position[1] < 2 or self.vehicle.speed < 0 or self.t>5.2
        return self.vehicle.crashed or self.vehicle.position[0] > 150 or self.vehicle.position[1] > 10 or \
            self.vehicle.position[1] < 2 or self.vehicle.speed < -0.3

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.
        :return: the road
        """
        net = RoadNetwork()

        # Highway lanes
        ends = [50, 50, 40, 150]  # Before, converging, merge, after
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH]
        p=StraightLane.DEFAULT_WIDTH
        # p = 3
        line_type = [[c, s], [n, c]]
        line_type_merge = [[c, s], [n, s]]
        line_type_straight=[c, c]
        for i in range(1):
            net.add_lane("a", "b", StraightLane([0, p], [sum(ends[:2])+5, p], line_types=line_type_straight))
            # net.add_lane("a", "b", StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type_straight))
            net.add_lane("b", "c", StraightLane([sum(ends[:2])+5, p], [sum(ends[:3]), p], line_types=line_type_merge[i]))
            net.add_lane("c", "d", StraightLane([sum(ends[:3]), p], [sum(ends), p], line_types=line_type_straight))

        # Merging lane
        amplitude = 1.5
        ljk = StraightLane([0, 3.0 + 3.5 + 3.5], [ends[0], 3.0 + 3.5 + 3.5], line_types=[c, c], forbidden=True)
        lkb = SineLane(ljk.position(ends[0], -amplitude), ljk.position(sum(ends[:2]), -amplitude),
                       amplitude, 2 * np.pi / (2*ends[1]), np.pi / 2, line_types=[c, c], forbidden=True)
        lbc = StraightLane(lkb.position(ends[1], 0), lkb.position(ends[1], 0) + [ends[2], 0],
                           line_types=[n, c], forbidden=True)

        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        :return: the ego-vehicle
        """
        # # # 训练时请解除注释
        # # 训练时请注释
        # print('wzmwzmwzmwmzmwmzmwmzmwmzmwmmwmzmwmzmwm')

        # # 定义IDM模型参数
        # a_max = 6.0  # 最大加速度 (m/s^2)
        # b = 1.5  # 舒适减速度 (m/s^2)
        # T = 1.0  # 安全时距 (s)
        # v0 = 5  # 期望速度 (m/s)
        # s0 = 2.0  # 最小间距 (m)

        # # 初始条件
        # num_cars = 13
        # char = []
        # length = []
        # y_pos = []
        # for i in range(num_cars):
        #     char_value = random.choices([1, 2], weights=[0.2, 0.8])[0]
        #     # char_value = 2
        #     char.append(char_value)
        # for i in range(num_cars):
        #     length_value = random.choices([5, 11], weights=[0.8, 0.2])[0]
        #     # length_value = 5
        #     length.append(length_value)
        # for i in range(num_cars):
        #     y_value = random_number = random.uniform(3.0, 4.0)
        #     # length_value = 5
        #     y_pos.append(y_value)

        # x1 = random.randint(10,30)
        # x2 = x1 + (1/2)*length[0] + (1/2)*length[1] + random.randint(2,7)
        # x3 = x2 + (1/2)*length[1] + (1/2)*length[2] + random.randint(2,7)
        # x4 = x3 + (1/2)*length[2] + (1/2)*length[3] + random.randint(2,7)
        # x5 = x4 + (1/2)*length[3] + (1/2)*length[4] + random.randint(2,7)
        # x6 = x5 + (1/2)*length[4] + (1/2)*length[5] + random.randint(2,7)
        # x7 = x6 + (1/2)*length[5] + (1/2)*length[6] + random.randint(2,7)
        # x8 = x7 + (1/2)*length[6] + (1/2)*length[7] + random.randint(2,7)
        # x9 = x8 + (1/2)*length[7] + (1/2)*length[8] + random.randint(2,7)
        # x10 = x9 + (1/2)*length[8] + (1/2)*length[9] + random.randint(2,7)
        # x11 = x10 + (1/2)*length[9] + (1/2)*length[10] + random.randint(2,7)
        # x12 = x11 + (1/2)*length[10] + (1/2)*length[11] + random.randint(2,7)
        # x13 = x12 + (1/2)*length[11] + (1/2)*length[12] + random.randint(2,7) + 2

        # initial_positions = np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13])

        # v0 = random.randint(3,6)
        # v1 = v0
        # v2 = v0
        # v3 = v0
        # v4 = v0
        # v5 = v0
        # v6 = v0
        # v7 = v0
        # v8 = v0
        # v9 = v0
        # v10 = v0
        # v11 = v0
        # v12 = v0
        # v13 = random.randint(3,5)

        # # All cars start with 5 m/s except vehicle 0 which starts with 7 m/s
        # initial_velocities = np.array([v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13])

        # # 记录频率和时间
        # frequency = 10  # 10Hz
        # dt = 1.0 / frequency  # 时间步长
        # total_time = 1.0  # 总时间1秒
        # timesteps = int(total_time * frequency)

        # # 初始化存储数组
        # positions = np.zeros((timesteps, num_cars))
        # velocities = np.zeros((timesteps, num_cars))
        # accelerations = np.zeros((timesteps, num_cars))

        # # 初始状态
        # positions[0, :] = initial_positions
        # velocities[0, :] = initial_velocities



        # # 模拟每个时间步长
        # for t in range(1, timesteps):
        #     for i in range(num_cars):
        #         if i == num_cars - 1:
        #             # 最前面的车恒速行驶
        #             velocities[t, i] = velocities[t-1, i]
        #             positions[t, i] = positions[t-1, i] + velocities[t, i] * dt
        #             accelerations[t, i] = 0
        #         else:
        #             # 其他车根据IDM模型行驶
        #             # delta_v = velocities[t-1, i] - velocities[t-1, i+1]
        #             # s = positions[t-1, i+1] - positions[t-1, i]
        #             # accelerations[t, i] = idm_acceleration(velocities[t-1, i], s, delta_v)
        #             # velocities[t, i] = velocities[t-1, i] + accelerations[t, i] * dt
        #             # positions[t, i] = positions[t-1, i] + velocities[t, i] * dt

        #             velocities[t, i] = velocities[t-1, i]
        #             positions[t, i] = positions[t-1, i] + velocities[t, i] * dt
        #             accelerations[t, i] = 0

        # # car_id为0的车的初始条件
        # car_0_initial_velocity = random.randint(4,6)  # 匀速7 m/s
        # car_0_initial_position = random.randint(90,95)  # 初始位置为0
        # car_0_position_y = 7  # 车道位置为3.5

        # # 计算car_id为0的车在每个时间步长的位置
        # car_0_positions = np.zeros(timesteps)
        # car_0_velocities = np.full(timesteps, car_0_initial_velocity)
        # car_0_accelerations = np.zeros(timesteps)

        # for t in range(1, timesteps):
        #     car_0_positions[t] = car_0_positions[t-1] + car_0_velocities[t-1] * dt 

        # # 创建包含所有车的数据
        # all_data = []

        # for t in range(timesteps):
        #     for i in range(num_cars):
        #         if length[i] == 5:
        #             width = 2
        #         else:
        #             width = 2.5
        #         all_data.append({
        #             'time': t * dt,
        #             'vehicle-ID': i + 1,
        #             'x': positions[t, i],
        #             'y': y_pos[i],
        #             'heading': 0.0,
        #             'v_x': velocities[t, i],
        #             'v_y': 0.0,
        #             'acc_x': accelerations[t, i],
        #             'acc_y': 0.0,
        #             'label': char[i],
        #             'length': length[i],
        #             'width': width         
        #         })
        #     all_data.append({
        #         'time': t * dt,
        #         'vehicle-ID': 0,
        #         'x': car_0_positions[t] + car_0_initial_position,
        #         'y': car_0_position_y,
        #         'heading': 0.0,
        #         'v_x': car_0_velocities[t],
        #         'v_y': 0.0,
        #         'acc_x': car_0_accelerations[t],
        #         'acc_y': 0.0,
        #         'label': 2,
        #         'length': 5.0,
        #         'width': 2
        #     })

        # # 创建DataFrame
        # df_all = pd.DataFrame(all_data)

        # # 保存为CSV文件
        # csv_file_path_all = '/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/trajectory_10.csv'
        # df_all.to_csv(csv_file_path_all, index=False)

        # # Extract the last frame
        # last_frame = df_all[df_all['time'] == df_all['time'].max()]

        # # Save the last frame to a separate CSV
        # last_frame.to_csv('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/trajectory_1.csv', index=False)
        
        # print('wzmwzmwzmwmzmwmzmwmzmwmzmwmmwmzmwmzmwm')


        # 初始条件
        with open('/data/wangzm/merge/Bench4Merge/cut_in2/data/trajectory_1.csv') as csvfile_length:
            reader_length = csv.DictReader(csvfile_length)
            for row in reader_length:
                if row['vehicle-ID'] == '1':
                    length_1 = float(row['length'])
                elif row['vehicle-ID'] == '2':
                    length_2 = float(row['length'])
                elif row['vehicle-ID'] == '3':
                    length_3 = float(row['length'])
                elif row['vehicle-ID'] == '4':
                    length_4 = float(row['length'])
                elif row['vehicle-ID'] == '5':
                    length_5 = float(row['length']) 
                elif row['vehicle-ID'] == '6':
                    length_6 = float(row['length']) 
                elif row['vehicle-ID'] == '7':
                    length_7 = float(row['length']) 
                elif row['vehicle-ID'] == '8':
                    length_8 = float(row['length']) 
                elif row['vehicle-ID'] == '9':
                    length_9 = float(row['length']) 
                elif row['vehicle-ID'] == '10':
                    length_10 = float(row['length']) 
                elif row['vehicle-ID'] == '11':
                    length_11 = float(row['length'])
                elif row['vehicle-ID'] == '12':
                    length_12 = float(row['length'])
                elif row['vehicle-ID'] == '13':
                    length_13 = float(row['length'])      

        self.t = 1.0

        road = self.road

        # 运行预测模型
        args = Args()
        # args.ckpt = '/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/output/Transformer-three-to-three/anmn8kdk/checkpoints/last.ckpt'   
        # tracks_df = pd.read_csv('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/trajectory_10_2.csv')
        tracks_df = pd.read_csv('/data/wangzm/merge/Bench4Merge/cut_in2/data/trajectory_10.csv')

        t_start = time.time()
        pred = run_Transformer(args, tracks_df)
        t_end = time.time()
        print('time total =', t_end - t_start)

        pred_pos = pred['pos'].detach().cpu().numpy()
        pred_heading = pred['heading'].detach().cpu().numpy()
        pred_vel = pred['vel'].detach().cpu().numpy()

        f1 = open('/data/wangzm/merge/Bench4Merge/cut_in2/data/prediction_1.csv', 'w', encoding='utf-8', newline="")
        csv_write_1 = csv.writer(f1)
        csv_write_1.writerow(['time','ID', 'v_x', 'v_y', 'heading', 'x', 'y'])

        for i in range(len(pred_pos)):

            pred_h = pred_heading[i][0]

            pred_v = pred_vel[i][0]
            pred_vx = pred_v[0]
            pred_vy = pred_v[1]

            pred_p = pred_pos[i][0]
            pred_x = pred_p[0]
            pred_y = pred_p[1]

            pred_h = float(pred_h)

            # pred_a = pred_acc[i][0]
            # pred_a = float(pred_a)

            # pred_s = pred_steering[i][0]
            # pred_s = float(pred_s)

            csv_write_1 = csv.writer(f1)
            csv_write_1.writerow([1.0, i, pred_vx, pred_vy, pred_h, pred_x, pred_y])

        # with open('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/trajectory_1_2.csv') as csvfile:
        with open('/data/wangzm/merge/Bench4Merge/cut_in2/data/trajectory_1.csv') as csvfile:

            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['vehicle-ID']=='0':
                    vehicle_0_x = float(row['x'])
                    vehicle_0_x = round(vehicle_0_x, 4)
                    vehicle_0_y = float(row['y'])
                    vehicle_0_y = round(vehicle_0_y, 4)
                    vehicle_0_v = float(row['v_x'])
                    vehicle_0_v = round(vehicle_0_v, 4)
                    vehicle_0_heading = float(row['heading'])
                    vehicle_0_heading = round(vehicle_0_heading, 4)
                    print(vehicle_0_x,vehicle_0_y,vehicle_0_v,vehicle_0_heading)
                if row['vehicle-ID']=='1':
                    vehicle_1_x=row['x']
                    vehicle_1_x=float(vehicle_1_x)
                    vehicle_1_y=row['y']
                    vehicle_1_y=float(vehicle_1_y)
                    vehicle_1_v=row['v_x']
                    vehicle_1_v=float(vehicle_1_v)
                    vehicle_1_heading=row['heading']
                    vehicle_1_heading=float(vehicle_1_heading)
                    print(vehicle_1_x,vehicle_1_y,vehicle_1_v,vehicle_1_heading)
                if row['vehicle-ID'] == '2':
                    vehicle_2_x = row['x']
                    vehicle_2_x = float(vehicle_2_x)
                    vehicle_2_y = row['y']
                    vehicle_2_y = float(vehicle_2_y)
                    vehicle_2_v = row['v_x']
                    vehicle_2_v = float(vehicle_2_v)
                    vehicle_2_heading = row['heading']
                    vehicle_2_heading = float(vehicle_2_heading)
                    print(vehicle_2_x, vehicle_2_y, vehicle_2_v, vehicle_2_heading)
                if row['vehicle-ID'] == '3':
                    vehicle_3_x = row['x']
                    vehicle_3_x = float(vehicle_3_x)
                    vehicle_3_y = row['y']
                    vehicle_3_y = float(vehicle_3_y)
                    vehicle_3_v = row['v_x']
                    vehicle_3_v = float(vehicle_3_v)
                    vehicle_3_heading = row['heading']
                    vehicle_3_heading = float(vehicle_3_heading)
                    print(vehicle_3_x, vehicle_3_y, vehicle_3_v, vehicle_3_heading)
                if row['vehicle-ID'] == '4':
                    vehicle_4_x = row['x']
                    vehicle_4_x = float(vehicle_4_x)
                    vehicle_4_y = row['y']
                    vehicle_4_y = float(vehicle_4_y)
                    vehicle_4_v = row['v_x']
                    vehicle_4_v = float(vehicle_4_v)
                    vehicle_4_heading = row['heading']
                    vehicle_4_heading = float(vehicle_4_heading)
                    print(vehicle_4_x, vehicle_4_y, vehicle_4_v, vehicle_4_heading)
                if row['vehicle-ID'] == '5':
                    vehicle_5_x = row['x']
                    vehicle_5_x = float(vehicle_5_x)
                    vehicle_5_y = row['y']
                    vehicle_5_y = float(vehicle_5_y)
                    vehicle_5_v = row['v_x']
                    vehicle_5_v = float(vehicle_5_v)
                    vehicle_5_heading = row['heading']
                    vehicle_5_heading = float(vehicle_5_heading)
                    print(vehicle_5_x, vehicle_5_y, vehicle_5_v, vehicle_5_heading)
                if row['vehicle-ID'] == '6':
                    vehicle_6_x = row['x']
                    vehicle_6_x = float(vehicle_6_x)
                    vehicle_6_y = row['y']
                    vehicle_6_y = float(vehicle_6_y)
                    vehicle_6_v = row['v_x']
                    vehicle_6_v = float(vehicle_6_v)
                    vehicle_6_heading = row['heading']
                    vehicle_6_heading = float(vehicle_6_heading)
                    print(vehicle_6_x, vehicle_6_y, vehicle_6_v, vehicle_6_heading)
                if row['vehicle-ID'] == '7':
                    vehicle_7_x = row['x']
                    vehicle_7_x = float(vehicle_7_x)
                    vehicle_7_y = row['y']
                    vehicle_7_y = float(vehicle_7_y)
                    vehicle_7_v = row['v_x']
                    vehicle_7_v = float(vehicle_7_v)
                    vehicle_7_heading = row['heading']
                    vehicle_7_heading = float(vehicle_7_heading)
                    print(vehicle_7_x, vehicle_7_y, vehicle_7_v, vehicle_7_heading)
                if row['vehicle-ID'] == '8':
                    vehicle_8_x = row['x']
                    vehicle_8_x = float(vehicle_8_x)
                    vehicle_8_y = row['y']
                    vehicle_8_y = float(vehicle_8_y)
                    vehicle_8_v = row['v_x']
                    vehicle_8_v = float(vehicle_8_v)
                    vehicle_8_heading = row['heading']
                    vehicle_8_heading = float(vehicle_8_heading)
                    print(vehicle_8_x, vehicle_8_y, vehicle_8_v, vehicle_8_heading)
                if row['vehicle-ID'] == '9':
                    vehicle_9_x = row['x']
                    vehicle_9_x = float(vehicle_9_x)
                    vehicle_9_y = row['y']
                    vehicle_9_y = float(vehicle_9_y)
                    vehicle_9_v = row['v_x']
                    vehicle_9_v = float(vehicle_9_v)
                    vehicle_9_heading = row['heading']
                    vehicle_9_heading = float(vehicle_9_heading)
                    print(vehicle_9_x, vehicle_9_y, vehicle_9_v, vehicle_9_heading)
                if row['vehicle-ID'] == '10':
                    vehicle_10_x = row['x']
                    vehicle_10_x = float(vehicle_10_x)
                    vehicle_10_y = row['y']
                    vehicle_10_y = float(vehicle_10_y)
                    vehicle_10_v = row['v_x']
                    vehicle_10_v = float(vehicle_10_v)
                    vehicle_10_heading = row['heading']
                    vehicle_10_heading = float(vehicle_10_heading)
                    print(vehicle_10_x, vehicle_10_y, vehicle_10_v, vehicle_10_heading)
                if row['vehicle-ID'] == '11':
                    vehicle_11_x = row['x']
                    vehicle_11_x = float(vehicle_11_x)
                    vehicle_11_y = row['y']
                    vehicle_11_y = float(vehicle_11_y)
                    vehicle_11_v = row['v_x']
                    vehicle_11_v = float(vehicle_11_v)
                    vehicle_11_heading = row['heading']
                    vehicle_11_heading = float(vehicle_11_heading)
                    print(vehicle_11_x, vehicle_11_y, vehicle_11_v, vehicle_11_heading)
                if row['vehicle-ID'] == '12':
                    vehicle_12_x = row['x']
                    vehicle_12_x = float(vehicle_12_x)
                    vehicle_12_y = row['y']
                    vehicle_12_y = float(vehicle_12_y)
                    vehicle_12_v = row['v_x']
                    vehicle_12_v = float(vehicle_12_v)
                    vehicle_12_heading = row['heading']
                    vehicle_12_heading = float(vehicle_12_heading)
                    print(vehicle_12_x, vehicle_12_y, vehicle_12_v, vehicle_12_heading)
                if row['vehicle-ID'] == '13':
                    vehicle_13_x = row['x']
                    vehicle_13_x = float(vehicle_13_x)
                    vehicle_13_y = row['y']
                    vehicle_13_y = float(vehicle_13_y)
                    vehicle_13_v = row['v_x']
                    vehicle_13_v = float(vehicle_13_v)
                    vehicle_13_heading = row['heading']
                    vehicle_13_heading = float(vehicle_13_heading)
                    print(vehicle_13_x, vehicle_13_y, vehicle_13_v, vehicle_13_heading)

                    
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        
        other_vehicles_type1 = utils.class_from_path(self.config["other_vehicles_type1"])
        other_vehicles_type_big = utils.class_from_path(self.config["other_vehicles_type_big"])
        
        if length_1 == 5:
            road.vehicles.append(
                other_vehicles_type1(ID=1, road=road,
                                    position=road.network.get_lane(("a", "b", 0)).position(vehicle_1_x, vehicle_1_y-3.5),
                                    heading=vehicle_1_heading, speed=vehicle_1_v))
        else:
            road.vehicles.append(
                other_vehicles_type_big(ID=1, road=road,
                                    position=road.network.get_lane(("a", "b", 0)).position(vehicle_1_x, vehicle_1_y-3.5),
                                    heading=vehicle_1_heading, speed=vehicle_1_v))

        if length_2 == 5:
            road.vehicles.append(
                other_vehicles_type1(ID=2, road=road,
                                    position=road.network.get_lane(("a", "b", 0)).position(vehicle_2_x, vehicle_2_y-3.5),
                                    heading=vehicle_2_heading, speed=vehicle_2_v))
        else:
            road.vehicles.append(
                other_vehicles_type_big(ID=2, road=road,
                                    position=road.network.get_lane(("a", "b", 0)).position(vehicle_2_x, vehicle_2_y-3.5),
                                    heading=vehicle_2_heading, speed=vehicle_2_v))

        if length_3 == 5:
            road.vehicles.append(
                other_vehicles_type1(ID=3, road=road,
                                    position=road.network.get_lane(("a", "b", 0)).position(vehicle_3_x, vehicle_3_y-3.5),
                                    heading=vehicle_3_heading, speed=vehicle_3_v))
        else:
            road.vehicles.append(
                other_vehicles_type_big(ID=3, road=road,
                                    position=road.network.get_lane(("a", "b", 0)).position(vehicle_3_x, vehicle_3_y-3.5),
                                    heading=vehicle_3_heading, speed=vehicle_3_v))
        
        if length_4 == 5:
            road.vehicles.append(
                other_vehicles_type1(ID=4, road=road,
                                    position=road.network.get_lane(("a", "b", 0)).position(vehicle_4_x, vehicle_4_y-3.5),
                                    heading=vehicle_4_heading, speed=vehicle_4_v))
        else:
            road.vehicles.append(
                other_vehicles_type_big(ID=4, road=road,
                                    position=road.network.get_lane(("a", "b", 0)).position(vehicle_4_x, vehicle_4_y-3.5),
                                    heading=vehicle_4_heading, speed=vehicle_4_v))

        if length_5 == 5:
            road.vehicles.append(
                other_vehicles_type1(ID=5, road=road,
                                    position=road.network.get_lane(("a", "b", 0)).position(vehicle_5_x, vehicle_5_y-3.5),
                                    heading=vehicle_5_heading, speed=vehicle_5_v))
        else:
            road.vehicles.append(
                other_vehicles_type_big(ID=5, road=road,
                                    position=road.network.get_lane(("a", "b", 0)).position(vehicle_5_x, vehicle_5_y-3.5),
                                    heading=vehicle_5_heading, speed=vehicle_5_v))

        if length_6 == 5:
            road.vehicles.append(
                other_vehicles_type1(ID=6, road=road,
                                    position=road.network.get_lane(("a", "b", 0)).position(vehicle_6_x, vehicle_6_y-3.5),
                                    heading=vehicle_6_heading, speed=vehicle_6_v))
        else:
            road.vehicles.append(
                other_vehicles_type_big(ID=6, road=road,
                                    position=road.network.get_lane(("a", "b", 0)).position(vehicle_6_x, vehicle_6_y-3.5),
                                    heading=vehicle_6_heading, speed=vehicle_6_v))

        if length_7 == 5:
            road.vehicles.append(
                other_vehicles_type1(ID=7, road=road,
                                    position=road.network.get_lane(("a", "b", 0)).position(vehicle_7_x, vehicle_7_y-3.5),
                                    heading=vehicle_7_heading, speed=vehicle_7_v))
        else:
            road.vehicles.append(
                other_vehicles_type_big(ID=7, road=road,
                                    position=road.network.get_lane(("a", "b", 0)).position(vehicle_7_x, vehicle_7_y-3.5),
                                    heading=vehicle_7_heading, speed=vehicle_7_v))

        if length_8 == 5:
            road.vehicles.append(
                other_vehicles_type1(ID=8, road=road,
                                    position=road.network.get_lane(("a", "b", 0)).position(vehicle_8_x, vehicle_8_y-3.5),
                                    heading=vehicle_8_heading, speed=vehicle_8_v))
        else:
            road.vehicles.append(
                other_vehicles_type_big(ID=8, road=road,
                                    position=road.network.get_lane(("a", "b", 0)).position(vehicle_8_x, vehicle_8_y-3.5),
                                    heading=vehicle_8_heading, speed=vehicle_8_v))

        if length_9 == 5:
            road.vehicles.append(
                other_vehicles_type1(ID=9, road=road,
                                    position=road.network.get_lane(("a", "b", 0)).position(vehicle_9_x, vehicle_9_y-3.5),
                                    heading=vehicle_9_heading, speed=vehicle_9_v))
        else:
            road.vehicles.append(
                other_vehicles_type_big(ID=9, road=road,
                                    position=road.network.get_lane(("a", "b", 0)).position(vehicle_9_x, vehicle_9_y-3.5),
                                    heading=vehicle_9_heading, speed=vehicle_9_v))

        if length_10 == 5:
            road.vehicles.append(
                other_vehicles_type1(ID=10, road=road,
                                    position=road.network.get_lane(("a", "b", 0)).position(vehicle_10_x, vehicle_10_y-3.5),
                                    heading=vehicle_10_heading, speed=vehicle_10_v))
        else:
            road.vehicles.append(
                other_vehicles_type_big(ID=10, road=road,
                                    position=road.network.get_lane(("a", "b", 0)).position(vehicle_10_x, vehicle_10_y-3.5),
                                    heading=vehicle_10_heading, speed=vehicle_10_v))

        if length_11 == 5:
            road.vehicles.append(
                other_vehicles_type1(ID=11, road=road,
                                    position=road.network.get_lane(("a", "b", 0)).position(vehicle_11_x, vehicle_11_y-3.5),
                                    heading=vehicle_11_heading, speed=vehicle_11_v))
        else:
            road.vehicles.append(
                other_vehicles_type_big(ID=11, road=road,
                                    position=road.network.get_lane(("a", "b", 0)).position(vehicle_11_x, vehicle_11_y-3.5),
                                    heading=vehicle_11_heading, speed=vehicle_11_v))

        if length_12 == 5:
            road.vehicles.append(
                other_vehicles_type1(ID=12, road=road,
                                    position=road.network.get_lane(("a", "b", 0)).position(vehicle_12_x, vehicle_12_y-3.5),
                                    heading=vehicle_12_heading, speed=vehicle_12_v))
        else:
            road.vehicles.append(
                other_vehicles_type_big(ID=12, road=road,
                                    position=road.network.get_lane(("a", "b", 0)).position(vehicle_12_x, vehicle_12_y-3.5),
                                    heading=vehicle_12_heading, speed=vehicle_12_v))
            
        if length_13 == 5:
            road.vehicles.append(
                other_vehicles_type1(ID=13, road=road,
                                    position=road.network.get_lane(("a", "b", 0)).position(vehicle_13_x, vehicle_13_y-3.5),
                                    heading=vehicle_13_heading, speed=vehicle_13_v))
        else:
            road.vehicles.append(
                other_vehicles_type_big(ID=13, road=road,
                                    position=road.network.get_lane(("a", "b", 0)).position(vehicle_13_x, vehicle_13_y-3.5),
                                    heading=vehicle_8_heading, speed=vehicle_13_v))


        ego_vehicle = self.action_type.vehicle_class(ID=0, road=road, position=road.network.get_lane(("b", "c", 0)).position(vehicle_0_x-107, vehicle_0_y-3.5), heading=vehicle_0_heading, speed=vehicle_0_v)
        # ego_vehicle = other_vehicles_type(ID=0, road=road, position=road.network.get_lane(("b", "c", 0)).position(vehicle_0_x-107, vehicle_0_y-3.5), heading=vehicle_0_heading, speed=vehicle_0_v)
        

        road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle
        ego_vehicle.SPEED_MIN = 0.5
        ego_vehicle.SPEED_MAX = 8


register(
    id='merge-v16',
    entry_point='highway_env.envs:MergeEnv16',
)