import numpy as np
import pandas as pd
import random

# 定义IDM模型参数
a_max = 1.0  # 最大加速度 (m/s^2)
b = 1.5  # 舒适减速度 (m/s^2)
T = 1.5  # 安全时距 (s)
v0 = 6  # 期望速度 (m/s)
s0 = 2.0  # 最小间距 (m)

# 初始条件

gap = random.randint(10,15)
x0 = random.randint(90,100)
x1 = random.randint(25,35)
x2 = x1 + random.randint(10,15)
x3 = x2 + random.randint(10,15)
x4 = x3 + random.randint(10,15)
x5 = x4 + random.randint(10,15)
x6 = x5 + random.randint(10,15)
x7 = x6 + random.randint(10,15)
x8 = x7 + random.randint(10,15)
x9 = x8 + random.randint(10,15)
x10 = x9 + random.randint(10,15)
x11 = x10 + random.randint(10,15)
x12 = x11 + random.randint(10,15)
x13 = x12 + random.randint(10,15)

initial_positions = np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13])

v0 = random.randint(6,9)
v1 = random.randint(4,7)
v2 = random.randint(4,7)
v3 = random.randint(4,7)
v4 = random.randint(4,7)
v5 = random.randint(4,7)
v6 = random.randint(4,7)
v7 = random.randint(4,7)
v8 = random.randint(4,7)
v9 = random.randint(4,7)
v10 = random.randint(4,7)
v11 = random.randint(4,7)
v12 = random.randint(4,7)
v13 = random.randint(4,7)

# All cars start with 5 m/s except vehicle 0 which starts with 7 m/s
initial_velocities = np.array([v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13])
num_cars = 13

# 记录频率和时间
frequency = 10  # 10Hz
dt = 1.0 / frequency  # 时间步长
total_time = 1.0  # 总时间1秒
timesteps = int(total_time * frequency)

# 初始化存储数组
positions = np.zeros((timesteps, num_cars))
velocities = np.zeros((timesteps, num_cars))
accelerations = np.zeros((timesteps, num_cars))

# 初始状态
positions[0, :] = initial_positions
velocities[0, :] = initial_velocities

def idm_acceleration(v, s, delta_v):
    """计算IDM模型下的加速度"""
    s_star = s0 + v * T + (v * delta_v) / (2 * np.sqrt(a_max * b))
    return a_max * (1 - (v / v0)**4 - (s_star / s)**2)

# 模拟每个时间步长
for t in range(1, timesteps):
    for i in range(num_cars):
        if i == num_cars - 1:
            # 最前面的车恒速行驶
            velocities[t, i] = velocities[t-1, i]
            positions[t, i] = positions[t-1, i] + velocities[t, i] * dt
            accelerations[t, i] = 0
        else:
            # 其他车根据IDM模型行驶
            delta_v = velocities[t-1, i] - velocities[t-1, i+1]
            s = positions[t-1, i+1] - positions[t-1, i]
            accelerations[t, i] = idm_acceleration(velocities[t-1, i], s, delta_v)
            velocities[t, i] = velocities[t-1, i] + accelerations[t, i] * dt
            positions[t, i] = positions[t-1, i] + velocities[t, i] * dt

# car_id为0的车的初始条件
car_0_initial_velocity = 7.0  # 匀速7 m/s
car_0_initial_position = 90.0  # 初始位置为0
car_0_position_y = 3.5  # 车道位置为3.5

# 计算car_id为0的车在每个时间步长的位置
car_0_positions = np.zeros(timesteps)
car_0_velocities = np.full(timesteps, car_0_initial_velocity)
car_0_accelerations = np.zeros(timesteps)

for t in range(1, timesteps):
    car_0_positions[t] = car_0_positions[t-1] + car_0_velocities[t-1] * dt 

# 创建包含所有车的数据
all_data = []

for t in range(timesteps):
    for i in range(num_cars):
        all_data.append({
            'time': t * dt,
            'vehicle_ID': i + 1,
            'x': positions[t, i],
            'y': 0.0,
            'width': 5.0,
            'height': 2.0,
            'v_x': velocities[t, i],
            'v_y': 0.0,
            'acc_x': accelerations[t, i],
            'acc_y': 0.0,
            'road-ID': 1,
            'heading': 0.0
        })
    all_data.append({
        'time': t * dt,
        'vehicle_ID': 0,
        'x': car_0_positions[t] + car_0_initial_position,
        'y': car_0_position_y,
        'width': 5.0,
        'height': 2.0,
        'v_x': car_0_velocities[t],
        'v_y': 0.0,
        'acc_x': car_0_accelerations[t],
        'acc_y': 0.0,
        'road-ID': 2,
        'heading': 0.0
    })

# 创建DataFrame
df_all = pd.DataFrame(all_data)

# 保存为CSV文件
csv_file_path_all = '/data/wangzm/merge/Merge-HighwayEnv-RL17/vehicles_model/car_simulation_data_corrected.csv'
df_all.to_csv(csv_file_path_all, index=False)

# Extract the last frame
last_frame = df_all[df_all['time'] == df_all['time'].max()]

# Save the last frame to a separate CSV
last_frame.to_csv('/data/wangzm/merge/Merge-HighwayEnv-RL17/vehicles_model/car_simulation_data_corrected_2.csv', index=False)
