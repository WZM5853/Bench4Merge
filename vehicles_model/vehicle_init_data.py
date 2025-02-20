import numpy as np
import pandas as pd

# IDM model parameters
params = {
    'v0': 5,  # Desired speed (m/s)
    'T': 1.5,  # Safe time headway (s)
    'a': 5.0,  # Maximum acceleration (m/s^2)
    'b': 3.5,  # Comfortable deceleration (m/s^2)
    's0': 1.0,  # Minimum gap (m)
    'delta': 4.0  # Acceleration exponent
}

# 定义IDM模型参数
a_max = 3.0  # 最大加速度 (m/s^2)
b = 1.5  # 舒适减速度 (m/s^2)
T = 1.5  # 安全时距 (s)
v0 = 5  # 期望速度 (m/s)
s0 = 2.0  # 最小间距 (m)

def calculate_idm_acceleration(v, v_lead, s, params):
    s_star = params['s0'] + v * params['T'] + (v * (v_lead - v)) / (2 * np.sqrt(params['a'] * params['b']))
    a = params['a'] * (1 - (v / params['v0'])**params['delta'] - (s_star / s)**2)
    return a

def idm_acceleration(v, s, delta_v):
        """计算IDM模型下的加速度"""
        s_star = s0 + v * T + (v * delta_v) / (2 * np.sqrt(a_max * b))
        return a_max * (1 - (v / v0)**4 - (s_star / s)**2)

def simulate_idm(positions, velocities, params, steps=10, dt=0.1):
    num_cars = len(positions)
    data = {'time': [], 'vehicle-ID': [], 'x': [], 'y': [], 'width': [], 'height': [], 'v_x': [], 'v_y': [], 'acc_x': [], 'acc_y': [], 'road_ID': [], 'heading': []}

    for step in range(steps):
        time = step * dt
        new_velocities = velocities.copy()
        new_positions = positions.copy()

        for i in reversed(range(num_cars)):
            if i == 0:
                # 0号车在另一条车道上保持恒定速度
                acceleration = 0.0
                new_velocities[i] = velocities[i]
                road_ID = 2
                y = 7.0
                width = 5
                height = 2
                v_y = 0
                acc_y = 0
                heading = 0

            elif i == 13:
                # 13号车是头车，以恒定速度行驶
                y = 3.5
                acceleration = 0.0
                new_velocities[i] = velocities[i]  # The leading car maintains constant speed
                road_ID = 1
                width = 5
                height = 2
                v_y = 0
                acc_y = 0
                heading = 0

            else:
                s = positions[i-1] - positions[i]  # 跟车距离
                v_lead = velocities[i-1]  # 前车速度
                acceleration = calculate_idm_acceleration(velocities[i], v_lead, s, params)
                new_velocities[i] = velocities[i] + acceleration * dt
                y = 3.5
                width = 5
                height = 2
                v_y = 0
                road_ID = 1
                acc_y = 0
                heading = 0

            new_positions[i] = positions[i] + new_velocities[i] * dt

            data['time'].append(time)
            data['vehicle-ID'].append(i)
            data['x'].append(new_positions[i])
            data['y'].append(y)
            data['width'].append(width)
            data['height'].append(height)
            data['v_x'].append(new_velocities[i])
            data['v_y'].append(v_y)
            data['acc_x'].append(acceleration)
            data['acc_y'].append(acc_y)
            data['road_ID'].append(road_ID)
            data['heading'].append(heading)

        velocities = new_velocities
        positions = new_positions

    return pd.DataFrame(data)