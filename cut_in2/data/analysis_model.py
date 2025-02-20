import pandas as pd
from statistics import stdev

def calculate_following_distance_acceleration_and_speed(file_path):
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 筛选出车编号为1-12的行
    df = df[df['vehicle-ID'].between(1, 12)]
    
    # 初始化结果字典
    results = {
        1: {'distances': [], 'acc_x': [], 'acc_y': [], 'speeds': [], 'y_deviations': []},
        2: {'distances': [], 'acc_x': [], 'acc_y': [], 'speeds': [], 'y_deviations': []},
        3: {'distances': [], 'acc_x': [], 'acc_y': [], 'speeds': [], 'y_deviations': []}
    }
    
    # 按时间分组
    grouped = df.groupby('time')
    
    for time, group in grouped:
        group = group.sort_values(by='x')
        for i, row in group.iterrows():
            vehicle_id = row['vehicle-ID']
            label = row['label']
            length = row['length']
            x = row['x']
            y = row['y']
            acc_x = row['acc_x']
            acc_y = row['acc_y']
            v_x = row['v_x']
            
            # 确定当前车的类别
            if length == 11:
                label = 3
            
            # 找到前车
            front_car = group[(group['x'] > x) & (group['y'] < y + 5) & (group['vehicle-ID'] != vehicle_id)]
            if not front_car.empty:
                front_car = front_car.iloc[0]
                distance = front_car['x'] - (1/2)*front_car['length'] - x - (1/2)*length
                results[label]['distances'].append(distance)
                results[label]['acc_x'].append(acc_x)
                results[label]['acc_y'].append(acc_y)
                results[label]['speeds'].append(v_x)
                results[label]['y_deviations'].append(y - 3.5)
    
    # 计算平均值和标准差
    avg_distance_label_1 = sum(results[1]['distances']) / len(results[1]['distances']) if results[1]['distances'] else 0
    avg_distance_label_2 = sum(results[2]['distances']) / len(results[2]['distances']) if results[2]['distances'] else 0
    avg_distance_label_3 = sum(results[3]['distances']) / len(results[3]['distances']) if results[3]['distances'] else 0

    
    avg_acc_x_label_1 = sum(results[1]['acc_x']) / len(results[1]['acc_x']) if results[1]['acc_x'] else 0
    avg_acc_x_label_2 = sum(results[2]['acc_x']) / len(results[2]['acc_x']) if results[2]['acc_x'] else 0
    avg_acc_x_label_3 = sum(results[3]['acc_x']) / len(results[3]['acc_x']) if results[3]['acc_x'] else 0
    
    avg_acc_y_label_1 = sum(results[1]['acc_y']) / len(results[1]['acc_y']) if results[1]['acc_y'] else 0
    avg_acc_y_label_2 = sum(results[2]['acc_y']) / len(results[2]['acc_y']) if results[2]['acc_y'] else 0
    avg_acc_y_label_3 = sum(results[3]['acc_y']) / len(results[3]['acc_y']) if results[3]['acc_y'] else 0
    
    std_acc_x_label_1 = stdev(results[1]['acc_x']) if len(results[1]['acc_x']) > 1 else 0
    std_acc_x_label_2 = stdev(results[2]['acc_x']) if len(results[2]['acc_x']) > 1 else 0
    std_acc_x_label_3 = stdev(results[3]['acc_x']) if len(results[3]['acc_x']) > 1 else 0
    
    std_acc_y_label_1 = stdev(results[1]['acc_y']) if len(results[1]['acc_y']) > 1 else 0
    std_acc_y_label_2 = stdev(results[2]['acc_y']) if len(results[2]['acc_y']) > 1 else 0
    std_acc_y_label_3 = stdev(results[3]['acc_y']) if len(results[3]['acc_y']) > 1 else 0
    
    avg_speed_label_1 = sum(results[1]['speeds']) / len(results[1]['speeds']) if results[1]['speeds'] else 0
    avg_speed_label_2 = sum(results[2]['speeds']) / len(results[2]['speeds']) if results[2]['speeds'] else 0
    avg_speed_label_3 = sum(results[3]['speeds']) / len(results[3]['speeds']) if results[3]['speeds'] else 0
    
    avg_y_deviation_label_1 = sum(results[1]['y_deviations']) / len(results[1]['y_deviations']) if results[1]['y_deviations'] else 0
    avg_y_deviation_label_2 = sum(results[2]['y_deviations']) / len(results[2]['y_deviations']) if results[2]['y_deviations'] else 0
    avg_y_deviation_label_3 = sum(results[3]['y_deviations']) / len(results[3]['y_deviations']) if results[3]['y_deviations'] else 0


    
    return (avg_distance_label_1, avg_distance_label_2, avg_distance_label_3, 
            avg_acc_x_label_1, avg_acc_x_label_2, avg_acc_x_label_3,
            std_acc_x_label_1, std_acc_x_label_2, std_acc_x_label_3,
            avg_acc_y_label_1, avg_acc_y_label_2, avg_acc_y_label_3,
            std_acc_y_label_1, std_acc_y_label_2, std_acc_y_label_3,
            avg_speed_label_1, avg_speed_label_2, avg_speed_label_3,
            avg_y_deviation_label_1, avg_y_deviation_label_2, avg_y_deviation_label_3)

# 使用示例
file_path = '/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/trajectory.csv'
(avg_distance_label_1, avg_distance_label_2, avg_distance_label_3, 
 avg_acc_x_label_1, avg_acc_x_label_2, avg_acc_x_label_3,
 std_acc_x_label_1, std_acc_x_label_2, std_acc_x_label_3,
 avg_acc_y_label_1, avg_acc_y_label_2, avg_acc_y_label_3,
 std_acc_y_label_1, std_acc_y_label_2, std_acc_y_label_3,
 avg_speed_label_1, avg_speed_label_2, avg_speed_label_3,
 avg_y_deviation_label_1, avg_y_deviation_label_2, avg_y_deviation_label_3) = calculate_following_distance_acceleration_and_speed(file_path)

print(f'平均跟车距离 (label=1): {avg_distance_label_1}')
print(f'平均跟车距离 (label=2): {avg_distance_label_2}')
print(f'平均跟车距离 (label=3): {avg_distance_label_3}')
print('---------------------------------------------------')

print(f'平均速度 (label=1): {avg_speed_label_1}')
print(f'平均速度 (label=2): {avg_speed_label_2}')
print(f'平均速度 (label=3): {avg_speed_label_3}')
print('---------------------------------------------------')

print(f'平均加速度 (label=1): {avg_acc_x_label_1}')
print(f'平均加速度 (label=2): {avg_acc_x_label_2}')
print(f'平均加速度 (label=3): {avg_acc_x_label_3}')
print('---------------------------------------------------')

print(f'加速度标准差 (label=1): {std_acc_x_label_1}')
print(f'加速度标准差 (label=2): {std_acc_x_label_2}')
print(f'加速度标准差 (label=3): {std_acc_x_label_3}')
print('---------------------------------------------------')

print(f'平均纵向加速度 (label=1): {avg_acc_y_label_1}')
print(f'平均纵向加速度 (label=2): {avg_acc_y_label_2}')
print(f'平均纵向加速度 (label=3): {avg_acc_y_label_3}')
print('---------------------------------------------------')

print(f'纵向加速度标准差 (label=1): {std_acc_y_label_1}')
print(f'纵向加速度标准差 (label=2): {std_acc_y_label_2}')
print(f'纵向加速度标准差 (label=3): {std_acc_y_label_3}')
print('---------------------------------------------------')

print(f'平均偏离中心线的距离 (label=1): {avg_y_deviation_label_1}')
print(f'平均偏离中心线的距离 (label=2): {avg_y_deviation_label_2}')
print(f'平均偏离中心线的距离 (label=3): {avg_y_deviation_label_3}')
