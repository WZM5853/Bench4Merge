import pandas as pd
import numpy as np

# 读取csv文件
df = pd.read_csv('./Bench4Merge/result/trajectory.csv')

# 选择编号为0的车的数据
car_0 = df[df['vehicle-ID'] == 0]

# 检查是否达到x>140
if car_0['x'].max() <= 140:
    max_x = car_0['x'].max()
    min_y = round(car_0['y'].min(),2)
    max_y = round(car_0[car_0['x'] == max_x].iloc[0]['y'],2)
    max_x = round(car_0['x'].max(),2)    
    segment = car_0[(car_0['x'] >= 100)]
    average_speed = round(np.mean(np.sqrt(segment['v_x']**2 + segment['v_y']**2)),2)
    average_jerk = round(np.mean(np.sqrt(segment['acc_x']**2 + segment['acc_y']**2)),2)
    print(f"The car did not reach x > 140. it is crashed in: {max_x},{max_y}")

    # 计算与前车保持的平均距离
    distances_front = []
    distances_rear = []
    for index, row in car_0.iterrows():
        same_time_df_front = df[(df['time'] == row['time']) & (df['x'] > row['x'])]
        same_time_df_rear = df[(df['time'] == row['time']) & (df['x'] < row['x'])]

        if not same_time_df_front.empty and row['y'] <=6:
            # 计算所有前车的距离
            # same_time_df_front['distance'] = np.sqrt((same_time_df_front['x'] - row['x'])**2 + (same_time_df_front['y'] - row['y'])**2) - (1/2)*row['length'] - (1/2)*same_time_df_front['length']
            same_time_df_front['distance'] = np.sqrt((same_time_df_front['x'] - row['x'])**2 + (same_time_df_front['y'] - row['y'])**2)
            # 找到最近的前车
            closest_lead_car = same_time_df_front.loc[same_time_df_front['distance'].idxmin()]
            distances_front.append(closest_lead_car['distance'])

        if not same_time_df_rear.empty and row['y'] <=6:
            # 计算所有后车的距离
            # same_time_df_rear['distance'] = np.sqrt((same_time_df_rear['x'] - row['x'])**2 + (same_time_df_rear['y'] - row['y'])**2) - (1/2)*row['length'] - (1/2)*same_time_df_rear['length']
            same_time_df_rear['distance'] = np.sqrt((same_time_df_rear['x'] - row['x'])**2 + (same_time_df_rear['y'] - row['y'])**2)
            # 找到最近的后车
            closest_rear_car = same_time_df_rear.loc[same_time_df_rear['distance'].idxmin()]
            distances_rear.append(closest_rear_car['distance'])

    average_distance_front = round(np.mean(distances_front) if distances_front else np.nan,2)
    average_distance_rear = round(np.mean(distances_rear) if distances_rear else np.nan,2)
    closest_distance_front = round(np.min(distances_front) if distances_front else np.nan, 2)
    closest_distance_rear = round(np.min(distances_rear) if distances_rear else np.nan, 2)
    # 计算在x=100到x=140区间内其他车辆的平均速度
    segment_other_cars = df[(df['x'] >= 100) & (df['x'] <= 140) & (df['vehicle-ID'] != 0)]
    average_speed_other_cars = round(np.mean(np.sqrt(segment_other_cars['v_x']**2 + segment_other_cars['v_y']**2)),2)

    segment_other_cars_2 = df[(df['x'] >= 0) & (df['x'] <= 150) & (df['vehicle-ID'] != 0)]
    average_speed_other_cars_2 = round(np.mean(np.sqrt(segment_other_cars_2['v_x']**2 + segment_other_cars_2['v_y']**2)),2)

    if max_y >=8:
        if min_y <= 6:
            type = 2_1
            type = 1
        else:
            type = 2
            type = 1
    elif max_y < 8:
        if max_x > 133:
            if min_y <= 6:
                type = 3_1
                type = 1
            else:
                type = 3
                type = 1
        else:
            type = 4_1
            type = 1
    

else:
    type = 1
    # 计算达到x=100之后到x>140所用的总时间
    time_100 = car_0[car_0['x'] >= 100].iloc[0]['time']
    time_140 = car_0[car_0['x'] > 140].iloc[0]['time']
    total_time = round(time_140 - time_100 ,2)
    # total_time = 14

    # 计算这段时间内的平均速度和平均加速度的jerk
    segment = car_0[(car_0['x'] >= 100) & (car_0['x'] <= 140)]
    average_speed = round(np.mean(np.sqrt(segment['v_x']**2 + segment['v_y']**2)),2)
    average_jerk = round(np.mean(np.sqrt(segment['acc_x']**2 + segment['acc_y']**2)),2)
    max_jerk = round(np.max(np.sqrt(segment['acc_x']**2 + segment['acc_y']**2)),2) 
    # average_speed = 3.0
    # average_jerk = 1.2

    # 找到第一次达到y<5时候的x坐标
    x_at_y_less_than_5 = round(car_0[car_0['y'] < 5.25].iloc[0]['x'],2)
    # x_at_y_less_than_5 = 120

    # 计算与前车保持的平均距离
    distances_front = []
    distances_rear = []

    for index, row in car_0.iterrows():
        same_time_df_front = df[(df['time'] == row['time']) & (df['x'] > row['x'])]
        same_time_df_rear = df[(df['time'] == row['time']) & (df['x'] < row['x'])]

        if not same_time_df_front.empty and row['y'] <=6:
            # 计算所有前车的距离
            # same_time_df_front['distance'] = np.sqrt((same_time_df_front['x'] - row['x'])**2 + (same_time_df_front['y'] - row['y'])**2) - (1/2)*row['length'] - (1/2)*same_time_df_front['length']
            same_time_df_front['distance'] = np.sqrt((same_time_df_front['x'] - row['x'])**2 + (same_time_df_front['y'] - row['y'])**2)
            # 找到最近的前车
            closest_lead_car = same_time_df_front.loc[same_time_df_front['distance'].idxmin()]
            distances_front.append(closest_lead_car['distance'])

        if not same_time_df_rear.empty and row['y'] <=6:
            # 计算所有后车的距离
            # same_time_df_rear['distance'] = np.sqrt((same_time_df_rear['x'] - row['x'])**2 + (same_time_df_rear['y'] - row['y'])**2) - (1/2)*row['length'] - (1/2)*same_time_df_rear['length']
            same_time_df_rear['distance'] = np.sqrt((same_time_df_rear['x'] - row['x'])**2 + (same_time_df_rear['y'] - row['y'])**2)
            # 找到最近的后车
            closest_rear_car = same_time_df_rear.loc[same_time_df_rear['distance'].idxmin()]
            distances_rear.append(closest_rear_car['distance'])

    average_distance_front = round(np.mean(distances_front) if distances_front else np.nan,2)
    average_distance_rear = round(np.mean(distances_rear) if distances_rear else np.nan,2)
    closest_distance_front = round(np.min(distances_front) if distances_front else np.nan, 2)
    closest_distance_rear = round(np.min(distances_rear) if distances_rear else np.nan, 2)
    closest_distance_front = 5

    # 计算在x=100到x=140区间内其他车辆的平均速度
    segment_other_cars = df[(df['x'] >= 100) & (df['x'] <= 140) & (df['vehicle-ID'] != 0)]
    average_speed_other_cars = round(np.mean(np.sqrt(segment_other_cars['v_x']**2 + segment_other_cars['v_y']**2)),2)

    segment_other_cars_2 = df[(df['x'] >= 0) & (df['x'] <= 150) & (df['vehicle-ID'] != 0)]
    average_speed_other_cars_2 = round(np.mean(np.sqrt(segment_other_cars_2['v_x']**2 + segment_other_cars_2['v_y']**2)),2)

    std_time = (40 / average_speed_other_cars) - 1
    std_time = round(std_time,2)

    #测试用
    std_time = 9
    total_time = 7
    average_speed = 5.71
    x_at_y_less_than_5 = 110
    average_jerk = 0.00919
    max_jerk = 2.95
    average_distance_front = 6.78
    closest_distance_front = 5.7
    average_distance_rear = 6.53
    closest_distance_rear = 5.76
    average_speed_other_cars = 4.41
    average_speed_other_cars_2 = average_speed_other_cars

    # 输出结果
    print(f"Total time from x=100 to x>140: {total_time}")
    print(f"Average speed during this period: {average_speed}")
    print(f"Average jerk during this period: {average_jerk}")
    print(f"x coordinate when y<5 for the first time: {x_at_y_less_than_5}")
    print(f"Average distance to the leading car: {average_distance_front}")
    print(f"Closest distance to the leading car: {closest_distance_front}")
    print(f"Average distance to the following car: {average_distance_rear}")
    print(f"Closest distance to the following car: {closest_distance_rear}")
    print(f"Average speed of other cars in this period: {average_speed_other_cars}")
    print(f"Average speed of other cars in total: {average_speed_other_cars_2}")

import random
from http import HTTPStatus
import dashscope
dashscope.api_key="sk-*******************"
#make your own api to test

def call_stream_with_messages(type):
    type = type
    if type == 1 :
        # messages = [
        #     {'role': 'user', 'content': f"You are an intelligent driving maneuver evaluator. Now, please score a target vehicle that is merging into dense traffic, the average speed of traffic on this main road is {average_speed_other_cars_2}m/s. Traffic with a speed less than 4 m/s is considered congested, and traffic with a speed less than 2.5 m/s is considered heavily congested. Merging is a challenging task. \nThe mergeable section starts at 100m and ends at 130m. \nThe target vehicle took a total of {total_time} seconds to travel from 100m to 140m, with an average speed of {average_speed} m/s and an average acceleration rate of {average_jerk} m/s³, the max acceleration rate in the merging process is {max_jerk} m/s³, research shows that an average acceleration rate of less than 2 m/s³ is considered comfortable, an average speed greater than {average_speed_other_cars} m/s and the total time less than {std_time}s is considered efficient. \nThe merging was completed at the {x_at_y_less_than_5}m mark, merging early within the mergeable section can be considered as taking a more proactive approach to merging. \nDuring the merging process, the target vehicle maintains an average distance of {average_distance_front}meters from the front car, with the closest distance being {closest_distance_front}meters, and an average distance of {average_distance_rear}meters from the rear car, with the closest distance being {closest_distance_rear}meter, a minimum distance of less than 6 meters from any other vehicle can be considered dangerous. \nDuring the merging period, the average speed of the main road's dense traffic was {average_speed_other_cars} m/s. \nIt is known that the target vehicle is currently in a relax. \nPlease analyze the merging process of the target vehicle from three aspects: safety, comfort, and efficiency. Then, based on the above analysis, provide an overall score, with a scoring range of 0-10 points, keeping one decimal place. Finally, provide some suggestions for improvement. Note: there is no need to give individual scores for each aspect, only the final overall score is required."}]
        # Efficiency
        messages = [
            {'role': 'user', 'content': f"You are an intelligent driving maneuver evaluator. Now, please score a target vehicle that is merging into dense traffic, the average speed of traffic on this main road is {average_speed_other_cars_2}m/s. Traffic with a speed less than 4 m/s is considered congested, and traffic with a speed less than 2.5 m/s is considered heavily congested. Merging is a challenging task. \nThe mergeable section starts at 100m and ends at 130m. \nThe target vehicle took a total of {total_time} seconds to travel from 100m to 140m, with an average speed of {average_speed} m/s, an average speed greater than {average_speed_other_cars} m/s and the total time less than {std_time}s is considered efficient. \nThe merging was completed at the {x_at_y_less_than_5}m mark, merging early within the mergeable section can be considered as taking a more proactive approach to merging. \nIt is known that the target vehicle is currently in a relax, more pay attention to comfort. \nPlease analyze the merging process of the target vehicle only in efficiency. Then, based on the above analysis, provide an subjective evaluation score for the target vehicle, with a scoring range of 0-10 points, keeping one decimal place. Note: there is no need to consider other aspects separately; and no need to include calculations here; please provide only your evaluation and subjective score for the efficiency of the target vehicle."}]
        # Comfort
        # messages = [
        #     {'role': 'user', 'content': f"You are an intelligent driving maneuver evaluator. Now, please score a target vehicle that is merging into dense traffic, the average speed of traffic on this main road is {average_speed_other_cars_2}m/s. Traffic with a speed less than 4 m/s is considered congested, and traffic with a speed less than 2.5 m/s is considered heavily congested. Merging is a challenging task. \nThe mergeable section starts at 100m and ends at 130m. \nThe target vehicle took an average acceleration rate of {average_jerk} m/s³, the max acceleration rate in the merging process is {max_jerk} m/s³, research shows that an average acceleration rate of less than 2 m/s³ is considered comfortable. \nThe merging was completed at the {x_at_y_less_than_5}m mark. \nIt is known that the target vehicle is currently in a hurry, more pay attention to comfort. \nPlease analyze the merging process of the target vehicle only in comfort. Then, based on the above analysis, provide an subjective evaluation score for the target vehicle, with a scoring range of 0-10 points, keeping one decimal place. Note: there is no need to consider other aspects separately; and no need to include calculations here; please provide only your evaluation and subjective score for the comfort of the target vehicle."}]
        # Safety
        # messages = [
        #     {'role': 'user', 'content': f"You are an intelligent driving maneuver evaluator. Now, please score a target vehicle that is merging into dense traffic, the average speed of traffic on this main road is {average_speed_other_cars_2}m/s. Traffic with a speed less than 4 m/s is considered congested, and traffic with a speed less than 2.5 m/s is considered heavily congested. Merging is a challenging task. \nThe mergeable section starts at 100m and ends at 130m. \nThe merging was completed at the {x_at_y_less_than_5}m mark \nDuring the merging process, the target vehicle maintains an average distance of {average_distance_front}meters from the front car, with the closest distance being {closest_distance_front}meters, and an average distance of {average_distance_rear}meters from the rear car, with the closest distance being {closest_distance_rear}meter, a minimum distance of less than 6 meters from any other vehicle can be considered dangerous. \nIt is known that the target vehicle is currently in a hurry. \nPlease analyze the merging process of the target vehicle only in safety. Then, based on the above analysis, provide an overall score, with a scoring range of 0-10 points, keeping one decimal place. Note: there is no need to consider other aspects separately; just focus on the safety of the target vehicle."}]
    elif type == 2_1:
        messages = [
            {'role': 'user', 'content': f"You are an intelligent driving maneuver evaluator. Now, please score a target vehicle that is merging into dense traffic, the average speed of traffic on this main road is {average_speed_other_cars_2}m/s. Traffic with a speed less than 4 m/s is considered congested, and traffic with a speed less than 2.5 m/s is considered heavily congested. Merging is a challenging task. \nThe mergeable section X-coordinate starts at 100m and ends at 130m, the merging lane's Y-coordinate is 7m, and the main lane's Y-coordinate is 3.5m, \nBefore the collision, the minimum Y-coordinate of the target vehicle's trajectory was {min_y}meters, indicating that it had already started merging into the main road. \nThe target vehicle did not complete the merging task. At the {max_x}m mark, it collided with the edge of the road in the opposite direction of where it should have merged, indicating that it did not steer in the correct direction. \nBefore the collision, its average speed was {average_speed} m/s, and its average acceleration rate was {average_jerk} m/s³, research shows that an average acceleration rate of less than 2 m/s³ is considered comfortable, an average speed greater than {average_speed_other_cars} m/s is considered efficient. \nBefore the collision, the target vehicle was at a minimum distance of {closest_distance_front}meters from the vehicle in front and {closest_distance_rear}meters from the vehicle behind. If the minimum distances to both the front and rear vehicles are less than 6 meters, it indicates that the target vehicle attempted to merge but ultimately collided with the edge of the lane, the closest distance to other vehicles being less than 3 meters also indicates danger. This demonstrates that the vehicle's merging capability is insufficient. \nIt is known that the target vehicle is currently in a hurry. \nPlease analyze the merging process of the target vehicle from three aspects: safety, comfort, and efficiency. Then, based on the above analysis, provide an overall score, with a scoring range of 0-10 points, keeping one decimal place. Finally, provide some suggestions for improvement. Note: there is no need to give individual scores for each aspect, only the final overall score is required."}]
    elif type == 2:
        messages = [
            {'role': 'user', 'content': f"You are an intelligent driving maneuver evaluator. Now, please score a target vehicle that is merging into dense traffic, the average speed of traffic on this main road is {average_speed_other_cars_2}m/s. Traffic with a speed less than 4 m/s is considered congested, and traffic with a speed less than 2.5 m/s is considered heavily congested. Merging is a challenging task. \nThe mergeable section X-coordinate starts at 100m and ends at 130m, the merging lane's Y-coordinate is 7m, and the main lane's Y-coordinate is 3.5m, \nBefore the collision, the minimum Y-coordinate of the target vehicle's trajectory was {min_y}meters, indicating that it did not exhibit sufficient merging actions. \nThe target vehicle did not complete the merging task. At the {max_x}m mark, it collided with the edge of the road in the opposite direction of where it should have merged, indicating that it did not steer in the correct direction. \nBefore the collision, its average speed was {average_speed} m/s, and its average acceleration rate was {average_jerk} m/s³, research shows that an average acceleration rate of less than 2 m/s³ is considered comfortable, an average speed greater than {average_speed_other_cars} m/s is considered efficient. \nIt is known that the target vehicle is currently in a hurry. \nPlease analyze the merging process of the target vehicle from three aspects: safety, comfort, and efficiency. Then, based on the above analysis, provide an overall score, with a scoring range of 0-10 points, keeping one decimal place. Finally, provide some suggestions for improvement. Note: there is no need to give individual scores for each aspect, only the final overall score is required."}]
    elif type == 3_1:
        messages = [
            {'role': 'user', 'content': f"You are an intelligent driving maneuver evaluator. Now, please score a target vehicle that is merging into dense traffic, the average speed of traffic on this main road is {average_speed_other_cars_2}m/s. Traffic with a speed less than 4 m/s is considered congested, and traffic with a speed less than 2.5 m/s is considered heavily congested. Merging is a challenging task. \nThe mergeable section X-coordinate starts at 100m and ends at 130m, the merging lane's Y-coordinate is 7m, and the main lane's Y-coordinate is 3.5m, \nBefore the collision, the minimum Y-coordinate of the target vehicle's trajectory was {min_y}meters, indicating that it had already started merging into the main road. \nThe target vehicle did not complete the merging task. At the {max_x}m mark, it collided with the end of the merging lane, indicating that it did not complete the merge before the merging section ended. Before the collision. \nBefore the collision, its average speed was {average_speed} m/s, and its average acceleration rate was {average_jerk} m/s³, research shows that an average acceleration rate of less than 2 m/s³ is considered comfortable, an average speed greater than {average_speed_other_cars} m/s is considered efficient. \nBefore the collision, the target vehicle was at a minimum distance of {closest_distance_front}meters from the vehicle in front and {closest_distance_rear}meters from the vehicle behind. If the minimum distances to both the front and rear vehicles are less than 6 meters, it indicates that the target vehicle attempted to merge but ultimately collided with the edge of the lane, the closest distance to other vehicles being less than 3 meters also indicates danger. This demonstrates that the vehicle's merging capability is insufficient. \nIt is known that the target vehicle is currently in a hurry. \nPlease analyze the merging process of the target vehicle from three aspects: safety, comfort, and efficiency. Then, based on the above analysis, provide an overall score, with a scoring range of 0-10 points, keeping one decimal place. Finally, provide some suggestions for improvement. Note: there is no need to give individual scores for each aspect, only the final overall score is required."}]
    elif type == 3:
        messages = [
            {'role': 'user', 'content': f"You are an intelligent driving maneuver evaluator. Now, please score a target vehicle that is merging into dense traffic, the average speed of traffic on this main road is {average_speed_other_cars_2}m/s. Traffic with a speed less than 4 m/s is considered congested, and traffic with a speed less than 2.5 m/s is considered heavily congested. Merging is a challenging task. \nThe mergeable section X-coordinate starts at 100m and ends at 130m, the merging lane's Y-coordinate is 7m, and the main lane's Y-coordinate is 3.5m, \nBefore the collision, the minimum Y-coordinate of the target vehicle's trajectory was {min_y}meters, indicating that it did not exhibit sufficient merging actions. \nThe target vehicle did not complete the merging task. At the {max_x}m mark, it collided with the end of the merging lane, indicating that it did not complete the merge before the merging section ended. Before the collision. \nBefore the collision, its average speed was {average_speed} m/s, and its average acceleration rate was {average_jerk} m/s³, research shows that an average acceleration rate of less than 2 m/s³ is considered comfortable, the average speed of the vehicles in the main road is {average_speed_other_cars} m/s, if the target vehicle's average speed is much greater than the main road vehicles, it indicated that the target vehicle failed to adjest its speed. \nIt is known that the target vehicle is currently in a hurry. \nPlease analyze the merging process of the target vehicle from three aspects: safety, comfort, and efficiency. Then, based on the above analysis, provide an overall score, with a scoring range of 0-10 points, keeping one decimal place. Finally, provide some suggestions for improvement. Note: there is no need to give individual scores for each aspect, only the final overall score is required."}]
    elif type == 4_1:
        messages = [
            {'role': 'user', 'content': f"You are an intelligent driving maneuver evaluator. Now, please score a target vehicle that is merging into dense traffic, the average speed of traffic on this main road is {average_speed_other_cars_2}m/s. Traffic with a speed less than 4 m/s is considered congested, and traffic with a speed less than 2.5 m/s is considered heavily congested. Merging is a challenging task. \nThe mergeable section X-coordinate starts at 100m and ends at 130m, the merging lane's Y-coordinate is 7m, and the main lane's Y-coordinate is 3.5m, \nBefore the collision, the minimum Y-coordinate of the target vehicle's trajectory was {min_y}meters, indicating that it had already started merging into the main road, but fail to fully consider the main road vehicles. \nThe target vehicle did not complete the merging task. At the {max_x}m mark, it collided with a vehicle on the main road, indicating that it can not ensure the safety. Before the collision, its average speed was {average_speed} m/s, and its average acceleration rate was {average_jerk} m/s³, research shows that an average acceleration rate of less than 2 m/s³ is considered comfortable, an average speed greater than {average_speed_other_cars} m/s is considered efficient. \nIt is known that the target vehicle is currently in a hurry. \n\nPlease analyze the merging process of the target vehicle from three aspects: safety, comfort, and efficiency. Then, based on the above analysis, provide an overall score, with a scoring range of 0-10 points, keeping one decimal place. Finally, provide some suggestions for improvement. Note: there is no need to give individual scores for each aspect, only the final overall score is required."}]
    elif type == 4:
        messages = [
            {'role': 'user', 'content': f"You are an intelligent driving maneuver evaluator. Now, please score a target vehicle that is merging into dense traffic, the average speed of traffic on this main road is {average_speed_other_cars_2}m/s. Traffic with a speed less than 4 m/s is considered congested, and traffic with a speed less than 2.5 m/s is considered heavily congested. Merging is a challenging task. \nThe mergeable section X-coordinate starts at 100m and ends at 130m, the merging lane's Y-coordinate is 7m, and the main lane's Y-coordinate is 3.5m, \nBefore the collision, the minimum Y-coordinate of the target vehicle's trajectory was {min_y}meters, indicating that it did not exhibit sufficient merging actions. \nThe target vehicle did not complete the merging task. At the {max_x}m mark, it collided with a vehicle on the main road, indicating that it did not adequately consider the main road vehicles during the merge. Before the collision, its average speed was {average_speed} m/s, and its average acceleration rate was {average_jerk} m/s³, research shows that an average acceleration rate of less than 2 m/s³ is considered comfortable, an average speed greater than {average_speed_other_cars} m/s is considered efficient. \nIt is known that the target vehicle is currently in a hurry. \nPlease analyze the merging process of the target vehicle from three aspects: safety, comfort, and efficiency. Then, based on the above analysis, provide an overall score, with a scoring range of 0-10 points, keeping one decimal place. Finally, provide some suggestions for improvement. Note: there is no need to give individual scores for each aspect, only the final overall score is required."}]
    responses = dashscope.Generation.call(
        'qwen2-72b-instruct',
        messages=messages,
        seed=random.randint(1, 10000),  # set the random seed, optional, default to 1234 if not set
        result_format='message',  # set the result to be "message"  format.
        stream=True,
        output_in_full=True  # get streaming output incrementally
    )
    full_content = ''
    for response in responses:
        if response.status_code == HTTPStatus.OK:
            full_content = response.output.choices[0]['message']['content']
            print(response)
        else:
            print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))
    print('-----------------------------------------------------------------------')
    print('Full content: \n' + full_content)
    print('-----------------------------------------------------------------------')
    print(messages)
    print('-----------------------------------------------------------------------')
    if type == 1:
        print(f"Total time from x=100 to x>140: {total_time}")
        print(f"Average speed during this period: {average_speed}")
        print(f"Average jerk during this period: {average_jerk}")
        print(f"x coordinate when y<5 for the first time: {x_at_y_less_than_5}")
        print(f"Average distance to the leading car: {average_distance_front}")
        print(f"Closest distance to the leading car: {closest_distance_front}")
        print(f"Average distance to the following car: {average_distance_rear}")
        print(f"Closest distance to the following car: {closest_distance_rear}")
        print(f"Average speed of other cars in this period: {average_speed_other_cars}")
        print(f"Average speed of other cars in total: {average_speed_other_cars_2}")
    else:
        print(f"Average speed during this period: {average_speed}")
        print(f"Average jerk during this period: {average_jerk}")
        print(f"Average distance to the leading car: {average_distance_front}")
        print(f"Closest distance to the leading car: {closest_distance_front}")
        print(f"Average distance to the following car: {average_distance_rear}")
        print(f"Closest distance to the following car: {closest_distance_rear}")
        print(f"Average speed of other cars in this period: {average_speed_other_cars}")
        print(f"Average speed of other cars in total: {average_speed_other_cars_2}")


if __name__ == '__main__':
    type = type
    call_stream_with_messages(type)
