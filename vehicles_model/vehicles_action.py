from cut_in2.test import run,parse_args
import pandas as pd
from vehicles_model.calculate_acc_steer import calculate_dynamics, VehicleState
import csv

# args = parse_args()
# args.ckpt = 'cut_in2/output/cut-in/3af3a0gl/checkpoints/epoch=505-val_fde=0.36.ckpt'

# # main(args)

# tracks_df = pd.read_csv('cut_in2/data/trajectory_1.csv')  
# map_df = pd.read_csv('cut_in2/data/map.csv')
# pred = run(args, tracks_df, map_df)
# index = 0
# pred_heading = pred['heading'][index].detach().cpu().numpy()
# pred_pos = pred['pos'][index].detach().cpu().numpy()
# for i in range(13):
#     print(pred_pos[i, 0, 0], pred_pos[i, 0, 1])
dt = 0.1

with open('cut_in2/data/trajectory_1.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row['vehicle-ID']=='1':
            vehicle_1_vx=row['v_x']
            vehicle_1_vx=float(vehicle_1_vx)
            vehicle_1_vy=row['v_y']
            vehicle_1_vy=float(vehicle_1_vy)
            vehicle_1_heading=row['heading']
            vehicle_1_heading=float(vehicle_1_heading)
            vehicle_1_current_state = VehicleState(vehicle_1_vx, vehicle_1_vy, vehicle_1_heading)
        if row['vehicle-ID']=='2':
            vehicle_2_vx=row['v_x']
            vehicle_2_vx=float(vehicle_2_vx)
            vehicle_2_vy=row['v_y']
            vehicle_2_vy=float(vehicle_2_vy)
            vehicle_2_heading=row['heading']
            vehicle_2_heading=float(vehicle_2_heading)
            vehicle_2_current_state = VehicleState(vehicle_2_vx, vehicle_2_vy, vehicle_2_heading)
        if row['vehicle-ID']=='3':
            vehicle_3_vx=row['v_x']
            vehicle_3_vx=float(vehicle_3_vx)
            vehicle_3_vy=row['v_y']
            vehicle_3_vy=float(vehicle_3_vy)
            vehicle_3_heading=row['heading']
            vehicle_3_heading=float(vehicle_3_heading)
            vehicle_3_current_state = VehicleState(vehicle_3_vx, vehicle_3_vy, vehicle_3_heading)
        if row['vehicle-ID']=='4':
            vehicle_4_vx=row['v_x']
            vehicle_4_vx=float(vehicle_4_vx)
            vehicle_4_vy=row['v_y']
            vehicle_4_vy=float(vehicle_4_vy)
            vehicle_4_heading=row['heading']
            vehicle_4_heading=float(vehicle_4_heading)
            vehicle_4_current_state = VehicleState(vehicle_4_vx, vehicle_4_vy, vehicle_4_heading)
        if row['vehicle-ID']=='5':
            vehicle_5_vx=row['v_x']
            vehicle_5_vx=float(vehicle_5_vx)
            vehicle_5_vy=row['v_y']
            vehicle_5_vy=float(vehicle_5_vy)
            vehicle_5_heading=row['heading']
            vehicle_5_heading=float(vehicle_5_heading)
            vehicle_5_current_state = VehicleState(vehicle_5_vx, vehicle_5_vy, vehicle_5_heading)
        if row['vehicle-ID']=='6':
            vehicle_6_vx=row['v_x']
            vehicle_6_vx=float(vehicle_6_vx)
            vehicle_6_vy=row['v_y']
            vehicle_6_vy=float(vehicle_6_vy)
            vehicle_6_heading=row['heading']
            vehicle_6_heading=float(vehicle_6_heading)
            vehicle_6_current_state = VehicleState(vehicle_6_vx, vehicle_6_vy, vehicle_6_heading)
        if row['vehicle-ID']=='7':
            vehicle_7_vx=row['v_x']
            vehicle_7_vx=float(vehicle_7_vx)
            vehicle_7_vy=row['v_y']
            vehicle_7_vy=float(vehicle_7_vy)
            vehicle_7_heading=row['heading']
            vehicle_7_heading=float(vehicle_7_heading)
            vehicle_7_current_state = VehicleState(vehicle_7_vx, vehicle_7_vy, vehicle_7_heading)
        if row['vehicle-ID']=='8':
            vehicle_8_vx=row['v_x']
            vehicle_8_vx=float(vehicle_8_vx)
            vehicle_8_vy=row['v_y']
            vehicle_8_vy=float(vehicle_8_vy)
            vehicle_8_heading=row['heading']
            vehicle_8_heading=float(vehicle_8_heading)
            vehicle_8_current_state = VehicleState(vehicle_8_vx, vehicle_8_vy, vehicle_8_heading)
        if row['vehicle-ID']=='9':
            vehicle_9_vx=row['v_x']
            vehicle_9_vx=float(vehicle_9_vx)
            vehicle_9_vy=row['v_y']
            vehicle_9_vy=float(vehicle_9_vy)
            vehicle_9_heading=row['heading']
            vehicle_9_heading=float(vehicle_9_heading)
            vehicle_9_current_state = VehicleState(vehicle_9_vx, vehicle_9_vy, vehicle_9_heading)
        if row['vehicle-ID']=='10':
            vehicle_10_vx=row['v_x']
            vehicle_10_vx=float(vehicle_10_vx)
            vehicle_10_vy=row['v_y']
            vehicle_10_vy=float(vehicle_10_vy)
            vehicle_10_heading=row['heading']
            vehicle_10_heading=float(vehicle_10_heading)
            vehicle_10_current_state = VehicleState(vehicle_10_vx, vehicle_10_vy, vehicle_10_heading)
        if row['vehicle-ID']=='11':
            vehicle_11_vx=row['v_x']
            vehicle_11_vx=float(vehicle_11_vx)
            vehicle_11_vy=row['v_y']
            vehicle_11_vy=float(vehicle_11_vy)
            vehicle_11_heading=row['heading']
            vehicle_11_heading=float(vehicle_11_heading)
            vehicle_11_current_state = VehicleState(vehicle_11_vx, vehicle_11_vy, vehicle_11_heading)
        if row['vehicle-ID']=='12':
            vehicle_12_vx=row['v_x']
            vehicle_12_vx=float(vehicle_12_vx)
            vehicle_12_vy=row['v_y']
            vehicle_12_vy=float(vehicle_12_vy)
            vehicle_12_heading=row['heading']
            vehicle_12_heading=float(vehicle_12_heading)
            vehicle_12_current_state = VehicleState(vehicle_12_vx, vehicle_12_vy, vehicle_12_heading)
        if row['vehicle-ID']=='13':
            vehicle_13_vx=row['v_x']
            vehicle_13_vx=float(vehicle_13_vx)
            vehicle_13_vy=row['v_y']
            vehicle_13_vy=float(vehicle_13_vy)
            vehicle_13_heading=row['heading']
            vehicle_13_heading=float(vehicle_13_heading)
            vehicle_13_current_state = VehicleState(vehicle_13_vx, vehicle_13_vy, vehicle_13_heading)

with open('cut_in2/data/prediction_1.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row['ID']=='0':
            vehicle_1_vx=row['v_x']
            vehicle_1_vx=float(vehicle_1_vx)
            vehicle_1_vy=row['v_y']
            vehicle_1_vy=float(vehicle_1_vy)
            vehicle_1_heading=row['heading']
            vehicle_1_heading=float(vehicle_1_heading)
            vehicle_1_next_state = VehicleState(vehicle_1_vx, vehicle_1_vy, vehicle_1_heading)
        if row['ID']=='1':
            vehicle_2_vx=row['v_x']
            vehicle_2_vx=float(vehicle_2_vx)
            vehicle_2_vy=row['v_y']
            vehicle_2_vy=float(vehicle_2_vy)
            vehicle_2_heading=row['heading']
            vehicle_2_heading=float(vehicle_2_heading)
            vehicle_2_next_state = VehicleState(vehicle_2_vx, vehicle_2_vy, vehicle_2_heading)
        if row['ID']=='2':
            vehicle_3_vx=row['v_x']
            vehicle_3_vx=float(vehicle_3_vx)
            vehicle_3_vy=row['v_y']
            vehicle_3_vy=float(vehicle_3_vy)
            vehicle_3_heading=row['heading']
            vehicle_3_heading=float(vehicle_3_heading)
            vehicle_3_next_state = VehicleState(vehicle_3_vx, vehicle_3_vy, vehicle_3_heading)
        if row['ID']=='3':
            vehicle_4_vx=row['v_x']
            vehicle_4_vx=float(vehicle_4_vx)
            vehicle_4_vy=row['v_y']
            vehicle_4_vy=float(vehicle_4_vy)
            vehicle_4_heading=row['heading']
            vehicle_4_heading=float(vehicle_4_heading)
            vehicle_4_next_state = VehicleState(vehicle_4_vx, vehicle_4_vy, vehicle_4_heading)
        if row['ID']=='4':
            vehicle_5_vx=row['v_x']
            vehicle_5_vx=float(vehicle_5_vx)
            vehicle_5_vy=row['v_y']
            vehicle_5_vy=float(vehicle_5_vy)
            vehicle_5_heading=row['heading']
            vehicle_5_heading=float(vehicle_5_heading)
            vehicle_5_next_state = VehicleState(vehicle_5_vx, vehicle_5_vy, vehicle_5_heading)
        if row['ID']=='5':
            vehicle_6_vx=row['v_x']
            vehicle_6_vx=float(vehicle_6_vx)
            vehicle_6_vy=row['v_y']
            vehicle_6_vy=float(vehicle_6_vy)
            vehicle_6_heading=row['heading']
            vehicle_6_heading=float(vehicle_6_heading)
            vehicle_6_next_state = VehicleState(vehicle_6_vx, vehicle_6_vy, vehicle_6_heading)
        if row['ID']=='6':
            vehicle_7_vx=row['v_x']
            vehicle_7_vx=float(vehicle_7_vx)
            vehicle_7_vy=row['v_y']
            vehicle_7_vy=float(vehicle_7_vy)
            vehicle_7_heading=row['heading']
            vehicle_7_heading=float(vehicle_7_heading)
            vehicle_7_next_state = VehicleState(vehicle_7_vx, vehicle_7_vy, vehicle_7_heading)
        if row['ID']=='7':
            vehicle_8_vx=row['v_x']
            vehicle_8_vx=float(vehicle_8_vx)
            vehicle_8_vy=row['v_y']
            vehicle_8_vy=float(vehicle_8_vy)
            vehicle_8_heading=row['heading']
            vehicle_8_heading=float(vehicle_8_heading)
            vehicle_8_next_state = VehicleState(vehicle_8_vx, vehicle_8_vy, vehicle_8_heading)
        if row['ID']=='8':
            vehicle_9_vx=row['v_x']
            vehicle_9_vx=float(vehicle_9_vx)
            vehicle_9_vy=row['v_y']
            vehicle_9_vy=float(vehicle_9_vy)
            vehicle_9_heading=row['heading']
            vehicle_9_heading=float(vehicle_9_heading)
            vehicle_9_next_state = VehicleState(vehicle_9_vx, vehicle_9_vy, vehicle_9_heading)
        if row['ID']=='9':
            vehicle_10_vx=row['v_x']
            vehicle_10_vx=float(vehicle_10_vx)
            vehicle_10_vy=row['v_y']
            vehicle_10_vy=float(vehicle_10_vy)
            vehicle_10_heading=row['heading']
            vehicle_10_heading=float(vehicle_10_heading)
            vehicle_10_next_state = VehicleState(vehicle_10_vx, vehicle_10_vy, vehicle_10_heading)
        if row['ID']=='10':
            vehicle_11_vx=row['v_x']
            vehicle_11_vx=float(vehicle_11_vx)
            vehicle_11_vy=row['v_y']
            vehicle_11_vy=float(vehicle_11_vy)
            vehicle_11_heading=row['heading']
            vehicle_11_heading=float(vehicle_11_heading)
            vehicle_11_next_state = VehicleState(vehicle_11_vx, vehicle_11_vy, vehicle_11_heading)
        if row['ID']=='11':
            vehicle_12_vx=row['v_x']
            vehicle_12_vx=float(vehicle_12_vx)
            vehicle_12_vy=row['v_y']
            vehicle_12_vy=float(vehicle_12_vy)
            vehicle_12_heading=row['heading']
            vehicle_12_heading=float(vehicle_12_heading)
            vehicle_12_next_state = VehicleState(vehicle_12_vx, vehicle_12_vy, vehicle_12_heading)
        if row['ID']=='12':
            vehicle_13_vx=row['v_x']
            vehicle_13_vx=float(vehicle_13_vx)
            vehicle_13_vy=row['v_y']
            vehicle_13_vy=float(vehicle_13_vy)
            vehicle_13_heading=row['heading']
            vehicle_13_heading=float(vehicle_13_heading)
            vehicle_13_next_state = VehicleState(vehicle_13_vx, vehicle_13_vy, vehicle_13_heading)


vehicle_1_acceleration, vehicle_1_steering = calculate_dynamics(vehicle_1_current_state, vehicle_1_next_state, dt)
vehicle_2_acceleration, vehicle_2_steering = calculate_dynamics(vehicle_2_current_state, vehicle_2_next_state, dt)
vehicle_3_acceleration, vehicle_3_steering = calculate_dynamics(vehicle_3_current_state, vehicle_3_next_state, dt)
vehicle_4_acceleration, vehicle_4_steering = calculate_dynamics(vehicle_4_current_state, vehicle_4_next_state, dt)
vehicle_5_acceleration, vehicle_5_steering = calculate_dynamics(vehicle_5_current_state, vehicle_5_next_state, dt)
vehicle_6_acceleration, vehicle_6_steering = calculate_dynamics(vehicle_6_current_state, vehicle_6_next_state, dt)
vehicle_7_acceleration, vehicle_7_steering = calculate_dynamics(vehicle_7_current_state, vehicle_7_next_state, dt)
vehicle_8_acceleration, vehicle_8_steering = calculate_dynamics(vehicle_8_current_state, vehicle_8_next_state, dt)
vehicle_9_acceleration, vehicle_9_steering = calculate_dynamics(vehicle_9_current_state, vehicle_9_next_state, dt)
vehicle_10_acceleration, vehicle_10_steering = calculate_dynamics(vehicle_10_current_state, vehicle_10_next_state, dt)
vehicle_11_acceleration, vehicle_11_steering = calculate_dynamics(vehicle_11_current_state, vehicle_11_next_state, dt)
vehicle_12_acceleration, vehicle_12_steering = calculate_dynamics(vehicle_12_current_state, vehicle_12_next_state, dt)
vehicle_13_acceleration, vehicle_13_steering = calculate_dynamics(vehicle_13_current_state, vehicle_13_next_state, dt)


replace_acceleration = 0
replace_steering = -0.1
vehicles_acceleration = []
vehicles_steering = [0,0,0,0,0,0,0,0,0,0,0,0,0]
# for i in range (13):
#     vehicles_acceleration.append(0)
#     vehicles_acceleration.append(0)
