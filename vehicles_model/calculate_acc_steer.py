import math

# 定义车辆当前状态和下一状态的数据结构
class VehicleState:
    def __init__(self, longitudinal_velocity, lateral_velocity, heading_angle, longitudinal_place, lateral_place):
        self.longitudinal_velocity = longitudinal_velocity  # 纵向速度
        self.lateral_velocity = lateral_velocity            # 横向速度
        self.heading_angle = heading_angle                  # 航向角，单位为度
        self.longitudinal_place = longitudinal_place
        self.lateral_place = lateral_place 

# 计算加速度和转向角的函数
def calculate_dynamics(current_state, next_state, dt):
    # 转换航向角度为弧度
    current_heading_rad = math.radians(current_state.heading_angle)
    next_heading_rad = math.radians(next_state.heading_angle)

    # 计算速度向量的变化
    delta_x = (next_state.longitudinal_place - current_state.longitudinal_place)
    delta_vx = 2 * (delta_x - current_state.longitudinal_velocity * dt) / (dt**2)

    delta_y = (next_state.lateral_place - current_state.lateral_place)
    delta_vy = 2 * (delta_y - current_state.lateral_velocity * dt) / (dt**2)

    # delta_vx = (next_state.longitudinal_velocity * math.cos(next_heading_rad) -
    #             current_state.longitudinal_velocity * math.cos(current_heading_rad))
    # delta_vy = (next_state.longitudinal_velocity * math.sin(next_heading_rad) -
    #             current_state.longitudinal_velocity * math.sin(current_heading_rad))

    # 计算总加速度
    acceleration = math.sqrt(delta_vx**2 + delta_vy**2) / dt
    # 根据速度变化的符号确定加速度方向
    # if (next_state.longitudinal_velocity - current_state.longitudinal_velocity) < 0:
    if delta_x < 0:
    
        acceleration = -acceleration
    # print(acceleration)

    # 计算转向角（角速度的近似）
    delta_heading = (next_heading_rad - current_heading_rad)
    turning_angle = delta_heading / dt
    # print(turning_angle)

    return acceleration, math.degrees(turning_angle)

# 示例数据
# current_state = VehicleState(20.0, 0.0, 0)  # 当前速度为20 m/s, 航向0度
# next_state = VehicleState(22.0, 1.0, 5)     # 下一状态速度为22 m/s, 横向速度1 m/s, 航向5度
# dt = 0.1  # 时间间隔

# # 计算加速度和转向角
# acceleration, turning_angle = calculate_dynamics(current_state, next_state, dt)
# print(f"Calculated Acceleration: {acceleration:.2f} m/s^2")
# print(f"Calculated Turning Angle: {turning_angle:.2f} degrees")