from typing import TYPE_CHECKING, Optional, Union, Tuple, Callable
from gym import spaces
import numpy as np

from highway_env import utils
from highway_env.vehicle.dynamics import BicycleVehicle
from highway_env.vehicle.kinematics import Vehicle

from scipy.interpolate import CubicSpline
import pandas as pd
import cvxpy as cp

if TYPE_CHECKING:
    from highway_env.envs.common.abstract import AbstractEnv

Action = Union[int, np.ndarray]


class ActionType(object):

    """A type of action specifies its definition space, and how actions are executed in the environment"""

    def __init__(self, env: 'AbstractEnv', **kwargs) -> None:
        self.env = env
        self.__controlled_vehicle = None

    @property
    def controlled_vehicle(self):
        """The vehicle acted upon.

        If not set, the first controlled vehicle is used by default."""
        return self.__controlled_vehicle or self.env.vehicle

    @controlled_vehicle.setter
    def controlled_vehicle(self, vehicle):
        self.__controlled_vehicle = vehicle


class ContinuousAction(ActionType):

    """
    An continuous action space for throttle and/or steering angle.

    If both throttle and steering are enabled, they are set in this order: [throttle, steering]

    The space intervals are always [-1, 1], but are mapped to throttle/steering intervals through configurations.
    """

    ACCELERATION_RANGE = (-7, 4.0)
    """Acceleration range: [-x, x], in m/s²."""

    STEERING_RANGE = (-np.pi / 6, np.pi / 6)
    """Steering angle range: [-x, x], in rad."""

    def __init__(self,
                 env: 'AbstractEnv',
                 acceleration_range: Optional[Tuple[float, float]] = None,
                 steering_range: Optional[Tuple[float, float]] = None,
                 longitudinal: bool = True,
                 lateral: bool = True,
                 dynamical: bool = False,
                 clip: bool = True,
                 **kwargs) -> None:
        """
        Create a continuous action space.

        :param env: the environment
        :param acceleration_range: the range of acceleration values [m/s²]
        :param steering_range: the range of steering values [rad]
        :param longitudinal: enable throttle control
        :param lateral: enable steering control
        :param dynamical: whether to simulate dynamics (i.e. friction) rather than kinematics
        :param clip: clip action to the defined range
        """
        super().__init__(env)
        self.acceleration_range = acceleration_range if acceleration_range else self.ACCELERATION_RANGE
        self.steering_range = steering_range if steering_range else self.STEERING_RANGE
        self.lateral = lateral
        self.longitudinal = longitudinal
        if not self.lateral and not self.longitudinal:
            raise ValueError("Either longitudinal and/or lateral control must be enabled")
        self.dynamical = dynamical
        self.clip = clip
        self.last_action = np.zeros(self.space().shape)

    def space(self) -> spaces.Box:
        size = 2 if self.lateral and self.longitudinal else 1
        return spaces.Box(-1., 1., shape=(size,), dtype=np.float32)

    @property
    def vehicle_class(self) -> Callable:
        return Vehicle if not self.dynamical else BicycleVehicle

    def act(self, action: np.ndarray) -> None:
        perdicted_v = self.controlled_vehicle.speed + utils.lmap(action[0], [-1, 1], self.acceleration_range)*0.1
        max_speed = 7
        min_speed = 0.1

        if self.clip:
            action = np.clip(action, -1, 1)

        if perdicted_v >=max_speed:
            if self.longitudinal and self.lateral:
                self.controlled_vehicle.act({
                    "acceleration":(max_speed - self.controlled_vehicle.speed)/0.1,
                    "steering": utils.lmap(action[1], [-1, 1], self.steering_range),
                })
        elif perdicted_v <=min_speed:
            if self.longitudinal and self.lateral:
                self.controlled_vehicle.act({
                    "acceleration": -(self.controlled_vehicle.speed - min_speed)/0.1,
                    "steering": utils.lmap(action[1], [-1, 1], self.steering_range),
                })
        else:
            if self.longitudinal and self.lateral:
                self.controlled_vehicle.act({
                    "acceleration": utils.lmap(action[0], [-1, 1], self.acceleration_range),
                    "steering": utils.lmap(action[1], [-1, 1], self.steering_range),
                })
            elif self.longitudinal:
                self.controlled_vehicle.act({
                    "acceleration": utils.lmap(action[0], [-1, 1], self.acceleration_range),
                    "steering": 0,
                })
            elif self.lateral:
                self.controlled_vehicle.act({
                    "acceleration": 0,
                    "steering": utils.lmap(action[0], [-1, 1], self.steering_range)
                })
        

        # # 利用时空联合的方法方法规划
        # file_path = '/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/trajectory_1.csv'  # 假设CSV文件的路径
        # data = pd.read_csv(file_path)

        # # 找到自车信息
        # ego_vehicle = data[data['vehicle-ID'] == 0].iloc[0]
        # ego_x = ego_vehicle['x']
        # ego_y = ego_vehicle['y']
        # ego_heading = ego_vehicle['heading']
        # ego_vx = ego_vehicle['v_x']
        # ego_vy = ego_vehicle['v_y']
        # ego_speed = np.sqrt(ego_vx**2 + ego_vy**2)

        # # 找到x值比自车小的车里x值最大的车辆（后车）
        # rear_vehicle = data[(data['x'] < ego_x) & (data['vehicle-ID'] != 0)].nlargest(1, 'x').iloc[0]

        # # 找到x值比自车大的车里x值最小的车辆（前车）
        # front_vehicle = data[(data['x'] > ego_x) & (data['vehicle-ID'] != 0)].nsmallest(1, 'x').iloc[0]

        # # 提取障碍物信息
        # obstacles = [rear_vehicle, front_vehicle]

        # # 目标点
        # target_x, target_y = 140, 3.5

        # # 时间步长固定为0.1秒
        # dt = 0.1

        # # 最大的轨迹点数假设
        # max_points = 1000

        # # 优化变量
        # x = cp.Variable(max_points)
        # y = cp.Variable(max_points)
        # vx = cp.Variable(max_points)
        # vy = cp.Variable(max_points)
        # total_time = cp.Variable()

        # # 目标函数：最小化总用时
        # objective = cp.Minimize(total_time)

        # # 初始条件约束
        # constraints = [
        #     x[0] == ego_x,
        #     y[0] == ego_y,
        #     vx[0] == ego_vx,
        #     vy[0] == ego_vy
        # ]

        # # 动力学约束：匀速运动模型
        # for t in range(max_points - 1):
        #     constraints += [
        #         x[t + 1] == x[t] + vx[t] * dt,
        #         y[t + 1] == y[t] + vy[t] * dt
        #     ]

        # # 终止条件：设置为距离目标点小于0.5米时终止
        # for t in range(max_points):
        #     constraints += [
        #         cp.norm(cp.hstack([x[t] - target_x, y[t] - target_y]), 2) <= 0.5 + 1000 * (t * dt - total_time),
        #         total_time >= t * dt
        #     ]
        #     # 障碍物避让约束
        #     for obstacle in obstacles:
        #         obs_x = obstacle['x']
        #         obs_y = obstacle['y']
        #         obs_vx = obstacle['v_x']
        #         obs_vy = obstacle['v_y']
        #         obs_pred_x = obs_x + obs_vx * t * dt
        #         obs_pred_y = obs_y + obs_vy * t * dt
                
        #         constraints += [
        #             cp.norm(cp.hstack([x[t] - obs_pred_x, y[t] - obs_pred_y]), 2) >= 2.0
        #         ]

        # # 求解优化问题
        # prob = cp.Problem(objective, constraints)
        # prob.solve()

        # # 提取优化后的轨迹
        # trajectory_x = x.value
        # trajectory_y = y.value
        
        self.last_action = action





def action_factory(env: 'AbstractEnv', config: dict) -> ActionType:
    if config["type"] == "ContinuousAction":
        return ContinuousAction(env, **config)
