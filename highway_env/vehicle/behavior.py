import os
from typing import Tuple, Union
import pandas as pd

import numpy as np
import random
from highway_env.road.road import Road, Route, LaneIndex
from highway_env.types import Vector
from highway_env.vehicle.controller import ControlledVehicle, ControlledVehicle2
from highway_env import utils
from highway_env.vehicle.kinematics import Vehicle, Vehicle2
from highway_env.vehicle.desired_gap import desired_gap
from highway_env.road.objects import RoadObject
# from vehicles_model.vehicles_action import replace_acceleration, replace_steering, vehicles_steering,vehicle_1_acceleration,vehicle_1_steering,vehicle_2_acceleration,vehicle_2_steering,vehicle_3_acceleration,vehicle_3_steering,vehicle_4_acceleration,vehicle_4_steering,vehicle_5_acceleration,vehicle_5_steering,vehicle_6_acceleration,vehicle_6_steering,vehicle_7_acceleration,vehicle_7_steering,vehicle_8_acceleration,vehicle_8_steering,vehicle_9_acceleration,vehicle_9_steering,vehicle_10_acceleration,vehicle_10_steering,vehicle_11_acceleration,vehicle_11_steering,vehicle_12_acceleration,vehicle_12_steering,vehicle_13_acceleration,vehicle_13_steering
from vehicles_model.calculate_acc_steer import calculate_dynamics, VehicleState

import csv



class IDMVehicle(ControlledVehicle):
    """
    A vehicle using both a longitudinal and a lateral decision policies.

    - Longitudinal: the IDM model computes an acceleration given the preceding vehicle's distance and speed.
    - Lateral: the MOBIL model decides when to change lane by maximizing the acceleration of nearby vehicles.
    """

    # Longitudinal policy parameters
    ACC_MAX = 8.0  # [m/s2]
    """Maximum acceleration."""

    COMFORT_ACC_MAX = 3.0  # [m/s2]
    """Desired maximum acceleration."""

    COMFORT_ACC_MIN = -6.0  # [m/s2]
    """Desired maximum deceleration."""

    DISTANCE_WANTED = 2.0 + ControlledVehicle.LENGTH  # [m]
    """Desired jam distance to the front vehicle."""

    TIME_WANTED = 1.5  # [s]
    """Desired time gap to the front vehicle."""

    DELTA = 4.0  # []
    """Exponent of the velocity term."""
    LESS_DELTA_SPEED = 0

    # Lateral policy parameters
    POLITENESS = 0.  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]
    # print(POLITENESS)
    # level = random.uniform(1, 2)

    def __init__(self,
                 ID,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: int = None,
                 target_speed: float = None,
                 route: Route = None,
                 enable_lane_change: bool = True,
                 timer: float = None):
        super().__init__(ID, road, position, heading, speed, target_lane_index, target_speed, route)
        self.enable_lane_change = enable_lane_change
        self.timer = timer or (np.sum(self.position)*np.pi) % self.LANE_CHANGE_DELAY

    def randomize_behavior(self):
        pass

    def act(self, action: Union[dict, str] = None):
        """
        Execute an action.

        For now, no action is supported because the vehicle takes all decisions
        of acceleration and lane changes on its own, based on the IDM and MOBIL models.

        :param action: the action
        """
        if self.crashed:
            return
        action = {}
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)

        # Lateral: MOBIL
        self.follow_road()
        if self.enable_lane_change:
            self.change_lane_policy()
        action['steering'] = self.steering_control(self.target_lane_index)
        action['steering'] = np.clip(action['steering'], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        # action['steering'] = replace_steering

        # Longitudinal: IDM
        action['acceleration'] = self.acceleration(ego_vehicle=self,
                                                   front_vehicle=front_vehicle,
                                                   rear_vehicle=rear_vehicle)
        # action['acceleration'] = self.recover_from_stop(action['acceleration'])
        action['acceleration'] = np.clip(action['acceleration'], -self.ACC_MAX, self.ACC_MAX)
        # action['acceleration'] = replace_acceleration
        Vehicle.act(self, action)  # Skip ControlledVehicle.act(), or the command will be overriden.
        # print('acc1 =', action['acceleration'])


    def acceleration(self,
                     ego_vehicle: ControlledVehicle,
                     front_vehicle: Vehicle = None,
                     rear_vehicle: Vehicle = None) -> float:
        """
        Compute an acceleration command with the Intelligent Driver Model.

        The acceleration is chosen so as to:
        - reach a target speed;
        - maintain a minimum safety distance (and safety time) w.r.t the front vehicle.

        :param ego_vehicle: the vehicle whose desired acceleration is to be computed. It does not have to be an
                            IDM vehicle, which is why this method is a class method. This allows an IDM vehicle to
                            reason about other vehicles behaviors even though they may not IDMs.
        :param front_vehicle: the vehicle preceding the ego-vehicle
        :param rear_vehicle: the vehicle following the ego-vehicle
        :return: the acceleration command for the ego-vehicle [m/s2]
        """
        if not ego_vehicle or isinstance(ego_vehicle, RoadObject):
            return 0
        ego_target_speed = utils.not_zero(getattr(ego_vehicle, "target_speed", 0))
        acceleration = self.COMFORT_ACC_MAX * (
                1 - np.power(max(ego_vehicle.speed, 0) / ego_target_speed, self.DELTA))

        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle)
            acceleration -= self.COMFORT_ACC_MAX * \
                np.power(self.desired_gap(ego_vehicle, front_vehicle) / utils.not_zero(d), 2)
        return acceleration

    def desired_gap(self, ego_vehicle: Vehicle, front_vehicle: Vehicle = None, projected: bool = True) -> float:
        """
        Compute the desired distance between a vehicle and its leading vehicle.

        :param ego_vehicle: the vehicle being controlled
        :param front_vehicle: its leading vehicle
        :param projected: project 2D velocities in 1D space
        :return: the desired distance between the two [m]
        """
        d0 = self.DISTANCE_WANTED
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = np.dot(ego_vehicle.velocity - front_vehicle.velocity, ego_vehicle.direction) if projected \
            else ego_vehicle.speed - front_vehicle.speed
        d_star = d0 + ego_vehicle.speed * tau + ego_vehicle.speed * dv / (2 * np.sqrt(ab))
        return d_star

    def change_lane_policy(self) -> None:
        """
        Decide when to change lane.

        Based on:
        - frequency;
        - closeness of the target lane;
        - MOBIL model.
        """
        # If a lane change already ongoing
        if self.lane_index != self.target_lane_index:
            # If we are on correct route but bad lane: abort it if someone else is already changing into the same lane
            if self.lane_index[:2] == self.target_lane_index[:2]:
                for v in self.road.vehicles:
                    if v is not self \
                            and v.lane_index != self.target_lane_index \
                            and isinstance(v, ControlledVehicle) \
                            and v.target_lane_index == self.target_lane_index:
                        d = self.lane_distance_to(v)
                        d_star = self.desired_gap(self, v)
                        if 0 < d < d_star:
                            self.target_lane_index = self.lane_index
                            break
            return

        # else, at a given frequency,
        if not utils.do_every(self.LANE_CHANGE_DELAY, self.timer):
            return
        self.timer = 0

        # decide to make a lane change
        for lane_index in self.road.network.side_lanes(self.lane_index):
            # Is the candidate lane close enough?
            if not self.road.network.get_lane(lane_index).is_reachable_from(self.position):
                continue
            # Does the MOBIL model recommend a lane change?
            if self.mobil(lane_index):
                self.target_lane_index = lane_index

    def mobil(self, lane_index: LaneIndex) -> bool:
        """
        MOBIL lane change model: Minimizing Overall Braking Induced by a Lane change

            The vehicle should change lane only if:
            - after changing it (and/or following vehicles) can accelerate more;
            - it doesn't impose an unsafe braking on its new following vehicle.

        :param lane_index: the candidate lane for the change
        :return: whether the lane change should be performed
        """
        # Is the maneuver unsafe for the new following vehicle?
        new_preceding, new_following = self.road.neighbour_vehicles(self, lane_index)
        new_following_a = self.acceleration(ego_vehicle=new_following, front_vehicle=new_preceding)
        new_following_pred_a = self.acceleration(ego_vehicle=new_following, front_vehicle=self)
        if new_following_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return False

        # Do I have a planned route for a specific lane which is safe for me to access?
        old_preceding, old_following = self.road.neighbour_vehicles(self)
        self_pred_a = self.acceleration(ego_vehicle=self, front_vehicle=new_preceding)
        if self.route and self.route[0][2]:
            # Wrong direction
            if np.sign(lane_index[2] - self.target_lane_index[2]) != np.sign(self.route[0][2] - self.target_lane_index[2]):
                return False
            # Unsafe braking required
            elif self_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
                return False

        # Is there an acceleration advantage for me and/or my followers to change lane?
        else:
            self_a = self.acceleration(ego_vehicle=self, front_vehicle=old_preceding)
            old_following_a = self.acceleration(ego_vehicle=old_following, front_vehicle=self)
            old_following_pred_a = self.acceleration(ego_vehicle=old_following, front_vehicle=old_preceding)
            jerk = self_pred_a - self_a + self.POLITENESS * (new_following_pred_a - new_following_a
                                                             + old_following_pred_a - old_following_a)
            if jerk < self.LANE_CHANGE_MIN_ACC_GAIN:
                return False

        # All clear, let's go!
        return True

class IDMVehicle_big(ControlledVehicle2):
    # Longitudinal policy parameters
    DELTA = 4.0  # []
    LESS_DELTA_SPEED = 0
    POLITENESS = 0.  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]
    dt = 0.1


    def __init__(self,
                 ID,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: int = None,
                 target_speed: float = None,
                 route: Route = None,
                 enable_lane_change: bool = True,
                 timer: float = None):
        super().__init__(ID, road, position, heading, speed, target_lane_index, target_speed, route)
        self.enable_lane_change = enable_lane_change
        self.timer = timer or (np.sum(self.position)*np.pi) % self.LANE_CHANGE_DELAY
        self.aq = 0

    def randomize_behavior(self):
        pass

    def act(self, action: Union[dict, str] = None):

        if self.crashed:
            return
        action = {}
        self.follow_road()

        self.ACC_MAX = 1.5
        self.ACC_MAX_de = -5
        self.COMFORT_ACC_MAX = 3.0  # [m/s2]
        self.COMFORT_ACC_MIN = -6.0  # [m/s2]
        self.DISTANCE_WANTED = 5.0
        self.TIME_WANTED = 1.5  # [s]
        self.MAX_STEERING_ANGLE = np.pi/8

        # 在 trajectory_1_2.csv 文件中找到 vehicle-ID 为 vehicle_id_with_length_11 的行
        # with open('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/trajectory_1_2.csv', 'r', encoding='utf-8') as csvfile:
        with open('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/trajectory_1.csv', 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            vehicle_1_current_state = None
            for row in reader:
                if int(row['vehicle-ID']) == self.ID:
                    vehicle_1_vx = float(row['v_x'])
                    vehicle_1_vy = float(row['v_y'])
                    vehicle_1_heading = float(row['heading'])
                    vehicle_1_x = float(row['x'])
                    vehicle_1_y = float(row['y'])
                    vehicle_1_current_state = VehicleState(vehicle_1_vx, vehicle_1_vy, vehicle_1_heading, vehicle_1_x, vehicle_1_y)
                    break

        # 在 prediction_1.csv 文件中找到 vehicle-ID 为 vehicle_id_with_length_11 的行
        prediction_path = '/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/prediction_1.csv'
        with open(prediction_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            vehicle_1_next_state = None
            vehicle_1_steering = None
            vehicle_1_acceleration = None
            for row in reader:
                if int(row['ID']) == (self.ID - 1) :
                    vehicle_1_vx = float(row['v_x'])
                    vehicle_1_vy = float(row['v_y'])
                    vehicle_1_heading = float(row['heading'])
                    vehicle_1_x = float(row['x'])
                    vehicle_1_y = float(row['y'])
                    vehicle_1_next_state = VehicleState(vehicle_1_vx, vehicle_1_vy, vehicle_1_heading, vehicle_1_x, vehicle_1_y)
                    # vehicle_1_steering = float(row['steering'])
                    # vehicle_1_acceleration = float(row['acceleration'])
                    break

        vehicle_big_acceleration, vehicle_big_steering = calculate_dynamics(vehicle_1_current_state, vehicle_1_next_state,
                                                                        self.dt)
        action['steering'] = vehicle_big_steering
        # action['steering'] = 0 
        action['steering'] = np.clip(action['steering'], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

        # action['acceleration'] = vehicle_1_acceleration
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles_2(self)
        # Calculate the desired acceleration
        desired_acceleration_2 = self.acceleration(ego_vehicle=self,
                                                 front_vehicle=front_vehicle,
                                                 rear_vehicle=rear_vehicle)
        desired_acceleration = vehicle_big_acceleration
        if vehicle_big_acceleration >= desired_acceleration_2 :
            desired_acceleration = desired_acceleration_2

        next_speed = self.velocity[0] + desired_acceleration * self.dt
        # Adjust acceleration if next speed would be less than 0
        if next_speed < 0:
            desired_acceleration = -self.velocity[0] / self.dt
        if self.ID == 13:
            desired_acceleration = 0
        # desired_acceleration = desired_acceleration_2 
        # Clip the acceleration to ensure it is within the allowable range
        action['acceleration'] = max(self.ACC_MAX_de, min(desired_acceleration, self.ACC_MAX))
        # print('1', action)
        self.aq = self.aq + 1
        Vehicle.act(self, action)
    
    def acceleration(self,
                     ego_vehicle: ControlledVehicle2,
                     front_vehicle: Vehicle2 = None,
                     rear_vehicle: Vehicle = None) -> float:
        if not ego_vehicle or isinstance(ego_vehicle, RoadObject):
            return 0

        target_speed = 6
        ego_target_speed = utils.not_zero(getattr(ego_vehicle, "target_speed", target_speed))
        ego_target_speed = target_speed
        acceleration = self.COMFORT_ACC_MAX * (
                1 - np.power(max(ego_vehicle.speed, 0) / ego_target_speed, self.DELTA))

        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle) - (1/2)*self.LENGTH - (1/2)*front_vehicle.LENGTH
            desired_gap = self.desired_gap(ego_vehicle, front_vehicle)
            if d < desired_gap:
                acceleration -= self.COMFORT_ACC_MAX * \
                    np.power(desired_gap / utils.not_zero(d), 2)
            else:
                acceleration -= self.COMFORT_ACC_MAX * \
                    np.power(desired_gap / utils.not_zero(d), 4)
        # Predict the next speed
        next_speed = ego_vehicle.velocity[0] + acceleration * self.dt
        # Adjust acceleration if next speed would be less than 0
        if next_speed < 0:
            acceleration = -ego_vehicle.velocity[0] / self.dt
        return acceleration

    def desired_gap(self, ego_vehicle: Vehicle, front_vehicle: Vehicle = None, projected: bool = True) -> float:
        d0 = self.DISTANCE_WANTED + (1/2)*self.LENGTH + (1/2)*front_vehicle.LENGTH
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = np.dot(ego_vehicle.velocity - front_vehicle.velocity, ego_vehicle.direction) if projected \
            else ego_vehicle.speed - front_vehicle.speed
        d_star = d0 + 0.5*ego_vehicle.speed * tau + 0.5* ego_vehicle.speed * dv / (2 * np.sqrt(ab)) - (1/2)*self.LENGTH - (1/2)*front_vehicle.LENGTH
        return d_star

class IDMVehicle1(ControlledVehicle):
    # Longitudinal policy parameters
    DELTA = 4.0  # []
    LESS_DELTA_SPEED = 0
    POLITENESS = 0.  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]
    dt = 0.1


    def __init__(self,
                 ID,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: int = None,
                 target_speed: float = None,
                 route: Route = None,
                 enable_lane_change: bool = True,
                 timer: float = None):
        super().__init__(ID, road, position, heading, speed, target_lane_index, target_speed, route)
        self.enable_lane_change = enable_lane_change
        self.timer = timer or (np.sum(self.position)*np.pi) % self.LANE_CHANGE_DELAY
        self.aq = 0

    def randomize_behavior(self):
        pass

    def act(self, action: Union[dict, str] = None):

        if self.crashed:
            return
        action = {}
        self.follow_road()

        # with open('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/trajectory_1_2.csv', 'r', encoding='utf-8') as csvfile:
        with open('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/trajectory_1.csv', 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if int(row['vehicle-ID']) == self.ID:
                # if row['vehicle-ID'] == '1':
                    vehicle_1_vx = float(row['v_x'])
                    vehicle_1_vy = float(row['v_y'])
                    vehicle_1_heading = float(row['heading'])
                    vehicle_1_x = float(row['x'])
                    vehicle_1_y = float(row['y'])
                    vehicle_1_current_state = VehicleState(vehicle_1_vx, vehicle_1_vy, vehicle_1_heading,vehicle_1_x,vehicle_1_y)
                    self.char_value = float(row['label'])
        
        if self.char_value == 1:
            self.ACC_MAX = 4.0  # [m/s2]
            self.ACC_MAX_de = -18
            self.COMFORT_ACC_MAX = 6.0  # [m/s2]
            self.COMFORT_ACC_MIN = -8.0  # [m/s2]
            self.DISTANCE_WANTED = 2.0  # [m]
            self.TIME_WANTED = 1.0  # [s]
            self.MAX_STEERING_ANGLE = np.pi/5
        elif self.char_value == 2:
            self.ACC_MAX = 2.0
            self.ACC_MAX_de = -18
            self.COMFORT_ACC_MAX = 3.0  # [m/s2]
            self.COMFORT_ACC_MIN = -6.0  # [m/s2]
            self.DISTANCE_WANTED = 3.0  # [m]
            self.TIME_WANTED = 2.0  # [s]
            self.MAX_STEERING_ANGLE = np.pi/8

        with open('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/prediction_1.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if int(row['ID']) == self.ID - 1:
                    vehicle_1_vx = row['v_x']
                    vehicle_1_vx = float(vehicle_1_vx)
                    vehicle_1_vy = row['v_y']
                    vehicle_1_vy = float(vehicle_1_vy)
                    vehicle_1_heading = row['heading']
                    vehicle_1_heading = float(vehicle_1_heading)
                    vehicle_1_x = row['x']
                    vehicle_1_x = float(vehicle_1_x)
                    vehicle_1_y = row['y']
                    vehicle_1_y = float(vehicle_1_y)
                    vehicle_1_next_state = VehicleState(vehicle_1_vx, vehicle_1_vy, vehicle_1_heading,vehicle_1_x,vehicle_1_y)
                    # vehicle_1_steering = row['steering']
                    # vehicle_1_steering = float(vehicle_1_steering)
                    # vehicle_1_acceleration = row['acceleration']
                    # vehicle_1_acceleration = float(vehicle_1_acceleration)

        vehicle_1_acceleration, vehicle_1_steering = calculate_dynamics(vehicle_1_current_state, vehicle_1_next_state,
                                                                        self.dt)
        action['steering'] = vehicle_1_steering
        # action['steering'] = 0 
        action['steering'] = np.clip(action['steering'], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

        # action['acceleration'] = vehicle_1_acceleration
        if self.char_value == 1:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)
        elif self.char_value == 2:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles_2(self)
        # Calculate the desired acceleration
        desired_acceleration_2 = self.acceleration(ego_vehicle=self,
                                                 front_vehicle=front_vehicle,
                                                 rear_vehicle=rear_vehicle)
        
        desired_acceleration = vehicle_1_acceleration
        if vehicle_1_acceleration >= desired_acceleration_2 :
            desired_acceleration = desired_acceleration_2

        next_speed = self.velocity[0] + desired_acceleration * self.dt
        # Adjust acceleration if next speed would be less than 0
        if next_speed < 0:
            desired_acceleration = -self.velocity[0] / self.dt
        # desired_acceleration = desired_acceleration_2        
        # Clip the acceleration to ensure it is within the allowable range
        if self.ID == 13:
            desired_acceleration = 0
        action['acceleration'] = max(self.ACC_MAX_de, min(desired_acceleration, self.ACC_MAX))
        # print('1', action)
        self.aq = self.aq + 1
        Vehicle.act(self, action)
    
    def acceleration(self,
                     ego_vehicle: ControlledVehicle,
                     front_vehicle: Vehicle = None,
                     rear_vehicle: Vehicle = None) -> float:
        if not ego_vehicle or isinstance(ego_vehicle, RoadObject):
            return 0
        if self.char_value == 1:
            target_speed = 8
        elif self.char_value == 2:
            target_speed = 6
        ego_target_speed = utils.not_zero(getattr(ego_vehicle, "target_speed", target_speed))
        ego_target_speed = target_speed
        acceleration = self.COMFORT_ACC_MAX * (
                1 - np.power(max(ego_vehicle.speed, 0) / ego_target_speed, self.DELTA))

        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle) - (1/2)*self.LENGTH - (1/2)*front_vehicle.LENGTH
            desired_gap = self.desired_gap(ego_vehicle, front_vehicle)
            if d < desired_gap:
                acceleration -= self.COMFORT_ACC_MAX * \
                    np.power(desired_gap / utils.not_zero(d), 2)
            else:
                acceleration -= self.COMFORT_ACC_MAX * \
                    np.power(desired_gap / utils.not_zero(d), 4)
        
        # Predict the next speed
        next_speed = ego_vehicle.velocity[0] + acceleration * self.dt
        # Adjust acceleration if next speed would be less than 0
        if next_speed < 0:
            acceleration = -ego_vehicle.velocity[0] / self.dt
        return acceleration

    def desired_gap(self, ego_vehicle: Vehicle, front_vehicle: Vehicle = None, projected: bool = True) -> float:
        d0 = self.DISTANCE_WANTED + (1/2)*self.LENGTH + (1/2)*front_vehicle.LENGTH
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = np.dot(ego_vehicle.velocity - front_vehicle.velocity, ego_vehicle.direction) if projected \
            else ego_vehicle.speed - front_vehicle.speed
        d_star = d0 + 0.5*ego_vehicle.speed * tau + 0.5* ego_vehicle.speed * dv / (2 * np.sqrt(ab)) - (1/2)*self.LENGTH - (1/2)*front_vehicle.LENGTH
        return d_star

class IDMVehicle2(ControlledVehicle):
    # Longitudinal policy parameters
    DELTA = 4.0  # []
    LESS_DELTA_SPEED = 0
    POLITENESS = 0.  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]
    dt = 0.1


    def __init__(self,
                 ID,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: int = None,
                 target_speed: float = None,
                 route: Route = None,
                 enable_lane_change: bool = True,
                 timer: float = None):
        super().__init__(ID, road, position, heading, speed, target_lane_index, target_speed, route)
        self.enable_lane_change = enable_lane_change
        self.timer = timer or (np.sum(self.position)*np.pi) % self.LANE_CHANGE_DELAY
        self.aq = 0

    def randomize_behavior(self):
        pass

    def act(self, action: Union[dict, str] = None):

        if self.crashed:
            return
        action = {}
        self.follow_road()

        df_2 = pd.read_csv('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/vehicle_char.csv')
        vehicle_row = df_2[df_2['vehicle-ID'] == 2]
        self.char_value = vehicle_row['char'].values[0]

        if self.char_value == 1:
            self.ACC_MAX = 4.0  # [m/s2]
            self.ACC_MAX_de = -18
            self.COMFORT_ACC_MAX = 6.0  # [m/s2]
            self.COMFORT_ACC_MIN = -8.0  # [m/s2]
            self.DISTANCE_WANTED = 2.0   # [m]
            self.TIME_WANTED = 1.0  # [s]
            self.MAX_STEERING_ANGLE = np.pi/5
        elif self.char_value == 2:
            self.ACC_MAX =2.0
            self.ACC_MAX_de = -10
            self.COMFORT_ACC_MAX = 3.0  # [m/s2]
            self.COMFORT_ACC_MIN = -6.0  # [m/s2]
            self.DISTANCE_WANTED = 3.0  # [m]
            self.TIME_WANTED = 2.0  # [s]
            self.MAX_STEERING_ANGLE = np.pi/8

        # with open('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/trajectory_1_2.csv', 'r', encoding='utf-8') as csvfile:
        with open('/data/wangzm/merge/DJI_select/DJI_init/trajectory_1.csv', 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['vehicle-ID'] == '2':
                    vehicle_2_vx = row['v_x']
                    vehicle_2_vx = float(vehicle_2_vx)
                    vehicle_2_vy = row['v_y']
                    vehicle_2_vy = float(vehicle_2_vy)
                    vehicle_2_heading = row['heading']
                    vehicle_2_heading = float(vehicle_2_heading)
                    vehicle_2_x = row['x']
                    vehicle_2_x = float(vehicle_2_x)
                    vehicle_2_y = row['y']
                    vehicle_2_y = float(vehicle_2_y)
                    vehicle_2_current_state = VehicleState(vehicle_2_vx, vehicle_2_vy, vehicle_2_heading,vehicle_2_x,vehicle_2_y)

        with open('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/prediction_1.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['ID'] == '1':
                    vehicle_2_vx = row['v_x']
                    vehicle_2_vx = float(vehicle_2_vx)
                    vehicle_2_vy = row['v_y']
                    vehicle_2_vy = float(vehicle_2_vy)
                    vehicle_2_heading = row['heading']
                    vehicle_2_heading = float(vehicle_2_heading)
                    vehicle_2_x = row['x']
                    vehicle_2_x = float(vehicle_2_x)
                    vehicle_2_y = row['y']
                    vehicle_2_y = float(vehicle_2_y)
                    vehicle_2_next_state = VehicleState(vehicle_2_vx, vehicle_2_vy, vehicle_2_heading,vehicle_2_x,vehicle_2_y)
                    # vehicle_2_steering = row['steering']
                    # vehicle_2_steering = float(vehicle_2_steering)
                    # vehicle_2_acceleration = row['acceleration']
                    # vehicle_2_acceleration = float(vehicle_2_acceleration)

        vehicle_2_acceleration, vehicle_2_steering = calculate_dynamics(vehicle_2_current_state, vehicle_2_next_state,
                                                                        self.dt)
        action['steering'] = vehicle_2_steering
        # action['steering'] = 0
        action['steering'] = np.clip(action['steering'], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

        # action['acceleration'] = vehicle_2_acceleration
        if self.char_value == 1:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)
        elif self.char_value == 2:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles_2(self)
        # Calculate the desired acceleration
        desired_acceleration_2 = self.acceleration(ego_vehicle=self,
                                                 front_vehicle=front_vehicle,
                                                 rear_vehicle=rear_vehicle)
        desired_acceleration = vehicle_2_acceleration
        if vehicle_2_acceleration >= desired_acceleration_2 :
            desired_acceleration = desired_acceleration_2
        next_speed = self.velocity[0] + desired_acceleration * self.dt
        # Adjust acceleration if next speed would be less than 0
        if next_speed < 0:
            desired_acceleration = -self.velocity[0] / self.dt
        # desired_acceleration = desired_acceleration_2 
        # Clip the acceleration to ensure it is within the allowable range
        action['acceleration'] = max(self.ACC_MAX_de, min(desired_acceleration, self.ACC_MAX))
        # print('2', action)
        self.aq = self.aq + 1
        Vehicle.act(self, action)

    def acceleration(self,
                     ego_vehicle: ControlledVehicle,
                     front_vehicle: Vehicle = None,
                     rear_vehicle: Vehicle = None) -> float:
        if not ego_vehicle or isinstance(ego_vehicle, RoadObject):
            return 0
        if self.char_value == 1:
            target_speed = 8
        elif self.char_value == 2:
            target_speed = 6
        ego_target_speed = utils.not_zero(getattr(ego_vehicle, "target_speed", target_speed))
        ego_target_speed = target_speed
        acceleration = self.COMFORT_ACC_MAX * (
                1 - np.power(max(ego_vehicle.speed, 0) / ego_target_speed, self.DELTA))

        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle) - (1/2)*self.LENGTH - (1/2)*front_vehicle.LENGTH
            desired_gap = self.desired_gap(ego_vehicle, front_vehicle)
            if d < desired_gap:
                acceleration -= self.COMFORT_ACC_MAX * \
                    np.power(desired_gap / utils.not_zero(d), 2)
            else:
                acceleration -= self.COMFORT_ACC_MAX * \
                    np.power(desired_gap / utils.not_zero(d), 4)
        # Predict the next speed
        next_speed = ego_vehicle.velocity[0] + acceleration * self.dt
        # Adjust acceleration if next speed would be less than 0
        if next_speed < 0:
            acceleration = -ego_vehicle.velocity[0] / self.dt
        return acceleration

    def desired_gap(self, ego_vehicle: Vehicle, front_vehicle: Vehicle = None, projected: bool = True) -> float:
        d0 = self.DISTANCE_WANTED + (1/2)*self.LENGTH + (1/2)*front_vehicle.LENGTH
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = np.dot(ego_vehicle.velocity - front_vehicle.velocity, ego_vehicle.direction) if projected \
            else ego_vehicle.speed - front_vehicle.speed
        d_star = d0 + 0.1*ego_vehicle.speed * tau + 0.1* ego_vehicle.speed * dv / (2 * np.sqrt(ab)) - (1/2)*self.LENGTH - (1/2)*front_vehicle.LENGTH
        return d_star

class IDMVehicle3(ControlledVehicle):
    # Longitudinal policy parameters
    DELTA = 4.0  # []
    LESS_DELTA_SPEED = 0
    POLITENESS = 0.  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]
    dt = 0.1

    def __init__(self,
                 ID,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: int = None,
                 target_speed: float = None,
                 route: Route = None,
                 enable_lane_change: bool = True,
                 timer: float = None):
        super().__init__(ID, road, position, heading, speed, target_lane_index, target_speed, route)
        self.enable_lane_change = enable_lane_change
        self.timer = timer or (np.sum(self.position)*np.pi) % self.LANE_CHANGE_DELAY
        self.aq = 0

    def randomize_behavior(self):
        pass

    def act(self, action: Union[dict, str] = None):

        if self.crashed:
            return
        action = {}
        self.follow_road()

        df_3 = pd.read_csv('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/vehicle_char.csv')
        vehicle_row = df_3[df_3['vehicle-ID'] == 3]
        self.char_value = vehicle_row['char'].values[0]

        if self.char_value == 1:
            self.ACC_MAX = 4.0  # [m/s2]
            self.ACC_MAX_de = -18
            self.COMFORT_ACC_MAX = 6.0  # [m/s2]
            self.COMFORT_ACC_MIN = -8.0  # [m/s2]
            self.DISTANCE_WANTED = 2.0   # [m]
            self.TIME_WANTED = 1.0  # [s]
            self.MAX_STEERING_ANGLE = np.pi/5
        elif self.char_value == 2:
            self.ACC_MAX = 2.0
            self.ACC_MAX_de = -10
            self.COMFORT_ACC_MAX = 3.0  # [m/s2]
            self.COMFORT_ACC_MIN = -6.0  # [m/s2]
            self.DISTANCE_WANTED = 3.0  # [m]
            self.TIME_WANTED = 2.0  # [s]
            self.MAX_STEERING_ANGLE = np.pi/8

        # with open('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/trajectory_1_2.csv', 'r', encoding='utf-8') as csvfile:
        with open('/data/wangzm/merge/DJI_select/DJI_init/trajectory_1.csv', 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['vehicle-ID'] == '3':
                    vehicle_3_vx = row['v_x']
                    vehicle_3_vx = float(vehicle_3_vx)
                    vehicle_3_vy = row['v_y']
                    vehicle_3_vy = float(vehicle_3_vy)
                    vehicle_3_heading = row['heading']
                    vehicle_3_heading = float(vehicle_3_heading)
                    vehicle_3_x = row['x']
                    vehicle_3_x = float(vehicle_3_x)
                    vehicle_3_y = row['y']
                    vehicle_3_y = float(vehicle_3_y)
                    vehicle_3_current_state = VehicleState(vehicle_3_vx, vehicle_3_vy, vehicle_3_heading,vehicle_3_x,vehicle_3_y)

        with open('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/prediction_1.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['ID'] == '2':
                    vehicle_3_vx = row['v_x']
                    vehicle_3_vx = float(vehicle_3_vx)
                    vehicle_3_vy = row['v_y']
                    vehicle_3_vy = float(vehicle_3_vy)
                    vehicle_3_heading = row['heading']
                    vehicle_3_heading = float(vehicle_3_heading)
                    vehicle_3_x = row['x']
                    vehicle_3_x = float(vehicle_3_x)
                    vehicle_3_y = row['y']
                    vehicle_3_y = float(vehicle_3_y)
                    vehicle_3_next_state = VehicleState(vehicle_3_vx, vehicle_3_vy, vehicle_3_heading,vehicle_3_x,vehicle_3_y)
                    # vehicle_3_steering = row['steering']
                    # vehicle_3_steering = float(vehicle_3_steering)
                    # vehicle_3_acceleration = row['acceleration']
                    # vehicle_3_acceleration = float(vehicle_3_acceleration)

        vehicle_3_acceleration, vehicle_3_steering = calculate_dynamics(vehicle_3_current_state, vehicle_3_next_state,
                                                                        self.dt)

        action['steering'] = vehicle_3_steering
        # action['steering'] = 0
        action['steering'] = np.clip(action['steering'], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

        # action['acceleration'] = vehicle_3_acceleration
        if self.char_value == 1:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)
        elif self.char_value == 2:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles_2(self)
        # Calculate the desired acceleration
        desired_acceleration_2 = self.acceleration(ego_vehicle=self,
                                                 front_vehicle=front_vehicle,
                                                 rear_vehicle=rear_vehicle)
        desired_acceleration = vehicle_3_acceleration
        if vehicle_3_acceleration >= desired_acceleration_2 :
            desired_acceleration = desired_acceleration_2
        next_speed = self.velocity[0] + desired_acceleration * self.dt
        # Adjust acceleration if next speed would be less than 0
        if next_speed < 0:
            desired_acceleration = -self.velocity[0] / self.dt
        # desired_acceleration = desired_acceleration_2  
        # Clip the acceleration to ensure it is within the allowable range
        action['acceleration'] = max(self.ACC_MAX_de, min(desired_acceleration, self.ACC_MAX))
        # print('3', action)
        self.aq = self.aq + 1
        Vehicle.act(self, action)

    def acceleration(self,
                     ego_vehicle: ControlledVehicle,
                     front_vehicle: Vehicle = None,
                     rear_vehicle: Vehicle = None) -> float:
        if not ego_vehicle or isinstance(ego_vehicle, RoadObject):
            return 0
        if self.char_value == 1:
            target_speed = 8
        elif self.char_value == 2:
            target_speed = 6
        ego_target_speed = utils.not_zero(getattr(ego_vehicle, "target_speed", target_speed))
        ego_target_speed = target_speed
        acceleration = self.COMFORT_ACC_MAX * (
                1 - np.power(max(ego_vehicle.speed, 0) / ego_target_speed, self.DELTA))

        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle) - (1/2)*self.LENGTH - (1/2)*front_vehicle.LENGTH
            desired_gap = self.desired_gap(ego_vehicle, front_vehicle)
            if d < desired_gap:
                acceleration -= self.COMFORT_ACC_MAX * \
                    np.power(desired_gap / utils.not_zero(d), 2)
                acceleration -= 2
            else:
                acceleration -= self.COMFORT_ACC_MAX * \
                    np.power(desired_gap / utils.not_zero(d), 4)
        # Predict the next speed
        next_speed = ego_vehicle.velocity[0] + acceleration * self.dt
        # Adjust acceleration if next speed would be less than 0
        if next_speed < 0:
            acceleration = -ego_vehicle.velocity[0] / self.dt
        return acceleration

    def desired_gap(self, ego_vehicle: Vehicle, front_vehicle: Vehicle = None, projected: bool = True) -> float:
        d0 = self.DISTANCE_WANTED + (1/2)*self.LENGTH + (1/2)*front_vehicle.LENGTH
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = np.dot(ego_vehicle.velocity - front_vehicle.velocity, ego_vehicle.direction) if projected \
            else ego_vehicle.speed - front_vehicle.speed
        d_star = d0 + 0.1*ego_vehicle.speed * tau + 0.1* ego_vehicle.speed * dv / (2 * np.sqrt(ab)) - (1/2)*self.LENGTH - (1/2)*front_vehicle.LENGTH
        return d_star


class IDMVehicle4(ControlledVehicle):
    # Longitudinal policy parameters
    DELTA = 4.0  # []
    LESS_DELTA_SPEED = 0
    POLITENESS = 0.  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]
    dt = 0.1

    def __init__(self,
                 ID,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: int = None,
                 target_speed: float = None,
                 route: Route = None,
                 enable_lane_change: bool = True,
                 timer: float = None):
        super().__init__(ID, road, position, heading, speed, target_lane_index, target_speed, route)
        self.enable_lane_change = enable_lane_change
        self.timer = timer or (np.sum(self.position)*np.pi) % self.LANE_CHANGE_DELAY
        self.aq = 0

    def randomize_behavior(self):
        pass

    def act(self, action: Union[dict, str] = None):

        if self.crashed:
            return
        action = {}
        self.follow_road()

        df_4 = pd.read_csv('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/vehicle_char.csv')
        vehicle_row = df_4[df_4['vehicle-ID'] == 4]
        self.char_value = vehicle_row['char'].values[0]

        if self.char_value == 1:
            self.ACC_MAX = 4.0  # [m/s2]
            self.ACC_MAX_de = -18
            self.COMFORT_ACC_MAX = 6.0  # [m/s2]
            self.COMFORT_ACC_MIN = -8.0  # [m/s2]
            self.DISTANCE_WANTED = 2.0   # [m]
            self.TIME_WANTED = 1.0  # [s]
            self.MAX_STEERING_ANGLE = np.pi/5
        elif self.char_value == 2:
            self.ACC_MAX = 2.0
            self.ACC_MAX_de = -10
            self.COMFORT_ACC_MAX = 3.0  # [m/s2]
            self.COMFORT_ACC_MIN = -6.0  # [m/s2]
            self.DISTANCE_WANTED = 3.0   # [m]
            self.TIME_WANTED = 2.0  # [s]
            self.MAX_STEERING_ANGLE = np.pi/8

        # with open('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/trajectory_1_2.csv', 'r', encoding='utf-8') as csvfile:
        with open('/data/wangzm/merge/DJI_select/DJI_init/trajectory_1.csv', 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['vehicle-ID'] == '4':
                    vehicle_4_vx = row['v_x']
                    vehicle_4_vx = float(vehicle_4_vx)
                    vehicle_4_vy = row['v_y']
                    vehicle_4_vy = float(vehicle_4_vy)
                    vehicle_4_heading = row['heading']
                    vehicle_4_heading = float(vehicle_4_heading)
                    vehicle_4_x = row['x']
                    vehicle_4_x = float(vehicle_4_x)
                    vehicle_4_y = row['y']
                    vehicle_4_y = float(vehicle_4_y)
                    vehicle_4_current_state = VehicleState(vehicle_4_vx, vehicle_4_vy, vehicle_4_heading,vehicle_4_x,vehicle_4_y)

        with open('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/prediction_1.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['ID'] == '3':
                    vehicle_4_vx = row['v_x']
                    vehicle_4_vx = float(vehicle_4_vx)
                    vehicle_4_vy = row['v_y']
                    vehicle_4_vy = float(vehicle_4_vy)
                    vehicle_4_heading = row['heading']
                    vehicle_4_heading = float(vehicle_4_heading)
                    vehicle_4_x = row['x']
                    vehicle_4_x = float(vehicle_4_x)
                    vehicle_4_y = row['y']
                    vehicle_4_y = float(vehicle_4_y)
                    vehicle_4_next_state = VehicleState(vehicle_4_vx, vehicle_4_vy, vehicle_4_heading,vehicle_4_x,vehicle_4_y)
                    # vehicle_4_steering = row['steering']
                    # vehicle_4_steering = float(vehicle_4_steering)
                    # vehicle_4_acceleration = row['acceleration']
                    # vehicle_4_acceleration = float(vehicle_4_acceleration)

        vehicle_4_acceleration, vehicle_4_steering = calculate_dynamics(vehicle_4_current_state, vehicle_4_next_state,
                                                                        self.dt)

        action['steering'] = vehicle_4_steering
        # action['steering'] = 0
        action['steering'] = np.clip(action['steering'], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

        # action['acceleration'] = vehicle_4_acceleration
        if self.char_value == 1:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)
        elif self.char_value == 2:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles_2(self)
        # front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)
        # Calculate the desired acceleration
        desired_acceleration_2 = self.acceleration(ego_vehicle=self,
                                                 front_vehicle=front_vehicle,
                                                 rear_vehicle=rear_vehicle)
        desired_acceleration = vehicle_4_acceleration
        if vehicle_4_acceleration >= desired_acceleration_2 :
            desired_acceleration = desired_acceleration_2
        next_speed = self.velocity[0] + desired_acceleration * self.dt
        # Adjust acceleration if next speed would be less than 0
        if next_speed < 0:
            desired_acceleration = -self.velocity[0] / self.dt
        # desired_acceleration = desired_acceleration_2  
        # Clip the acceleration to ensure it is within the allowable range
        action['acceleration'] = max(self.ACC_MAX_de, min(desired_acceleration, self.ACC_MAX))
        # print('4',action)
        self.aq = self.aq + 1
        Vehicle.act(self, action)

    def acceleration(self,
                     ego_vehicle: ControlledVehicle,
                     front_vehicle: Vehicle = None,
                     rear_vehicle: Vehicle = None) -> float:
        if not ego_vehicle or isinstance(ego_vehicle, RoadObject):
            return 0
        if self.char_value == 1:
            target_speed = 8
        elif self.char_value == 2:
            target_speed = 6
        ego_target_speed = utils.not_zero(getattr(ego_vehicle, "target_speed", target_speed))
        ego_target_speed = target_speed
        acceleration = self.COMFORT_ACC_MAX * (
                1 - np.power(max(ego_vehicle.speed, 0) / ego_target_speed, self.DELTA))

        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle) - (1/2)*self.LENGTH - (1/2)*front_vehicle.LENGTH
            desired_gap = self.desired_gap(ego_vehicle, front_vehicle)
            if d < desired_gap:
                acceleration -= self.COMFORT_ACC_MAX * \
                    np.power(desired_gap / utils.not_zero(d), 2)
            else:
                acceleration -= self.COMFORT_ACC_MAX * \
                    np.power(desired_gap / utils.not_zero(d), 4)
        # Predict the next speed
        next_speed = ego_vehicle.velocity[0] + acceleration * self.dt
        # Adjust acceleration if next speed would be less than 0
        if next_speed < 0:
            acceleration = -ego_vehicle.velocity[0] / self.dt
        return acceleration

    def desired_gap(self, ego_vehicle: Vehicle, front_vehicle: Vehicle = None, projected: bool = True) -> float:
        d0 = self.DISTANCE_WANTED + (1/2)*self.LENGTH + (1/2)*front_vehicle.LENGTH
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = np.dot(ego_vehicle.velocity - front_vehicle.velocity, ego_vehicle.direction) if projected \
            else ego_vehicle.speed - front_vehicle.speed
        d_star = d0 + 0.1*ego_vehicle.speed * tau + 0.1* ego_vehicle.speed * dv / (2 * np.sqrt(ab)) - (1/2)*self.LENGTH - (1/2)*front_vehicle.LENGTH
        return d_star

class IDMVehicle5(ControlledVehicle):
    # Longitudinal policy parameters
    DELTA = 4.0  # []
    LESS_DELTA_SPEED = 0
    POLITENESS = 0.  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]
    dt = 0.1

    def __init__(self,
                 ID,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: int = None,
                 target_speed: float = None,
                 route: Route = None,
                 enable_lane_change: bool = True,
                 timer: float = None):
        super().__init__(ID, road, position, heading, speed, target_lane_index, target_speed, route)
        self.enable_lane_change = enable_lane_change
        self.timer = timer or (np.sum(self.position)*np.pi) % self.LANE_CHANGE_DELAY
        self.aq = 0

    def randomize_behavior(self):
        pass

    def act(self, action: Union[dict, str] = None):

        if self.crashed:
            return
        action = {}
        self.follow_road()

        df_5 = pd.read_csv('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/vehicle_char.csv')
        vehicle_row = df_5[df_5['vehicle-ID'] == 5]
        self.char_value = vehicle_row['char'].values[0]

        if self.char_value == 1:
            self.ACC_MAX = 4.0  # [m/s2]
            self.ACC_MAX_de = -18
            self.COMFORT_ACC_MAX = 6.0  # [m/s2]
            self.COMFORT_ACC_MIN = -8.0  # [m/s2]
            self.DISTANCE_WANTED = 2.0   # [m]
            self.TIME_WANTED = 0.5  # [s]
            self.MAX_STEERING_ANGLE = np.pi/5
        elif self.char_value == 2:
            self.ACC_MAX = 2.0
            self.ACC_MAX_de = -10
            self.COMFORT_ACC_MAX = 3.0  # [m/s2]
            self.COMFORT_ACC_MIN = -6.0  # [m/s2]
            self.DISTANCE_WANTED = 3.0   # [m]
            self.TIME_WANTED = 2.0  # [s]
            self.MAX_STEERING_ANGLE = np.pi/8

        # with open('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/trajectory_1_2.csv', 'r', encoding='utf-8') as csvfile:
        with open('/data/wangzm/merge/DJI_select/DJI_init/trajectory_1.csv', 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['vehicle-ID'] == '5':
                    vehicle_5_vx = row['v_x']
                    vehicle_5_vx = float(vehicle_5_vx)
                    vehicle_5_vy = row['v_y']
                    vehicle_5_vy = float(vehicle_5_vy)
                    vehicle_5_heading = row['heading']
                    vehicle_5_heading = float(vehicle_5_heading)
                    vehicle_5_x = row['x']
                    vehicle_5_x = float(vehicle_5_x)
                    vehicle_5_y = row['y']
                    vehicle_5_y = float(vehicle_5_y)
                    vehicle_5_current_state = VehicleState(vehicle_5_vx, vehicle_5_vy, vehicle_5_heading,vehicle_5_x,vehicle_5_y)

        with open('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/prediction_1.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['ID'] == '4':
                    vehicle_5_vx = row['v_x']
                    vehicle_5_vx = float(vehicle_5_vx)
                    vehicle_5_vy = row['v_y']
                    vehicle_5_vy = float(vehicle_5_vy)
                    vehicle_5_heading = row['heading']
                    vehicle_5_heading = float(vehicle_5_heading)
                    vehicle_5_x = row['x']
                    vehicle_5_x = float(vehicle_5_x)
                    vehicle_5_y = row['y']
                    vehicle_5_y = float(vehicle_5_y)
                    vehicle_5_next_state = VehicleState(vehicle_5_vx, vehicle_5_vy, vehicle_5_heading,vehicle_5_x,vehicle_5_y)
                    # vehicle_5_steering = row['steering']
                    # vehicle_5_steering = float(vehicle_5_steering)
                    # vehicle_5_acceleration = row['acceleration']
                    # vehicle_5_acceleration = float(vehicle_5_acceleration)

        vehicle_5_acceleration, vehicle_5_steering = calculate_dynamics(vehicle_5_current_state, vehicle_5_next_state,
                                                                        self.dt)
        action['steering'] = vehicle_5_steering
        # action['steering'] = 0
        action['steering'] = np.clip(action['steering'], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

        # action['acceleration'] = vehicle_5_acceleration
        if self.char_value == 1:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)
        elif self.char_value == 2:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles_2(self)
        # Calculate the desired acceleration
        desired_acceleration_2 = self.acceleration(ego_vehicle=self,
                                                 front_vehicle=front_vehicle,
                                                 rear_vehicle=rear_vehicle)
        desired_acceleration = vehicle_5_acceleration
        if vehicle_5_acceleration >= desired_acceleration_2 :
            desired_acceleration = desired_acceleration_2
        next_speed = self.velocity[0] + desired_acceleration * self.dt
        # Adjust acceleration if next speed would be less than 0
        if next_speed < 0:
            desired_acceleration = -self.velocity[0] / self.dt
        # desired_acceleration = desired_acceleration_2 
        # Clip the acceleration to ensure it is within the allowable range
        action['acceleration'] = max(self.ACC_MAX_de, min(desired_acceleration, self.ACC_MAX))
        # print('5',action)
        self.aq = self.aq + 1
        Vehicle.act(self, action)

    def acceleration(self,
                     ego_vehicle: ControlledVehicle,
                     front_vehicle: Vehicle = None,
                     rear_vehicle: Vehicle = None) -> float:
        if not ego_vehicle or isinstance(ego_vehicle, RoadObject):
            return 0
        if self.char_value == 1:
            target_speed = 8
        elif self.char_value == 2:
            target_speed = 5
        ego_target_speed = utils.not_zero(getattr(ego_vehicle, "target_speed", target_speed))
        ego_target_speed = target_speed
        acceleration = self.COMFORT_ACC_MAX * (
                1 - np.power(max(ego_vehicle.speed, 0) / ego_target_speed, self.DELTA))

        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle) - (1/2)*self.LENGTH - (1/2)*front_vehicle.LENGTH
            desired_gap = self.desired_gap(ego_vehicle, front_vehicle)
            if d < desired_gap:
                acceleration -= self.COMFORT_ACC_MAX * \
                    np.power(desired_gap / utils.not_zero(d), 2)
            else:
                acceleration -= self.COMFORT_ACC_MAX * \
                    np.power(desired_gap / utils.not_zero(d), 4)
        # Predict the next speed
        next_speed = ego_vehicle.velocity[0] + acceleration * self.dt
        # Adjust acceleration if next speed would be less than 0
        if next_speed < 0:
            acceleration = -ego_vehicle.velocity[0] / self.dt
        return acceleration

    def desired_gap(self, ego_vehicle: Vehicle, front_vehicle: Vehicle = None, projected: bool = True) -> float:
        d0 = self.DISTANCE_WANTED + (1/2)*self.LENGTH + (1/2)*front_vehicle.LENGTH
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = np.dot(ego_vehicle.velocity - front_vehicle.velocity, ego_vehicle.direction) if projected \
            else ego_vehicle.speed - front_vehicle.speed
        d_star = d0 + 0.5*ego_vehicle.speed * tau + 0.5* ego_vehicle.speed * dv / (2 * np.sqrt(ab)) - (1/2)*self.LENGTH - (1/2)*front_vehicle.LENGTH
        return d_star

class IDMVehicle6(ControlledVehicle):
    # Longitudinal policy parameters
    DELTA = 4.0  # []
    LESS_DELTA_SPEED = 0
    POLITENESS = 0.  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]
    dt = 0.1

    def __init__(self,
                 ID,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: int = None,
                 target_speed: float = None,
                 route: Route = None,
                 enable_lane_change: bool = True,
                 timer: float = None):
        super().__init__(ID, road, position, heading, speed, target_lane_index, target_speed, route)
        self.enable_lane_change = enable_lane_change
        self.timer = timer or (np.sum(self.position)*np.pi) % self.LANE_CHANGE_DELAY
        self.aq = 0

    def randomize_behavior(self):
        pass

    def act(self, action: Union[dict, str] = None):

        if self.crashed:
            return
        action = {}
        self.follow_road()

        df_6 = pd.read_csv('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/vehicle_char.csv')
        vehicle_row = df_6[df_6['vehicle-ID'] == 6]
        self.char_value = vehicle_row['char'].values[0]

        if self.char_value == 1:
            self.ACC_MAX = 4.0  # [m/s2]
            self.ACC_MAX_de = -18
            self.COMFORT_ACC_MAX = 6.0  # [m/s2]
            self.COMFORT_ACC_MIN = -8.0  # [m/s2]
            self.DISTANCE_WANTED = 2.0   # [m]
            self.TIME_WANTED = 1.0  # [s]
            self.MAX_STEERING_ANGLE = np.pi/5
        elif self.char_value == 2:
            self.ACC_MAX = 2.0
            self.ACC_MAX_de = -10
            self.COMFORT_ACC_MAX = 3.0  # [m/s2]
            self.COMFORT_ACC_MIN = -6.0  # [m/s2]
            self.DISTANCE_WANTED = 3.0  # [m]
            self.TIME_WANTED = 2.0  # [s]
            self.MAX_STEERING_ANGLE = np.pi/8

        # with open('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/trajectory_1_2.csv', 'r', encoding='utf-8') as csvfile:
        with open('/data/wangzm/merge/DJI_select/DJI_init/trajectory_1.csv', 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['vehicle-ID'] == '6':
                    vehicle_6_vx = row['v_x']
                    vehicle_6_vx = float(vehicle_6_vx)
                    vehicle_6_vy = row['v_y']
                    vehicle_6_vy = float(vehicle_6_vy)
                    vehicle_6_heading = row['heading']
                    vehicle_6_heading = float(vehicle_6_heading)
                    vehicle_6_x = row['x']
                    vehicle_6_x = float(vehicle_6_x)
                    vehicle_6_y = row['y']
                    vehicle_6_y = float(vehicle_6_y)
                    vehicle_6_current_state = VehicleState(vehicle_6_vx, vehicle_6_vy, vehicle_6_heading,vehicle_6_x,vehicle_6_y)

        with open('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/prediction_1.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['ID'] == '5':
                    vehicle_6_vx = row['v_x']
                    vehicle_6_vx = float(vehicle_6_vx)
                    vehicle_6_vy = row['v_y']
                    vehicle_6_vy = float(vehicle_6_vy)
                    vehicle_6_heading = row['heading']
                    vehicle_6_heading = float(vehicle_6_heading)
                    vehicle_6_x = row['x']
                    vehicle_6_x = float(vehicle_6_x)
                    vehicle_6_y = row['y']
                    vehicle_6_y = float(vehicle_6_y)
                    vehicle_6_next_state = VehicleState(vehicle_6_vx, vehicle_6_vy, vehicle_6_heading,vehicle_6_x,vehicle_6_y)
                    # vehicle_6_steering = row['steering']
                    # vehicle_6_steering = float(vehicle_6_steering)
                    # vehicle_6_acceleration = row['acceleration']
                    # vehicle_6_acceleration = float(vehicle_6_acceleration)

        vehicle_6_acceleration, vehicle_6_steering = calculate_dynamics(vehicle_6_current_state, vehicle_6_next_state,
                                                                        self.dt)

        action['steering'] = vehicle_6_steering
        # action['steering'] = 0
        action['steering'] = np.clip(action['steering'], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

        # action['acceleration'] = vehicle_6_acceleration
        if self.char_value == 1:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)
        elif self.char_value == 2:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles_2(self)
        # Calculate the desired acceleration
        desired_acceleration_2 = self.acceleration(ego_vehicle=self,
                                                 front_vehicle=front_vehicle,
                                                 rear_vehicle=rear_vehicle)
        desired_acceleration = vehicle_6_acceleration
        if vehicle_6_acceleration >= desired_acceleration_2 :
            desired_acceleration = desired_acceleration_2
        next_speed = self.velocity[0] + desired_acceleration * self.dt
        # Adjust acceleration if next speed would be less than 0
        if next_speed < 0:
            desired_acceleration = -self.velocity[0] / self.dt
        # desired_acceleration = desired_acceleration_2 
        # Clip the acceleration to ensure it is within the allowable range
        action['acceleration'] = max(self.ACC_MAX_de, min(desired_acceleration, self.ACC_MAX))
        # print('6',action)
        self.aq = self.aq + 1
        Vehicle.act(self, action)

    def acceleration(self,
                     ego_vehicle: ControlledVehicle,
                     front_vehicle: Vehicle = None,
                     rear_vehicle: Vehicle = None) -> float:
        if not ego_vehicle or isinstance(ego_vehicle, RoadObject):
            return 0
        if self.char_value == 1:
            target_speed = 8
        elif self.char_value == 2:
            target_speed = 6
        ego_target_speed = utils.not_zero(getattr(ego_vehicle, "target_speed", target_speed))
        ego_target_speed = target_speed
        acceleration = self.COMFORT_ACC_MAX * (
                1 - np.power(max(ego_vehicle.speed, 0) / ego_target_speed, self.DELTA))

        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle) - (1/2)*self.LENGTH - (1/2)*front_vehicle.LENGTH
            desired_gap = self.desired_gap(ego_vehicle, front_vehicle)
            if d < desired_gap:
                acceleration -= self.COMFORT_ACC_MAX * \
                    np.power(desired_gap / utils.not_zero(d), 2)
            else:
                acceleration -= self.COMFORT_ACC_MAX * \
                    np.power(desired_gap / utils.not_zero(d), 4)
        # Predict the next speed
        next_speed = ego_vehicle.velocity[0] + acceleration * self.dt
        # Adjust acceleration if next speed would be less than 0
        if next_speed < 0:
            acceleration = -ego_vehicle.velocity[0] / self.dt
        return acceleration

    def desired_gap(self, ego_vehicle: Vehicle, front_vehicle: Vehicle = None, projected: bool = True) -> float:
        d0 = self.DISTANCE_WANTED + (1/2)*self.LENGTH + (1/2)*front_vehicle.LENGTH
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = np.dot(ego_vehicle.velocity - front_vehicle.velocity, ego_vehicle.direction) if projected \
            else ego_vehicle.speed - front_vehicle.speed
        d_star = d0 + 0.5*ego_vehicle.speed * tau + 0.5* ego_vehicle.speed * dv / (2 * np.sqrt(ab)) - (1/2)*self.LENGTH - (1/2)*front_vehicle.LENGTH
        return d_star

class IDMVehicle7(ControlledVehicle):
    # Longitudinal policy parameters
    DELTA = 4.0  # []
    LESS_DELTA_SPEED = 0
    POLITENESS = 0.  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]
    dt = 0.1

    def __init__(self,
                 ID,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: int = None,
                 target_speed: float = None,
                 route: Route = None,
                 enable_lane_change: bool = True,
                 timer: float = None):
        super().__init__(ID, road, position, heading, speed, target_lane_index, target_speed, route)
        self.enable_lane_change = enable_lane_change
        self.timer = timer or (np.sum(self.position)*np.pi) % self.LANE_CHANGE_DELAY
        self.aq = 0

    def randomize_behavior(self):
        pass

    def act(self, action: Union[dict, str] = None):

        if self.crashed:
            return
        action = {}
        self.follow_road()

        df_7 = pd.read_csv('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/vehicle_char.csv')
        vehicle_row = df_7[df_7['vehicle-ID'] == 7]
        self.char_value = vehicle_row['char'].values[0]

        if self.char_value == 1:
            self.ACC_MAX = 4.0  # [m/s2]
            self.ACC_MAX_de = -18
            self.COMFORT_ACC_MAX = 6.0  # [m/s2]
            self.COMFORT_ACC_MIN = -8.0  # [m/s2]
            self.DISTANCE_WANTED = 2.0  # [m]
            self.TIME_WANTED = 1.0  # [s]
            self.MAX_STEERING_ANGLE = np.pi/5
        elif self.char_value == 2:
            self.ACC_MAX = 2.0
            self.ACC_MAX_de = -10
            self.COMFORT_ACC_MAX = 3.0  # [m/s2]
            self.COMFORT_ACC_MIN = -6.0  # [m/s2]
            self.DISTANCE_WANTED = 3.0   # [m]
            self.TIME_WANTED = 2.0  # [s]
            self.MAX_STEERING_ANGLE = np.pi/8

        # with open('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/trajectory_1_2.csv', 'r', encoding='utf-8') as csvfile:
        with open('/data/wangzm/merge/DJI_select/DJI_init/trajectory_1.csv', 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['vehicle-ID'] == '7':
                    vehicle_7_vx = row['v_x']
                    vehicle_7_vx = float(vehicle_7_vx)
                    vehicle_7_vy = row['v_y']
                    vehicle_7_vy = float(vehicle_7_vy)
                    vehicle_7_heading = row['heading']
                    vehicle_7_heading = float(vehicle_7_heading)
                    vehicle_7_x = row['x']
                    vehicle_7_x = float(vehicle_7_x)
                    vehicle_7_y = row['y']
                    vehicle_7_y = float(vehicle_7_y)
                    vehicle_7_current_state = VehicleState(vehicle_7_vx, vehicle_7_vy, vehicle_7_heading,vehicle_7_x,vehicle_7_y)

        with open('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/prediction_1.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['ID'] == '6':
                    vehicle_7_vx = row['v_x']
                    vehicle_7_vx = float(vehicle_7_vx)
                    vehicle_7_vy = row['v_y']
                    vehicle_7_vy = float(vehicle_7_vy)
                    vehicle_7_heading = row['heading']
                    vehicle_7_heading = float(vehicle_7_heading)
                    vehicle_7_x = row['x']
                    vehicle_7_x = float(vehicle_7_x)
                    vehicle_7_y = row['y']
                    vehicle_7_y = float(vehicle_7_y)
                    vehicle_7_next_state = VehicleState(vehicle_7_vx, vehicle_7_vy, vehicle_7_heading,vehicle_7_x,vehicle_7_y)
                    # vehicle_7_steering = row['steering']
                    # vehicle_7_steering = float(vehicle_7_steering)
                    # vehicle_7_acceleration = row['acceleration']
                    # vehicle_7_acceleration = float(vehicle_7_acceleration)

        vehicle_7_acceleration, vehicle_7_steering = calculate_dynamics(vehicle_7_current_state, vehicle_7_next_state,
                                                                        self.dt)
        action['steering'] = vehicle_7_steering
        # action['steering'] = 0
        action['steering'] = np.clip(action['steering'], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

        # action['acceleration'] = vehicle_7_acceleration
        if self.char_value == 1:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)
        elif self.char_value == 2:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles_2(self)
        # Calculate the desired acceleration
        desired_acceleration_2 = self.acceleration(ego_vehicle=self,
                                                 front_vehicle=front_vehicle,
                                                 rear_vehicle=rear_vehicle)
        desired_acceleration = vehicle_7_acceleration
        if vehicle_7_acceleration >= desired_acceleration_2 :
            desired_acceleration = desired_acceleration_2
        next_speed = self.velocity[0] + desired_acceleration * self.dt
        # Adjust acceleration if next speed would be less than 0
        if next_speed < 0:
            desired_acceleration = -self.velocity[0] / self.dt
        # desired_acceleration = desired_acceleration_2  
        # Clip the acceleration to ensure it is within the allowable range
        action['acceleration'] = max(self.ACC_MAX_de, min(desired_acceleration, self.ACC_MAX))
        # print('7', action)
        self.aq = self.aq + 1
        Vehicle.act(self, action)

    def acceleration(self,
                     ego_vehicle: ControlledVehicle,
                     front_vehicle: Vehicle = None,
                     rear_vehicle: Vehicle = None) -> float:
        if not ego_vehicle or isinstance(ego_vehicle, RoadObject):
            return 0
        if self.char_value == 1:
            target_speed = 8
        elif self.char_value == 2:
            target_speed = 6
        ego_target_speed = utils.not_zero(getattr(ego_vehicle, "target_speed", target_speed))
        ego_target_speed = target_speed
        acceleration = self.COMFORT_ACC_MAX * (
                1 - np.power(max(ego_vehicle.speed, 0) / ego_target_speed, self.DELTA))

        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle) - (1/2)*self.LENGTH - (1/2)*front_vehicle.LENGTH
            desired_gap = self.desired_gap(ego_vehicle, front_vehicle)
            if d < desired_gap:
                acceleration -= self.COMFORT_ACC_MAX * \
                    np.power(desired_gap / utils.not_zero(d), 2)
            else:
                acceleration -= self.COMFORT_ACC_MAX * \
                    np.power(desired_gap / utils.not_zero(d), 4)
        # Predict the next speed
        next_speed = ego_vehicle.velocity[0] + acceleration * self.dt
        # Adjust acceleration if next speed would be less than 0
        if next_speed < 0:
            acceleration = -ego_vehicle.velocity[0] / self.dt
        return acceleration

    def desired_gap(self, ego_vehicle: Vehicle, front_vehicle: Vehicle = None, projected: bool = True) -> float:
        d0 = self.DISTANCE_WANTED + (1/2)*self.LENGTH + (1/2)*front_vehicle.LENGTH
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = np.dot(ego_vehicle.velocity - front_vehicle.velocity, ego_vehicle.direction) if projected \
            else ego_vehicle.speed - front_vehicle.speed
        d_star = d0 + 0.5*ego_vehicle.speed * tau + 0.5* ego_vehicle.speed * dv / (2 * np.sqrt(ab)) - (1/2)*self.LENGTH - (1/2)*front_vehicle.LENGTH
        return d_star

class IDMVehicle8(ControlledVehicle):
    # Longitudinal policy parameters
    DELTA = 4.0  # []
    LESS_DELTA_SPEED = 0
    POLITENESS = 0.  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]
    dt = 0.1

    def __init__(self,
                 ID,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: int = None,
                 target_speed: float = None,
                 route: Route = None,
                 enable_lane_change: bool = True,
                 timer: float = None):
        super().__init__(ID, road, position, heading, speed, target_lane_index, target_speed, route)
        self.enable_lane_change = enable_lane_change
        self.timer = timer or (np.sum(self.position)*np.pi) % self.LANE_CHANGE_DELAY
        self.aq = 0

    def randomize_behavior(self):
        pass

    def act(self, action: Union[dict, str] = None):

        if self.crashed:
            return
        action = {}
        self.follow_road()

        df_8 = pd.read_csv('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/vehicle_char.csv')
        vehicle_row = df_8[df_8['vehicle-ID'] == 8]
        self.char_value = vehicle_row['char'].values[0]

        if self.char_value == 1:
            self.ACC_MAX = 4.0  # [m/s2]
            self.ACC_MAX_de = -18
            self.COMFORT_ACC_MAX = 6.0  # [m/s2]
            self.COMFORT_ACC_MIN = -8.0  # [m/s2]
            self.DISTANCE_WANTED = 2.0   # [m]
            self.TIME_WANTED = 1.0  # [s]
            self.MAX_STEERING_ANGLE = np.pi/5
        elif self.char_value == 2:
            self.ACC_MAX = 2.0
            self.ACC_MAX_de = -10
            self.COMFORT_ACC_MAX = 3.0  # [m/s2]
            self.COMFORT_ACC_MIN = -6.0  # [m/s2]
            self.DISTANCE_WANTED = 3.0   # [m]
            self.TIME_WANTED = 2.0  # [s]
            self.MAX_STEERING_ANGLE = np.pi/8

        # with open('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/trajectory_1_2.csv', 'r', encoding='utf-8') as csvfile:
        with open('/data/wangzm/merge/DJI_select/DJI_init/trajectory_1.csv', 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['vehicle-ID'] == '8':
                    vehicle_8_vx = row['v_x']                                  
                    vehicle_8_vx = float(vehicle_8_vx)
                    vehicle_8_vy = row['v_y']
                    vehicle_8_vy = float(vehicle_8_vy)
                    vehicle_8_heading = row['heading']
                    vehicle_8_heading = float(vehicle_8_heading)
                    vehicle_8_x = row['x']
                    vehicle_8_x = float(vehicle_8_x)
                    vehicle_8_y = row['y']
                    vehicle_8_y = float(vehicle_8_y)
                    vehicle_8_current_state = VehicleState(vehicle_8_vx, vehicle_8_vy, vehicle_8_heading,vehicle_8_x,vehicle_8_y)

        with open('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/prediction_1.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['ID'] == '7':
                    vehicle_8_vx = row['v_x']
                    vehicle_8_vx = float(vehicle_8_vx)
                    vehicle_8_vy = row['v_y']
                    vehicle_8_vy = float(vehicle_8_vy)
                    vehicle_8_heading = row['heading']
                    vehicle_8_heading = float(vehicle_8_heading)
                    vehicle_8_x = row['x']
                    vehicle_8_x = float(vehicle_8_x)
                    vehicle_8_y = row['y']
                    vehicle_8_y = float(vehicle_8_y)
                    vehicle_8_next_state = VehicleState(vehicle_8_vx, vehicle_8_vy, vehicle_8_heading,vehicle_8_x,vehicle_8_y)
                    # vehicle_8_steering = row['steering']
                    # vehicle_8_steering = float(vehicle_8_steering)
                    # vehicle_8_acceleration = row['acceleration']
                    # vehicle_8_acceleration = float(vehicle_8_acceleration)

        vehicle_8_acceleration, vehicle_8_steering = calculate_dynamics(vehicle_8_current_state, vehicle_8_next_state,
                                                                        self.dt)

        action['steering'] = vehicle_8_steering
        # action['steering'] = 0
        action['steering'] = np.clip(action['steering'], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

        # action['acceleration'] = vehicle_8_acceleration
        if self.char_value == 1:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)
        elif self.char_value == 2:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles_2(self)
        # Calculate the desired acceleration
        desired_acceleration_2 = self.acceleration(ego_vehicle=self,
                                                 front_vehicle=front_vehicle,
                                                 rear_vehicle=rear_vehicle)
        desired_acceleration = vehicle_8_acceleration
        if vehicle_8_acceleration >= desired_acceleration_2 :
            desired_acceleration = desired_acceleration_2
        next_speed = self.velocity[0] + desired_acceleration * self.dt
        # Adjust acceleration if next speed would be less than 0
        if next_speed < 0:
            desired_acceleration = -self.velocity[0] / self.dt
        # desired_acceleration = desired_acceleration_2  
        # Clip the acceleration to ensure it is within the allowable range
        action['acceleration'] = max(self.ACC_MAX_de, min(desired_acceleration, self.ACC_MAX))
        # print('8',action)
        self.aq = self.aq + 1
        Vehicle.act(self, action)

    def acceleration(self,
                     ego_vehicle: ControlledVehicle,
                     front_vehicle: Vehicle = None,
                     rear_vehicle: Vehicle = None) -> float:
        if not ego_vehicle or isinstance(ego_vehicle, RoadObject):
            return 0
        if self.char_value == 1:
            target_speed = 8
        elif self.char_value == 2:
            target_speed = 6
        ego_target_speed = utils.not_zero(getattr(ego_vehicle, "target_speed", target_speed))
        ego_target_speed = target_speed
        acceleration = self.COMFORT_ACC_MAX * (
                1 - np.power(max(ego_vehicle.speed, 0) / ego_target_speed, self.DELTA))

        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle) - (1/2)*self.LENGTH - (1/2)*front_vehicle.LENGTH
            desired_gap = self.desired_gap(ego_vehicle, front_vehicle)
            if d < desired_gap:
                acceleration -= self.COMFORT_ACC_MAX * \
                    np.power(desired_gap / utils.not_zero(d), 2)
                acceleration -= 2
            else:
                acceleration -= self.COMFORT_ACC_MAX * \
                    np.power(desired_gap / utils.not_zero(d), 4)
        # Predict the next speed
        next_speed = ego_vehicle.velocity[0] + acceleration * self.dt
        # Adjust acceleration if next speed would be less than 0
        if next_speed < 0:
            acceleration = -ego_vehicle.velocity[0] / self.dt
        return acceleration

    def desired_gap(self, ego_vehicle: Vehicle, front_vehicle: Vehicle = None, projected: bool = True) -> float:
        d0 = self.DISTANCE_WANTED + (1/2)*self.LENGTH + (1/2)*front_vehicle.LENGTH
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = np.dot(ego_vehicle.velocity - front_vehicle.velocity, ego_vehicle.direction) if projected \
            else ego_vehicle.speed - front_vehicle.speed
        d_star = d0 + 0.5*ego_vehicle.speed * tau + 0.5* ego_vehicle.speed * dv / (2 * np.sqrt(ab)) - (1/2)*self.LENGTH - (1/2)*front_vehicle.LENGTH
        return d_star

class IDMVehicle9(ControlledVehicle):
    # Longitudinal policy parameters
    DELTA = 4.0  # []
    LESS_DELTA_SPEED = 0
    POLITENESS = 0.  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]
    dt = 0.1

    def __init__(self,
                 ID,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: int = None,
                 target_speed: float = None,
                 route: Route = None,
                 enable_lane_change: bool = True,
                 timer: float = None):
        super().__init__(ID, road, position, heading, speed, target_lane_index, target_speed, route)
        self.enable_lane_change = enable_lane_change
        self.timer = timer or (np.sum(self.position)*np.pi) % self.LANE_CHANGE_DELAY
        self.aq = 0

    def randomize_behavior(self):
        pass

    def act(self, action: Union[dict, str] = None):

        if self.crashed:
            return
        action = {}
        self.follow_road()

        df_9 = pd.read_csv('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/vehicle_char.csv')
        vehicle_row = df_9[df_9['vehicle-ID'] == 9]
        self.char_value = vehicle_row['char'].values[0]

        if self.char_value == 1:
            self.ACC_MAX = 4.0  # [m/s2]
            self.ACC_MAX_de = -18
            self.COMFORT_ACC_MAX = 6.0  # [m/s2]
            self.COMFORT_ACC_MIN = -8.0  # [m/s2]
            self.DISTANCE_WANTED = 2.0   # [m]
            self.TIME_WANTED = 1.0  # [s]
            self.MAX_STEERING_ANGLE = np.pi/5
        elif self.char_value == 2:
            self.ACC_MAX = 2.0
            self.ACC_MAX_de = -10
            self.COMFORT_ACC_MAX = 3.0  # [m/s2]
            self.COMFORT_ACC_MIN = -6.0  # [m/s2]
            self.DISTANCE_WANTED = 3.0  # [m]
            self.TIME_WANTED = 2.0  # [s]
            self.MAX_STEERING_ANGLE = np.pi/8

        # with open('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/trajectory_1_2.csv', 'r', encoding='utf-8') as csvfile:
        with open('/data/wangzm/merge/DJI_select/DJI_init/trajectory_1.csv', 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['vehicle-ID'] == '9':
                    vehicle_9_vx = row['v_x']
                    vehicle_9_vx = float(vehicle_9_vx)
                    vehicle_9_vy = row['v_y']
                    vehicle_9_vy = float(vehicle_9_vy)
                    vehicle_9_heading = row['heading']
                    vehicle_9_heading = float(vehicle_9_heading)
                    vehicle_9_x = row['x']
                    vehicle_9_x = float(vehicle_9_x)
                    vehicle_9_y = row['y']
                    vehicle_9_y = float(vehicle_9_y)
                    vehicle_9_current_state = VehicleState(vehicle_9_vx, vehicle_9_vy, vehicle_9_heading,vehicle_9_x,vehicle_9_y)

        with open('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/prediction_1.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['ID'] == '8':
                    vehicle_9_vx = row['v_x']
                    vehicle_9_vx = float(vehicle_9_vx)
                    vehicle_9_vy = row['v_y']
                    vehicle_9_vy = float(vehicle_9_vy)
                    vehicle_9_heading = row['heading']
                    vehicle_9_heading = float(vehicle_9_heading)
                    vehicle_9_x = row['x']
                    vehicle_9_x = float(vehicle_9_x)
                    vehicle_9_y = row['y']
                    vehicle_9_y = float(vehicle_9_y)
                    vehicle_9_next_state = VehicleState(vehicle_9_vx, vehicle_9_vy, vehicle_9_heading,vehicle_9_x,vehicle_9_y)
                    # vehicle_9_steering = row['steering']
                    # vehicle_9_steering = float(vehicle_9_steering)
                    # vehicle_9_acceleration = row['acceleration']
                    # vehicle_9_acceleration = float(vehicle_9_acceleration)

        vehicle_9_acceleration, vehicle_9_steering = calculate_dynamics(vehicle_9_current_state, vehicle_9_next_state, self.dt)

        action['steering'] = vehicle_9_steering
        # action['steering'] = 0
        action['steering'] = np.clip(action['steering'], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

        # action['acceleration'] = vehicle_9_acceleration
        if self.char_value == 1:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)
        elif self.char_value == 2:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles_2(self)
        # Calculate the desired acceleration
        desired_acceleration_2 = self.acceleration(ego_vehicle=self,
                                                 front_vehicle=front_vehicle,
                                                 rear_vehicle=rear_vehicle)
        desired_acceleration = vehicle_9_acceleration
        if vehicle_9_acceleration >= desired_acceleration_2 :
            desired_acceleration = desired_acceleration_2
        next_speed = self.velocity[0] + desired_acceleration * self.dt
        # Adjust acceleration if next speed would be less than 0
        if next_speed < 0:
            desired_acceleration = -self.velocity[0] / self.dt
        # desired_acceleration = desired_acceleration_2  
        # Clip the acceleration to ensure it is within the allowable range
        action['acceleration'] = max(self.ACC_MAX_de, min(desired_acceleration, self.ACC_MAX))
        # print('9',action)
        self.aq = self.aq + 1
        Vehicle.act(self, action)

    def acceleration(self,
                     ego_vehicle: ControlledVehicle,
                     front_vehicle: Vehicle = None,
                     rear_vehicle: Vehicle = None) -> float:
        if not ego_vehicle or isinstance(ego_vehicle, RoadObject):
            return 0
        if self.char_value == 1:
            target_speed = 8
        elif self.char_value == 2:
            target_speed = 6
        ego_target_speed = utils.not_zero(getattr(ego_vehicle, "target_speed", target_speed))
        ego_target_speed = target_speed
        acceleration = self.COMFORT_ACC_MAX * (
                1 - np.power(max(ego_vehicle.speed, 0) / ego_target_speed, self.DELTA))

        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle) - (1/2)*self.LENGTH - (1/2)*front_vehicle.LENGTH
            desired_gap = self.desired_gap(ego_vehicle, front_vehicle)
            if d < desired_gap:
                acceleration -= self.COMFORT_ACC_MAX * \
                    np.power(desired_gap / utils.not_zero(d), 2)
                acceleration -= 2
            else:
                acceleration -= self.COMFORT_ACC_MAX * \
                    np.power(desired_gap / utils.not_zero(d), 4)
        # Predict the next speed
        next_speed = ego_vehicle.velocity[0] + acceleration * self.dt
        # Adjust acceleration if next speed would be less than 0
        if next_speed < 0:
            acceleration = -ego_vehicle.velocity[0] / self.dt
        return acceleration

    def desired_gap(self, ego_vehicle: Vehicle, front_vehicle: Vehicle = None, projected: bool = True) -> float:
        d0 = self.DISTANCE_WANTED + (1/2)*self.LENGTH + (1/2)*front_vehicle.LENGTH
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = np.dot(ego_vehicle.velocity - front_vehicle.velocity, ego_vehicle.direction) if projected \
            else ego_vehicle.speed - front_vehicle.speed
        d_star = d0 + 0.5*ego_vehicle.speed * tau + 0.5* ego_vehicle.speed * dv / (2 * np.sqrt(ab)) - (1/2)*self.LENGTH - (1/2)*front_vehicle.LENGTH
        return d_star

class IDMVehicle10(ControlledVehicle):
    # Longitudinal policy parameters
    DELTA = 4.0  # []
    LESS_DELTA_SPEED = 0
    POLITENESS = 0.  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]
    dt = 0.1

    def __init__(self,
                 ID,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: int = None,
                 target_speed: float = None,
                 route: Route = None,
                 enable_lane_change: bool = True,
                 timer: float = None):
        super().__init__(ID, road, position, heading, speed, target_lane_index, target_speed, route)
        self.enable_lane_change = enable_lane_change
        self.timer = timer or (np.sum(self.position)*np.pi) % self.LANE_CHANGE_DELAY
        self.aq = 0

    def randomize_behavior(self):
        pass

    def act(self, action: Union[dict, str] = None):

        if self.crashed:
            return
        action = {}
        self.follow_road()

        df_10 = pd.read_csv('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/vehicle_char.csv')
        vehicle_row = df_10[df_10['vehicle-ID'] == 10]
        self.char_value = vehicle_row['char'].values[0]

        if self.char_value == 1:
            self.ACC_MAX = 4.0  # [m/s2]
            self.ACC_MAX_de = -18
            self.COMFORT_ACC_MAX = 6.0  # [m/s2]
            self.COMFORT_ACC_MIN = -8.0  # [m/s2]
            self.DISTANCE_WANTED = 2.0   # [m]
            self.TIME_WANTED = 1.0  # [s]
            self.MAX_STEERING_ANGLE = np.pi/5
        elif self.char_value == 2:
            self.ACC_MAX = 2.0
            self.ACC_MAX_de = -10
            self.COMFORT_ACC_MAX = 3.0  # [m/s2]
            self.COMFORT_ACC_MIN = -6.0  # [m/s2]
            self.DISTANCE_WANTED = 3.0   # [m]
            self.TIME_WANTED = 2.0  # [s]
            self.MAX_STEERING_ANGLE = np.pi/8

        # with open('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/trajectory_1_2.csv', 'r', encoding='utf-8') as csvfile:
        with open('/data/wangzm/merge/DJI_select/DJI_init/trajectory_1.csv', 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['vehicle-ID'] == '10':
                    vehicle_10_vx = row['v_x']
                    vehicle_10_vx = float(vehicle_10_vx)
                    vehicle_10_vy = row['v_y']
                    vehicle_10_vy = float(vehicle_10_vy)
                    vehicle_10_heading = row['heading']
                    vehicle_10_heading = float(vehicle_10_heading)
                    vehicle_10_x = row['x']
                    vehicle_10_x = float(vehicle_10_x)
                    vehicle_10_y = row['y']
                    vehicle_10_y = float(vehicle_10_y)
                    vehicle_10_current_state = VehicleState(vehicle_10_vx, vehicle_10_vy, vehicle_10_heading,vehicle_10_x,vehicle_10_y)

        with open('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/prediction_1.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['ID'] == '9':
                    vehicle_10_vx = row['v_x']
                    vehicle_10_vx = float(vehicle_10_vx)
                    vehicle_10_vy = row['v_y']
                    vehicle_10_vy = float(vehicle_10_vy)
                    vehicle_10_heading = row['heading']
                    vehicle_10_heading = float(vehicle_10_heading)
                    vehicle_10_x = row['x']
                    vehicle_10_x = float(vehicle_10_x)
                    vehicle_10_y = row['y']
                    vehicle_10_y = float(vehicle_10_y)
                    vehicle_10_next_state = VehicleState(vehicle_10_vx, vehicle_10_vy, vehicle_10_heading,vehicle_10_x,vehicle_10_y)
                    # vehicle_10_steering = row['steering']
                    # vehicle_10_steering = float(vehicle_10_steering)
                    # vehicle_10_acceleration = row['acceleration']
                    # vehicle_10_acceleration = float(vehicle_10_acceleration)

        vehicle_10_acceleration, vehicle_10_steering = calculate_dynamics(vehicle_10_current_state,
                                                                          vehicle_10_next_state, self.dt)

        action['steering'] = vehicle_10_steering
        # action['steering'] = 0
        action['steering'] = np.clip(action['steering'], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

        # action['acceleration'] = vehicle_10_acceleration
        if self.char_value == 1:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)
        elif self.char_value == 2:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles_2(self)
        # Calculate the desired acceleration
        desired_acceleration_2 = self.acceleration(ego_vehicle=self,
                                                 front_vehicle=front_vehicle,
                                                 rear_vehicle=rear_vehicle)
        desired_acceleration = vehicle_10_acceleration
        if vehicle_10_acceleration >= desired_acceleration_2 :
            desired_acceleration = desired_acceleration_2
        next_speed = self.velocity[0] + desired_acceleration * self.dt
        # Adjust acceleration if next speed would be less than 0
        if next_speed < 0:
            desired_acceleration = -self.velocity[0] / self.dt
        # desired_acceleration = desired_acceleration_2  
        # Clip the acceleration to ensure it is within the allowable range
        action['acceleration'] = max(self.ACC_MAX_de, min(desired_acceleration, self.ACC_MAX))
        # print('10',action)
        self.aq = self.aq + 1
        Vehicle.act(self, action)

    def acceleration(self,
                     ego_vehicle: ControlledVehicle,
                     front_vehicle: Vehicle = None,
                     rear_vehicle: Vehicle = None) -> float:
        if not ego_vehicle or isinstance(ego_vehicle, RoadObject):
            return 0
        if self.char_value == 1:
            target_speed = 8
        elif self.char_value == 2:
            target_speed = 6
        ego_target_speed = utils.not_zero(getattr(ego_vehicle, "target_speed", target_speed))
        ego_target_speed = target_speed
        acceleration = self.COMFORT_ACC_MAX * (
                1 - np.power(max(ego_vehicle.speed, 0) / ego_target_speed, self.DELTA))

        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle) - (1/2)*self.LENGTH - (1/2)*front_vehicle.LENGTH
            desired_gap = self.desired_gap(ego_vehicle, front_vehicle)
            if d < desired_gap:
                acceleration -= self.COMFORT_ACC_MAX * \
                    np.power(desired_gap / utils.not_zero(d), 2)
                acceleration -= 2
            else:
                acceleration -= self.COMFORT_ACC_MAX * \
                    np.power(desired_gap / utils.not_zero(d), 4)
        # Predict the next speed
        next_speed = ego_vehicle.velocity[0] + acceleration * self.dt
        # Adjust acceleration if next speed would be less than 0
        if next_speed < 0:
            acceleration = -ego_vehicle.velocity[0] / self.dt
        return acceleration

    def desired_gap(self, ego_vehicle: Vehicle, front_vehicle: Vehicle = None, projected: bool = True) -> float:
        d0 = self.DISTANCE_WANTED + (1/2)*self.LENGTH + (1/2)*front_vehicle.LENGTH
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = np.dot(ego_vehicle.velocity - front_vehicle.velocity, ego_vehicle.direction) if projected \
            else ego_vehicle.speed - front_vehicle.speed
        d_star = d0 + 0.5*ego_vehicle.speed * tau + 0.5* ego_vehicle.speed * dv / (2 * np.sqrt(ab)) - (1/2)*self.LENGTH - (1/2)*front_vehicle.LENGTH
        return d_star

class IDMVehicle11(ControlledVehicle):
    # Longitudinal policy parameters
    DELTA = 4.0  # []
    LESS_DELTA_SPEED = 0
    POLITENESS = 0.  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]
    dt = 0.1

    def __init__(self,
                 ID,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: int = None,
                 target_speed: float = None,
                 route: Route = None,
                 enable_lane_change: bool = True,
                 timer: float = None):
        super().__init__(ID, road, position, heading, speed, target_lane_index, target_speed, route)
        self.enable_lane_change = enable_lane_change
        self.timer = timer or (np.sum(self.position)*np.pi) % self.LANE_CHANGE_DELAY
        self.aq = 0

    def randomize_behavior(self):
        pass

    def act(self, action: Union[dict, str] = None):

        if self.crashed:
            return
        action = {}
        self.follow_road()

        df_11 = pd.read_csv('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/vehicle_char.csv')
        vehicle_row = df_11[df_11['vehicle-ID'] == 11]
        self.char_value = vehicle_row['char'].values[0]

        if self.char_value == 1:
            self.ACC_MAX = 4.0  # [m/s2]
            self.ACC_MAX_de = -18
            self.COMFORT_ACC_MAX = 6.0  # [m/s2]
            self.COMFORT_ACC_MIN = -8.0  # [m/s2]
            self.DISTANCE_WANTED = 2.0  # [m]
            self.TIME_WANTED = 0.5  # [s]
            self.MAX_STEERING_ANGLE = np.pi/5
        elif self.char_value == 2:
            self.ACC_MAX = 2.0
            self.ACC_MAX_de = -10
            self.COMFORT_ACC_MAX = 3.0  # [m/s2]
            self.COMFORT_ACC_MIN = -6.0  # [m/s2]
            self.DISTANCE_WANTED = 3.0  # [m]
            self.TIME_WANTED = 2.0  # [s]
            self.MAX_STEERING_ANGLE = np.pi/8

        # with open('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/trajectory_1_2.csv', 'r', encoding='utf-8') as csvfile:
        with open('/data/wangzm/merge/DJI_select/DJI_init/trajectory_1.csv', 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['vehicle-ID'] == '11':
                    vehicle_11_vx = row['v_x']
                    vehicle_11_vx = float(vehicle_11_vx)
                    vehicle_11_vy = row['v_y']
                    vehicle_11_vy = float(vehicle_11_vy)
                    vehicle_11_heading = row['heading']
                    vehicle_11_heading = float(vehicle_11_heading)
                    vehicle_11_x = row['x']
                    vehicle_11_x = float(vehicle_11_x)
                    vehicle_11_y = row['y']
                    vehicle_11_y = float(vehicle_11_y)
                    vehicle_11_current_state = VehicleState(vehicle_11_vx, vehicle_11_vy, vehicle_11_heading,vehicle_11_x,vehicle_11_y)

        with open('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/prediction_1.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['ID'] == '10':
                    vehicle_11_vx = row['v_x']
                    vehicle_11_vx = float(vehicle_11_vx)
                    vehicle_11_vy = row['v_y']
                    vehicle_11_vy = float(vehicle_11_vy)
                    vehicle_11_heading = row['heading']
                    vehicle_11_heading = float(vehicle_11_heading)
                    vehicle_11_x = row['x']
                    vehicle_11_x = float(vehicle_11_x)
                    vehicle_11_y = row['y']
                    vehicle_11_y = float(vehicle_11_y)
                    vehicle_11_next_state = VehicleState(vehicle_11_vx, vehicle_11_vy, vehicle_11_heading,vehicle_11_x,vehicle_11_y)
                    # vehicle_11_steering = row['steering']
                    # vehicle_11_steering = float(vehicle_11_steering)
                    # vehicle_11_acceleration = row['acceleration']
                    # vehicle_11_acceleration = float(vehicle_11_acceleration)

        vehicle_11_acceleration, vehicle_11_steering = calculate_dynamics(vehicle_11_current_state,
                                                                          vehicle_11_next_state, self.dt)

        action['steering'] = vehicle_11_steering
        # action['steering'] = 0
        action['steering'] = np.clip(action['steering'], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

        # action['acceleration'] = vehicle_11_acceleration
        if self.char_value == 1:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)
        elif self.char_value == 2:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles_2(self)
        # Calculate the desired acceleration
        desired_acceleration_2 = self.acceleration(ego_vehicle=self,
                                                 front_vehicle=front_vehicle,
                                                 rear_vehicle=rear_vehicle)
        desired_acceleration = vehicle_11_acceleration
        if vehicle_11_acceleration >= desired_acceleration_2 :
            desired_acceleration = desired_acceleration_2
        next_speed = self.velocity[0] + desired_acceleration * self.dt
        # Adjust acceleration if next speed would be less than 0
        if next_speed < 0:
            desired_acceleration = -self.velocity[0] / self.dt
        # desired_acceleration = desired_acceleration_2  
        # Clip the acceleration to ensure it is within the allowable range
        action['acceleration'] = max(self.ACC_MAX_de, min(desired_acceleration, self.ACC_MAX))
        # print('11',action)
        self.aq = self.aq + 1
        Vehicle.act(self, action)

    def acceleration(self,
                     ego_vehicle: ControlledVehicle,
                     front_vehicle: Vehicle = None,
                     rear_vehicle: Vehicle = None) -> float:
        if not ego_vehicle or isinstance(ego_vehicle, RoadObject):
            return 0
        if self.char_value == 1:
            target_speed = 8
        elif self.char_value == 2:
            target_speed = 6
        ego_target_speed = utils.not_zero(getattr(ego_vehicle, "target_speed", target_speed))
        ego_target_speed = target_speed
        acceleration = self.COMFORT_ACC_MAX * (
                1 - np.power(max(ego_vehicle.speed, 0) / ego_target_speed, self.DELTA))

        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle) - (1/2)*self.LENGTH - (1/2)*front_vehicle.LENGTH
            desired_gap = self.desired_gap(ego_vehicle, front_vehicle)
            if d < desired_gap:
                acceleration -= self.COMFORT_ACC_MAX * \
                    np.power(desired_gap / utils.not_zero(d), 2)
                acceleration -= 2
            else:
                acceleration -= self.COMFORT_ACC_MAX * \
                    np.power(desired_gap / utils.not_zero(d), 4)
        # Predict the next speed
        next_speed = ego_vehicle.velocity[0] + acceleration * self.dt
        # Adjust acceleration if next speed would be less than 0
        if next_speed < 0:
            acceleration = -ego_vehicle.velocity[0] / self.dt
        return acceleration

    def desired_gap(self, ego_vehicle: Vehicle, front_vehicle: Vehicle = None, projected: bool = True) -> float:
        d0 = self.DISTANCE_WANTED + (1/2)*self.LENGTH + (1/2)*front_vehicle.LENGTH
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = np.dot(ego_vehicle.velocity - front_vehicle.velocity, ego_vehicle.direction) if projected \
            else ego_vehicle.speed - front_vehicle.speed
        d_star = d0 + 0.5*ego_vehicle.speed * tau + 0.5* ego_vehicle.speed * dv / (2 * np.sqrt(ab)) - (1/2)*self.LENGTH - (1/2)*front_vehicle.LENGTH
        return d_star

class IDMVehicle12(ControlledVehicle):
    # Longitudinal policy parameters
    DELTA = 4.0  # []
    LESS_DELTA_SPEED = 0
    POLITENESS = 0.  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]
    dt = 0.1

    def __init__(self,
                 ID,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: int = None,
                 target_speed: float = None,
                 route: Route = None,
                 enable_lane_change: bool = True,
                 timer: float = None):
        super().__init__(ID, road, position, heading, speed, target_lane_index, target_speed, route)
        self.enable_lane_change = enable_lane_change
        self.timer = timer or (np.sum(self.position)*np.pi) % self.LANE_CHANGE_DELAY
        self.aq = 0

    def randomize_behavior(self):
        pass

    def act(self, action: Union[dict, str] = None):

        if self.crashed:
            return
        action = {}
        self.follow_road()

        df_12 = pd.read_csv('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/vehicle_char.csv')
        vehicle_row = df_12[df_12['vehicle-ID'] == 12]
        self.char_value = vehicle_row['char'].values[0]

        if self.char_value == 1:
            self.ACC_MAX = 4.0  # [m/s2]
            self.ACC_MAX_de = -18
            self.COMFORT_ACC_MAX = 6.0  # [m/s2]
            self.COMFORT_ACC_MIN = -8.0  # [m/s2]
            self.DISTANCE_WANTED = 2.0  # [m]
            self.TIME_WANTED = 0.5  # [s]
            self.MAX_STEERING_ANGLE = np.pi/5
        elif self.char_value == 2:
            self.ACC_MAX = 2.0
            self.ACC_MAX_de = -10
            self.COMFORT_ACC_MAX = 3.0  # [m/s2]
            self.COMFORT_ACC_MIN = -6.0  # [m/s2]
            self.DISTANCE_WANTED = 3.0  # [m]
            self.TIME_WANTED = 2.0  # [s]
            self.MAX_STEERING_ANGLE = np.pi/8

        # with open('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/trajectory_1_2.csv', 'r', encoding='utf-8') as csvfile:
        with open('/data/wangzm/merge/DJI_select/DJI_init/trajectory_1.csv', 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['vehicle-ID'] == '12':
                    vehicle_12_vx = row['v_x']
                    vehicle_12_vx = float(vehicle_12_vx)
                    vehicle_12_vy = row['v_y']
                    vehicle_12_vy = float(vehicle_12_vy)
                    vehicle_12_heading = row['heading']
                    vehicle_12_heading = float(vehicle_12_heading)
                    vehicle_12_x = row['x']
                    vehicle_12_x = float(vehicle_12_x)
                    vehicle_12_y = row['y']
                    vehicle_12_y = float(vehicle_12_y)
                    vehicle_12_current_state = VehicleState(vehicle_12_vx, vehicle_12_vy, vehicle_12_heading,vehicle_12_x,vehicle_12_y)

        with open('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/prediction_1.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['ID'] == '11':
                    vehicle_12_vx = row['v_x']
                    vehicle_12_vx = float(vehicle_12_vx)
                    vehicle_12_vy = row['v_y']
                    vehicle_12_vy = float(vehicle_12_vy)
                    vehicle_12_heading = row['heading']
                    vehicle_12_heading = float(vehicle_12_heading)
                    vehicle_12_x = row['x']
                    vehicle_12_x = float(vehicle_12_x)
                    vehicle_12_y = row['y']
                    vehicle_12_y = float(vehicle_12_y)
                    vehicle_12_next_state = VehicleState(vehicle_12_vx, vehicle_12_vy, vehicle_12_heading,vehicle_12_x,vehicle_12_y)
                    # vehicle_12_steering = row['steering']
                    # vehicle_12_steering = float(vehicle_12_steering)
                    # vehicle_12_acceleration = row['acceleration']
                    # vehicle_12_acceleration = float(vehicle_12_acceleration)

        vehicle_12_acceleration, vehicle_12_steering = calculate_dynamics(vehicle_12_current_state,
                                                                          vehicle_12_next_state, self.dt)

        action['steering'] = vehicle_12_steering
        # action['steering'] = 0
        action['steering'] = np.clip(action['steering'], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

        # action['acceleration'] = vehicle_12_acceleration
        if self.char_value == 1:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)
        elif self.char_value == 2:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles_2(self)
        # Calculate the desired acceleration
        desired_acceleration_2 = self.acceleration(ego_vehicle=self,
                                                 front_vehicle=front_vehicle,
                                                 rear_vehicle=rear_vehicle)
        desired_acceleration = vehicle_12_acceleration
        if vehicle_12_acceleration >= desired_acceleration_2 :
            desired_acceleration = desired_acceleration_2
        next_speed = self.velocity[0] + desired_acceleration * self.dt
        # Adjust acceleration if next speed would be less than 0
        if next_speed < 0:
            desired_acceleration = -self.velocity[0] / self.dt
        # desired_acceleration = desired_acceleration_2  
        # Clip the acceleration to ensure it is within the allowable range
        action['acceleration'] = max(self.ACC_MAX_de, min(desired_acceleration, self.ACC_MAX))
        # print('12',action)
        self.aq = self.aq + 1
        Vehicle.act(self, action)

    def acceleration(self,
                     ego_vehicle: ControlledVehicle,
                     front_vehicle: Vehicle = None,
                     rear_vehicle: Vehicle = None) -> float:
        if not ego_vehicle or isinstance(ego_vehicle, RoadObject):
            return 0
        if self.char_value == 1:
            target_speed = 8
        elif self.char_value == 2:
            target_speed = 6
        ego_target_speed = utils.not_zero(getattr(ego_vehicle, "target_speed", target_speed))
        ego_target_speed = target_speed
        acceleration = self.COMFORT_ACC_MAX * (
                1 - np.power(max(ego_vehicle.speed, 0) / ego_target_speed, self.DELTA))

        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle) - (1/2)*self.LENGTH - (1/2)*front_vehicle.LENGTH
            desired_gap = self.desired_gap(ego_vehicle, front_vehicle)
            if d < desired_gap:
                acceleration -= self.COMFORT_ACC_MAX * \
                    np.power(desired_gap / utils.not_zero(d), 2)
                acceleration -= 2
            else:
                acceleration -= self.COMFORT_ACC_MAX * \
                    np.power(desired_gap / utils.not_zero(d), 4)
        # Predict the next speed
        next_speed = ego_vehicle.velocity[0] + acceleration * self.dt
        # Adjust acceleration if next speed would be less than 0
        if next_speed < 0:
            acceleration = -ego_vehicle.velocity[0] / self.dt
        return acceleration

    def desired_gap(self, ego_vehicle: Vehicle, front_vehicle: Vehicle = None, projected: bool = True) -> float:
        d0 = self.DISTANCE_WANTED + (1/2)*self.LENGTH + (1/2)*front_vehicle.LENGTH
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = np.dot(ego_vehicle.velocity - front_vehicle.velocity, ego_vehicle.direction) if projected \
            else ego_vehicle.speed - front_vehicle.speed
        d_star = d0 + 0.5*ego_vehicle.speed * tau + 0.5* ego_vehicle.speed * dv / (2 * np.sqrt(ab)) - (1/2)*self.LENGTH - (1/2)*front_vehicle.LENGTH
        return d_star

class IDMVehicle13(ControlledVehicle):
    # Longitudinal policy parameters
    ACC_MAX = 2.0  # [m/s2]
    ACC_MAX_de = 9.0
    COMFORT_ACC_MAX = 3.0  # [m/s2]
    COMFORT_ACC_MIN = -6.0  # [m/s2]
    DISTANCE_WANTED = 2.0 + ControlledVehicle.LENGTH  # [m]
    TIME_WANTED = 1.5  # [s]
    DELTA = 4.0  # []
    LESS_DELTA_SPEED = 0
    POLITENESS = 0.  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]
    dt = 0.1
    MAX_STEERING_ANGLE = np.pi/8

    def __init__(self,
                 ID,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: int = None,
                 target_speed: float = None,
                 route: Route = None,
                 enable_lane_change: bool = True,
                 timer: float = None):
        super().__init__(ID, road, position, heading, speed, target_lane_index, target_speed, route)
        self.enable_lane_change = enable_lane_change
        self.timer = timer or (np.sum(self.position)*np.pi) % self.LANE_CHANGE_DELAY
        self.aq = 0

    def randomize_behavior(self):
        pass

    def act(self, action: Union[dict, str] = None):

        if self.crashed:
            return
        action = {}
        self.follow_road()

        df_13 = pd.read_csv('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/vehicle_char.csv')
        vehicle_row = df_13[df_13['vehicle-ID'] == 13]
        self.char_value = vehicle_row['char'].values[0]

        if self.char_value == 1:
            self.ACC_MAX = 4.0  # [m/s2]
            self.ACC_MAX_de = -18
            self.COMFORT_ACC_MAX = 6.0  # [m/s2]
            self.COMFORT_ACC_MIN = -8.0  # [m/s2]
            self.DISTANCE_WANTED = 2.0  # [m]
            self.TIME_WANTED = 0.5  # [s]
            self.MAX_STEERING_ANGLE = np.pi/5
        elif self.char_value == 2:
            self.ACC_MAX = 2.0
            self.ACC_MAX_de = -10
            self.COMFORT_ACC_MAX = 3.0  # [m/s2]
            self.COMFORT_ACC_MIN = -6.0  # [m/s2]
            self.DISTANCE_WANTED = 3.0  # [m]
            self.TIME_WANTED = 2.0  # [s]
            self.MAX_STEERING_ANGLE = np.pi/8


        # with open('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/trajectory_1_2.csv') as csvfile:
        with open('/data/wangzm/merge/DJI_select/DJI_init/trajectory_1.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['vehicle-ID'] == '13':
                    vehicle_13_vx = row['v_x']
                    vehicle_13_vx = float(vehicle_13_vx)
                    vehicle_13_vy = row['v_y']
                    vehicle_13_vy = float(vehicle_13_vy)
                    vehicle_13_heading = row['heading']
                    vehicle_13_heading = float(vehicle_13_heading)
                    vehicle_13_x = row['x']
                    vehicle_13_x = float(vehicle_13_x)
                    vehicle_13_y = row['y']
                    vehicle_13_y = float(vehicle_13_y)
                    vehicle_13_current_state = VehicleState(vehicle_13_vx, vehicle_13_vy, vehicle_13_heading,vehicle_13_x,vehicle_13_y)

        with open('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/prediction_1.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['ID'] == '12':
                    vehicle_13_vx = row['v_x']
                    vehicle_13_vx = float(vehicle_13_vx)
                    vehicle_13_vy = row['v_y']
                    vehicle_13_vy = float(vehicle_13_vy)
                    vehicle_13_heading = row['heading']
                    vehicle_13_heading = float(vehicle_13_heading)
                    vehicle_13_x = row['x']
                    vehicle_13_x = float(vehicle_13_x)
                    vehicle_13_y = row['y']
                    vehicle_13_y = float(vehicle_13_y)
                    vehicle_13_next_state = VehicleState(vehicle_13_vx, vehicle_13_vy, vehicle_13_heading,vehicle_13_x,vehicle_13_y)

        vehicle_13_acceleration, vehicle_13_steering = calculate_dynamics(vehicle_13_current_state,
                                                                          vehicle_13_next_state, self.dt)
        action['steering'] = vehicle_13_steering
        # action['steering'] = 0
        action['steering'] = np.clip(action['steering'], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

        # action['acceleration'] = vehicle_13_acceleration
        if self.char_value == 1:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)
        elif self.char_value == 2:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles_2(self)
        # front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)
        desired_acceleration_2 = self.acceleration(ego_vehicle=self,
                                                   front_vehicle=front_vehicle,
                                                   rear_vehicle=rear_vehicle)
        desired_acceleration = vehicle_13_acceleration
        if vehicle_13_acceleration >= desired_acceleration_2 :
            desired_acceleration = desired_acceleration_2
        next_speed = self.velocity[0] + desired_acceleration * self.dt
        # Adjust acceleration if next speed would be less than 0
        if next_speed < 0:
            desired_acceleration = -self.velocity[0] / self.dt
        action['acceleration'] = max(self.ACC_MAX_de, min(desired_acceleration, self.ACC_MAX))
        # print('13',action)
        self.aq = self.aq + 1
        Vehicle.act(self, action)

    def acceleration(self,
                     ego_vehicle: ControlledVehicle,
                     front_vehicle: Vehicle = None,
                     rear_vehicle: Vehicle = None) -> float:
        if not ego_vehicle or isinstance(ego_vehicle, RoadObject):
            return 0
        if self.char_value == 1:
            target_speed = 7
        elif self.char_value == 2:
            target_speed = 5
        ego_target_speed = utils.not_zero(getattr(ego_vehicle, "target_speed", target_speed))
        acceleration = self.COMFORT_ACC_MAX * (
                1 - np.power(max(ego_vehicle.speed, 0) / ego_target_speed, self.DELTA))

        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle) - 5
            # acceleration -= self.COMFORT_ACC_MAX * \
            #     np.power(self.desired_gap(ego_vehicle, front_vehicle) / utils.not_zero(d), 2)
            d_desired = self.desired_gap(ego_vehicle, front_vehicle) - 5
            acceleration = self.COMFORT_ACC_MAX * (1 - 0.2*(ego_vehicle.velocity[0] / target_speed) ** self.DELTA - (d_desired / d) ** 2)
        # Predict the next speed
        next_speed = ego_vehicle.velocity[0] + acceleration * self.dt
        # Adjust acceleration if next speed would be less than 0
        if next_speed < 0:
            acceleration = -ego_vehicle.velocity[0] / self.dt
        return acceleration

    def desired_gap(self, ego_vehicle: Vehicle, front_vehicle: Vehicle = None, projected: bool = True) -> float:
        d0 = self.DISTANCE_WANTED
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = np.dot(ego_vehicle.velocity - front_vehicle.velocity, ego_vehicle.direction) if projected \
            else ego_vehicle.speed - front_vehicle.speed
        d_star = d0 + 0.1*ego_vehicle.speed * tau + 0.1*ego_vehicle.speed * dv / (2 * np.sqrt(ab))
        return d_star