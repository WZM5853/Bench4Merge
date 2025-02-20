import csv
from typing import List, Tuple, Union

import numpy as np
import copy
from highway_env import utils
from highway_env.road.road import Road, LaneIndex, Route
from highway_env.types import Vector
from highway_env.vehicle.kinematics import Vehicle, Vehicle2


class ControlledVehicle(Vehicle):
    """
    A vehicle piloted by two low-level controller, allowing high-level actions such as cruise control and lane changes.

    - The longitudinal controller is a speed controller;
    - The lateral controller is a heading controller cascaded with a lateral position controller.
    """

    target_speed: float
    """ Desired velocity."""

    TAU_A = 0.6  # [s]
    TAU_DS = 0.2  # [s]
    PURSUIT_TAU = 0.5*TAU_DS  # [s]
    KP_A = 1.2 / TAU_A
    KP_HEADING = 1 / TAU_DS
    KP_LATERAL = 1/3 * KP_HEADING  # [1/s]
    MAX_STEERING_ANGLE = np.pi / 1  # [rad]
    DELTA_SPEED = 1  # [m/s]
    LESS_DELTA_SPEED = 0.5
    MORE_DELTA_SPEED = 1.5

    def __init__(self,
                 ID,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: LaneIndex = None,
                 target_speed: float = None,
                 route: Route = None):
        super().__init__(ID, road, position, heading, speed)
        self.target_lane_index = target_lane_index or self.lane_index
        self.target_speed = target_speed or self.speed
        self.route = route
        # self.f = open('result/act6.csv', 'w', encoding='utf-8', newline="")
        # self.csv_write = csv.writer(self.f)
        # self.csv_write.writerow(['time', 'yaw', 'speed'])
        # self.i = 0


    def follow_road(self) -> None:
        """At the end of a lane, automatically switch to a next one."""
        if self.road.network.get_lane(self.target_lane_index).after_end(self.position):
            self.target_lane_index = self.road.network.next_lane(self.target_lane_index,
                                                                 route=self.route,
                                                                 position=self.position,
                                                                 np_random=self.road.np_random)



    def steering_control(self, target_lane_index: LaneIndex) -> float:
        """
        Steer the vehicle to follow the center of an given lane.

        1. Lateral position is controlled by a proportional controller yielding a lateral speed command
        2. Lateral speed command is converted to a heading reference
        3. Heading is controlled by a proportional controller yielding a heading rate command
        4. Heading rate command is converted to a steering angle

        :param target_lane_index: index of the lane to follow
        :return: a steering wheel angle command [rad]
        """
        target_lane = self.road.network.get_lane(target_lane_index)
        # print(target_lane)
        lane_coords = target_lane.local_coordinates(self.position)
        lane_next_coords = lane_coords[0] + self.speed * self.PURSUIT_TAU
        lane_future_heading = target_lane.heading_at(lane_next_coords)

        # Lateral position control
        lateral_speed_command = - self.KP_LATERAL * lane_coords[1]
        # Lateral speed to heading
        heading_command = np.arcsin(np.clip(lateral_speed_command / utils.not_zero(self.speed), -1, 1))
        heading_ref = lane_future_heading + np.clip(heading_command, -np.pi/4, np.pi/4)
        # Heading control
        heading_rate_command = self.KP_HEADING * utils.wrap_to_pi(heading_ref - self.heading)
        # Heading rate to steering angle
        steering_angle = np.arcsin(np.clip(self.LENGTH / 2 / utils.not_zero(self.speed) * heading_rate_command,
                                           -1, 1))
        # print(self.speed_control(self.speed))

        # steering_constraint = (np.pi/3)*(np.log10(0.02*(self.speed)+0.1))
        # steering_angle = np.clip(steering_angle, -steering_constraint, steering_constraint)
        steering_angle = np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        # print(steering_angle)
        return float(steering_angle)

class ControlledVehicle2(Vehicle2):
    """
    A vehicle piloted by two low-level controller, allowing high-level actions such as cruise control and lane changes.

    - The longitudinal controller is a speed controller;
    - The lateral controller is a heading controller cascaded with a lateral position controller.
    """

    target_speed: float
    """ Desired velocity."""

    TAU_A = 0.6  # [s]
    TAU_DS = 0.2  # [s]
    PURSUIT_TAU = 0.5*TAU_DS  # [s]
    KP_A = 1.2 / TAU_A
    KP_HEADING = 1 / TAU_DS
    KP_LATERAL = 1/3 * KP_HEADING  # [1/s]
    MAX_STEERING_ANGLE = np.pi / 1  # [rad]
    DELTA_SPEED = 1  # [m/s]
    LESS_DELTA_SPEED = 0.5
    MORE_DELTA_SPEED = 1.5

    def __init__(self,
                 ID,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: LaneIndex = None,
                 target_speed: float = None,
                 route: Route = None):
        super().__init__(ID, road, position, heading, speed)
        self.target_lane_index = target_lane_index or self.lane_index
        self.target_speed = target_speed or self.speed
        self.route = route
        # self.f = open('result/act6.csv', 'w', encoding='utf-8', newline="")
        # self.csv_write = csv.writer(self.f)
        # self.csv_write.writerow(['time', 'yaw', 'speed'])
        # self.i = 0


    def follow_road(self) -> None:
        """At the end of a lane, automatically switch to a next one."""
        if self.road.network.get_lane(self.target_lane_index).after_end(self.position):
            self.target_lane_index = self.road.network.next_lane(self.target_lane_index,
                                                                 route=self.route,
                                                                 position=self.position,
                                                                 np_random=self.road.np_random)



    def steering_control(self, target_lane_index: LaneIndex) -> float:
        """
        Steer the vehicle to follow the center of an given lane.

        1. Lateral position is controlled by a proportional controller yielding a lateral speed command
        2. Lateral speed command is converted to a heading reference
        3. Heading is controlled by a proportional controller yielding a heading rate command
        4. Heading rate command is converted to a steering angle

        :param target_lane_index: index of the lane to follow
        :return: a steering wheel angle command [rad]
        """
        target_lane = self.road.network.get_lane(target_lane_index)
        # print(target_lane)
        lane_coords = target_lane.local_coordinates(self.position)
        lane_next_coords = lane_coords[0] + self.speed * self.PURSUIT_TAU
        lane_future_heading = target_lane.heading_at(lane_next_coords)

        # Lateral position control
        lateral_speed_command = - self.KP_LATERAL * lane_coords[1]
        # Lateral speed to heading
        heading_command = np.arcsin(np.clip(lateral_speed_command / utils.not_zero(self.speed), -1, 1))
        heading_ref = lane_future_heading + np.clip(heading_command, -np.pi/4, np.pi/4)
        # Heading control
        heading_rate_command = self.KP_HEADING * utils.wrap_to_pi(heading_ref - self.heading)
        # Heading rate to steering angle
        steering_angle = np.arcsin(np.clip(self.LENGTH / 2 / utils.not_zero(self.speed) * heading_rate_command,
                                           -1, 1))
        # print(self.speed_control(self.speed))

        # steering_constraint = (np.pi/3)*(np.log10(0.02*(self.speed)+0.1))
        # steering_angle = np.clip(steering_angle, -steering_constraint, steering_constraint)
        steering_angle = np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        # print(steering_angle)
        return float(steering_angle)

class MDPVehicle(ControlledVehicle):

    """A controlled vehicle with a specified discrete range of allowed target speeds."""

    SPEED_COUNT: int = 2  # []
    SPEED_MIN: float = 5  # [m/s]
    SPEED_MAX: float = 15  # [m/s]
