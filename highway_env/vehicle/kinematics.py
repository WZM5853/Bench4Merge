from typing import Union, TYPE_CHECKING, Optional
import numpy as np
import pandas as pd
from collections import deque

from highway_env import utils
from highway_env.road.lane import AbstractLane
from highway_env.road.road import Road, LaneIndex
from highway_env.road.objects import Obstacle, Landmark
from highway_env.types import Vector

import csv

if TYPE_CHECKING:
    from highway_env.road.objects import RoadObject

class Vehicle(object):

    """
    A moving vehicle on a road, and its kinematics.

    The vehicle is represented by a dynamical system: a modified bicycle model.
    It's state is propagated depending on its steering and acceleration actions.
    """

    COLLISIONS_ENABLED = True
    """ Enable collision detection between vehicles """

    LENGTH = 5.0
    """ Vehicle length [m] """
    WIDTH = 2.0
    """ Vehicle width [m] """
    DEFAULT_SPEEDS = [4, 6]
    """ Range for random initial speeds [m/s] """
    MAX_SPEED = 8
    """ Maximum reachable speed [m/s] """
    

    def __init__(self,
                 ID,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0):
        self.ID = ID
        self.road = road
        self.position = np.array(position).astype('float')
        self.heading = heading
        self.speed = speed
        self.lane_index = self.road.network.get_closest_lane_index(self.position, self.heading) if self.road else np.nan
        self.lane = self.road.network.get_lane(self.lane_index) if self.road else None
        self.action = {'steering': 0, 'acceleration': 0}
        self.crashed = False
        self.log = []
        self.history = deque(maxlen=30)



    def act(self, action: Union[dict, str] = None) -> None:
        """
        Store an action to be repeated.

        :param action: the input action
        """
        if action:
            self.action = action

    def step(self, dt: float) -> None:
        """
        Propagate the vehicle state given its actions.

        Integrate a modified bicycle model with a 1st-order response on the steering wheel dynamics.
        If the vehicle is crashed, the actions are overridden with erratic steering and braking until complete stop.
        The vehicle's current lane is updated.

        :param dt: timestep of integration of the model [s]
        """
        self.clip_actions()
        delta_f = self.action['steering']
        beta = np.arctan(1 / 2 * np.tan(delta_f))
        v = self.speed * np.array([np.cos(self.heading + beta),
                                   np.sin(self.heading + beta)])
        self.position += v * dt
        self.heading += self.speed * np.sin(beta) / (self.LENGTH / 2) * dt
        self.speed += self.action['acceleration'] * dt
        self.on_state_update()

        # print(self.ID, self.position[0], self.position[1], self.speed, self.heading)

        

        

    def clip_actions(self) -> None:
        if self.crashed:
            self.action['steering'] = 0
            self.action['acceleration'] = -1.0*self.speed
        # self.action['steering'] = float(self.action['steering'])
        self.action['acceleration'] = float(self.action['acceleration'])
        if self.speed > self.MAX_SPEED:
            self.action['acceleration'] = min(self.action['acceleration'], 1.0 * (self.MAX_SPEED - self.speed))
        elif self.speed < -self.MAX_SPEED:
            self.action['acceleration'] = max(self.action['acceleration'], 1.0 * (self.MAX_SPEED - self.speed))

    def on_state_update(self) -> None:
        if self.road:
            self.lane_index = self.road.network.get_closest_lane_index(self.position, self.heading)
            self.lane = self.road.network.get_lane(self.lane_index)
            if self.road.record_history:
                self.history.appendleft(self.create_from(self))

    def lane_distance_to(self, vehicle: "Vehicle", lane: AbstractLane = None) -> float:
        """
        Compute the signed distance to another vehicle along a lane.

        :param vehicle: the other vehicle
        :param lane: a lane
        :return: the distance to the other vehicle [m]
        """
        if not vehicle:
            return np.nan
        if not lane:
            lane = self.lane
        return lane.local_coordinates(vehicle.position)[0] - lane.local_coordinates(self.position)[0]

    def check_collision(self, other: Union['Vehicle', 'RoadObject']) -> None:
        """
        Check for collision with another vehicle.

        :param other: the other vehicle or object
        """
        if self.crashed or other is self:
            return

        if isinstance(other, Vehicle):
            if not self.COLLISIONS_ENABLED or not other.COLLISIONS_ENABLED:
                return

            if self._is_colliding(other):
                self.speed = other.speed = min([self.speed, other.speed], key=abs)
                self.crashed = other.crashed = True
        elif isinstance(other, Obstacle):
            if not self.COLLISIONS_ENABLED:
                return

            if self._is_colliding(other):
                self.speed = min([self.speed, 0], key=abs)
                self.crashed = other.hit = True
        elif isinstance(other, Landmark):
            if self._is_colliding(other):
                other.hit = True

    def _is_colliding(self, other):
        # Fast spherical pre-check
        if np.linalg.norm(other.position - self.position) > self.LENGTH:
            return False
        # Accurate rectangular check
        return utils.rotated_rectangles_intersect((self.position, 0.9*self.LENGTH, 0.9*self.WIDTH, self.heading),
                                                  (other.position, 0.9*other.LENGTH, 0.9*other.WIDTH, other.heading))

    @property
    def direction(self) -> np.ndarray:
        return np.array([np.cos(self.heading), np.sin(self.heading)])

    @property
    def velocity(self) -> np.ndarray:
        return self.speed * self.direction  # TODO: slip angle beta should be used here

    @property
    def destination(self) -> np.ndarray:
        if getattr(self, "route", None):
            last_lane = self.road.network.get_lane(self.route[-1])
            return last_lane.position(last_lane.length, 0)
        else:
            return self.position

    @property
    def destination_direction(self) -> np.ndarray:
        if (self.destination != self.position).any():
            return (self.destination - self.position) / np.linalg.norm(self.destination - self.position)
        else:
            return np.zeros((2,))

    def to_dict(self, origin_vehicle: "Vehicle" = None, observe_intentions: bool = True) -> dict:
        d = {
            'presence': 1,
            'x': self.position[0],
            'y': self.position[1],
            'vx': self.velocity[0],
            'vy': self.velocity[1],
            'heading': self.heading,
            'cos_h': self.direction[0],
            'sin_h': self.direction[1],
            'cos_d': self.destination_direction[0],
            'sin_d': self.destination_direction[1]
        }
        if not observe_intentions:
            d["cos_d"] = d["sin_d"] = 0
        if origin_vehicle:
            origin_dict = origin_vehicle.to_dict()
            for key in ['x', 'y', 'vx', 'vy']:
                d[key] -= origin_dict[key]
        return d
    

class Vehicle2(object):

    """
    A moving vehicle on a road, and its kinematics.

    The vehicle is represented by a dynamical system: a modified bicycle model.
    It's state is propagated depending on its steering and acceleration actions.
    """

    COLLISIONS_ENABLED = True
    """ Enable collision detection between vehicles """

    LENGTH = 11.0
    """ Vehicle length [m] """
    WIDTH = 2.5
    """ Vehicle width [m] """
    DEFAULT_SPEEDS = [4, 6]
    """ Range for random initial speeds [m/s] """
    MAX_SPEED = 8
    """ Maximum reachable speed [m/s] """
    

    def __init__(self,
                 ID,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0):
        self.ID = ID
        self.road = road
        self.position = np.array(position).astype('float')
        self.heading = heading
        self.speed = speed
        self.lane_index = self.road.network.get_closest_lane_index(self.position, self.heading) if self.road else np.nan
        self.lane = self.road.network.get_lane(self.lane_index) if self.road else None
        self.action = {'steering': 0, 'acceleration': 0}
        self.crashed = False
        self.log = []
        self.history = deque(maxlen=30)



    def act(self, action: Union[dict, str] = None) -> None:
        """
        Store an action to be repeated.

        :param action: the input action
        """
        if action:
            self.action = action

    def step(self, dt: float) -> None:
        """
        Propagate the vehicle state given its actions.

        Integrate a modified bicycle model with a 1st-order response on the steering wheel dynamics.
        If the vehicle is crashed, the actions are overridden with erratic steering and braking until complete stop.
        The vehicle's current lane is updated.

        :param dt: timestep of integration of the model [s]
        """
        self.clip_actions()
        delta_f = self.action['steering']
        beta = np.arctan(1 / 2 * np.tan(delta_f))
        v = self.speed * np.array([np.cos(self.heading + beta),
                                   np.sin(self.heading + beta)])
        self.position += v * dt
        self.heading += self.speed * np.sin(beta) / (self.LENGTH / 2) * dt
        self.speed += self.action['acceleration'] * dt
        self.on_state_update()

        # print(self.ID, self.position[0], self.position[1], self.speed, self.heading)

        

        

    def clip_actions(self) -> None:
        if self.crashed:
            self.action['steering'] = 0
            self.action['acceleration'] = -1.0*self.speed
        # self.action['steering'] = float(self.action['steering'])
        self.action['acceleration'] = float(self.action['acceleration'])
        if self.speed > self.MAX_SPEED:
            self.action['acceleration'] = min(self.action['acceleration'], 1.0 * (self.MAX_SPEED - self.speed))
        elif self.speed < -self.MAX_SPEED:
            self.action['acceleration'] = max(self.action['acceleration'], 1.0 * (self.MAX_SPEED - self.speed))

    def on_state_update(self) -> None:
        if self.road:
            self.lane_index = self.road.network.get_closest_lane_index(self.position, self.heading)
            self.lane = self.road.network.get_lane(self.lane_index)
            if self.road.record_history:
                self.history.appendleft(self.create_from(self))

    def lane_distance_to(self, vehicle: "Vehicle", lane: AbstractLane = None) -> float:
        """
        Compute the signed distance to another vehicle along a lane.

        :param vehicle: the other vehicle
        :param lane: a lane
        :return: the distance to the other vehicle [m]
        """
        if not vehicle:
            return np.nan
        if not lane:
            lane = self.lane
        return lane.local_coordinates(vehicle.position)[0] - lane.local_coordinates(self.position)[0]

    def check_collision(self, other: Union['Vehicle', 'RoadObject']) -> None:
        """
        Check for collision with another vehicle.

        :param other: the other vehicle or object
        """
        if self.crashed or other is self:
            return

        if isinstance(other, Vehicle):
            if not self.COLLISIONS_ENABLED or not other.COLLISIONS_ENABLED:
                return

            if self._is_colliding(other):
                self.speed = other.speed = min([self.speed, other.speed], key=abs)
                self.crashed = other.crashed = True
        elif isinstance(other, Obstacle):
            if not self.COLLISIONS_ENABLED:
                return

            if self._is_colliding(other):
                self.speed = min([self.speed, 0], key=abs)
                self.crashed = other.hit = True
        elif isinstance(other, Landmark):
            if self._is_colliding(other):
                other.hit = True

    def _is_colliding(self, other):
        # Fast spherical pre-check
        if np.linalg.norm(other.position - self.position) > self.LENGTH:
            return False
        # Accurate rectangular check
        return utils.rotated_rectangles_intersect((self.position, 0.9*self.LENGTH, 0.9*self.WIDTH, self.heading),
                                                  (other.position, 0.9*other.LENGTH, 0.9*other.WIDTH, other.heading))

    @property
    def direction(self) -> np.ndarray:
        return np.array([np.cos(self.heading), np.sin(self.heading)])

    @property
    def velocity(self) -> np.ndarray:
        return self.speed * self.direction  # TODO: slip angle beta should be used here

    @property
    def destination(self) -> np.ndarray:
        if getattr(self, "route", None):
            last_lane = self.road.network.get_lane(self.route[-1])
            return last_lane.position(last_lane.length, 0)
        else:
            return self.position

    @property
    def destination_direction(self) -> np.ndarray:
        if (self.destination != self.position).any():
            return (self.destination - self.position) / np.linalg.norm(self.destination - self.position)
        else:
            return np.zeros((2,))

    def to_dict(self, origin_vehicle: "Vehicle" = None, observe_intentions: bool = True) -> dict:
        d = {
            'presence': 1,
            'x': self.position[0],
            'y': self.position[1],
            'vx': self.velocity[0],
            'vy': self.velocity[1],
            'heading': self.heading,
            'cos_h': self.direction[0],
            'sin_h': self.direction[1],
            'cos_d': self.destination_direction[0],
            'sin_d': self.destination_direction[1]
        }
        if not observe_intentions:
            d["cos_d"] = d["sin_d"] = 0
        if origin_vehicle:
            origin_dict = origin_vehicle.to_dict()
            for key in ['x', 'y', 'vx', 'vy']:
                d[key] -= origin_dict[key]
        return d

