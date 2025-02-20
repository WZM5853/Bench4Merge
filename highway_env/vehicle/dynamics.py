from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from highway_env.road.road import Road
from highway_env.types import Vector
from highway_env.vehicle.kinematics import Vehicle


class BicycleVehicle(Vehicle):
    """
    A dynamical bicycle model, with tire friction and slipping.

    See Chapter 2 of Lateral Vehicle Dynamics. Vehicle Dynamics and Control. Rajamani, R. (2011)
    """
    MASS: float = 1  # [kg]
    LENGTH_A: float = Vehicle.LENGTH / 2  # [m]
    LENGTH_B: float = Vehicle.LENGTH / 2  # [m]
    INERTIA_Z: float = 1/12 * MASS * (Vehicle.LENGTH ** 2 + 3 * Vehicle.WIDTH ** 2)  # [kg.m2]
    FRICTION_FRONT: float = 15.0 * MASS  # [N]
    FRICTION_REAR: float = 15.0 * MASS  # [N]

    MAX_ANGULAR_SPEED: float = 2 * np.pi  # [rad/s]
    MAX_SPEED: float = 15  # [m/s]

    def __init__(self, road: Road, position: Vector, heading: float = 0, speed: float = 0) -> None:
        super().__init__(road, position, heading, speed)
        self.lateral_speed = 0
        self.yaw_rate = 0
        self.theta = None
        self.A_lat, self.B_lat = self.lateral_lpv_dynamics()



