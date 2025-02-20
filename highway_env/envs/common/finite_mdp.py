from typing import TYPE_CHECKING, Optional

import numpy as np

from highway_env import utils
from highway_env.vehicle.kinematics import Vehicle

if TYPE_CHECKING:
    from highway_env.envs import AbstractEnv


def finite_mdp(env: 'AbstractEnv',
               time_quantization: float = 1.,
               horizon: float = 10.) -> object:
    """
    Time-To-Collision (TTC) representation of the state.

    The state reward is defined from a occupancy grid over different TTCs and lanes. The grid cells encode the
    probability that the ego-vehicle will collide with another vehicle if it is located on a given lane in a given
    duration, under the hypothesis that every vehicles observed will maintain a constant speed (including the
    ego-vehicle) and not change lane (excluding the ego-vehicle).

    For instance, in a three-lane road with a vehicle on the left lane with collision predicted in 5s the grid will
    be:
    [0, 0, 0, 0, 1, 0, 0,
     0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0]
    The TTC-state is a coordinate (lane, time) within this grid.

    If the ego-vehicle has the ability to change its speed, an additional layer is added to the occupancy grid
    to iterate over the different speed choices available.

    Finally, this state is flattened for compatibility with the FiniteMDPEnv environment.

    :param AbstractEnv env: an environment
    :param time_quantization: the time quantization used in the state representation [s]
    :param horizon: the horizon on which the collisions are predicted [s]
    """
    # Compute TTC grid
    grid = compute_ttc_grid(env, time_quantization, horizon)

    # Compute current state
    grid_state = (env.vehicle.speed_index, env.vehicle.lane_index[2], 0)
    state = np.ravel_multi_index(grid_state, grid.shape)

    # Compute reward function
    v, l, t = grid.shape
    lanes = np.arange(l)/max(l - 1, 1)
    speeds = np.arange(v)/max(v - 1, 1)
    state_reward = \
        + env.COLLISION_REWARD * grid \
        + env.RIGHT_LANE_REWARD * np.tile(lanes[np.newaxis, :, np.newaxis], (v, 1, t)) \
        + env.HIGH_SPEED_REWARD * np.tile(speeds[:, np.newaxis, np.newaxis], (1, l, t))
    state_reward = np.ravel(state_reward)
    action_reward = [env.LANE_CHANGE_REWARD, 0, env.LANE_CHANGE_REWARD, 0, 0]
    reward = np.fromfunction(np.vectorize(lambda s, a: state_reward[s] + action_reward[a]),
                             (np.size(state_reward), np.size(action_reward)),  dtype=int)

    # Compute terminal states
    collision = grid == 1
    end_of_horizon = np.fromfunction(lambda h, i, j: j == grid.shape[2] - 1, grid.shape, dtype=int)
    terminal = np.ravel(collision | end_of_horizon)


def compute_ttc_grid(env: 'AbstractEnv',
                     time_quantization: float,
                     horizon: float,
                     vehicle: Optional[Vehicle] = None) -> np.ndarray:
    """
    Compute the grid of predicted time-to-collision to each vehicle within the lane

    For each ego-speed and lane.
    :param env: environment
    :param time_quantization: time step of a grid cell
    :param horizon: time horizon of the grid
    :param vehicle: the observer vehicle
    :return: the time-co-collision grid, with axes SPEED x LANES x TIME
    """
    vehicle = vehicle or env.vehicle
    road_lanes = env.road.network.all_side_lanes(env.vehicle.lane_index)
    grid = np.zeros((vehicle.SPEED_COUNT, len(road_lanes), int(horizon / time_quantization)))
    for speed_index in range(grid.shape[0]):
        ego_speed = vehicle.index_to_speed(speed_index)
        for other in env.road.vehicles:
            if (other is vehicle) or (ego_speed == other.speed):
                continue
            margin = other.LENGTH / 2 + vehicle.LENGTH / 2
            collision_points = [(0, 1), (-margin, 0.5), (margin, 0.5)]
            for m, cost in collision_points:
                distance = vehicle.lane_distance_to(other) + m
                other_projected_speed = other.speed * np.dot(other.direction, vehicle.direction)
                time_to_collision = distance / utils.not_zero(ego_speed - other_projected_speed)
                if time_to_collision < 0:
                    continue
                if env.road.network.is_connected_road(vehicle.lane_index, other.lane_index,
                                                      route=vehicle.route, depth=3):
                    # Same road, or connected road with same number of lanes
                    if len(env.road.network.all_side_lanes(other.lane_index)) == len(env.road.network.all_side_lanes(vehicle.lane_index)):
                        lane = [other.lane_index[2]]
                    # Different road of different number of lanes: uncertainty on future lane, use all
                    else:
                        lane = range(grid.shape[1])
                    # Quantize time-to-collision to both upper and lower values
                    for time in [int(time_to_collision / time_quantization),
                                 int(np.ceil(time_to_collision / time_quantization))]:
                        if 0 <= time < grid.shape[2]:
                            # TODO: check lane overflow (e.g. vehicle with higher lane id than current road capacity)
                            grid[speed_index, lane, time] = np.maximum(grid[speed_index, lane, time], cost)
    return grid

