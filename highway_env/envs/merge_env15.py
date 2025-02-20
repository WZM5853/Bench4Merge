import numpy as np
import random
import csv
import pandas as pd
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle, MDPVehicle
from highway_env.road.objects import Obstacle
# from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.behavior_1 import IDMVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.envs.logdata import CircularCSVLogger, CircularCSVLogger2
import matplotlib.pyplot as plt

import csv
# from os.path import exists



class MergeEnv15(AbstractEnv):

    """
    A highway merge negotiation environment.
    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """

    COLLISION_REWARD: float = -80
    RIGHT_LANE_REWARD: float = 0.02
    HIGH_SPEED_REWARD: float = 0.15
    MERGING_SPEED_REWARD: float = -1
    LANE_CHANGE_REWARD: float = -0.05
    OVER_SPOT_REWARD: float = 10
    STOP_REWARD: float = -5

    onramp = True

    # f3 = open('result/trajecotry.csv', 'w', encoding='utf-8', newline="")
    # csv_write = csv.writer(f3)
    # csv_write.writerow(['Time', 'Vehicle-ID', 'X', 'Y', 'Speed', 'Heading'])
    t=0.1
    # f4 = open('result/temporary_trajecotry.csv', 'w', encoding='utf-8', newline="")
    # csv_write = csv.writer(f4)
    # csv_write.writerow(['Time', 'Vehicle-ID', 'X', 'Y', 'Speed', 'Heading'])
    f5 = open('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/map.csv','w',encoding='utf-8', newline="")
    csv_write = csv.writer(f5)
    csv_write.writerow(['Road-ID', 'x', 'y'])
    for x1 in range(115):
        csv_write = csv.writer(f5)
        csv_write.writerow([1,x1,7])
    for x2 in range(20):
        csv_write = csv.writer(f5)
        csv_write.writerow([1,x2+115,7-0.175*x2])
    for x3 in range(155):
        csv_write = csv.writer(f5)
        csv_write.writerow([2,x3+135,3.5])
    for x4 in range(290):
        csv_write = csv.writer(f5)
        csv_write.writerow([2,x4,3.5])

    f6 = open('/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/data/trajectory.csv','w',encoding='utf-8', newline="")
    csv_write = csv.writer(f6)
    csv_write.writerow(['time', 'vehicle-ID', 'x', 'y', 'width', 'height','v_x','v_y','acc_x','acc_y','road_ID','heading'])


    def default_config(self) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "ContinuousAction",
            },
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "reward_speed_range": [1, 7],
            "offroad_terminal": False
        })
        


        return config

    def _reward(self, action: int) -> float:
        """
        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions
        But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.
        :param action: the action performed
        :return: the reward of the state-action transition
        """


        reward = self.COLLISION_REWARD * self.vehicle.crashed \
                 + self.HIGH_SPEED_REWARD * self.vehicle.speed

        if self.vehicle.speed < 2:
            reward += 2 * (self.vehicle.speed - 2)
        if self.vehicle.speed >= 2 and self.vehicle.speed < 11:
            reward += 1.5 * (np.cos(((np.pi)/9)*(self.vehicle.speed - 2) - np.pi) + 1)

        if self.vehicle.position[1] > 7.5:
            reward -= 2
        if self.vehicle.position[1] < 3.0:
            reward -= 2
        if self.vehicle.position[0] > 138:
            reward += 0.9
            if self.vehicle.position[1] > 4.0:
                reward -=5
        
        #merging reward
        distance = self.vehicle.position[0] - 100
        merging_reward = 1 - np.exp(-((distance - 40) * (distance - 40))/(100))

        if self.vehicle.position[1] < 5:
            if self.vehicle.position[0] < 140:
                reward += 2 * merging_reward


        v_list = []
        print(self.vehicle.position)
        
        #记录轨迹，训练时关闭
        for i in range(len(self.road.vehicles)):
            logger = CircularCSVLogger('cut_in2/data/trajectory_10.csv')


            delta_f = self.road.vehicles[i].action['steering']
            beta = np.arctan(1 / 2 * np.tan(delta_f))
            v_x = self.road.vehicles[i].speed * np.cos(self.road.vehicles[i].heading + beta)
            v_y = self.road.vehicles[i].speed * np.sin(self.road.vehicles[i].heading + beta)
            a_x = self.road.vehicles[i].action['acceleration'] * np.cos(self.road.vehicles[i].heading + beta)
            a_y = self.road.vehicles[i].action['acceleration'] * np.sin(self.road.vehicles[i].heading + beta)
            if self.road.vehicles[i].position[1] > 6 :
                Road_ID = 1
            else:
                Road_ID = 2

            self.csv_write = csv.writer(self.f6)
            self.csv_write.writerow([self.t,
                            self.road.vehicles[i].ID,
                            self.road.vehicles[i].position[0],
                            self.road.vehicles[i].position[1],
                            Vehicle.LENGTH,
                            Vehicle.WIDTH,
                            v_x,
                            v_y,
                            a_x,
                            a_y,
                            Road_ID,
                            self.road.vehicles[i].heading])

            logger.add_row([self.t,
                            self.road.vehicles[i].ID,
                            self.road.vehicles[i].position[0],
                            self.road.vehicles[i].position[1],
                            Vehicle.LENGTH,
                            Vehicle.WIDTH,
                            v_x,
                            v_y,
                            a_x,
                            a_y,
                            Road_ID,
                            self.road.vehicles[i].heading])

        for i in range(len(self.road.vehicles)):
            logger = CircularCSVLogger2('cut_in2/data/trajectory_1.csv')
            delta_f = self.road.vehicles[i].action['steering']
            beta = np.arctan(1 / 2 * np.tan(delta_f))
            v_x = self.road.vehicles[i].speed * np.cos(self.road.vehicles[i].heading + beta)
            v_y = self.road.vehicles[i].speed * np.sin(self.road.vehicles[i].heading + beta)
            a_x = self.road.vehicles[i].action['acceleration'] * np.cos(self.road.vehicles[i].heading + beta)
            a_y = self.road.vehicles[i].action['acceleration'] * np.sin(self.road.vehicles[i].heading + beta)
            if self.road.vehicles[i].position[1] > 6 :
                Road_ID = 1
            else:
                Road_ID = 2
            logger.add_row([self.t,
                            self.road.vehicles[i].ID,
                            self.road.vehicles[i].position[0],
                            self.road.vehicles[i].position[1],
                            Vehicle.LENGTH,
                            Vehicle.WIDTH,
                            v_x,
                            v_y,
                            a_x,
                            a_y,
                            Road_ID,
                            self.road.vehicles[i].heading])
        
        self.t=0.1+self.t



        # Altruistic penalty
        for vehicle in self.road.vehicles:
            if vehicle.lane_index == ("b", "c", 1)and isinstance(vehicle, ControlledVehicle):
                reward += self.MERGING_SPEED_REWARD * (1 - (vehicle.speed)/10)
                reward -= 1
                if self.vehicle.speed < 4:
                    reward += 1


        for vehicle in self.road.vehicles:
            if vehicle.lane_index == ("b", "c", 2)and isinstance(vehicle, ControlledVehicle):
                reward += 2


        with open('last_action.csv', "r") as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                last_action = row[0]
        current_action = self.vehicle.action['steering']
        difference_steering = float(current_action)-float(last_action)
        # print(difference_steering)
        reward -= ((6/np.pi)*np.square(difference_steering)) * 5


        with open('last_speed.csv', 'r') as csvfile2:
            csvreader2 = csv.reader(csvfile2)
            for row in csvreader2:
                last_speed = row[0]
        current_speed = self.vehicle.speed
        difference_speed = float(current_speed)-float(last_speed)
        # print(abs(difference_speed))
        reward -= abs(difference_speed)


        print(self.vehicle.speed, reward)

        current_action = self.vehicle.action['steering']
        f = open('last_action.csv','w',encoding='utf-8')
        csv_write = csv.writer(f)
        csv_write.writerow([current_action])
        f.close

        current_speed = self.vehicle.speed
        f2 = open('last_speed.csv', 'w', encoding='utf-8')
        csv_write2 = csv.writer(f2)
        csv_write2.writerow([current_speed])
        f2.close

        

        return reward

    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return self.vehicle.crashed or self.vehicle.position[0] > 150 or self.vehicle.position[1] > 8.5 or self.vehicle.position[1] < 2 or self.vehicle.speed < 0 or self.t>2.0
        # return self.vehicle.crashed or self.vehicle.position[0] > 150 or self.vehicle.position[1] > 8.5 or self.vehicle.position[1] < 2 or self.vehicle.speed < 0


    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.
        :return: the road
        """
        net = RoadNetwork()

        # Highway lanes
        ends = [50, 50, 40, 150]  # Before, converging, merge, after
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH]
        p=StraightLane.DEFAULT_WIDTH
        line_type = [[c, s], [n, c]]
        line_type_merge = [[c, s], [n, s]]
        line_type_straight=[c, c]
        for i in range(1):
            net.add_lane("a", "b", StraightLane([0, p], [sum(ends[:2])+5, p], line_types=line_type_straight))
            # net.add_lane("a", "b", StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type_straight))
            net.add_lane("b", "c", StraightLane([sum(ends[:2])+5, p], [sum(ends[:3]), p], line_types=line_type_merge[i]))
            net.add_lane("c", "d", StraightLane([sum(ends[:3]), p], [sum(ends), p], line_types=line_type_straight))

        # Merging lane
        amplitude = 1.5
        ljk = StraightLane([0, 3.0 + 3.5 + 3.5], [ends[0], 3.0 + 3.5 + 3.5], line_types=[c, c], forbidden=True)
        lkb = SineLane(ljk.position(ends[0], -amplitude), ljk.position(sum(ends[:2]), -amplitude),
                       amplitude, 2 * np.pi / (2*ends[1]), np.pi / 2, line_types=[c, c], forbidden=True)
        lbc = StraightLane(lkb.position(ends[1], 0), lkb.position(ends[1], 0) + [ends[2], 0],
                           line_types=[n, c], forbidden=True)

        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        :return: the ego-vehicle
        """
        road = self.road

        ret=random.randint(70,90)
        pet=-5
        center=random.randint(8,12)
        spd=random.randint(5,8)
        vel=random.randint(4,7)
        type = 1

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        road.vehicles.append(
            other_vehicles_type(ID=1, road=road, position=road.network.get_lane(("a", "b", 0)).position(ret - center * 4, 0), speed=vel))
        # print("generate vehicle13", "position = ", ret - center * 4, "speed = ", vel)
        road.vehicles.append(
            other_vehicles_type(ID=2, road=road, position=road.network.get_lane(("a", "b", 0)).position(ret - center * 3, 0), speed=vel))
        # print("generate vehicle12", "position = ", ret - center * 3, "speed = ", vel)
        road.vehicles.append(
            other_vehicles_type(ID=3, road=road, position=road.network.get_lane(("a", "b", 0)).position(ret - center * 2, 0), speed=vel))
        # print("generate vehicle11", "position = ", ret - center * 2, "speed = ", vel)
        road.vehicles.append(
            other_vehicles_type(ID=4, road=road, position=road.network.get_lane(("a", "b", 0)).position(ret - center * 1, 0), speed=vel))
        # print("generate vehicle10", "position = ", ret - center * 1, "speed = ", vel)
        road.vehicles.append(
            other_vehicles_type(ID=5, road=road, position=road.network.get_lane(("a", "b", 0)).position(ret, 0), speed=vel))
        # print("generate vehicle9", "position = ", ret, "speed = ", vel)
        road.vehicles.append(
            other_vehicles_type(ID=6, road=road, position=road.network.get_lane(("a", "b", 0)).position(ret + center * 1, 0), speed=vel))
        # print("generate vehicle8", "position = ", ret + center * 1, "speed = ", vel)
        road.vehicles.append(
            other_vehicles_type(ID=7, road=road, position=road.network.get_lane(("a", "b", 0)).position(ret + center * 2, 0), speed=vel))
        # print("generate vehicle7", "position = ", ret + center * 2, "speed = ", vel)
        road.vehicles.append(
            other_vehicles_type(ID=8, road=road, position=road.network.get_lane(("a", "b", 0)).position(ret + center * 3, 0), speed=vel))
        # print("generate vehicle6", "position = ", ret + center * 3, "speed = ", vel)
        road.vehicles.append(
            other_vehicles_type(ID=9, road=road, position=road.network.get_lane(("a", "b", 0)).position(ret + center * 4, 0), speed=vel))
        # print("generate vehicle5", "position = ", ret + center * 4, "speed = ", vel)
        road.vehicles.append(
            other_vehicles_type(ID=10, road=road, position=road.network.get_lane(("a", "b", 0)).position(ret + center * 5, 0), speed=vel))
        # print("generate vehicle4", "position = ", ret + center * 5, "speed = ", vel)
        road.vehicles.append(
            other_vehicles_type(ID=11, road=road, position=road.network.get_lane(("a", "b", 0)).position(ret + center * 6, 0), speed=vel))
        # print("generate vehicle3", "position = ", ret + center * 6, "speed = ", vel)
        road.vehicles.append(
            other_vehicles_type(ID=12, road=road, position=road.network.get_lane(("a", "b", 0)).position(ret + center * 7, 0), speed=vel))
        # print("generate vehicle2", "position = ", ret + center * 7, "speed = ", vel)
        road.vehicles.append(
            other_vehicles_type(ID=13, road=road, position=road.network.get_lane(("a", "b", 0)).position(ret + center * 8, 0), speed=spd))
        # print("generate vehicle1", "position = ", ret + center * 8, "speed = ", vel)



        ego_vehicle = self.action_type.vehicle_class(0, road, road.network.get_lane(("b", "c", 0)).position(pet-7, 3.5), speed=5)

        road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle
        ego_vehicle.SPEED_MIN = 0.5
        ego_vehicle.SPEED_MAX = 8






register(
    id='merge-v15',
    entry_point='highway_env.envs:MergeEnv15',
)