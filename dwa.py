import numpy as np
import math
from math import degrees, pi
import random

from utils import quat2euler
from igibson.utils.utils import quatToXYZW, parse_config
from transforms3d.euler import euler2quat

"""
Fetch Robot 

Max velocity = 1.0 m/s
Webpage: https://fetchrobotics.com/fetch-mobile-manipulator/
White paper on robot: https://fetch3.wpenginepowered.com/wp-content/uploads/2021/06/Fetch-and-Freight-Workshop-Paper.pdf


"""

"""
DWA

Paper: https://www.ri.cmu.edu/pub_files/pub1/fox_dieter_1997_1/fox_dieter_1997_1.pdf
Youtube video with algorithm: https://www.youtube.com/watch?v=tNtUgMBCh2g&t=1s
Youtube video demoing it with attached code: https://www.youtube.com/watch?v=Mdg9ElewwA0
Kinematics of differential drive: https://www.cs.columbia.edu/~allen/F19/NOTES/icckinematics.pdf

"""

class DWA():
    def __init__(self, env):
        # 0.2, 0.1
        # 5.22, 1.73
        # Constants for Fetch robot
        self.WHEEL_RADIUS = 0.0613
        self.WHEEL_AXLE_LENGTH = 0.372
        self.MIN_WHEEL_VELOCITY = -17.4
        self.MAX_WHEEL_VELOCITY = -1 * self.MIN_WHEEL_VELOCITY
        self.MAX_LIN_VEL = 1.06662
        self.MAX_ANG_VEL = 5.7345161
        self.MAX_ACCELERATION = 400
        self.env = env
        self.dt = self.env.action_timestep
        self.num_dt_to_predict = 70
        self.robot = env.robots[0]
        # set the robot gripper as closed (-1) or open (1).
        self.gripper = 1

        # theta  = random.uniform(-0.8 * math.pi, 0.8 * math.pi)
        # rotation = quatToXYZW(euler2quat(0, 0, theta), "wxyz")
        # self.robot.set_orientation(rotation)

    # action map for Fetch robot - 11 controls
    # 0 - forward/back
    # 1 - rotation
    def step(self, state, destination):
        x = state['proprioception'][0]
        y = state['proprioception'][1]
        qx, qy, qz, qw = self.robot.get_orientation()
        _, _, theta = quat2euler(qx, qy, qz, qw)
        # self.test_ori(x, y, destination)
        vr = state['proprioception'][2]
        vl = state['proprioception'][3]
        vl_possible_vels, vr_possible_vels = self.get_dynamic_window_velocities(vl, vr)
        new_vl, new_vr = self.find_best_velocity(x, y, theta, vl_possible_vels, vr_possible_vels, destination)
        action_l, action_a = self.calculate_action_from_wheel_vel(new_vl, new_vr) 

        action = np.zeros((11,))
        action[0] = action_l
        action[1] = action_a
        action[10] = self.gripper
        self.robot.apply_action(action)

    def grab(self):
        self.gripper = -1
        action = np.zeros((11,))
        action[10] = self.gripper
        self.robot.apply_action(action)
        for i in range(10):
            self.env.step(None)
    
    def release(self):
        self.gripper = 1

    # def test_ori(self, x, y, destination):
    #     qx, qy, qz, qw = self.robot.get_orientation()
    #     _, _, theta = quat2euler(qx, qy, qz, qw)
    #     print(theta)
    #     x_dist = destination[0] - x
    #     y_dist = destination[1] - y
    #     optimal_heading =  np.arctan2(y_dist, x_dist)
    #     print(optimal_heading)
    #     if optimal_heading < 0:
    #         optimal_heading = 2 * math.pi + optimal_heading
    #     print(optimal_heading)

    def get_dynamic_window_velocities(self, vl, vr):
        # Velocities that are limited by acceleration and min/max velocities
        vl_start = max(vl - self.MAX_ACCELERATION * self.dt, self.MIN_WHEEL_VELOCITY)
        vl_end = min(vl + self.MAX_ACCELERATION * self.dt, self.MAX_WHEEL_VELOCITY)

        vr_start = max(vr - self.MAX_ACCELERATION * self.dt, self.MIN_WHEEL_VELOCITY)
        vr_end = min(vr + self.MAX_ACCELERATION * self.dt, self.MAX_WHEEL_VELOCITY)

        vl_possible_vels = np.linspace(vl_start, vl_end, 7)
        vr_possible_vels = np.linspace(vr_start, vr_end, 7)

        return vl_possible_vels, vr_possible_vels

    def find_best_velocity(self, x, y, theta, vls, vrs, destination):
        selected_vl = None
        selected_vr = None
        selected_heading = None
        best_theta_predict = None
        best_benefit = -1000000
        HEADING_WEIGHT = 5
        OBSTACLE_WEIGHT = 90
        VELOCITY_WEIGHT = 50

        for vl in vls:
            for vr in vrs:
                positions = self.predict_positions(x, y, theta, vl, vr)
                x_predict = positions[-1][0]
                y_predict = positions[-1][1]
                theta_predict = positions[-1][2]
                heading_metric = self.heading_metric(theta_predict, x, y, destination)
                obstacle_clearance_distance = self.obstacle_clearance_metric(positions)
                velocity_metric = self.velocity_metric(x_predict, y_predict, destination)
                benefit = HEADING_WEIGHT * heading_metric  +  OBSTACLE_WEIGHT * obstacle_clearance_distance + VELOCITY_WEIGHT * velocity_metric
                # print(str(vl) + "-" + str(vr) + " : " + str(benefit) +  "|" +  str(degrees(theta_predict)))
                if benefit > best_benefit:
                    best_benefit = benefit
                    selected_vl = vl
                    selected_vr = vr
                    best_theta_predict = theta_predict
                    selected_heading = heading_metric
        # print("-----------")
        # print(selected_vl, selected_vr, best_benefit, degrees(best_theta_predict), degrees(theta))
        # print("-----------")
        return selected_vl, selected_vr


    # def predict_position(self, x, y, theta, vl, vr):
    #     vl = vl * self.WHEEL_RADIUS
    #     vr = vr * self.WHEEL_RADIUS
    #     t = self.dt * self.num_dt_to_predict
    #     x_predict = None
    #     y_predict = None
    #     theta_predict = None
    #     if (round(vl, 3) == round(vr, 3)):
    #         x_predict = x + vl * t * math.cos(theta)
    #         y_predict = y + vl * t * math.sin(theta)
    #         theta_predict = theta
    #     else:
    #         R = self.WHEEL_AXLE_LENGTH / 2.0 * (vr + vl) / (vr - vl)
    #         delta_theta = (vr - vl) * t / self.WHEEL_AXLE_LENGTH
    #         # Need to review this
    #         x_predict = x + R * (math.sin(delta_theta + theta) - math.sin(theta))
    #         y_predict = y - R * (math.cos(delta_theta + theta) - math.cos(theta))
    #         theta_predict = theta + delta_theta
    #     return x_predict, y_predict, theta_predict

    def predict_positions(self, x, y, theta, vl, vr):
        vl = vl * self.WHEEL_RADIUS
        vr = vr * self.WHEEL_RADIUS
        positions = []
        for step in range(self.num_dt_to_predict):
            x_predict = None
            y_predict = None
            theta_predict = None
            t = self.dt * step
            if (round(vl, 3) == round(vr, 3)):
                x_predict = x + vl * t * math.cos(theta)
                y_predict = y + vl * t * math.sin(theta)
                theta_predict = theta
            else:
                R = self.WHEEL_AXLE_LENGTH / 2.0 * (vr + vl) / (vr - vl)
                delta_theta = (vr - vl) * t / self.WHEEL_AXLE_LENGTH
                # Need to review this
                x_predict = x + R * (math.sin(delta_theta + theta) - math.sin(theta))
                y_predict = y - R * (math.cos(delta_theta + theta) - math.cos(theta))
                theta_predict = theta + delta_theta
            positions.append((x_predict, y_predict, theta_predict))
        return positions

    def heading_metric(self, theta_predict, x, y, destination):
        x_dist = destination[0] - x
        y_dist = destination[1] - y
        optimal_heading =  np.arctan2(y_dist, x_dist)
        optimal_heading = self.normalize_radians(optimal_heading)
        theta_predict = self.normalize_radians(theta_predict)
        theta_1 = abs(optimal_heading - theta_predict)
        theta_2 = 2 * math.pi - theta_1
        # print("####")
        # print("Optimal: " + str(degrees(optimal_heading)))
        # print("theta predict: " + str(degrees(theta_predict)))
        # print("####")
        return -1 * min(theta_1, theta_2)

    # def obstacle_clearance_metric(self, x, y):
    #     SAFE_DIST = 1.8
    #     closest_distance = 100000
    #     for robot in self.env.robots[1:]:
    #         x_robot, y_robot, z_robot = robot.get_position()
    #         distance =  math.dist([x,y], [x_robot, y_robot])
    #         if distance < closest_distance:
    #             closest_distance = distance
    #     metric = SAFE_DIST - closest_distance
    #     if metric < 0:
    #         return 0.0
    #     return -1 * metric

    def obstacle_clearance_metric(self, positions):
        SAFE_DIST = 1.8
        closest_distance = 100000
        for pos_predict in positions:
            for robot in self.env.robots[1:]:
                x_robot, y_robot, z_robot = robot.get_position()
                distance =  math.dist([pos_predict[0], pos_predict[1]], [x_robot, y_robot])
                if distance < closest_distance:
                    closest_distance = distance
        metric = SAFE_DIST - closest_distance
        if metric < 0:
            return 0.0
        return -1 * metric

    def velocity_metric(self, x_predict, y_predict, destination):
        return -1 * math.dist([x_predict, y_predict], destination) 

    def calculate_action_from_wheel_vel(self, vl, vr):
        a = (vr - vl) * self.WHEEL_RADIUS / (-2 * self.WHEEL_AXLE_LENGTH / 2)
        l = self.WHEEL_RADIUS * (vl + vr) / 2
        action_a = a / self.MAX_ANG_VEL
        action_l = l / self.MAX_LIN_VEL
        return action_l, action_a

    def normalize_radians(self, rad):
        # Convert radians to value between 0 and 2 * pi
        rad = rad % (2 * math.pi)
        if rad < 0:
            rad = rad + 2 * math.pi
        return rad

    # right_wheel_joint_vel = (lin_vel - ang_vel * wheel_axle_halflength) / wheel_radius
    # left_wheel_joint_vel = (lin_vel + ang_vel * wheel_axle_halflength) / wheel_radius
    # circle_radius = wheel_axle_halflength * (left_wheel_joint_vel + right_wheel_joint_vel) / (left_wheel_joint_vel - right_wheel_joint_vel)