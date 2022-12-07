import numpy as np
import math

from utils import quat2euler

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
        self.MAX_ACCELERATION = 100
        self.env = env
        self.dt = self.env.action_timestep
        self.num_dt_to_predict = 20
        self.robot = env.robots[0]

    # action map for Fetch robot - 11 controls
    # 0 - forward/back
    # 1 - rotation
    def step(self, state, destination):
        x = state['proprioception'][0]
        y = state['proprioception'][1]
        qx, qy, qz, qw = self.robot.get_orientation()
        _, _, theta = quat2euler(qx, qy, qz, qw)
        vr = state['proprioception'][2]
        vl = state['proprioception'][3]
        vl_possible_vels, vr_possible_vels = self.get_dynamic_window_velocities(vl, vr)
        new_vl, new_vr = self.find_best_velocity(x, y, theta, vl_possible_vels, vr_possible_vels, destination)
        action_l, action_a = self.calculate_action_from_wheel_vel(new_vl, new_vr) 

        action = np.zeros((11,))
        action[0] = action_l
        action[1] = action_a
        self.robot.apply_action(action)

    def get_dynamic_window_velocities(self, vl, vr):
        # Velocities that are limited by acceleration and min/max velocities
        vl_start = max(vl - self.MAX_ACCELERATION * self.dt, self.MIN_WHEEL_VELOCITY)
        vl_end = min(vl + self.MAX_ACCELERATION * self.dt, self.MAX_WHEEL_VELOCITY)

        vr_start = max(vr - self.MAX_ACCELERATION * self.dt, self.MIN_WHEEL_VELOCITY)
        vr_end = min(vr + self.MAX_ACCELERATION * self.dt, self.MAX_WHEEL_VELOCITY)

        vl_possible_vels = np.linspace(vl_start, vl_end, 5)
        vr_possible_vels = np.linspace(vr_start, vr_end, 5)

        return vl_possible_vels, vr_possible_vels

    def find_best_velocity(self, x, y, theta, vls, vrs, destination):
        selected_vl = None
        selected_vr = None
        best_benefit = -1000000
        
        FORWARD_WEIGHT = -100
        OBSTACLE_WEIGHT = 0

        for vl in vls:
            for vr in vrs:
                x_predict, y_predict, theta_predict = self.predict_position(x, y, theta, vl, vr)
                obstacle_clearance_distance = self.obstacle_clearance_metric(x_predict, y_predict)
                forward_progress = self.destination_metric(x_predict, y_predict, destination)
                benefit = FORWARD_WEIGHT * forward_progress  +  OBSTACLE_WEIGHT * obstacle_clearance_distance
                print(benefit)
                if benefit > best_benefit:
                    benefit = best_benefit
                    selected_vl = vl
                    selected_vr = vr
        print(selected_vl, selected_vr)
        return selected_vl, selected_vr


    def predict_position(self, x, y, theta, vl, vr):
        t = self.dt * self.num_dt_to_predict
        x_predict = None
        y_predict = None
        theta_predict = None
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
        return x_predict, y_predict, theta_predict

    def destination_metric(self, x, y, destination):
        return math.dist([x, y], destination)

    def obstacle_clearance_metric(self, x, y):
        closest_distance = 100000
        for robot in self.env.robots[1:]:
            x_robot, y_robot, z_robot = robot.get_position()
            distance =  math.dist([x,y], [x_robot, y_robot])
            if distance < closest_distance:
                closest_distance = distance
        return distance

    def calculate_action_from_wheel_vel(self, vl, vr):
        a = (vr - vl) * self.WHEEL_RADIUS / (-2 * self.WHEEL_AXLE_LENGTH / 2)
        l = self.WHEEL_RADIUS * (vl + vr) / 2
        action_a = a / self.MAX_ANG_VEL
        action_l = l / self.MAX_LIN_VEL
        return action_l, action_a

    # right_wheel_joint_vel = (lin_vel - ang_vel * wheel_axle_halflength) / wheel_radius
    # left_wheel_joint_vel = (lin_vel + ang_vel * wheel_axle_halflength) / wheel_radius
    # circle_radius = wheel_axle_halflength * (left_wheel_joint_vel + right_wheel_joint_vel) / (left_wheel_joint_vel - right_wheel_joint_vel)