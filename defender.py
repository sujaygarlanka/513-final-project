from igibson.utils.utils import quatToXYZW, parse_config
from transforms3d.euler import euler2quat
from igibson.robots.behavior_robot import BehaviorRobot
from igibson.robots.fetch import Fetch

import numpy as np
import random
import math

def quat2euler(x, y, z, w):
        """
        https://automaticaddison.com/how-to-convert-a-quaternion-into-euler-angles-in-python/
        """

        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

class Defenders():
    # action map for Behavior robot
    # 0 - forward/back
    # 1 - left/right
    # 2 - up/down
    # 3 - roll
    # 4 - pitch
    # 5 - yaw

    def __init__(self, env, x_range, y_range, num_defenders):
        self.env = env
        self.x_range = x_range
        self.y_range = y_range
        self.num_defenders = num_defenders
        self.config = parse_config("dwa_config.yaml")
        self.defenders = []
        self.num_steps_counter = 0
        self.num_steps_update = 100
        self.actions = np.zeros((num_defenders,28))
        self.init()
    
    def init(self):
        for _ in range(self.num_defenders):
            defender = BehaviorRobot()
            self.defenders.append(defender)
            self.env.simulator.import_object(defender)
            rand_x = random.randrange(self.x_range[0], self.x_range[1])
            rand_y = random.randrange(self.y_range[0], self.y_range[1])
            self.env.set_pos_orn_with_z_offset(defender, [rand_x, rand_y, 0], [0, 0, 0])

    def step(self):
        speed_lower_bound = -0.2
        speed_upper_bound = 0.2
        
        for idx, d in enumerate(self.defenders):
            # x, y, z, w = d.get_orientation()
            # x, y, z = quat2euler(x, y, z, w)
            # print(x, y, z)
            if self.num_steps_counter == self.num_steps_update:
                forward_back = random.uniform(speed_lower_bound, speed_upper_bound)
                theta  = random.uniform(-0.8 * math.pi, 0.8 * math.pi)
                rotation = quatToXYZW(euler2quat(0, 0, theta), "wxyz")
                d.set_orientation(rotation)
                self.actions[idx][0] = 0.1
            self.keep_in_range(d)
            d.apply_action(self.actions[idx])
        
        if self.num_steps_counter == self.num_steps_update:
            self.num_steps_counter = 0
        self.num_steps_counter += 1


    def keep_in_range(self, d):
        [x, y, z] = d.get_position()
        update_orientation = False
        x_rot, y_rot, z_rot, w_rot = d.get_orientation()
        x_rot, y_rot, z_rot = quat2euler(x_rot, y_rot, z_rot, w_rot)
        if x < self.x_range[0]:
            x = self.x_range[0]
            update_orientation = True
        elif x > self.x_range[1]:
            x = self.x_range[1]
            update_orientation = True
        elif y < self.y_range[0]:
            y = self.y_range[0]
            update_orientation = True
        elif y > self.y_range[1]:
            y = self.y_range[1]
            update_orientation = True
            
        if update_orientation:
            x_rot, y_rot, z_rot, w_rot = d.get_orientation()
            x_rot, y_rot, z_rot = quat2euler(x_rot, y_rot, z_rot, w_rot)
            z_rot += math.pi
            rotation = quatToXYZW(euler2quat(x_rot, y_rot, z_rot), "wxyz")
            d.set_orientation(rotation)
        
        d.set_position([x, y, z])