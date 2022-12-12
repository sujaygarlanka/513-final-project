import argparse
import os

import numpy as np

import igibson
from igibson.envs.igibson_env import iGibsonEnv
from igibson.utils.motion_planning_wrapper import MotionPlanningWrapper
import pybullet as p
import pybullet_data
from time import time
from time import sleep
from igibson.objects.ycb_object import YCBObject
from igibson.objects.articulated_object import URDFObject, URDFObject
from igibson.robots.behavior_robot import BehaviorRobot
import math
from multiprocessing import Process
import logging
from igibson.utils.utils import quatToXYZW, parse_config
from transforms3d.euler import euler2quat

from igibson.objects.articulated_object import ArticulatedObject
from igibson.objects.pedestrian import  Pedestrian
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from igibson.render.mesh_renderer.mesh_renderer_vr import VrSettings
from igibson.scenes.empty_scene import EmptyScene
# HDR files for PBR rendering
from igibson.simulator_vr import SimulatorVR
from igibson.simulator_vr import Simulator
from igibson.utils.utils import parse_config
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from defender import Defenders
from dwa import DWA

import random
from arm_controller import ArmController
import time


def main():
    ############################################# Setting environment ############################################
    robot_x, robot_y = 0, 0
    # mid_n = random.randint(1, 3)
    # df_n = random.randint(1, 3)
    mid_n = 5
    df_n = 5
    
    # soccer_y = random.randrange(-10, 10)
    ball = (12, 0)
    midpoint = (random.randrange(20, 23), random.randrange(-2, 2))
    midpoint = (23, 0)
    
    endpoint = (random.randrange(23, 26), random.randrange(-10, 10))
    endpoint = (22, -3)
    
    # Env size operating size
    # x -> 0 - 25
    # y -> -10 - 10
    env = iGibsonEnv(
        config_file="dwa_config.yaml", mode="headless", action_timestep=1.0 / 30.0, physics_timestep=1.0 / 120.0, use_pb_gui=True
    ) 
   
    robot = env.robots[0]
    theta  = random.uniform(-0.3 * math.pi, 0.3 * math.pi)
    # rotation = quatToXYZW(euler2quat(0, 0, theta), "wxyz")
    # robot.set_orientation(rotation)
    env.set_pos_orn_with_z_offset(robot, [robot_x, robot_y, 0], [0, 0, 0.61])
    

    # config = parse_config("dwa_config.yaml")
    # human = BehaviorRobot(**config["human"])
    # env.simulator.import_object(human)
    # env.simulator.switch_main_vr_robot(human)
    # env.set_pos_orn_with_z_offset(human, [-2, 0, 0], [0, 0, 0])
    
    # Instantiate mid fielders
    mid_fielders = Defenders(env, (2, 10), (-3, 3), mid_n)
    # Instantiate defenders and goalie
    # defenders = Defenders(env, (15, 20), (-5, 5), df_n)
    # Instantiate objects
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    objects = [
        ("soccerball.urdf", (ball[0], ball[1], 1.000000), (0.000000, 0.000000, 0.707107, 0.707107), 0.3),
    ]
    scene_objects = []
    for item in objects:
        fpath = item[0]
        pos = item[1]
        orn = item[2]
        scale = item[3]
        item_ob = ArticulatedObject(fpath, scale=scale, rendering_params={"use_pbr": False, "use_pbr_mapping": False})
        env.simulator.import_object(item_ob)
        item_ob.set_position(pos)
        item_ob.set_orientation(orn)
        scene_objects.append(item_ob)
    ############################################# Running program ############################################
    waypoints = [ball, midpoint, endpoint]
    curr_destination = 0
    state = env.get_state()
    dwa_planner = DWA(env)
    arm_controller = ArmController(robot)
    SAFE_DIST = 1.8
    GOAL_DIST = 0.5
    COLL_DIST = 0.5

    benefits = [[] for i in range(mid_n + df_n + 1)]
    time_step = 0
    arrive_timing = []
    input()
    while(True):
        # actionStep = env.simulator.gen_vr_robot_action()
        # human.apply_action(actionStep)
        # robot.apply_action([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        mid_fielders.step()
        # defenders.step()
        dwa_planner.step(state, waypoints[curr_destination])
        if curr_destination == 1:
            pos = robot.get_eef_position(arm="default")
            pos[2] = pos[2] - 0.18
            scene_objects[0].set_position(pos)
        state, _, _, _ = env.step(None)
        
        x = state['proprioception'][0]
        y = state['proprioception'][1]

        # robustness for each requirement
        for id, d in enumerate(env.robots[1:]):
            x_robot, y_robot, _ = d.get_position()
            benefits[id].append(math.dist([x, y], [x_robot, y_robot]) - COLL_DIST)

        benefits[-1].append(GOAL_DIST - \
            math.dist([waypoints[curr_destination][0], waypoints[curr_destination][1]], [x, y]))

        if math.dist([waypoints[curr_destination][0], waypoints[curr_destination][1]], [x, y]) < GOAL_DIST:
            if curr_destination == 0:
                arm_controller.move_to_location([waypoints[0][0], waypoints[0][1], 0.3])
                time.sleep(2)
                arm_controller.untuck()
                mid_fielders.move_defenders((15, 20), (-5, 5))
            curr_destination += 1
            arrive_timing.append(time_step)

        if curr_destination == len(waypoints):
            break

        time_step += 1
    
    # overall robustness
    robustness = min(map(min, benefits[:mid_n + df_n]))
    print(robustness)
    for i in range(curr_destination):
        robustness = min(robustness, benefits[-1][arrive_timing[i]])
        print(benefits[-1][arrive_timing[i]])
    return robustness

if __name__ == "__main__":
    # metrics = []
    # for i in range(100):
    #     print("------------------------------------------------------------------")
    #     print("Running at stage"+str(i)+":")
        
    #     result = main()
    #     with open("results.txt", "a") as f:
    #         f.write(str(result)+"\n")
        
    #     metrics.append(result)
    #     time.sleep(1)
    #     print("Robustness Result: "+str(metrics[-1]))
    # print(min(metrics))
    main()


# data = state['occupancy_grid']
# # data = env.scene.floor_map[0]
# data = np.array(data)
# data = data.reshape(128, 128)
# length = data.shape[0]
# width = data.shape[1]
# x, y = np.meshgrid(np.arange(length), np.arange(width))

# fig = plt.figure()
# ax = fig.add_subplot(1,1,1, projection='3d')
# ax.plot_surface(x, y, data)
# plt.show()