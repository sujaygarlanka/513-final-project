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


def main():
    ############################################# Setting environment ############################################
    # Env size operating size
    # x -> 0 - 25
    # y -> -10 - 10
    env = iGibsonEnv(
        config_file="dwa_config.yaml", mode="headless", action_timestep=1.0 / 30.0, physics_timestep=1.0 / 120.0, use_pb_gui=True
    ) 

    robot = env.robots[0]
    robot_x, robot_y = 0, 0
    env.set_pos_orn_with_z_offset(robot, [robot_x, robot_y, 0], [0, 0, 0])
    
    # Instantiate mid fielders
    mid_fielders = Defenders(env, (0, 10), (-5, 5), 5)
    # Instantiate defenders and goalie
    defenders = Defenders(env, (15, 20), (-5, 5), 0)
    # Instantiate objects
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    objects = [
        ("soccerball.urdf", (12.000000, 0.0, 0.000000), (0.000000, 0.000000, 0.707107, 0.707107)),
    ]
    for item in objects:
        fpath = item[0]
        pos = item[1]
        orn = item[2]
        item_ob = ArticulatedObject(fpath, scale=1, rendering_params={"use_pbr": False, "use_pbr_mapping": False})
        env.simulator.import_object(item_ob)
        item_ob.set_position(pos)
        item_ob.set_orientation(orn)

    ############################################# Running program ############################################
    waypoints = [(12, 0), (22, 0), (22, 5)]
    curr_destination = 0
    state = env.get_state()
    dwa_planner = DWA(env)
    while(True):
        # actionStep = env.simulator.gen_vr_robot_action()
        # human.apply_action(actionStep)

        mid_fielders.step()
        defenders.step()
        dwa_planner.step(state, waypoints[curr_destination])
        state, _, _, _ = env.step(None)
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
        x = state['proprioception'][0]
        y = state['proprioception'][1]
        # print(state['proprioception'][3])
        # print(state['proprioception'][4])
        if x > waypoints[0][0] - 0.5 and x < waypoints[0][0] + 0.5 and y > waypoints[0][1] - 0.5 and y < waypoints[0][1] + 0.5:
            break
        break

if __name__ == "__main__":
    main()


