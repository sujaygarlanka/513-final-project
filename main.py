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
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from igibson.render.mesh_renderer.mesh_renderer_vr import VrSettings
from igibson.scenes.empty_scene import EmptyScene
# HDR files for PBR rendering
from igibson.simulator_vr import SimulatorVR
from igibson.simulator_vr import Simulator
from igibson.utils.utils import parse_config
import time

def main():
    
    robot_x, robot_y = 0, 0
    human_x, human_y = 3, 3
    env = iGibsonEnv(
        config_file="dwa_config.yaml", mode="headless", action_timestep=1.0 / 30.0, physics_timestep=1.0 / 120.0, use_pb_gui=True
    ) 

    print("**************loading objects***************")
    print(os.path.join(igibson.configs_path, "fetch_motion_planning_3d_lsi.yaml"))
    config = parse_config(os.path.join(igibson.configs_path, "fetch_motion_planning_3d_lsi.yaml"))
    human = BehaviorRobot(**config["human"])
    env.simulator.import_object(human)
    # env.simulator.switch_main_vr_robot(human)
    env.set_pos_orn_with_z_offset(human, [human_x, human_y, 0], [0, 0, 0])
    env.set_pos_orn_with_z_offset(env.robots[0], [robot_x, robot_y, 0], [0, 0, 0])
    
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    objects = [
        ("jenga/jenga.urdf", (1.300000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000, 0.707107)),
        ("duck_vhacd.urdf", (1.050000, -0.500000, 0.700000), (0.000000, 0.000000, 0.707107, 0.707107)),
        ("sphere_small.urdf", (-1.0, 0.0, 2), (0.000000, 0.000000, 0.707107, 0.707107)),
    ]

    for item in objects:
        fpath = item[0]
        pos = item[1]
        orn = item[2]
        item_ob = ArticulatedObject(fpath, scale=1, rendering_params={"use_pbr": False, "use_pbr_mapping": False})
        env.simulator.import_object(item_ob)
        item_ob.set_position(pos)
        item_ob.set_orientation(orn)
    print(env.robots)
    print("**************running program***************")
    while(True):
        # actionStep = env.simulator.gen_vr_robot_action()
        # human.apply_action(actionStep)

        action = env.action_space.sample()
        # print(env.action_space.shape)
        state, reward, done, _ = env.step(action)
        print(state)
        if done:
            break

if __name__ == "__main__":
   main()
