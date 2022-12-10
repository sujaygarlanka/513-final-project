import argparse
import logging
import os
import time

import numpy as np
import pybullet as p

import igibson
from igibson.external.pybullet_tools.utils import (
    get_max_limits,
    get_min_limits,
    get_sample_fn,
    joints_from_names,
    set_joint_positions,
)
from igibson.objects.visual_marker import VisualMarker
from igibson.robots.fetch import Fetch
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.utils.utils import l2_distance, parse_config, restoreState

class ArmController():

    def __init__(self, fetch_robot):
        self.fetch_robot = fetch_robot
        body_ids = self.fetch_robot.get_body_ids()
        self.robot_id = body_ids[0]
        robot_joint_names = [
            "r_wheel_joint",
            "l_wheel_joint",
            "torso_lift_joint",
            "head_pan_joint",
            "head_tilt_joint",
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "upperarm_roll_joint",
            "elbow_flex_joint",
            "forearm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
            "r_gripper_finger_joint",
            "l_gripper_finger_joint",
        ]
        arm_joints_names = [
            "torso_lift_joint",
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "upperarm_roll_joint",
            "elbow_flex_joint",
            "forearm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
        ]

        # Indices of the joints of the arm in the vectors returned by IK and motion planning (excluding wheels, head, fingers)
        self.robot_arm_indices = [robot_joint_names.index(arm_joint_name) for arm_joint_name in arm_joints_names]

        # PyBullet ids of the joints corresponding to the joints of the arm
        self.arm_joint_ids = joints_from_names(self.robot_id, arm_joints_names)
        self.all_joint_ids = joints_from_names(self.robot_id, robot_joint_names)

        arm_default_joint_positions = (
            0.10322468280792236,
            -1.414019864768982,
            1.5178184935241699,
            0.8189625336474915,
            2.200358942909668,
            2.9631312579803466,
            -1.2862852996643066,
            0.0008453550418615341,
        )

        self.robot_default_joint_positions = (
            [0.0, 0.0]
            + [arm_default_joint_positions[0]]
            + [0.0, 0.0]
            + list(arm_default_joint_positions[1:])
            + [0.01, 0.01]
        )

    def move_to_location(self, loc):
        threshold = 0.03
        max_iter = 100
        joint_pos = self.accurate_calculate_inverse_kinematics(
            self.robot_id, self.fetch_robot.eef_links[self.fetch_robot.default_arm].link_id, loc, threshold, max_iter
        )
        if joint_pos is not None and len(joint_pos) > 0:
            print("Solution found. Setting new arm configuration.")
            set_joint_positions(self.robot_id, self.arm_joint_ids, joint_pos)
        else:
            print(
                "No configuration to reach that point. Move the marker to a different configuration and try again."
            )

    def release(self):
        pass
    
    def grab(self):
        self.fetch_robot.apply_action([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    def accurate_calculate_inverse_kinematics(self, robot_id, eef_link_id, target_pos, threshold, max_iter):
        max_limits = get_max_limits(robot_id, self.all_joint_ids)
        min_limits = get_min_limits(robot_id, self.all_joint_ids)
        rest_position = self.robot_default_joint_positions
        joint_range = list(np.array(max_limits) - np.array(min_limits))
        joint_range = [item + 1 for item in joint_range]
        joint_damping = [0.1 for _ in joint_range]
        print("IK solution to end effector position {}".format(target_pos))
        # Save initial robot pose
        state_id = p.saveState()

        max_attempts = 5
        solution_found = False
        joint_poses = None
        for attempt in range(1, max_attempts + 1):
            print("Attempt {} of {}".format(attempt, max_attempts))
            # Get a random robot pose to start the IK solver iterative process
            # We attempt from max_attempt different initial random poses
            sample_fn = get_sample_fn(robot_id, self.arm_joint_ids)
            sample = np.array(sample_fn())
            # Set the pose of the robot there
            set_joint_positions(robot_id, self.arm_joint_ids, sample)

            it = 0
            # Query IK, set the pose to the solution, check if it is good enough repeat if not
            while it < max_iter:

                joint_poses = p.calculateInverseKinematics(
                    robot_id,
                    eef_link_id,
                    target_pos,
                    lowerLimits=min_limits,
                    upperLimits=max_limits,
                    jointRanges=joint_range,
                    restPoses=rest_position,
                    jointDamping=joint_damping,
                )
                joint_poses = np.array(joint_poses)[self.robot_arm_indices]

                set_joint_positions(robot_id, self.arm_joint_ids, joint_poses)

                dist = l2_distance(self.fetch_robot.get_eef_position(), target_pos)
                if dist < threshold:
                    solution_found = True
                    break
                logging.debug("Dist: " + str(dist))
                it += 1

            if solution_found:
                print("Solution found at iter: " + str(it) + ", residual: " + str(dist))
                break
            else:
                print("Attempt failed. Retry")
                joint_poses = None

        restoreState(state_id)
        p.removeState(state_id)
        return joint_poses