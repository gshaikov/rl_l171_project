# Code adapted from https://github.com/jimmyyhwu/tidybot2/blob/main/mujoco_env.py
# Code to actually move the base and arm based on a given command

import math
import numpy as np
from ruckig import InputParameter, OutputParameter, Result, Ruckig
from rl_l171.utils.ik_solver_piper import IKSolver as IKSolverPiper


class BaseController:
    def __init__(self, qpos, qvel, ctrl, timestep, initial_position, randomisation=False,
                 np_random=None):
        self.qpos = qpos
        self.qvel = qvel
        self.ctrl = ctrl
        self.initial_position = initial_position
        self.randomisation = randomisation

        # OTG (online trajectory generation)
        num_dofs = 3
        self.last_command_time = None
        self.otg = Ruckig(num_dofs, timestep)
        self.otg_inp = InputParameter(num_dofs)
        self.otg_out = OutputParameter(num_dofs)
        self.otg_inp.max_velocity = [0.5, 0.5, 3.14]
        self.otg_inp.max_acceleration = [0.5, 0.5, 2.36]
        self.otg_res = None

        assert np_random is not None, "np_random must be provided for BaseController"
        self.np_random = np_random

    def reset(self):
        # Initialize base at origin
        if self.randomisation:
            self.qpos[:] = np.array([
                self.initial_position[0] + self.np_random.uniform(-0.1, 0.1),
                self.initial_position[1] + self.np_random.uniform(-0.1, 0.1),
                self.initial_position[2] + self.np_random.uniform(-0.3, 0.3)
            ])
        else:
            self.qpos[:] = np.array(self.initial_position)
        self.ctrl[:] = self.qpos

        # Initialize OTG
        self.otg_inp.current_position = self.qpos
        self.otg_inp.current_velocity = self.qvel
        self.otg_inp.target_position = self.qpos
        self.otg_res = Result.Finished

    def control_callback(self, command):
        if command is not None:
            if 'base_pose' in command:
                # Set target base qpos
                self.otg_inp.target_position = command['base_pose']
                self.otg_res = Result.Working

        # Update OTG
        if self.otg_res == Result.Working:
            self.otg_res = self.otg.update(self.otg_inp, self.otg_out)
            self.otg_out.pass_to_input(self.otg_inp)
            self.ctrl[:] = self.otg_out.new_position


class CamralArmController:
    def __init__(self, qpos, qvel, ctrl, qpos_gripper, ctrl_gripper, timestep, reset_qpos, wbc=False):
        self.qpos = qpos
        self.qvel = qvel
        self.ctrl = ctrl
        self.qpos_gripper = qpos_gripper
        self.ctrl_gripper = ctrl_gripper
        self.reset_qpos = reset_qpos

        # OTG (online trajectory generation) for 6-DOF Piper arm
        num_dofs = 6
        self.otg = Ruckig(num_dofs, timestep)
        self.otg_inp = InputParameter(num_dofs)
        self.otg_out = OutputParameter(num_dofs)
        # Piper arm velocity and acceleration limits - increased for faster movement
        self.otg_inp.max_velocity = [math.radians(120), math.radians(120), math.radians(120),
                                     math.radians(180), math.radians(180), math.radians(180)]
        self.otg_inp.max_acceleration = [math.radians(360), math.radians(360), math.radians(360),
                                        math.radians(540), math.radians(540), math.radians(540)]
        self.otg_res = None

        self.ik_solver = IKSolverPiper()

    def reset(self):
        # Initialize arm in "home" configuration for Piper
        self.qpos[:] = np.array(self.reset_qpos)
        self.ctrl[:] = self.qpos
        self.ctrl_gripper[:] = 0.0

        # Initialize OTG
        self.otg_inp.current_position = self.qpos
        self.otg_inp.current_velocity = self.qvel
        self.otg_inp.target_position = self.qpos
        self.otg_res = Result.Finished

    def control_callback(self, command):
        if command is not None:
            if 'arm_pos' in command:
                # Run inverse kinematics on new target pose
                qpos = self.ik_solver.solve(command['arm_pos'], command['arm_quat_xyzw'], self.qpos)
                qpos = qpos[:6]  # Take only first 6 joints for Piper arm
                qpos = self.qpos + np.mod((qpos - self.qpos) + np.pi, 2 * np.pi) - np.pi  # Unwrapped joint angles

                # Set target arm qpos
                self.otg_inp.target_position = qpos
                self.otg_res = Result.Working

            if 'gripper_pos' in command:
                # Set target gripper pos for Piper gripper (joint7)
                self.ctrl_gripper[:] = 0.035 * command['gripper_pos']  # gripper range [0, 0.035]

        # Update OTG
        if self.otg_res == Result.Working:
            self.otg_res = self.otg.update(self.otg_inp, self.otg_out)
            self.otg_out.pass_to_input(self.otg_inp)
            self.ctrl[:] = self.otg_out.new_position
        elif self.otg_res == Result.Finished:
            self.ctrl[:] = self.otg_out.new_position
