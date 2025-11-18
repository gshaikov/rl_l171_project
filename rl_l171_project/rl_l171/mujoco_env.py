# Adapted from: https://github.com/camral/tidybot2_camral_sim/blob/main/mujoco_env.py

import random
import mujoco
import mujoco.viewer
import numpy as np
import logging

from rl_l171.utils.constants import NR_GRIPPER_PIPER_JOINTS, ARM_RESET_QPOS
from rl_l171.utils.controllers import BaseController, CamralArmController

from rl_l171.utils.mujoco_xml import MujocoXML

logger = logging.getLogger(__name__)


def _load_scene(nr_cubes):
    scene_xml_path = 'cubes/scene_cubes.xml'
    xml = MujocoXML()

    xml.add_default_compiler_directive()

    xml.append(MujocoXML.parse(scene_xml_path))

    for i in range(nr_cubes):
        letter_xml = f"cubes/cube.xml"
        xml.append(MujocoXML.parse(letter_xml)
                   .add_name_prefix(f"cube_{i}:"))

    model = xml.build()
    data = mujoco.MjData(model)

    return model, data


class CubesMujocoSim:
    """
    Manages the MuJoCo simulation state, robot controllers, and scene objects.

    This class wraps the low-level MuJoCo model and data, providing a
    higher-level API to step the simulation, send commands, and get state.
    """

    def __init__(self, randomise_initial_position=False, seed=None, nr_cubes=5):
        """
        :args:
            randomise_initial_position (bool): Whether to randomise the initial robot position on reset.
                                                The robot is actually randomised in controllers.py but the seed
                                                is set here for consistency.
                                                Cubes are always fixed.
            seed (int): Random seed for reproducibility
        """
        self.model, self.data = _load_scene(nr_cubes=nr_cubes)

        self.randomise_initial_position = randomise_initial_position

        # Cache references to array slices
        self.base_dofs = self.model.body('base_link').jntnum.item()
        self.arm_dofs = 6
        gripper_dofs = NR_GRIPPER_PIPER_JOINTS

        self.qpos_arm = self.data.qpos[self.base_dofs:(self.base_dofs + self.arm_dofs)]
        qpos_arm = self.data.qpos[self.base_dofs:(self.base_dofs + self.arm_dofs)]
        self.qvel_arm = self.data.qvel[self.base_dofs:(self.base_dofs + self.arm_dofs)]
        ctrl_arm = self.data.ctrl[self.base_dofs:(self.base_dofs + self.arm_dofs)]
        self.qpos_gripper = self.data.qpos[(self.base_dofs + self.arm_dofs):(self.base_dofs + self.arm_dofs + gripper_dofs)]
        ctrl_gripper = self.data.ctrl[(self.base_dofs + self.arm_dofs):(self.base_dofs + self.arm_dofs + gripper_dofs)]
        self.qpos_base = self.data.qpos[:self.base_dofs]
        qvel_base = self.data.qvel[:self.base_dofs]
        ctrl_base = self.data.ctrl[:self.base_dofs]

        self.np_random = np.random.default_rng(seed)

        self.nr_cubes = nr_cubes

        if nr_cubes > 5:
            base_initial_pos = np.array([-1.2, -0.1, 0.0])
        else:
            base_initial_pos = np.array([-1, 0.0, 0.0])

        self.base_controller = BaseController(self.qpos_base, qvel_base, ctrl_base, self.model.opt.timestep,
                                              initial_position=base_initial_pos,
                                              randomisation=randomise_initial_position,
                                              np_random=self.np_random)
        self.arm_controller = CamralArmController(qpos_arm, self.qvel_arm, ctrl_arm, self.qpos_gripper, ctrl_gripper,
                                                  self.model.opt.timestep, ARM_RESET_QPOS, wbc=False)

        # Variables for calculating arm pos and quat
        site_id = self.model.site('pinch_site').id
        self.site_xpos = self.data.site(site_id).xpos
        self.site_xmat = self.data.site(site_id).xmat
        self.site_quat = np.empty(4)
        self.base_height = self.model.body('piper_base_link').pos[2]
        self.arm_forward = self.model.body('piper_base_link').pos[0]
        self.base_rot_axis = np.array([0.0, 0.0, 1.0])
        self.base_quat_inv = np.empty(4)

        # Current command to execute
        self.current_command = None

        # Cache last target and resolved command
        self.last_arm_target_pos_global = None
        self.last_arm_target_quat_global = None
        self.last_base_target_pos = None
        self.last_gripper_pos = None
        self.cached_resolved_command = None  # Stores command with base_pose/arm_qpos
        self.reset()

    def reset(self, seed=None):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        # Reset controllers
        self.base_controller.reset()
        self.arm_controller.reset()

        cube_positions = [
                [-0.6287730809442508, -0.13576445902147924, 0.05489224457978388],
                [-0.4748852395267025, -0.18896169588197023, 0.05489224457978384],
                [-0.5397146831619227, 0.021428717628009353, 0.054892244579783755],
                [-0.38311660155149596, 0.2103195554514166, 0.05489224457978378],
                [-0.6557746293540929, 0.291931660631359, 0.05489224457978393],
                [-0.4296109029490817, -0.015421040860003234, 0.05489224457978378],
                [-0.6788908392511658, 0.01364989739894522, 0.05489224457978392],
                [-0.8956461870270883, 0.030838133878249893, 0.0548922445797838],
                [-0.7911053198151642, -0.10476567357593729, 0.0548922445797839],
                [-0.7703161359020488, -0.42882539564862334, 0.054892244579783755],
        ]

        cube_quaternions = [
            [-0.612757048045442, -0.612757048045442, 0.35288638408223716, 0.3528863840822364],
            [0.5568045108541378, 0.5568045108541387, 0.4358540314055662, 0.43585403140556545],
            [-0.3443038126878328, -0.3443038126878328, 0.6176203401513105, 0.617620340151312],
            [-0.33874950308679486, 0.33874950308679364, -0.6206841178558135, 0.620684117855814],
            [-0.6442164317998623, 0.29152219297860066, 0.6442164317998638, 0.2915221929786004],
            [0.5146920558339604, 0.5146920558339622, -0.48486295760906595, -0.4848629576090658],
            [-0.36563152474371713, -0.6052384555805954, -0.3656315247437179, 0.6052384555805953],
            [0.5880520582560342, -0.3926764275849024, -0.5880520582560359, -0.3926764275849025],
            [0.2560757808252185, 0.6591093949222336, -0.2560757808252183, 0.659109394922235],
            [0.9999999999371122, 2.4507530841173426e-18, -2.1775598762200762e-19, -1.1214989255568561e-05],
        ]

        for i in range(self.nr_cubes):
            cube_joint_name = f'cube_{i}:joint'

            # Get the joint address in qpos
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, cube_joint_name)
            joint_qposadr = self.model.jnt_qposadr[joint_id]

            # Set position (first 3 elements of free joint qpos)
            self.data.qpos[joint_qposadr:joint_qposadr + 3] = cube_positions[i]
            # Set orientation (next 4 elements of free joint qpos)
            self.data.qpos[joint_qposadr + 3:joint_qposadr + 7] = cube_quaternions[i]

        # Step the simulation to propagate changes
        mujoco.mj_forward(self.model, self.data)

        # Clear cached targets and commands
        self.last_arm_target_pos_global = None
        self.last_arm_target_quat_global = None
        self.last_base_target_pos = None
        self.last_gripper_pos = None

        self.cached_resolved_command = None

        if not self.randomise_initial_position:
            return

        # apply small force to move the cubes
        valid_body_ids = []
        FORCE_MAGNITUDE = 0.5
        for i in range(self.nr_cubes):
            cube_body_name = f'cube_{i}:middle'

            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, cube_body_name)

            # store the is so we can clear the force later
            valid_body_ids.append(body_id)

            # generate a unique random force and torque for this cube
            random_force = (np.random.rand(3) - 0.5) * 2 * FORCE_MAGNITUDE

            # self.data.xfrc_applied is (nbody, 6) -> [fx, fy, fz, tx, ty, tz]
            self.data.xfrc_applied[body_id, 0:3] = random_force

        # tep the simulation to let all forces take effect
        # The simulation applies all non-zero forces in xfrc_applied at once
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        # clear the forces for all cubes
        for body_id in valid_body_ids:
            self.data.xfrc_applied[body_id, :] = 0.0


    def _update_controls(self, command):
        if command is not None:

            assert 'arm_pos_global' in command and 'base_pose' in command, "Command must include 'arm_pos_global' and 'base_pos'"

            target_base_pose = command['base_pose']
            target_arm_pos_global = command['arm_pos_global']
            target_arm_quat_global_xyzw = command['arm_quat_global_xyzw']
            target_gripper_pos = command['gripper_pos']

            POSITION_DEADBAND = 0.001  # 1mm
            ORIENTATION_DEADBAND = 0.01  # radians

            target_changed = (
                    (self.last_arm_target_pos_global is None
                     or np.linalg.norm(target_arm_pos_global - self.last_arm_target_pos_global) > POSITION_DEADBAND) or
                    (self.last_arm_target_quat_global is None
                        or np.linalg.norm(target_arm_quat_global_xyzw - self.last_arm_target_quat_global) > ORIENTATION_DEADBAND) or
                    (self.last_base_target_pos is None
                        or np.linalg.norm(target_base_pose - self.last_base_target_pos) > POSITION_DEADBAND) or
                    (self.last_gripper_pos is None
                        or np.linalg.norm(target_gripper_pos - self.last_gripper_pos) > POSITION_DEADBAND)
            )

            if target_changed:
                # Only solve IK when target actually changes

                tmp = target_arm_pos_global.copy()
                tmp[2] -= self.base_height  # Base height offset
                tmp[:2] -= self.qpos_base[:2]  # Base position inverse
                tmp[0] -= self.arm_forward  # Arm base offset
                mujoco.mju_axisAngle2Quat(self.base_quat_inv, self.base_rot_axis,
                                          -self.qpos_base[2])  # Base orientation inverse
                target_arm_pos = np.empty(3)
                mujoco.mju_rotVecQuat(target_arm_pos, tmp, self.base_quat_inv)  # Arm pos in local frame

                # Update arm quat
                # convert target_arm_quat_global from (x, y, z, w) to (w, x, y, z) for mujoco
                target_global_quat_wxyz = target_arm_quat_global_xyzw[[3, 0, 1, 2]]

                target_local_quat_wxyz = np.empty(4)
                mujoco.mju_mulQuat(target_local_quat_wxyz, self.base_quat_inv, target_global_quat_wxyz)
                target_arm_quat_xyzw = target_local_quat_wxyz[[1, 2, 3, 0]]  # back to (x,y,z,w) as IK piper expects this

                # Create resolved command with IK solution
                self.cached_resolved_command = {
                    'base_pose': target_base_pose,
                    'arm_pos': target_arm_pos,
                    'arm_quat_xyzw': target_arm_quat_xyzw,
                    'gripper_pos': target_gripper_pos
                }

                # Cache the target
                self.last_base_target_pos = target_base_pose
                self.last_arm_target_pos_global = target_arm_pos_global
                self.last_arm_target_quat_global = target_arm_quat_global_xyzw
                self.last_gripper_pos = target_gripper_pos

            command_to_execute = self.cached_resolved_command
        else:
            command_to_execute = command

        self.base_controller.control_callback(command_to_execute)
        self.arm_controller.control_callback(command_to_execute)

    def step_simulation(self, command=None):
        """Execute one simulation step with optional command"""
        self._update_controls(command)
        mujoco.mj_step(self.model, self.data)

    def get_state(self):
        """Get current state observations"""
        # EE pos
        site_xpos = self.site_xpos.copy() # global frame

        # Update gripper pos
        gripper_pos = self.qpos_gripper / 0.035  # joint7, joint range [0, 0.035]

        # this populates self.site_quat with the CURRENT (world) orientation
        mujoco.mju_mat2Quat(self.site_quat, self.site_xmat)

        state = {
            'base_pose': self.qpos_base.copy(),
            'arm_pos_global': site_xpos.copy(),
            'arm_quat_global_wxyz': self.site_quat.copy(), # in [w, x, y, z] since based on mujoco
            'gripper_pos': gripper_pos.copy(),
        }

        # Update cube positions
        cube_positions = []
        cube_orientations = []
        for i in range(self.nr_cubes):
            cube_i_pos = self.data.body(f'cube_{i}:middle').xpos.copy()
            cube_positions.append(cube_i_pos)
            cube_orientations.append(self.data.body(f'cube_{i}:middle').xquat.copy())

            state[f'cube_{i}_pos'] = cube_i_pos

        return state


