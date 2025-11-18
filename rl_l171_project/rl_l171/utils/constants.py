from dataclasses import dataclass, field

import numpy as np

################################################################################
# Mobile base

# Vehicle center to steer axis (m)
h_x, h_y = 0.140150 * np.array([1.0, 1.0, -1.0, -1.0]), 0.120150 * np.array([-1.0, 1.0, 1.0, -1.0])  # ARX5-camral-bot1

# Encoder magnet offsets
ENCODER_MAGNET_OFFSETS = [1575.0 / 4096, -3608.0 / 4096, 347.0 / 4096, 1542.0 / 4096]

################################################################################
#
POLICY_CONTROL_FREQ = 20.0  # Hz
POLICY_CONTROL_PERIOD = 1.0 / POLICY_CONTROL_FREQ

NR_GRIPPER_PIPER_JOINTS = 2

# also update in the "home" keyframe in the xml
ARM_RESET_QPOS = np.array([0., 2.686114078696833, -1.1633143063317806, 0., 0., 0.])
# distance from the center for a cube to be considered fully inside the target area
INSIDE_THRESHOLD = 0.3
