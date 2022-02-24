import numpy as np


def forward_kinematics(params: dict, state: np.ndarray, odometry: np.ndarray) -> np.ndarray:
    """
    Forward kinematics for differential platform

    delta_state (3x1) = A (3x2) @ angle_wheel (2x1)
    delta_state.T (1x3) = angle_wheel.T (1x2) @ A.T (2x3)
    angle_wheel (scalar) = counts / counts_per_rotation * 2 * pi
    angle_robot (scalar) = angle_wheel * r / R (if rotation on place)

    :param params: Dict, robot geometry parameters
    :param state: Array-like object (1x3), robot current state [x, y, theta]
    :param odometry: Array-like object (1x2), ticks from odometry [right, left]
    """
    assert state.shape == (3,)
    assert odometry.shape == (2,)
    assert params.get('diag_length')
    assert params.get('wheel_radius')
    assert params.get('counts_per_rotation')

    l = params['diag_length']
    r = params['wheel_radius']
    counts_2pi = params['counts_per_rotation']

    x, y, th = state
    A = 1/2 * np.array([[np.cos(th), np.cos(th)],
                        [np.sin(th), np.sin(th)],
                        [1/l, -1/l]]).T

    return 2 * r * np.pi / counts_2pi * odometry @ A


# metres
params = {'diag_length': 2,
          'wheel_radius': 0.5,
          'counts_per_rotation': 256}
# initial state
state = np.array([0., 0., np.pi/4])

# ticks (counts on every wheel)
odometry = np.array([4*256., 2*256.])

state += forward_kinematics(params, state, odometry)

print(state)
