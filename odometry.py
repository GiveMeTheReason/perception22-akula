import numpy as np


def forward_kinematics(params: dict, state: np.ndarray, odometry: np.ndarray) -> np.ndarray:
    """
    Forward kinematics for differential platform
    Arc Kinematic motion model (Prob.Rob.Ch5.3)
    Assumption: motion consists of arcs

    Arc length = k * (V1 + V2) / 2
    Arc angle = k * (V1 - V2) / (2 * l)
    k = 2 * pi * r / counts_2pi

    angle_wheel (scalar) = counts / counts_per_rotation * 2 * pi
    angle_robot (scalar) = angle_wheel * r / R (R - rotation radius (L))

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

    k = 2 * np.pi * r / counts_2pi
    d_len_right = odometry[0] * k
    d_len_left = odometry[1] * k
    d_len = (d_len_right + d_len_left) / 2
    d_angle = (d_len_right - d_len_left) / (2 * l)

    d_state = np.array([d_len/d_angle * (-np.sin(th) + np.sin(th + d_angle)),
                        d_len/d_angle * (np.cos(th) - np.cos(th + d_angle)),
                        d_angle])

    return state + d_state


# metres
params = {'diag_length': 2,
          'wheel_radius': 0.5,
          'counts_per_rotation': 256}
# initial state
state = np.array([0., 0., 0.])

# ticks (counts on every wheel)
odometry = np.array([4*256., 2*256.])

state += forward_kinematics(params, state, odometry)

print(state)
