import numpy as np


def forward_kinematics(params: dict, state: np.ndarray, odometry: np.ndarray) -> np.ndarray:
    """
    Forward kinematics for differential platform
    Arc Kinematic motion model (Prob.Rob.Ch5.3)
    Assumption: motion consists of arcs

    Arc length = k * (V1 + V2) / 2
    Arc angle = k * (V1 - V2) / l
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
    d_angle = (d_len_right - d_len_left) / l

    if d_angle != 0:
        d_state = np.array([d_len/d_angle * (-np.sin(th) + np.sin(th + d_angle)),
                            d_len/d_angle * (np.cos(th) - np.cos(th + d_angle)),
                            d_angle])
    else:
        d_state = np.array([d_len * np.cos(th),
                            d_len * np.sin(th),
                            d_angle])

    return d_state


def inverse_kinematics(params: dict, state: np.ndarray, d_state: np.ndarray) -> np.ndarray:
    """
    Inverse kinematics for differential platform
    Arc Kinematic motion model (Prob.Rob.Ch5.3)
    Assumption: motion consists of arcs

    Arc length = k * (V1 + V2) / 2
    Arc angle = k * (V1 - V2) / l
    k = 2 * pi * r / counts_2pi

    angle_wheel (scalar) = counts / counts_per_rotation * 2 * pi
    angle_robot (scalar) = angle_wheel * r / R (R - rotation radius (L))

    :param params: Dict, robot geometry parameters
    :param state: Array-like object (1x3), robot current state [x, y, theta]
    :param d_state: Array-like object (1x3), delta of current state [x, y, theta]
    """
    assert d_state.shape == (3,)
    assert params.get('diag_length')
    assert params.get('wheel_radius')
    assert params.get('counts_per_rotation')

    l = params['diag_length']
    r = params['wheel_radius']
    counts_2pi = params['counts_per_rotation']
    th = state[2]
    d_x, d_y, d_th = d_state

    k = 2 * np.pi * r / counts_2pi

    scal_prod = d_x * np.cos(th) + d_y * np.sin(th)
    
    if d_th != 0:
        d_wheel = d_th/(2 * k) * np.array([np.linalg.norm([d_x, d_y]) / np.sin(d_th/2) + l * np.sign(scal_prod),
                                           np.linalg.norm([d_x, d_y]) / np.sin(d_th/2) - l * np.sign(scal_prod)])
    else:
        d_wheel = 1/k * np.array([np.linalg.norm([d_x, d_y]),
                                  np.linalg.norm([d_x, d_y])])
    d_wheel *= np.sign(scal_prod)

    return d_wheel


def wrap_angle(angle):
    """
    Wraps the given angle to the range [-pi, +pi].

    :param angle: The angle (in rad) to wrap (can be unbounded).
    :return: The wrapped angle (guaranteed to in [-pi, +pi]).
    """

    pi2 = 2 * np.pi

    while angle < -np.pi:
        angle += pi2

    while angle >= np.pi:
        angle -= pi2

    return angle


# metres
params = {'diag_length': 0.490,
          'wheel_radius': 0.130,
          'counts_per_rotation': 2394}
# initial state
state = np.array([0., 0., 0.])

# ticks (counts on every wheel)
right = 52
left = 157
odometry = np.array([right, left])

d_state = forward_kinematics(params, state, odometry)

print(d_state)

state += d_state
state[2] = wrap_angle(state[2])
print(state)

print(inverse_kinematics(params, state - d_state, d_state))

print(f"Right: {right} | Left: {left}")
