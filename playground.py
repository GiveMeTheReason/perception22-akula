import matplotlib.pyplot as plt
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



trajectory_filename = 'trajectory03.txt'
f = open(trajectory_filename, 'r')

odometry_filename = 'obs03.txt'
f_od = open(odometry_filename, 'r')

input_filename = 'input_log03.txt'
f_in = open(input_filename, 'r')

x = []
y = []
th = []

while True:
    if not f.readline().split():
        break
    x.append(float(f.readline().split()[-1]))
    y.append(float(f.readline().split()[-1]))
    th.append(float(f.readline().split()[-1]))



# metres
params = {'diag_length': 0.490,
          'wheel_radius': 0.130,
          'counts_per_rotation': 2394}
# initial state
state = np.array([0., 0., 0.])

_, left_last, right_last = list(map(int, f_od.readline().split()))

x_od = [state[0]]
y_od = [state[1]]
th_od = [state[2]]

# ticks (counts on every wheel)
while True:
    inp = f_od.readline().split()
    if not inp:
        break
    _, left, right = list(map(int, inp))
    left = left - left_last
    right = right - right_last
    left_last = left + left_last
    right_last = right + right_last
    odometry = np.array([right/200*1/1.5, left/200])
    d_state = forward_kinematics(params, state, odometry)
    state = state + d_state
    x_od.append(state[0])
    y_od.append(state[1])
    th_od.append(state[2])



# initial state
state = np.array([0., 0., 0.])

timestamp_last, left, right = list(map(int, f_in.readline().split()))

x_in = [state[0]]
y_in = [state[1]]
th_in = [state[2]]

# ticks (counts on every wheel)
while True:
    inp = list(map(int, f_in.readline().split()))
    if not inp:
        break
    timestamp = inp[0]
    d_time = (timestamp - timestamp_last) / 1000
    timestamp_last = timestamp
    odometry = np.array([right*d_time/25, left*d_time/25])
    left = inp[1]
    right = inp[2]
    d_state = forward_kinematics(params, state, odometry)
    state = state + d_state
    x_in.append(state[0])
    y_in.append(state[1])
    th_in.append(state[2])



f.close()
f_od.close()
f_in.close()

plt.figure()
plt.plot(-np.array(y), -np.array(x))
plt.plot(np.array(x_od), np.array(y_od))
plt.plot(np.array(x_in), np.array(y_in))
plt.legend(['Ground truth', 'Encoders', 'Input'])
plt.show()
