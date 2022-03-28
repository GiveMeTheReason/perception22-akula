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
    assert params.get('counts_per_rotation_right')
    assert params.get('counts_per_rotation_left')

    l = params['diag_length']
    r = params['wheel_radius']
    counts_2pi_r = params['counts_per_rotation_right']
    counts_2pi_l = params['counts_per_rotation_left']
    x, y, th = state

    k_r = 2 * np.pi * r / counts_2pi_r
    k_l = 2 * np.pi * r / counts_2pi_l
    d_len_right = odometry[0] * k_r
    d_len_left = odometry[1] * k_l
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



trajectory_filename = 'trajectory.txt'
f = open(trajectory_filename, 'r')

odometry_filename = 'obs07.txt'
f_od = open(odometry_filename, 'r')

input_filename = 'input_log07.txt'
f_in = open(input_filename, 'r')

x = []
y = []
z = []
th = []

while True:
    if not f.readline().split():
        break
    x.append(float(f.readline().split()[-1]))
    y.append(float(f.readline().split()[-1]))
    z.append(float(f.readline().split()[-1]))
    th.append(float(f.readline().split()[-1]))



# metres
params = {'diag_length': 0.490,
          'wheel_radius': 0.130,
          'counts_per_rotation_right': 3248,
          'counts_per_rotation_left': 2594}
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
    odometry = np.array([right, left])
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

# emperical
# delta_ticks / (delta_time * pwm_duty)
d_r = (250892 - 185913) / ( (28735 - 5797)/1000 * 180)
d_l = (133933 - 82051) / ( (49848 - 26384)/1000 * 180)

# inputs (counts on every wheel)
while True:
    inp = list(map(int, f_in.readline().split()))
    if not inp:
        break
    timestamp = inp[0]
    d_time = (timestamp - timestamp_last) / 1000
    timestamp_last = timestamp
    linspace = 20
    for t in range(linspace):
        odometry = np.array([right*d_time*d_r/linspace, left*d_time*d_l/linspace])
        d_state = forward_kinematics(params, state, odometry)
        state = state + d_state
        x_in.append(state[0])
        y_in.append(state[1])
        th_in.append(state[2])
    left = inp[1]
    right = inp[2]



f.close()
f_od.close()
f_in.close()

x = np.array(x) - x[0]
y = np.array(y) - y[0]
z = np.array(z) - z[0]
x, y = z, -y
alp = 0

plt.figure()
plt.plot(x*np.cos(alp) - y*np.sin(alp), x*np.sin(alp) + y*np.cos(alp))
plt.plot(np.array(x_od), np.array(y_od))
plt.plot(np.array(x_in), np.array(y_in))
plt.legend(['Ground truth', 'Raw Encoders prediction', 'Raw Input prediction'])
plt.axis('equal')
plt.show()
