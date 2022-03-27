from ekf import EKFarc
import numpy as np
import matplotlib.pyplot as plt
import argparse
import scipy.linalg
import os
from termcolor import colored
from scipy.interpolate import interp1d


def interpolate(val, t, t_new):
    if not isinstance(val, np.ndarray):
        val = np.array(val)

    if not isinstance(t, np.ndarray):
        t = np.array(t)

    f = interp1d(t, val)

    # print(t_new.min(), t_new.max(), t.min(), t.max())

    return f(t_new)

def plot2dcov(mean, covariance, k):
    points = [k*np.cos(np.linspace(0, 2*np.pi)), k*np.sin(np.linspace(0, 2*np.pi))]

    L = scipy.linalg.cholesky(covariance).T
    points = L @ points + mean
    return plt.plot(points[0], points[1])


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



def aplly_EKF(args):
    
    source_observations = args.source_observations
    source_actions = args.source_actions
    trajectory_filename = args.trajectory_filename

    obs = open(source_observations, 'r', encoding='utf-8')
    acts = open(source_actions, 'r', encoding='utf-8')

    f_in = open(source_actions, 'r', encoding='utf-8')
    f_od = open(source_observations, 'r')
    f = open(trajectory_filename, 'r')

    # metres
    params = {'diag_length': 0.490,
              'wheel_radius': 0.130,
              'counts_per_rotation_left': 2594,
              'counts_per_rotation_right': 3248}

    # find first observation
    ts_obs_prev, left_obs_prev, right_obs_prev = list(map(int, obs.readline().split()))
    while True:
        ts_obs, left_obs, right_obs = list(map(int, obs.readline().split()))

        if (left_obs - left_obs_prev) or (right_obs - right_obs_prev):
            break
        
        ts_obs_prev, left_obs_prev, right_obs_prev = ts_obs, left_obs, right_obs
    ts_obs_0 = ts_obs_prev
    ts_obs_prev = 0
    ts_obs -= ts_obs_0

    ts_acts_prev, left_acts_prev, right_acts_prev = list(map(int, acts.readline().split()))
    ts_acts_0 = ts_acts_prev
    ts_acts_prev = 0

    ts = 0

    state_initial = np.array([[0.], [0.], [0.]])
    sigma_initial = np.array([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 0., 0.]])
    alphas = np.array([500, 500, 500, 500])**2
    d = params['diag_length']
    r = params['wheel_radius']
    counts_2pi_l = params['counts_per_rotation_left']
    counts_2pi_r = params['counts_per_rotation_right']
    k_l = 2 * np.pi * r / counts_2pi_l
    k_r = 2 * np.pi * r / counts_2pi_r
    Q = np.array([[1000**2, 250**2],
                  [250**2, 1000**2]])

    ekf = EKFarc(state_initial, sigma_initial, alphas, d, k_l, k_r, Q)
    t_ekf = []
    x = []
    y = []
    th = []

    # emperical
    # delta_ticks / (delta_time * pwm_duty)
    d_l = (133933 - 82051) / ( (49848 - 26384)/1000 * 180)
    d_r = (250892 - 185913) / ( (28735 - 5797)/1000 * 180)

    while True:
        acts_line = acts.readline()
        if not acts_line:
            break
        ts_acts, left_acts, right_acts = list(map(int, acts_line.split()))
        ts_acts -= ts_acts_0

        v = (k_r * d_r * right_acts_prev + k_l * d_l * left_acts_prev) / 2
        w = (k_r * d_r * right_acts_prev - k_l * d_l * left_acts_prev) / d
        
        while ts_obs <= ts_acts:
            ekf.predict([v, w], (ts_obs - ts)/1000)
            ekf.update(np.array([[(left_obs-left_obs_prev)], [(right_obs-right_obs_prev)]]))
            ts = ts_obs

            t_ekf.append(ts_obs)
            x.append(ekf.mu[0][0])
            y.append(ekf.mu[1][0])
            th.append(ekf.mu[2][0])

            obs_line = obs.readline()
            if not obs_line:
                ts_obs = np.inf
                break
            
            ts_obs_prev, left_obs_prev, right_obs_prev = ts_obs, left_obs, right_obs
            ts_obs, left_obs, right_obs = list(map(int, obs_line.split()))
            ts_obs -= ts_obs_0

        ekf.predict([v, w], (ts_acts - ts)/1000)
        ts = ts_acts

        ts_acts_prev, left_acts_prev, right_acts_prev = ts_acts, left_acts, right_acts

    obs.close()
    acts.close()


    # Plot raw input
    # initial state
    state = np.array([0., 0., 0.])

    timestamp_last, left, right = list(map(int, f_in.readline().split()))

    x_in = [state[0]]
    y_in = [state[1]]
    th_in = [state[2]]

    # inputs pwm (counts on every wheel)
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

    f_in.close()


    # Plot raw encoders
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

    f_od.close()


    # Plot "Ground Truth"
    t_gt = []
    x_gt = []
    y_gt = []
    z_gt = []
    th_gt = []

    
    while True:
        t_gt_inp = f.readline()
        if not t_gt_inp:
            break
        t_gt.append(float(t_gt_inp.split()[-1]))

        x_gt.append(float(f.readline().split()[-1]))
        y_gt.append(float(f.readline().split()[-1]))
        z_gt.append(float(f.readline().split()[-1]))
        th_gt.append(float(f.readline().split()[-1]))

    # GT axis rotation
    t_gt = np.array(t_gt) - t_gt[0]
    x_gt = np.array(x_gt)
    y_gt = np.array(y_gt)
    z_gt = np.array(z_gt)
    x_gt, y_gt = z_gt, -y_gt
    alp = 0

    # print(ekf.sigma)

    plot_path_fname = os.path.join(args.folder_path_output, 'plot.png')
    plt.figure()
    plt.plot(x, y)
    plt.plot(np.array(x_in), np.array(y_in))
    plt.plot(np.array(x_od), np.array(y_od))
    plt.plot(x_gt*np.cos(alp) - y_gt*np.sin(alp), x_gt*np.sin(alp) + y_gt*np.cos(alp))
    plot2dcov(ekf.mu[:2], ekf.sigma[:2, :2], k=3)
    plt.legend(['EKF', 'Raw input prediction', 'Raw Encoders prediction', 'Ground truth', '3sigma iso-contour'])
    plt.axis('equal')
    plt.grid()
    plt.savefig(plot_path_fname, dpi=400)
    plt.show()
    
    print(colored('saved to: ', 'green', attrs=['bold']), plot_path_fname)

    # print(t_gt)
    # print(t_ekf)
    # print(max(np.array(t_ekf).min(), t_gt.min()), min(np.array(t_ekf).max(), t_gt.max()))
    t_sync = np.linspace(max(np.array(t_ekf).min(), t_gt.min()) + 10, min(np.array(t_ekf).max(), t_gt.max()) - 10, 100)
    # print('yes ')
    # print(np.array(np.array(t_ekf).max()), np.array(t_ekf).min(), t_sync.max(), t_sync.min())
    x_sync = interpolate(x, t_ekf, t_sync)
    # print('yes ')
    y_sync = interpolate(y, t_ekf, t_sync)
    # print('yes ')
    x_gt_sync = interpolate(x_gt, t_gt, t_sync)
    # print('yes ')
    y_gt_sync = interpolate(y_gt, t_gt, t_sync)
    # print('yes ')
    plot_err_fname = os.path.join(args.folder_path_output, 'err.png')

    plt.figure()
    plt.plot(t_sync, np.linalg.norm(np.vstack((x_sync, y_sync)) - np.vstack((x_gt_sync, y_gt_sync)), axis=0, ord=2), '-', label=r'$||pose_{ekf} - pose_{GT}||$')
    plt.legend(loc='best')
    plt.xlabel('time (ms)')
    plt.grid()
    plt.savefig(plot_err_fname, dpi=400)
    plt.show()

    print(colored('saved to: ', 'green', attrs=['bold']), plot_err_fname)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-src_obs',
                        type=str,
                        dest='source_observations',
                        action='store',
                        default='./data/obs07.txt',
                        help="Path to observation file")

    parser.add_argument('-src_act',
                        type=str,
                        dest='source_actions',
                        action='store',
                        default='./raw_data/input_log07.txt',
                        help="Path to actions file")

    parser.add_argument('-traj',
                        type=str,
                        dest='trajectory_filename',
                        action='store',
                        default='./data/trajectory07.txt',
                        help="Path to actions file")


    parser.add_argument('-save_to',
                        type=str,
                        dest='folder_path_output',
                        action='store',
                        default='./output')

    args = parser.parse_args()
    aplly_EKF(args)
