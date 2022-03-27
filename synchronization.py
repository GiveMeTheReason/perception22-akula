from ekf import EKFarc
import numpy as np

class OnlineEKFarc:

    def __init__(self, source_observation, source_motion,
                state_initial, sigma_initial, alphas, d, k, Q,
                speed_coeff_left, speed_coeff_right):

        self.ekf = EKFarc(state_initial, sigma_initial, alphas, d, k, Q)
        self.speed_coeff_left = speed_coeff_left
        self.speed_coeff_right = speed_coeff_right

        self.stream_observation = open(source_observation, "r", encoding="utf-8")
        self.stream_motion = open(source_motion, "r", encoding="utf-8")

        ts1_obs, ts2_obs, delta_obs_l, delta_obs_r, last_obs = self.find_motion_start(self.stream_observation)
        ts1_motion, ts2_motion, delta_l, delta_r, last_pwm = self.find_motion_start(self.stream_motion)


        self.last_ticks = last_obs
        self.last_pwm = last_pwm

        self.observation = np.array([delta_obs_r, delta_obs_l])
        self.controls = self.get_speeds(delta_r, delta_l)

        self.init_time_obs = ts1_obs
        self.init_time_motion = ts1_motion
        self.time_motion = ts2_motion - self.init_time_motion
        self.time_observation = ts2_obs - self.init_time_obs
        self.current_observation_ts = self.time_observation


    def find_motion_start(self, stream):
        while True:
            line1 = stream.readline()
            line2 = stream.readline()

            ts1, left_wheel1, right_wheel1 = self.parse_line(line1)
            ts2, left_wheel2, right_wheel2 = self.parse_line(line2)

            if ((left_wheel2 -  left_wheel1) != 0) or ((right_wheel2 -  right_wheel1) != 0):
                return ts1, ts2, left_wheel2 - left_wheel1, right_wheel2 - right_wheel1, np.array([left_wheel2, right_wheel2])


    def parse_line(self, line):
        return list(map(int, line.split()))


    def motion_step(self):
        line = self.stream_motion.readline()
        ts, left_wheel, right_wheel = self.parse_line(line)
        self.time_motion = ts - self.init_time_motion
        self.delta_pwn = np.array([right_wheel, left_wheel]) - self.last_pwm
        self.last_pwm = np.array([right_wheel, left_wheel])
        self.controls = self.get_speeds(*self.delta_pwn)


    def observation_step(self):
        line = self.stream_observation.readline()
        ts, left_wheel, right_wheel = self.parse_line(line)
        self.time_observation = ts - self.init_time_obs
        self.observation = np.array([right_wheel, left_wheel]) - self.last_ticks
        self.last_ticks = np.array([right_wheel, left_wheel])


    def get_speeds(self, delta_r, delta_l):
        v = (delta_r * self.speed_coeff_right + delta_l * self.speed_coeff_left) / 2
        w = (delta_r * self.speed_coeff_right - delta_l * self.speed_coeff_left) / (2 * self.ekf.d)
        return np.array([v, w])


    def online(self):
        delta_t_motion_obs = self.time_motion - self.time_observation
        if delta_t_motion_obs > 0:
            self.current_observation_ts = self.time_observation - self.init_time_obs
            self.ekf.predict(self.controls, delta_t_motion_obs)
            self.ekf.update(self.observation)
            self.observation_step()

        else:
            self.ekf.predict(self.controls, -delta_t_motion_obs)
            self.motion_step()


    def get_filtered_prediction(self):
        return self.ekf.mu, self.ekf.sigma, self.current_observation_ts


    def close(self):
        self.stream_observation.close()
        self.stream_motion.close()
