import numpy as np
from utils import wrap_angle
from scipy.linalg import inv

class EKFarc:
    """
    This class implements EKF for arc motion model.
    :param state_initial: initial state of the robot np.array([[x], [y], [theta]]])
    :param sigma_initial: initial covariance of the robot np.array(3 x 3)
    :param alphas: parameters for calcualtiion of covariance of noise in motion state
    :param d: inter-wheel distance in meters
    :param k :observation = (delta distance) / k
    :param Q: covariance matrix of observation
    """
    def __init__(self, state_initial, sigma_initial, alphas, d, k, Q):
        self.mu = state_initial
        self.sigma = sigma_initial
        self.alphas = alphas
        self.previous_state = state_initial
        self.d = d
        self.k = k
        self.Q = Q


    def predict(self, controls, delta_t):
        """
        Prediction step of EKF model: calculate bel_bar statistics
        :param controls: vector of controls, shape = (2, )
        """

        mu_theta = self.mu[2][0]
        v, w = controls

        #Calculate mean for prediction step
        if w:
            self.mu_bar = self.mu + np.array([[-v / w * (np.sin(mu_theta) - np.sin(mu_theta + w * delta_t))],
                                              [v / w * (np.cos(mu_theta) - np.cos(mu_theta + w * delta_t))],
                                              [w * delta_t]])

        else:
            self.mu_bar = self.mu + np.array([[v * delta_t * np.cos(mu_theta)],
                                              [v * delta_t * np.sin(mu_theta)],
                                              [0]])

        Gt = self.get_Gt(mu_theta, delta_t, v, w)
        Vt = self.get_Vt(mu_theta, delta_t, v, w)
        Mt = self.get_Mt(v, w)
        #Calculate covariance for prediction step
        self.sigma_bar = Gt @ self.sigma @ Gt.T + Vt @ Mt @ Vt.T


    def update(self, z):
        """
        Update step of EKF model: update belief statistics
        :param z: [distance which left wheel passed in tiks, distance which right wheel passed in tiks]
        """

        dx = self.mu_bar[0][0] - self.previous_state[0][0]
        dy = self.mu_bar[1][0] - self.previous_state[1][0]

        c = (dx ** 2 + dy ** 2) ** 0.5

        delta_theta = wrap_angle(self.mu_bar[2][0] - self.previous_state[2][0])

        # sign of scalar product between
        # robot orientation vector (delta_theta) and movement difference
        # says if robot moves forward or backward
        # (it changes the signes of sensor model, so sensor model is piecewise function)
        scal_prod = dx * np.cos(delta_theta) + dy * np.sin(delta_theta)

        #Calculate predicted observation    
        if delta_theta != 0:
            z_hat = delta_theta / (2 * self.k) * np.array([[c / np.sin(delta_theta/2) - self.d * np.sign(scal_prod)],
                                                           [c / np.sin(delta_theta/2) + self.d * np.sign(scal_prod)]])
        else:
            z_hat = 1 / self.k * np.array([[c],
                                           [c]])
        z_hat *= np.sign(scal_prod)

        # Calculate Jacobian of observation matrix
        Ht = self.get_Ht(dx, dy, delta_theta)
        Ht *= np.sign(scal_prod)

        St = Ht @ self.sigma_bar @ Ht.T + self.Q
        Kt = self.sigma_bar @ Ht.T @ inv(St)

        #Calculate mean and covariance of belief distribution
        self.mu = self.mu_bar + Kt @ (z - z_hat)
        self.sigma = (np.eye(3) - Kt @ Ht) @ self.sigma_bar
        self.previous_state = self.mu


    def get_Gt(self, mu_theta, delta_t, v, w):
        """
        Caculate jacobian matrix for arc model with respect to state
        """
        if w:
            Gt = np.array([[1, 0, -v / w * (np.cos(mu_theta) - np.cos(mu_theta + w * delta_t))],
                           [0, 1, -v / w * (np.sin(mu_theta) - np.sin(mu_theta + w * delta_t))],
                           [0, 0, 1]])
        else:
            Gt = np.array([[1, 0, -v * delta_t * np.sin(mu_theta)],
                            [0, 1, v * delta_t * np.cos(mu_theta)],
                            [0, 0, 1]])
        return Gt


    def get_Vt(self, mu_theta, delta_t, v, w):
        """
        Caculate jacobian matrix for arc model with respect to controls
        """
        
        if w:
            Vt = np.array([[(-np.sin(mu_theta) + np.sin(mu_theta + w * delta_t)) / w,
                    (v * (np.sin(mu_theta) - np.sin(mu_theta + w * delta_t))) / (w ** 2) + \
                    (v * np.cos(mu_theta + w * delta_t) * delta_t) / w],

                    [(np.cos(mu_theta) - np.cos(mu_theta + w * delta_t)) / w,
                    -(v * (np.cos(mu_theta) - np.cos(mu_theta + w * delta_t))) / (w ** 2) + \
                    (v * np.sin(mu_theta + w * delta_t) * delta_t) / w],

                    [0, delta_t]])
        else:
            Vt = np.array([[delta_t * np.cos(mu_theta), 0],
                           [delta_t * np.sin(mu_theta), 0],
                           [0, 0]])

        return Vt


    def get_Mt(self, v, w):
        """
        Calculate covariance matrix of noise in action space
        """
        a1, a2, a3, a4 = self.alphas
        Mt = np.diag([a1 * v ** 2 + a2 * w ** 2,
                      a3 * v ** 2 + a4 * w ** 2 ])
        return Mt


    def get_Ht(self, dx, dy, delta_theta):
        """
        Calculate observation matrix (Jacobian of Sensor model with respect to state space)
        """
        c = (dx ** 2 + dy ** 2) ** 0.5

        if not c:
            Ht = 1 / self.k * np.array([[0, 0, -delta_theta],
                                        [0, 0, delta_theta]])
        elif delta_theta:
            Ht = 1 / self.k * np.array([[delta_theta / (2 * self.k) * dx / c * 1 / np.sin(delta_theta/2), \
                                         delta_theta / (2 * self.k) * dy / c * 1 / np.sin(delta_theta/2), \
                                         (- delta_theta * np.sin(delta_theta) * np.sin(delta_theta/2) / np.power(np.cos(delta_theta)-1, 2) - 2 * np.sin(delta_theta / 2) / (np.cos(delta_theta)-1)) * \
                                             c / (2 * self.k)],
                                        [delta_theta / (2 * self.k) * dx / c * 1 / np.sin(delta_theta/2), \
                                         delta_theta / (2 * self.k) * dy / c * 1 / np.sin(delta_theta/2), \
                                         (- delta_theta * np.sin(delta_theta) * np.sin(delta_theta/2) / np.power(np.cos(delta_theta)-1, 2) - 2 * np.sin(delta_theta / 2) / (np.cos(delta_theta)-1)) * \
                                             c / (2 * self.k)]])
        else:
            Ht = 1 / self.k * np.array([[dx / c, dy / c, 0],
                                        [dx / c, dy / c, 0]])
        return Ht
