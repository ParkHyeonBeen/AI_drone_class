import numpy as np
import math, random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from WSs.WS_FIP.fip_config import FIP_config
from tools.logger import *

config = FIP_config()
log = logger()

class Env_FIP:
    def __init__(self):

        """Dynamics of Flying Inverted Pendulum"""
        #initial
        self.sample_time = FIP_config.sample_time
        self.gravity = FIP_config.gravity
        self.pendulum_length = FIP_config.pendulum_length
        self.goal_point = FIP_config.goal_point

        # states of Quadrotor
        self.position_ini = np.zeros(3)
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.attitude = np.zeros(3)

        # states of pendulum
        self.position_pd = np.zeros(2)
        self.velocity_pd = np.zeros(2)

        # middle states
        self.accel = np.zeros(3)
        self.angle_velocity = np.zeros(3)
        self.accel_pd = np.zeros(2)

        # actions
        self.thrust = np.zeros(1)
        self.omega = np.zeros(3)

        """Related to RL for Flying Inverted Pendulum"""

        self.states_len = 0
        self.actions_len = 0

        self.reward = 0
        self.goal_tolerance = FIP_config.goal_tolerance
        self.dist_limit = FIP_config.dist_limit
        self.done = False
        self.success = 0

    def get_len(self):
        self.set_state()
        self.set_action()

        return self.states_len, self.actions_len

    def set_state(self):
        states = np.hstack((self.position, self.velocity, np.cos(self.attitude[:2]), np.sin(self.attitude[:2]), self.angle_velocity[:2]))
        self.states_len = len(states)

        return states

    def set_action(self):
        actions = np.hstack((self.thrust, self.omega))
        self.actions_len = len(actions)

        return actions

    def reset(self):
        # states of Quadrotor
        self.position = np.zeros(3)
        # self.position = np.array([(2*random.random() - 1) for _ in range(3)])
        cheker = False
        while not cheker:
            for i in range(3):
                self.position[i] = 0.9*(2 * random.random() - 1)
                if math.sqrt(sum(self.position**2)) > 0.3:
                    cheker = True

        self.position_ini = self.position
        self.velocity = np.zeros(3)
        self.angle_velocity = np.zeros(3)

        # states of Quadrotor
        self.position_pd = np.zeros(2)
        self.velocity_pd = np.zeros(2)

        # middle states
        self.accel = np.zeros(3)
        self.attitude = np.zeros(3)
        self.accel_pd = np.zeros(2)

        # actions
        self.thrust = np.zeros(1)
        self.omega = np.zeros(3)

        # for RL
        self.reward = 0
        self.done = False
        self.success = 0

        states = self.set_state()

        return states, self.position

    def Attitude_dynamics(self, omega):

        gamma, beta, alpha = self.attitude
        self.omega = omega
        R = np.array([[math.cos(beta)*math.cos(gamma), -math.sin(gamma), 0.],\
                      [math.cos(beta)*math.sin(gamma), math.cos(gamma), 0.],\
                      [-math.sin(beta), 0., 1.]])
        R_inv = np.linalg.inv(R)
        omega = omega.reshape(-1,1)
        attitude_d = R_inv@omega
        self.angle_velocity = attitude_d.reshape(len(self.angle_velocity))
        self.attitude = (self.attitude.reshape(-1, 1) + self.sample_time*attitude_d).reshape(len(self.attitude))

        return self.attitude

    def Translational_dynamics(self, thrust):

        gamma, beta, alpha = self.attitude  # roll = gamma, pitch = beta, yaw = alpha

        R_x = np.array([[1., 0., 0.],\
                        [0., math.cos(gamma), -math.sin(gamma)], \
                        [0., math.sin(gamma), math.cos(gamma)]])

        R_y = np.array([[math.cos(beta), 0., math.sin(beta)],\
                        [0., 1., 0.], \
                        [-math.sin(beta), 0., math.cos(beta)]])

        R_z = np.array([[math.cos(alpha), -math.sin(alpha), 0.],\
                        [math.sin(alpha), math.cos(alpha), 0.],\
                        [0., 0., 1.]])

        R = R_z@R_y@R_x
        thrust_mtx = np.array([0., 0., thrust[0]+self.gravity]).reshape(-1,1)
        gravity_mtx = np.array([0., 0., self.gravity]).reshape(-1,1)

        accel = R@thrust_mtx - gravity_mtx
        self.accel = accel.reshape(len(self.accel))
        velocity = self.velocity.reshape(-1, 1) + self.sample_time * accel
        self.velocity = velocity.reshape(len(self.velocity))
        position = self.position.reshape(-1, 1) + self.sample_time * velocity
        self.position = position.reshape(len(self.position))

        return self.position, self.velocity, self.accel

    def IP_dynamics(self):

        x_dd, y_dd, z_dd = self.accel
        r, s = self.position_pd
        r_d, s_d = self.velocity_pd
        r_dd, s_dd = self.accel_pd

        zeta = math.sqrt(self.pendulum_length**2 - (r**2 + s**2))

        r_dd_new = (-r**4 * x_dd - (self.pendulum_length**2 - s**2)**2 * x_dd - 2 * r**2 * (s * r_d * s_d + (-self.pendulum_length**2 + s**2) * x_dd)\
                    + r**3 * (s_d**2 + s * s_dd**2 - zeta * (self.gravity + z_dd))\
                    + r * (-self.pendulum_length**2 * s * s_dd + s**3 * s_dd + s**2 * (r_d**2 - zeta * (self.gravity + z_dd))\
                    + self.pendulum_length**2 * (-r_d**2 - s_d**2 + zeta * (self.gravity + z_dd)))) / ((self.pendulum_length**2 - s**2) * zeta**2)

        s_dd_new = (-s**4 * y_dd - (self.pendulum_length**2 - r**2)**2 * y_dd - 2 * s**2 * (r * r_d * s_d + (-self.pendulum_length**2 + r**2) * y_dd)\
                    + s**3 * (r_d**2 + r * r_dd**2 - zeta * (self.gravity + z_dd))\
                    + s * (-self.pendulum_length**2 * r * r_dd + r**3 * r_dd + r**2 * (s_d**2 - zeta * (self.gravity + z_dd))\
                    + self.pendulum_length**2 * (-r_d**2 - s_d**2 + zeta * (self.gravity + z_dd)))) / ((self.pendulum_length**2 - r**2) * zeta**2);

        self.accel_pd = np.array([r_dd_new, s_dd_new])
        # print("accel : ", self.accel)
        # print("accel_pd : ", self.accel_pd)
        self.velocity_pd = self.velocity_pd + self.sample_time * self.accel_pd
        self.position_pd = self.position_pd + self.sample_time * self.velocity_pd


        return self.position_pd, self.velocity_pd

    def _get_reward(self, action):

        distance_ini = math.sqrt(sum((self.position_ini - self.goal_point) ** 2))
        distance_toGoal = math.sqrt(sum((self.position-self.goal_point)**2))
        radius = math.sqrt(sum(self.position_pd**2))
        # reward_pd = self.goal_tolerance/(radius+self.goal_tolerance) if radius < self.goal_tolerance else 0.0

        # self.done = True if radius > self.dist_limit else False

        if max(abs(self.position)) > 1:
            self.done = True
            # limit_penalty = -5.0 * self.done
            # print("The drone is out of line, current position : {0}".format(self.position))
            # fail_penalty = -2.0
            # self.reward += limit_penalty
        else:
            self.done = False

        if (distance_toGoal - distance_ini) > 0:
            reward_direction = -2-(distance_toGoal - distance_ini) ** 3 \
                     - 100.0 * self.done
        else:
            reward_direction = 2 - (distance_toGoal - distance_ini) ** 3

        reward_move = - 0.02 * math.sqrt(sum(action ** 2)) ** 0.5 - 0.05 * math.sqrt(sum(self.angle_velocity ** 2)) \

        reward = reward_direction + reward_move

        if distance_toGoal < 0.1:
            self.reward += 1000.0
            self.success += 1
            self.done = True
        # self.reward += reward_dist

        self.reward += reward

        return reward

    def step(self, actions):

        thrust = 1.0*actions[:len(self.thrust)]
        omega = 1.0*actions[len(self.thrust):]

        self.Attitude_dynamics(omega)
        self.Translational_dynamics(thrust)
        # self.IP_dynamics()

        s_prime = self.set_state()
        reward = self._get_reward(actions)

        return s_prime, reward, self.done, self.success

    # def call_matlab(self):
    #     eng = matlab.engine.start_matlab()
    #     print(eng.__file__)

    def close(self):
        pass

def main():

    env = Env_FIP()
    states, position_ini = env.reset()


    for _ in range(1000):
        action = np.array([0, 1, 1, 0])
        s_prime, reward, _, _ = env.step(action)
        env.render()
        log.get_data((s_prime, reward))


if __name__ == '__main__':
    main()

