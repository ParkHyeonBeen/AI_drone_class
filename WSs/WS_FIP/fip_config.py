import numpy as np
import math

class FIP_config:

    # Related initial setup
    sample_time = 0.01
    gravity = 9.81
    pendulum_length = 0.8
    goal_point = np.array([0, 0, 0])

    # Related to reward
    goal_tolerance = 0.1
    dist_limit = pendulum_length*math.cos(math.pi * (45 / 180))

    # Related to Algorithms
    buffer_limit = 1000000
    batch_size = 256
    learning_rate_pi = 3e-4        # default = 3e-4
    learning_rate_q = 3e-4         # default = 3e-4
    learning_rate_alpha = 3e-5    # default = 3e-4
    initial_alpha = 0.1
    target_min_entropy = -3.0
    gamma = 0.99
    tau = 0.005                      # default = 0.005
    iteration = 100000
    history_num = 5

    path = "C:/Users/owner/Desktop/Workspace_paper/"
    path_policy = path + "results/20211114-121758/policy/policy"

