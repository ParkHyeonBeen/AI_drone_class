import numpy as np

from WSs.WS_FIP.fip_config import FIP_config
from envs.FIP_envs import Env_FIP
from algorithms.SAC import *
from tools.logger import *

config = FIP_config()
log = logger()
memory = ReplayBuffer()

if __name__ == '__main__':

    env = gym.make('QuadRate-v0')
    # len_s, len_a = env.get_len()
    len_s = len(env.reset())*config.history_num
    len_a = len(env.action_space.sample())

    pi = PolicyNet(len_s, len_a)
    pi.load_state_dict(torch.load(config.path_policy))
    pi.cuda()

    score = 0.0
    num_success = 0
    print_interval = 10

    for n_epi in range(config.iteration):
        s = env.reset()
        done = False
        # first_step = True

        while not done:
            # s_hst = memory.make_history(s, config.history_num, first_step)
            # first_step = False
            _, a, _ = pi.forward(torch.from_numpy(s).cuda().float())
            a = a.cpu().detach().numpy()
            s_prime, r, done, _ = env.step(a)
            env.render()
            score += r
            s = s_prime
            # log.get_data((s, r))

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}, alpha : {:.4f}, num_success : {}".format(n_epi, score / print_interval,
                                                                             pi.log_alpha.exp(), num_success))
            score = 0.0

    env.close()

    # log.create_log_directories()
    # log.log_csv()
    # state_data = log.state_trajectory()
    # visualize_3D(state_data).render()

