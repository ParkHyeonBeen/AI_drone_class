import time

import numpy as np
from algorithms.SAC import *
from tools.logger import *
from tools.buffer import *

config = FIP_config()
log = logger()
memory = ReplayBuffer()

if __name__ == '__main__':

    env = gym.make('QuadRate-v0')
    len_s = len(env.reset())*config.history_num
    len_a = len(env.action_space.sample())

    memory = ReplayBuffer()
    q1, q2, q1_target, q2_target = Qnet(len_s, len_a), Qnet(len_s, len_a), Qnet(len_s, len_a), Qnet(len_s, len_a)
    q1.cuda(), q2.cuda(), q1_target.cuda(), q2_target.cuda()
    pi = PolicyNet(len_s, len_a)
    pi.cuda()

    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())

    score = 0.0
    print_interval = 10
    eval_interval = 50

    for n_epi in range(config.iteration):
        start_time = time.time()

        s = env.reset()
        done = False
        first_step = True
        local_step = 0

        for timestep in range(8000):
            s_hst = memory.make_history(s, config.history_num, first_step)
            first_step = False
            a = pi.forward(torch.from_numpy(s_hst).cuda().float())[0]
            a = a.cpu().detach().numpy()

            s_prime, r, done, _ = env.step(8.0*a)
            local_step += 1
            if local_step == 8000:
                done = False
            # env.render()
            memory.put((s, a, 10*r, s_prime, done))
            score += r
            s = s_prime

            if memory.size() > 1000:
                minibatch = memory.sample_history(config.batch_size, config.history_num)
                td_target = calc_target(pi, q1_target, q2_target, minibatch)
                q1.train_net(td_target, minibatch)
                q2.train_net(td_target, minibatch)
                pi.train_net(q1, q2, minibatch)
                q1.soft_update(q1_target)
                q2.soft_update(q2_target)

            if done:
                break

        if n_epi % print_interval == 0 and n_epi != 0:
            process_time = time.time() - start_time

            print("# of episode :{}, avg score : {:.1f}, alpha : {:.4f}, buffer size : {}".format(n_epi, score / print_interval,
                                                                             pi.log_alpha.exp(), memory.size()))
            score = 0.0

        if n_epi % eval_interval == 0 and n_epi != 0:
            total_reward = 0
            for _ in range(10):
                s = env.reset()
                first_step = True
                done = False
                while not done:
                    s_hst = memory.make_history(s, config.history_num, first_step)
                    first_step = False
                    a = pi.forward(torch.from_numpy(s_hst).cuda().float())[1]
                    a = a.cpu().detach().numpy()
                    s_prime, r, done, _ = env.step(a)
                    # env.render()
                    total_reward += r
                    s = s_prime

            print("[EVALUATION] TOTAL_REWARD : %.2f"%(total_reward/10))

    env.close()

    log.create_log_directories()
    # log.log_csv()
    # state_data = log.state_trajectory()
    # visualize2(state_data).render()
    log.log_policy(pi)

