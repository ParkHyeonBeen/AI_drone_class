import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Hyperparameters
learning_rate = 0.0002
gamma = 0.98
n_rollout = 10


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(4, 256)
        self.fc_pi = nn.Linear(256, 2)  # output 2개 이유 : action의 종류가 2개( 왼쪽, 오른쪽 )
        self.fc_v = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim) # dim = 0 : 1차원에 대해 softmax 적용
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], [] # done : episode 종료여부
        for transition in self.data:    # list에서 직접 for로 값을 가져올 경우, list의 한 원소씩 가져옴
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r / 100.0])   # 1/100은 별 의미 없음, 목적식에 다른 항이 있는 것이 아니니 learning rate를 바꿔준 것과 같음
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0    # 마지막에 0을 넣는거는 단순히 episode 넘어가기 위한 marking 용도
            done_lst.append([done_mask])

        s_batch, a_batch, r_batch, s_prime_batch, done_batch = torch.tensor(s_lst, dtype=torch.float), torch.tensor(
            a_lst), \
                                                               torch.tensor(r_lst, dtype=torch.float), torch.tensor(
            s_prime_lst, dtype=torch.float), \
                                                               torch.tensor(done_lst, dtype=torch.float)
        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def train_net(self):
        s, a, r, s_prime, done_mask = self.make_batch()
        print("done_mask : ", done_mask, done_mask.size())
        print("s : ", s, s.size())
        print("s_prime : ", s_prime, s_prime.size())
        td_target = r + gamma * self.v(s_prime) * done_mask
        delta = td_target - self.v(s)

        pi = self.pi(s, softmax_dim=1)  # dim = 0 일 경우, 전체 matrix에 대한 인덱스
                                        # 아래 방향으로 indexing ex_ (0,0) = 0, (0,1) = 1, (1,0) = 10
                                        # dim = 1인 경우, 1차원(벡터)에 대한 index
        pi_a = pi.gather(1, a)
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        # .Required_grad : True로 세팅하면, 텐서의 모든 연산에 대하여 추적을 시작
        # .backward() : 계산 작업 완료 후 호출하여 모든 gradient를 자동으로 계산
        # .grad : 텐서를 위한 gradient 속성이 누적되어 저장
        # .detach() : 텐서에 대하여 기록 추적을 중지, 현재의 계산 기록으로 부터 분리
        # .backward() : 도함수(derivatives)을 계산할 때 사용, Tensor가 scalar 라면
        #               아무 parameter도 필요하지 않음. scalar가 아니라면, 올바른 형태의 tensor가 필요


def main():
    env = gym.make('CartPole-v1')
    model = ActorCritic()
    print_interval = 20
    score = 0.0

    for n_epi in range(10000):
        print("n_epi : ", n_epi)
        done = False
        s = env.reset()
        env.render()
        while not done:
            print("while not done")
            for t in range(n_rollout):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)  # 카테고리 분포로 변환 : 확률 tuple에 따라 index를 추출
                a = m.sample().item()   # items()함수를 사용하면 딕셔너리에 있는 키와 값들의 쌍을 얻을 수 있다.
                s_prime, r, done, info = env.step(a)
                print("done : ", done)
                model.put_data((s, a, r, s_prime, done))

                s = s_prime
                score += r

                if done:
                    break

            model.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score / print_interval))
            score = 0.0
    env.close()


if __name__ == '__main__':
    main()