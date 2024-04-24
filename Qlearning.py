"""Q-Table learning algorithm.
Non deep learning - TD Learning, Off-Policy, e-Greedy Exploration
Q(S, A) <- Q(S, A) + alpha * (R + lambda * Q(newS, newA) - Q(S, A))
See David Silver RL Tutorial Lecture 5 - Q-Learning for more details.
For Q-Network, see tutorial_frozenlake_q_network.py
EN: https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0#.5m3361vlw
CN: https://zhuanlan.zhihu.com/p/25710327
tensorflow==2.0.0a0
tensorlayer==2.0.0
"""

import argparse
import os
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from env import QuantumBoardEnv 
from CnnNet import *


parser = argparse.ArgumentParser()
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_true', default=True)

parser.add_argument(
    '--save_path', default=None, help='folder to save if mode == train else model path,'
                                      'qnet will be saved once target net update'
)
parser.add_argument('--seed', help='random seed', type=int, default=0)
parser.add_argument('--env_id', default='FrozenLake-v1')
args = parser.parse_args()

## Load the environment
alg_name = 'Qlearning'
env_id = args.env_id
env = QuantumBoardEnv()
render = True  # display the game environment

##================= Implement Q-Table learning algorithm =====================##
## Initialize table with all zeros
Q = np.zeros([3 ** 16, 16])
## Set learning parameters
lr = .85  # alpha, if use value function approximation, we can ignore it
lambd = .99  # decay factor
num_episodes = 10000
t0 = time.time()

def do_train():
    if args.train:
        all_episode_reward = []
        for i in range(num_episodes):
            ## Reset environment and get first new observation
            s = env.reset()
            rAll = 0
            ## The Q-Table learning algorithm
            for j in range(9):
                # if render: env.render()
                ## Choose an action by greedily (with noise) picking from Q table
                a = np.argmax(Q[s, :] + np.random.randn(1, 16) * (1. / (i + 1)))

                ## Get new state and reward from environment
                s1, r, done, _ = env.step(a)
                ## Update Q-Table with new knowledge
                Q[s, a] = Q[s, a] + lr * (r + lambd * np.max(Q[s1, :]) - Q[s, a])
                rAll += r
                s = s1
                if done: break
            print(
                'Training  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    i + 1, num_episodes, rAll,
                    time.time() - t0
                )
            )
            if i == 0:
                all_episode_reward.append(rAll)
            else:
                all_episode_reward.append(all_episode_reward[-1] * 0.9 + rAll * 0.1)

        # save
        path = os.path.join('model', '_'.join([alg_name, env_id]))
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(os.path.join(path, 'Q_table.npy'), Q)
        print("保存reward图片")
        plt.plot(all_episode_reward)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', '_'.join([alg_name, env_id])))

        # print("Final Q-Table Values:/n %s" % Q)
def do_test():
    if args.test:
        # path = os.path.join('model', '_'.join([alg_name, env_id]))
        path = "/Users/liuyang/Desktop/课程/研一下/强化学习/大作业/RL_course/model/Qlearning_FrozenLake-v1"
        Q = np.load(os.path.join(path, 'Q_table.npy'))
        for i in range(100):
            ## Reset environment and get first new observation
            s = env.reset()
            rAll = 0
            ## The Q-Table learning algorithm
            for j in range(5):
                ## Choose an action by greedily (with noise) picking from Q table
                a = np.argmax(Q[s, :])
                ## Get new state and reward from environment
                s1, r, done, _= env.step(a)
                ## Update Q-Table with new knowledge
                rAll += r
                s = s1
                if done:
                    break
            print(
                'Testing  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    i + 1, num_episodes, rAll,
                    time.time() - t0
                )
            )


if __name__ == '__main__':
    dqn = DQN()
    dqn.PlayWidthHuman()
    # do_train()
    do_test()

