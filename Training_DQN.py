###2024.0416
## 每个agent只有一个网络结构，reward只有一个，


from enviroment import enviroment
import argparse
from Trainer_DQN import Trainer_dqn###MADQN
import torch
import numpy as np
from madqn import MADQN
from buffer_DQN import MultiAgentReplayBuffer
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot

def plot_acc_loss(loss,f,i):
    host = host_subplot(111)  # row=1 col=1 first pic
    plt.subplots_adjust(right=0.8)  # adjust the right boundary of the plot window

    # set labels
    host.set_xlabel("episode")
    host.set_ylabel("average_rewards")
    name = 'fanctory'+str(f)+'reward'+str(i)
    # plot curves
    p1, = host.plot(range(len(loss)), loss, label=name)

    # set location of the legend,
    # 1->rightup corner, 2->leftup corner, 3->leftdown corner
    # 4->rightdown corner, 5->rightmid ...
    host.legend(loc=5)

    # set label color
    host.axis["left"].label.set_color(p1.get_color())


    plt.draw()
    plt.show()


def running():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(torch.cuda.is_available())
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--state_num', type=int, default=7, help='the dimension of input')
    parser.add_argument('--rule_num', type=int, default=6, help='the number of rules')
    parser.add_argument('--buffer_size', type=int, default=10000, help='buffer size')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--Initial_jobs', type=int, default=30, help='Initial_jobs')

    parser.add_argument('--max_operation_time', type=int, default=20, help='max time of each operation')
    parser.add_argument('--c', type=int, default=8, help='max number of stage in each episode')
    parser.add_argument('--max_machine_number', type=int, default=3, help='max number of machine in each stage')
    parser.add_argument('--max_job', type=int, default=8)
    parser.add_argument('--max_stage', type=int, default=4)
    parser.add_argument('--DDT', type=int, default=1, choices=[0.5, 1, 1.5])
    parser.add_argument('--lam', type=int, default=20, choices=[50, 100, 200], help='posion distribution')
    parser.add_argument('--add_job', type=int, default=20, choices=[50, 100, 150])
    parser.add_argument('--epsilon', default=0.2)
    parser.add_argument('--max_energy_perunit_time', type=int, default=10, help='max energy consumption per unit time')
    parser.add_argument('--blocking_index', type=float, default=0.3, help='energy index for blocking time')
    parser.add_argument('--idle_index', type=float, default=0.2, help='energy index for idle time')
    parser.add_argument('--reward_size',type=int, default=2,help='reward vector length')
    parser.add_argument('--weight_num',type=int, default=32,help='number of weights')

    parser.add_argument('--epsilon_final', default=0.01)
    parser.add_argument('--epsilon_start', default=0.5)
    parser.add_argument('--epsilon_decay', type=int, default=300)
    parser.add_argument('--gamma', default=0.99)
    # parser.add_argument('--pr', default=pr)
    parser.add_argument('--print_interval', type=int, default=60)
    parser.add_argument('--episode', type=int, default=200)
    parser.add_argument('--learning_rate', default=0.01)
    parser.add_argument('--tor', default=0.001, help='the update level of target_net')
    parser.add_argument('--device', type=str, default=device)

    parser.add_argument('--shop_num', type=int, default=2)
    parser.add_argument('--embedding_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--input_size', type=int, default=11)
    parser.add_argument('--N', type=int, default=1, help='the number of transformer encoder')
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--clip_logits', type=int, default=10,
                        help='improve exploration; clipping logits')


    args = parser.parse_args()

    print('Start simulation')

    actor_dims = []

    for i in range(args.shop_num):
        actor_dims.append([args.max_job, args.max_stage + 3])
    critic_dims = [args.max_job * args.shop_num, args.max_stage + 3]

    n_actions = 1
    # name = 'ppo_e'+ str(times+1)
    maddpg_agents = MADQN(args, args.shop_num, n_actions, chkpt_dir='tmp/madqn/')

    memory = MultiAgentReplayBuffer(1000000, args,
                                        n_actions, batch_size=args.batch_size)

    PRINT_INTERVAL = 50
    N_GAMES = 3000
    total_steps = 0
    score_history = []
    evaluate = False
    best_score = -10000000000
    if evaluate:
        maddpg_agents.load_checkpoint()
    ##定义车间的机器数和加工时间系数以及机器的加工速度水平和机器的单位时间能耗
    machine_on_stage = np.array([[2, 3, 1, 4], [4, 3, 3, 2]])
    # factor = [0.7,1,1.3]
    fac_factor = np.array([[1, 1.3, 0.7, 1], [1.3, 1, 1, 0.7]])

    energy_standard = np.array([4, 7, 5, 9])
    ef = np.array([[1.3, 0.7, 1.3, 0.7], [0.7, 1, 1, 0.7]])
    energy_fac = np.zeros((args.shop_num, args.max_stage))
    for i in range(args.shop_num):
        for s in range(args.max_stage):
            # r = np.random.randint(len(factor))
            energy_fac[i, s] = round(ef[i, s] * energy_standard[s])

    ave_rew = np.zeros((args.episode, 2))


    for ep_num in range(args.episode):
        print('The', ep_num, 'episode')
        env = enviroment()
        device = torch.device(args.device)
        train = Trainer_dqn(env, device, args, ep_num, maddpg_agents, memory, total_steps, evaluate, machine_on_stage,
                            fac_factor, energy_fac)

        # 开始至执行
        env.run()
        total_steps = train.total_steps
        ave_rew[ep_num] = train.episode_rew
        # ep_num += 1



if __name__=='__main__':


    running()




