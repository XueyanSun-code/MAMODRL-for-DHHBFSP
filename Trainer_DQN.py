import copy

from Machine import Machine
from JobShop import JobShop
from DDQN import DDQN
import torch
from buffer import ReplayBuffer
from Gantt import Gante
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
from HGA import BigroupGeneticAlgorithm
from Job import Job
# from maddpg import MADDPG
from madqn import MADQN
from buffer_DDPG import MultiAgentReplayBuffer
import simpy
from Normalizations import RewardScaling, Normalization


# import matplotlib.pyplot as plt


class Trainer_dqn:
    def __init__(self, env, device, args, ep_num, maddpg_agents, memory, total_steps, evaluate, machine_on_stage,
                 fac_factor, energy_fac):
        self.args = args
        self.device = device
        self.ep_num = ep_num
        self.env = env
        self.maddpg = maddpg_agents
        self.memory = memory
        self.episode_end = self.env.event()
        # self.jobs=[]
        self.stage = 0
        self.numjob = 0
        self.machine_on_stage = machine_on_stage
        self.process_time = None
        self.job_list = None
        # 不同车间的加工时间系数，因为工件在不同车间的加工时间不同
        self.factor = fac_factor
        self.energy_fac = energy_fac
        self.reset_world()
        # self.state=torch.zeros(self.args.state_num).to(self.device)
        self.total_steps = total_steps
        self.evaluate = evaluate

        self.train_tardness = torch.zeros(args.episode).to(self.device)
        self.test_tardness = torch.zeros(args.episode).to(self.device)

        self.multiShop = [
            JobShop(self.device, env, self.args, self.stage, self.machine_on_stage[i], i, fac_factor[i], energy_fac[i])
            for i in range(args.shop_num)]
        self.gante = Gante()

        self.mean_loss = torch.zeros(args.episode).to(self.device)
        self.ep_count = 0
        self.step_num = 0
        self.epsilon_by_frame = lambda frame_idx: self.args.epsilon_final + frame_idx * (
                    self.args.epsilon_start - self.args.epsilon_final) / self.args.epsilon_decay
        self.new_job = []
        self.new_arrvial = False
        ##这里产生的是add_job个时间间隔，但是因为后面两个进程中都有new-comer。所以会产生同时产生两个新工件，导致多加了一个工件，因此，在产生新工件时，增加一个随机时间扰动
        self.exp_distribution = self.job_distribution(self.args.lam, self.args.add_job)
        self.add = 0
        # self.reward = 0
        self.epsilon = self.args.epsilon_start - ep_num * (
                    self.args.epsilon_start - self.args.epsilon_final) / self.args.epsilon_decay
        ##暂存buffer
        self.temp_size = 1000
        self.init_temp_reward_memory()
        self.gamma_reward = np.zeros(2)
        self.reward = np.zeros(2)

        self.episode_rew = np.zeros(2)
        self.objectives = np.zeros(2)
        self.actor_store_index = np.zeros(args.shop_num, dtype=int)
        self.init_rew_memory()
        self.step = 0

        self.multi_rew_scaling = [RewardScaling(2, 0.95) for _ in range(args.shop_num)]
        self.multi_obs = [Normalization((args.max_job, args.max_stage + 7)) for _ in range(args.shop_num)]

        ##全局状态

        for i in range(self.args.shop_num):
            self.env.process(self.whole(i))

        self.rule = 0
        self.jobshop_end = np.zeros(args.shop_num)

    def init_temp_reward_memory(self):

        self.actor_reward_mamory = []

        for i in range(self.args.shop_num):
            self.actor_reward_mamory.append([])

    def init_rew_memory(self):
        self.rew_store = []
        for i in range(self.args.shop_num):
            self.rew_store.append(np.zeros((1000, 2)))

    def reset_world(self):
        self.numjob = np.random.randint(20, self.args.Initial_jobs)
        self.stage = self.args.max_stage

        ##随机生成工件的标准加工时间,
        self.process_time = np.random.randint(1, self.args.max_operation_time, size=(self.numjob, self.stage))

        ##根据以上随机信息通过HGA产生初始方案
        HGA = BigroupGeneticAlgorithm(6, 7, 0.2, 0.1, 0.1, 0.3, self.args.shop_num, self.numjob, self.stage,
                                      self.machine_on_stage[0], self.process_time)
        self.job_list, fitness = HGA.BGA_running()


    # 根据截止对工件进行排序，知道所有机器无空闲
    def Initial_patching(self, shop_ID):

        candidate = list(range(len(self.multiShop[shop_ID].jobs)))
        # torch.tensor(candidate, dtype=torch.long).to(self.device)

        machine_state = torch.zeros(self.machine_on_stage[shop_ID][0]).to(self.device)
        for index in range(len(self.multiShop[shop_ID].jobs)):
            for i in range(self.multiShop[shop_ID].gantt_machine[0]):

                operation = self.env.process(
                    self.multiShop[shop_ID].jobs[candidate[index]].create_operation_process(self.multiShop[shop_ID],
                                                                                            i + 1, self.factor))
                self.multiShop[shop_ID].jobs[candidate[index]].update_operation_process(1, operation)
                self.env.process(self.operation_finished(operation, self.args.epsilon_start, shop_ID))
                # print('finish time', self.env.now)
                machine_state[i] = 1
                break
            if torch.sum(machine_state) == self.machine_on_stage[shop_ID][0]:
                break

    def whole(self, shop_ID):
        self.step_num = 0
        for _ in range(self.machine_on_stage[shop_ID][0]):
            self.multiShop[shop_ID].machines.append(Machine(self.env, self.device))

        yield self.env.process(self.train_and_test( shop_ID))


    def train_and_test(self, shop_ID):
        self.ep_count = 0
        # self.losses=torch.zeros(self.args.max_operation_number*(self.args.Initial_jobs+self.args.add_job)).to(self.device)
        self.env.reset_now(0)
        self.multiShop[shop_ID].jobs = []


        for number, job in enumerate(self.job_list[shop_ID]):
            print(self.process_time[job - 1])
            self.multiShop[shop_ID].jobs.append(
                self.multiShop[shop_ID].creat_job(self.args.DDT, self.process_time[job - 1], job, number,
                                                  self.env.now, 0))

        for machine in self.multiShop[shop_ID].machines:
            machine.reset_machine()

        self.Initial_patching(shop_ID)
        yield self.env.process(self.new_comer(self.evaluate))


    def new_comer(self, evaluate):

        while self.add < self.args.add_job:

            rand = np.random.randint(10)

            yield self.env.timeout(self.exp_distribution[self.add] + rand)
            # print('after new', self.env.now, 'shop_ID')
            self.add += 1
            if not evaluate:
                self.new_job.append(self.creat_new_job(self.args.DDT, self.args.max_operation_time,
                                                       self.numjob + self.add, self.env.now, 1))

            self.new_arrvial = True

        yield self.episode_end

    def creat_new_job(self, DDT, max_operation_time, job_number, time, original):
        processing_information = 0

        all_process_time = np.random.randint(1, max_operation_time, size=self.stage)

        processing_information = all_process_time[0]

        job = Job(self.env, processing_information, DDT, time, job_number, self.device, all_process_time, original)
        return job

    def job_distribution(self, lam, add_job):
        exp_distribution = np.around(np.random.exponential(lam, size=add_job)).astype(int)  # 50 个参数为 lam 的指数分布随机数
        # exp_distribution = np.concatenate(([0], exp_distribution))
        return exp_distribution


    def operation_finished(self, operation, epsilon, shop_ID):
        yield operation
        # print('finish time', self.env.now,'shop_ID',shop_ID)
        if not self.new_arrvial:
            self.env.process(self.executing_ddqn1(epsilon, shop_ID))
        else:
            self.env.process(self.executing_MADDPG(epsilon, shop_ID))
        end1 = 0
        for job in self.multiShop[shop_ID].jobs:
            end1 += job.finished
        if end1 == len(self.multiShop[shop_ID].jobs):
            self.jobshop_end[shop_ID] = 1
        end2 = 0
        for job in self.new_job:
            end2 += job.finished
        if min(self.jobshop_end) == 1 and end2 == self.args.add_job:
            self.episode_rew = self.episode_rew / self.step
            ##更新
            objectives = self.objectives / self.step
            reward = 1. / objectives
            self.rew_update(reward)
            self.memory.reward_update(self.rew_store)
            self.episode_end.succeed()
            self.episode_end = self.env.event()


    def rew_update(self, reward):
        # reward = copy.deepcopy(self.rew_store)
        for s in range(self.args.shop_num):
            for i in range(self.actor_store_index[s]):
                self.rew_store[s][i] = self.rew_store[s][i] + reward

    def reward_average(self):
        reward = copy.deepcopy(self.actor_reward_mamory)
        for s, shop_reward in enumerate(self.actor_reward_mamory):
            for i in range(len(shop_reward)):
                reward[s][i] = sum(shop_reward[i:]) / (len(shop_reward) - i) + shop_reward[i]
        return reward

    def get_obs(self, shop_ID, UC_job, rew_list):

        self.multiShop[shop_ID].activite = True

        obs, mask = self.multiShop[shop_ID].observations(UC_job, self.reward, rew_list)

        return obs, mask

    def get_reward(self, decided_job, shop_id, machine_num):
        ##当前step的即时reward

        reward = self.multiShop[shop_id].reward(decided_job, machine_num)
        return reward

    def get_done(self, shop_id):
        ##判断当前的车间的工件和新工件是否全部完成
        done = False
        end1 = 0

        for job in self.multiShop[shop_id].jobs:
            end1 += job.finished
        end2 = 0
        for job in self.new_job:
            end2 += job.finished
        if end1 == self.multiShop[shop_id].jobs and end2 == self.args.add_job:
            done = True
        return done

    def obs_list_to_state_vector(self, observation):
        state = observation[0]
        for obs in observation[1:]:
            state = np.concatenate([state, obs])
        return state

    def executing_MADDPG(self, epsilon, shop_ID):
        yield self.env.timeout(0)

        UC_job = self.multiShop[shop_ID].get_new_uncompleted_jobs(self.new_job)
        UC = []
        for job in UC_job:
            UC.append(job.number)

        if len(UC_job) != 0:
            reward_list = []
            obj_list = []
            for job in UC_job:
                decided_operation, decided_machine = self.multiShop[shop_ID].choose_machine(job)
                rew, obj = self.get_reward(job, shop_ID, decided_machine)
                reward_list.append(rew)
                obj_list.append(obj)
            obs, mask = self.get_obs(shop_ID, UC_job, reward_list)

            ##根据当前的状态选择工件进行加工，这里的action是UCjob的顺序号，需要转换,且目标值需要减去1
            action = self.maddpg.choose_action(obs, mask, shop_ID)
            # print(actions[shop_ID][0])
            decided_job = UC_job[action]
            decided_operation, decided_machine = self.multiShop[shop_ID].choose_machine(decided_job)
            reward = reward_list[action]
            self.objectives += obj_list[action]

            if decided_job.construct_count != decided_operation and action != None:
                operation = self.env.process(
                    decided_job.create_operation_process(self.multiShop[shop_ID], decided_machine, self.factor))
                decided_job.update_operation_process(decided_operation, operation)
                yield self.env.timeout(0)
                # 当工序进程结束后会执行规则
                self.env.process(self.operation_finished(operation, epsilon, shop_ID))
                yield self.env.timeout(0)
                obs_ = copy.deepcopy(obs)
                obs_[action] = np.zeros(self.args.max_stage + 7)
                mask_ = copy.deepcopy(mask)
                mask_[action] = 0

                reward_o = self.multi_rew_scaling[shop_ID].__call__(reward)
                # gamma = 0.95
                # reward = reward_o + self.args.pr * self.gamma_reward  ##经过scaling的单步reward加上前面的reward的影响
                self.reward = reward_o  ##保存当前一步获得的总reward值
                self.rew_store[shop_ID][self.actor_store_index[shop_ID]] = reward
                self.actor_store_index[shop_ID] += 1
                # self.gamma_reward = self.args.pr * (reward_o + self.gamma_reward)  ##更新前面所有reward的影响值
                self.episode_rew += reward
                self.step += 1

                # self.actor_reward_mamory[shop_ID].append(reward)
                done = self.get_done(shop_ID)

                ##这里的obs需要flatten
                self.memory.store_transition(obs.flatten(), mask, action, reward, obs_.flatten(), mask_, done, shop_ID)
                if self.total_steps % 20 == 0 and not self.evaluate:

                    self.maddpg.learn(self.memory)

                    print('######################################main' + str(self.total_steps))
                self.total_steps += 1
                if (self.total_steps) % 1000 == 0:
                    # date = datetime.now().strftime('%m%d_%H_%M')
                    print('save model-----------------------------')

                    self.maddpg.save_checkpoint()

    def executing_ddqn1(self, epsilon, shop_ID):
        yield self.env.timeout(0)
        rule = self.rule  # self.ddqn.act(self.state, epsilon)
        UC_job = self.multiShop[shop_ID].get_uncompleted_jobs()
        if len(UC_job) != 0:

            decided_job, decided_operation, decided_machine = self.multiShop[shop_ID].execute_rules(self.env, rule,
                                                                                                    self.multiShop[
                                                                                                        shop_ID].machines)
            reward, obj = self.get_reward(decided_job, shop_ID, decided_machine)
            self.objectives += obj
            # print('decided_job.',decided_job)
            if decided_job.construct_count != decided_operation:
                operation = self.env.process(
                    decided_job.create_operation_process(self.multiShop[shop_ID], decided_machine, self.factor))
                decided_job.update_operation_process(decided_operation, operation)
                yield self.env.timeout(0)
                # 当工序进程结束后会执行规则
                self.env.process(self.operation_finished(operation, epsilon, shop_ID))
                yield self.env.timeout(0)










