import copy
import time

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
from maddpg import MADDPG
from buffer_DDPG import MultiAgentReplayBuffer
import simpy
from DispatchingRules import Rules
from Normalizations import RewardScaling,Normalization

# import matplotlib.pyplot as plt



class Trainer_dispatch:
    def __init__(self, env, device, args,total_steps, evaluate, machine_on_stage, fac_factor,
                 energy_fac, job_list, process_time, add_jobs, process_time_new, ep_num):
        self.Dispatchrules = Rules(device,args,machine_on_stage,fac_factor, energy_fac)
        self.args = args
        self.device = device
        self.ep_num = ep_num
        self.env = env

        self.episode_end = self.env.event()
        self.evaluate = evaluate
        # self.jobs=[]
        self.stage = 4
        self.numjob = 20
        self.machine_on_stage = machine_on_stage
        # self.machine_on_stage = None
        self.process_time = process_time
        self.job_list = job_list
        # self.reset_world()
        # self.state=torch.zeros(self.args.state_num).to(self.device)
        self.total_steps = total_steps
        self.add_jobs = add_jobs  ##新到达工件个数
        self.process_time_new = process_time_new

        self.factor = fac_factor
        self.energy_fac = energy_fac
        # self.memory = ReplayBuffer(self.args.buffer_size,self.device)
        # self.losses=torch.zeros(args.max_operation_number*(args.Initial_jobs+args.add_job)).to(self.device)
        self.train_tardness = torch.zeros(args.episode).to(self.device)
        self.test_tardness = torch.zeros(args.episode).to(self.device)
        # 创建一个车间
        # self.JobShop = JobShop(self.device,env)
        # 创建多个车间
        self.multiShop = [
            JobShop(self.device, env, self.args, self.stage, self.machine_on_stage[i], i, fac_factor[i], energy_fac[i])
            for i in range(args.shop_num)]
        self.gante = Gante()
        # self.tb_writer = SummaryWriter(log_dir='logs')
        # self.ddqn = DDQN(args.learning_rate, args.tor, self.tb_writer, self.device)
        self.mean_loss = torch.zeros(args.episode).to(self.device)
        self.ep_count = 0
        self.step_num = 0
        self.epsilon_by_frame = lambda frame_idx: self.args.epsilon_final + frame_idx * (
                    self.args.epsilon_start - self.args.epsilon_final) / self.args.epsilon_decay
        self.new_job = []
        self.new_arrvial = False
        ##这里产生的是add_job个时间间隔，但是因为后面两个进程中都有new-comer。所以会产生同时产生两个新工件，导致多加了一个工件，因此，在产生新工件时，增加一个随机时间扰动
        self.exp_distribution = self.job_distribution(self.args.lam, self.add_jobs)
        self.add = 0
        self.reward = np.zeros(2)
        self.episode_rew = np.zeros((args.shop_num, 2))
        self.objectives = np.zeros(2)  ###存储
        self.step = 0
        self.all_t = 0
        ##暂存buffer,需要episode结束后计算adv再进行存储
        # self.temp_size = 1000
        # self.actor_store_index = np.zeros(args.shop_num,dtype=int)
        #
        # self.init_temp_memory()
        self.multi_rew_scaling = [RewardScaling(2, 0.95) for _ in range(args.shop_num)]
        self.multi_obs = [Normalization((args.max_job, args.max_stage + 9)) for _ in range(args.shop_num)]

        ##全局状态

        for i in range(self.args.shop_num):
            self.env.process(self.whole(i))

        self.rule = 0
        self.jobshop_end = np.zeros(args.shop_num)

        # self.test_jobs=[]
        # self.test_add_job=[]
        # self.test_exp_distribution = self.JobShop.job_distribution(args.lam, args.add_job)
        # self.test_exp_distribution_total=np.cumsum(self.test_exp_distribution)
        # self.test_JobShop()

    # def test_JobShop(self):
    #     for i in range(self.args.Initial_jobs):
    #         self.test_jobs.append(
    #             self.JobShop.creat_job(self.args.DDT, self.args.max_operation_time,
    #                                    self.args.machine_number, i + 1,0))
    #     for i in range(self.args.add_job):
    #         self.test_add_job.append(
    #             self.JobShop.creat_job(self.args.DDT, self.args.max_operation_time,
    #                                    self.args.machine_number, self.args.Initial_jobs+i + 1,self.test_exp_distribution_total[i]))

    def init_temp_reward_memory(self):

        self.actor_reward_mamory = []
        # self.actor_terminal = []
        # self.actor_store_index = []##因为输入全为0时actor不存储，为了与state保持一直，记录每次的存储序号

        for i in range(self.args.shop_num):

            self.actor_reward_mamory.append([])
            # self.actor_terminal.append(np.zeros(self.temp_size, dtype=bool))
            # self.actor_store_index.append(np.zeros((self.mem_size)))



    #根据截止对工件进行排序，知道所有机器无空闲
    def Initial_patching(self,env, train_or_test,shop_ID):
        # evaluate_value = torch.zeros(self.args.Initial_jobs).to(self.device)
        # for item,job in enumerate(self.jobs):
        #     sum=0
        #     for i in range(job.ni):
        #         sum+=job.calculate_mean_tij(i)
        #     evaluate_value[item]=sum
        candidate = list(range(len(self.multiShop[shop_ID].jobs)))
        # torch.tensor(candidate, dtype=torch.long).to(self.device)

        machine_state=torch.zeros(self.machine_on_stage[shop_ID][0]).to(self.device)
        for index in range(len(self.multiShop[shop_ID].jobs)):
            for i in range(self.multiShop[shop_ID].gantt_machine[0]):
                # if item != 0 and machine_state[i]==0:
                    # print('start time', self.env.now)
                operation=self.env.process(self.multiShop[shop_ID].jobs[candidate[index]].create_operation_process(self.multiShop[shop_ID], i+1,self.factor))
                self.multiShop[shop_ID].jobs[candidate[index]].update_operation_process(1,operation)
                self.env.process(self.operation_finished(operation, self.args.epsilon_start, train_or_test,shop_ID))
                # print('finish time', self.env.now)
                machine_state[i]=1
                break
            if torch.sum(machine_state)==self.machine_on_stage[shop_ID][0]:
                break

    def whole(self,shop_ID):
        self.step_num=0
        for _ in range(self.machine_on_stage[shop_ID][0]):
            self.multiShop[shop_ID].machines.append(Machine(self.env, self.device))
        # for ep_num in range(self.args.episode):
        train_or_test = True
        #print('start time', self.env.now)
        yield self.env.process(self.train_and_test(train_or_test,shop_ID))

        # self.rule+=1
        # self.gante.gante(self.multiShop[shop_ID].machines_start_finish_blocking_time,self.multiShop[shop_ID].machine_joblist)
    # self.ddqn.save_model()
    # self.gante.gante(self.machines)

    def train_and_test(self, train_or_test, shop_ID):
        self.ep_count = 0
        # self.losses=torch.zeros(self.args.max_operation_number*(self.args.Initial_jobs+self.args.add_job)).to(self.device)
        self.env.reset_now(0)
        self.multiShop[shop_ID].jobs = []
        # gc.collect()
        if train_or_test:
            # 初始化数据

            # print('joblist', self.job_list[shop_ID])
            for number, job in enumerate(self.job_list[shop_ID]):
                print(self.process_time[job - 1])
                self.multiShop[shop_ID].jobs.append(
                    self.multiShop[shop_ID].creat_job(self.args.DDT, self.process_time[job - 1], job, number,
                                                      self.env.now, 0))
            # print('shop_ID',shop_ID,len(self.multiShop[shop_ID].jobs))
        # else:
        #     self.test_reset_jobs(ep_num)
        #     self.jobs=self.test_jobs
        for machine in self.multiShop[shop_ID].machines:
            machine.reset_machine()

        self.Initial_patching(self.env, train_or_test, shop_ID)
        yield self.env.process(self.new_comer())

    def new_comer(self):

        while self.add < self.add_jobs - 1:
            # if add == 0:
            #
            #     yield self.env.timeout(exp_distribution[add])
            rand = np.random.randint(10)
            # print('before new', self.env.now, 'shop_ID')
            # print('add',self.add)
            yield self.env.timeout(self.exp_distribution[self.add] + rand)
            # print('after new', self.env.now, 'shop_ID')
            self.add += 1

            self.new_job.append(self.creat_new_job(self.args.DDT, self.process_time_new[self.add - 1],
                                                   self.numjob + self.add, self.env.now, 1))

            # else:
            #     self.new_job.append(self.test_add_job[self.add - 1])
            # epsilon = self.args.epsilon_final
            ##第一个新工件到达后，需要等到下一个决策点再调用executing_MADDPG，且只此一次
            self.new_arrvial = True
            # self.env.process(self.executing_MADDPG(epsilon,train_or_test,shop_ID))
        yield self.episode_end

    def creat_new_job(self, DDT, all_process_time, job_number, time, original):
        processing_information = 0

        # all_process_time = np.random.randint(1, max_operation_time, size=self.stage)
        ##继续产生其他阶段的加工时间

        # for i in range(self.machine_on_stage[shop_ID][0]):
        # print('all', all_process_time[0])
        processing_information = all_process_time[0]

        job = Job(self.env, processing_information, DDT, time, job_number, self.device, all_process_time, original)
        return job

    def job_distribution(self,lam, add_job):
        exp_distribution = np.around(np.random.exponential(lam, size=add_job)).astype(int)  # 50 个参数为 lam 的指数分布随机数
        #exp_distribution = np.concatenate(([0], exp_distribution))
        return exp_distribution

    def new_come(self,ep_num,train_or_test,shop_ID):
        add = 0
        if train_or_test:
            exp_distribution = self.multiShop[shop_ID].job_distribution(self.args.lam, self.args.add_job)
        else:
            exp_distribution=self.test_exp_distribution
        #print(exp_distribution)

        while add < self.args.add_job:
            # if add == 0:
            #
            #     yield self.env.timeout(exp_distribution[add])
            # print('before new',self.env.now,'shop_ID',shop_ID)
            yield self.env.timeout(exp_distribution[add])
            # print('after new', self.env.now,'shop_ID',shop_ID)
            add += 1
            if train_or_test:
                self.new_job.append(self.multiShop[shop_ID].new_job(self.args.DDT, self.args.max_operation_time,
                                      self.numjob + add,len(self.job_list[shop_ID])+add,self.env.now, 1 ))
                if ep_num>self.args.epsilon_decay:
                    epsilon=self.args.epsilon_final
                else:
                    epsilon = self.epsilon_by_frame(ep_num)
            else:
                self.multiShop[shop_ID].jobs.append(self.test_add_job[add-1])
                epsilon=self.args.epsilon_final
            ##第一个新工件到达后，需要等到下一个决策点再调用executing_MADDPG，且只此一次
            self.new_arrvial = True
            # self.env.process(self.executing_MADDPG(epsilon,train_or_test,shop_ID))
        yield self.episode_end

    def operation_finished(self,operation, epsilon,train_or_test,shop_ID):
        yield operation
        # print('finish time', self.env.now,'shop_ID',shop_ID)
        if not self.new_arrvial:
            self.env.process(self.executing_ddqn(epsilon,train_or_test,shop_ID))
        else:
            self.env.process(self.executing_MADDPG( epsilon,train_or_test, shop_ID))
        end1 = 0
        for job in self.multiShop[shop_ID].jobs:
            end1 += job.finished
        if end1 == len(self.multiShop[shop_ID].jobs):
            self.jobshop_end[shop_ID] = 1
        end2 = 0
        for job in self.new_job:
            end2 += job.finished
        if min(self.jobshop_end) == 1 and end2 == self.args.add_job:
            ##计算每个车间内后续reward的平均值
            # reward = self.reward_average()
            # ##更新buffer的reward
            # self.memory.reward_update(reward)
            self.episode_end.succeed()
            self.episode_end = self.env.event()

    def new_operation_finished(self,epsilon,operation,train_or_test,shop_ID):
        yield operation
        # print('new fififiif time', self.env.now,'shop ',shop_ID)
        self.env.process(self.executing_MADDPG(epsilon,train_or_test,shop_ID))
        end = 0
        for job in self.multiShop[shop_ID].jobs:
            end += job.finished
        if end == len(self.multiShop[shop_ID].jobs):
            self.jobshop_end[shop_ID] = 1
            # self.multiShop[shop_ID].done = True
        if min(self.jobshop_end) == 1:
            ##每个episode结束后reward更新
            ##需要一个暂存空间存储当前episode的buffer，更新后再存到训练的buffer
            ##计算每个车间内后续reward的平均值
            # reward = self.reward_average()
            # ##更新buffer的reward
            # self.memory.reward_update(reward)

            self.episode_end.succeed()
            self.episode_end = self.env.event()

    def reward_average(self):
        reward = copy.deepcopy(self.actor_reward_mamory)
        for s,shop_reward in enumerate(self.actor_reward_mamory):
            for i in range(len(shop_reward)):
                reward[s][i] = sum(shop_reward[i:])/(len(shop_reward)-i) + shop_reward[i]
        return reward



    def get_obs(self,shop_ID,UC_job):

        self.multiShop[shop_ID].activite = True

        obs, mask = self.multiShop[shop_ID].observations(UC_job,self.reward)

        return obs, mask

    def get_reward(self, decided_job, shop_id, machine_num):
        ##当前step的即时reward

        reward, obj = self.multiShop[shop_id].reward(decided_job, machine_num)
        return reward, obj

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

    def obs_list_to_state_vector(self,observation):
        state = observation[0]
        for obs in observation[1:]:
            state = np.concatenate([state, obs])
        return state

    def executing_MADDPG(self,epsilon,train_or_test,shop_ID):
        yield self.env.timeout(0)
        # rule = self.rule  # self.ddqn.act(self.state, epsilon)
        ##先确定需要加工的工件，再确定machine
        ##获取每个车间的当前时间的状态，剩余每个工件的加工时间加duedate加blocking加标志位（是否已经超过duedate）
        # print(self.multiShop[shop_ID])
        # print('MMA',self.env.now)
        ##增加全局的新工件
        UC_job = self.multiShop[shop_ID].get_new_uncompleted_jobs(self.new_job)
        UC = []
        for job in UC_job:
            UC.append(job.number)
        # print('be',UC)
        if len(UC_job) != 0 :

            # obs,mask = self.get_obs(shop_ID,UC_job)

            t = time.time()
            print(len(UC_job))
            ##根据当前的状态选择工件进行加工，这里的action是UCjob的顺序号，需要转换,且目标值需要减去1
            action = self.Dispatchrules.SPT(UC_job)
            self.all_t = self.all_t + time.time() - t
            print('action',action)
            
            decided_job = UC_job[action]
            decided_operation, decided_machine = self.multiShop[shop_ID].choose_machine(decided_job)
            reward, obj = self.get_reward(decided_job, shop_ID, decided_machine)
            self.objectives += obj


            # print('new decided_job.', decided_job.number,'shop_ID',shop_ID)

            if decided_job.construct_count != decided_operation and action != None:
                operation = self.env.process(
                    decided_job.create_operation_process(self.multiShop[shop_ID], decided_machine,self.factor))
                decided_job.update_operation_process(decided_operation, operation)
                yield self.env.timeout(0)
                # 当工序进程结束后会执行规则
                self.env.process(self.operation_finished(operation, epsilon, train_or_test, shop_ID))
                yield self.env.timeout(0)
                # obs_ = copy.deepcopy(obs)
                # obs_[action] = np.zeros(self.args.max_stage + 5)
                # mask_ = copy.deepcopy(mask)
                # mask_[action] = 0
                # #############
                # reward = self.get_reward(decided_job,shop_ID)
                # self.reward = reward
                # # self.actor_reward_mamory[shop_ID].append(reward)
                # done = self.get_done(shop_ID)
                # state = self.obs_list_to_state_vector(obs)
                # state_ = self.obs_list_to_state_vector(obs_)



                print('######################################main'+ str(self.total_steps))
                self.total_steps += 1

    def executing_ddqn(self, epsilon, train_or_test, shop_ID):
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
                self.env.process(self.operation_finished(operation, epsilon, train_or_test, shop_ID))
                yield self.env.timeout(0)








