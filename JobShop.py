import numpy as np
from DispatchingRules import Rules
import torch
from Job import Job


class JobShop:
    def __init__(self,device,env,args,stage,machine_on_stage,number,factor,energy_fac):
        self.rules = Rules(device,args,machine_on_stage,factor, energy_fac)
        self.objective_value = 0
        self.device=device
        self.env=env
        self.machines = []
        self.jobs = []
        self.stage = stage
        self.args = args
        self.gantt_machine = machine_on_stage
        self.machines_start_finish_blocking_time = self.init_time_count()
        self.machine_release_time, self.machine_joblist = self.ini_machine_release_time()
        self.episode_end = self.env.event()
        self.activite = True  ##表示当前的agent的状态可以进行工件选择，没有正在加工的工件
        self.obs = np.zeros((self.args.max_job, self.args.max_stage + 9))
        self.mask = np.zeros(self.args.max_job)
        self.done = False
        self.new_arrvial = False
        self.num = number
        self.factor = factor
        self.energy_fac = energy_fac#每个阶段的机器的能耗相同



    def ini_machine_release_time(self):
        machine_release_time = []
        machine_joblist = []
        for i, stage in enumerate(self.gantt_machine):
            time = [0 for j in range(stage)]
            machine_release_time.append(time)
            machine_list = [[] for j in range(stage)]
            machine_joblist.append(machine_list)

        return machine_release_time, machine_joblist

    def init_time_count(self):
        machines_start_finish_blocking_time = []
        # all_count = []
        for i, stage in enumerate(self.gantt_machine):
            stage_time = []

            for j in range(stage):
                stage_time.append(np.zeros(1000))
            machines_start_finish_blocking_time.append(stage_time)
        return machines_start_finish_blocking_time


    def get_machine_map(self):
        machine_number_stage = np.random.randint(1, self.args.max_machine_number, size=self.stage)
        return machine_number_stage


    def get_Tcur(self, machines):
        CT = torch.zeros(len(machines)).to(self.device)
        for index,machine in enumerate(machines):
            if machine.point==0:
                CT[index]=0
            else:
                CT[index]=machine.machine_finishing_time[machine.point-1]
        Tcur = torch.sum(CT) / len(machines)
        return Tcur


    def get_tardiness_jobs(self, jobs, machines):
        Tard_job = torch.zeros(len(jobs)).int().to(self.device)
        #OP = self.get_OP(jobs)
        Tcur = self.get_Tcur(machines)
        count=0
        for index, job in enumerate(jobs):
            if jobs[index].count < job.ni and job.D < Tcur:
                Tard_job[count]=index + 1
                count+=1
        return Tard_job[:count]

    def get_new_uncompleted_jobs(self,new_jobs):
        AUC_job = []

        job_pool = self.jobs + new_jobs
        for index, job in enumerate(job_pool):
            if job.count < job.ni:
                AUC_job.append(job)

        estimated_value = []
        for i, job in enumerate(AUC_job):
            estimated_value.append(job.D - self.env.now - sum(job.process_time))
        selected_job_list = range(len(AUC_job))  ##根据现有工件进行重新顺序编号，1，2，3，4，
        selected_job_list = sorted(selected_job_list, key=lambda k: estimated_value[k - 1])
        selected_job_list = selected_job_list[:self.args.max_job]
        ##按照顺序找出最终的UC——jobs
        UC_jobs = []
        for j in selected_job_list:
            UC_jobs.append(AUC_job[j])

        return UC_jobs

    def get_uncompleted_jobs(self):
        UC_job = []
        count = 0
        # OP = self.get_OP(jobs)
        for index, job in enumerate(self.jobs):
            if self.jobs[index].count < job.ni:
                UC_job.append(job)
                count += 1
        return UC_job

    def calculate_input_sequence(self, jobs, machines):
        n = len(jobs)
        m = len(machines)
        U = torch.zeros(m).to(self.device)
        CT = torch.zeros(m).to(self.device)
        for index,machine in enumerate(machines):
            starting_time = machine.machine_starting_time
            finishing_time = machine.machine_finishing_time
            U[index]=torch.sum(finishing_time-starting_time)
            CT[index]=max(machine.machine_finishing_time[machine.point-1],self.env.now)
            if CT[index] == 0:
                CT[index] = 1
        U = torch.div(U, CT)
        '''机器的平均利用率、机器利用率的标准差'''
        Uave = torch.div(torch.sum(U),torch.tensor(m))
        Ustd = torch.std(U)

        OP = torch.zeros(n).to(self.device)
        ni = torch.zeros(n).to(self.device)
        for index,job in enumerate(jobs):
            OP[index]=job.count
            ni[index]=job.ni
        CRJ = torch.div(OP,ni)
        '''工序的平均完成率、工件的平均完成率、攻坚完成率的方差'''
        CROave = torch.div(torch.sum(OP),torch.sum(ni))
        CRJave = torch.div(torch.sum(CRJ),torch.tensor(n))
        CRJstd = torch.std(CRJ)

        Tcur = torch.div(torch.sum(CT), torch.tensor(m))
        Ntard = 0
        Nleft = 0
        for i in range(n):
            if OP[i] < ni[i]:
                Nleft += ni[i] - OP[i]
                Tleft = 0
                for j in range(OP[i].int().item()+1, ni[i].int().item()+1):
                    tij_ave = jobs[i].calculate_mean_tij(j)
                    Tleft += tij_ave
                    if Tcur + Tleft > jobs[i].D:
                        Ntard += ni[i] - j + 1
                        break
        '''估计的延迟率  估计延迟的未加工序/所有未加工的工序'''
        if Nleft == 0:
            Tard_e = 1
        else:
            Tard_e = Ntard/Nleft

        '''实际的延迟率  实际延迟的未加工序/所有未加工的工序'''
        Ntard = 0
        for i in range(n):
            if OP[i] < ni[i]:
                if max(jobs[i].operation_finishing_time[OP[i].int().item() - 1],self.env.now) > jobs[i].D:
                    Ntard += ni[i] - OP[i]
        if Nleft == 0:
            Tard_a=1
        else:
            Tard_a = Ntard/Nleft

        input_sequence = torch.tensor([Uave, Ustd, CROave, CRJave, CRJstd, Tard_e, Tard_a]).to(self.device)
        return input_sequence

    def choose_machine(self,decided_job):
        '''选择工序'''
        decided_operation = int(decided_job.count + 1)
        '''选择机器'''
        # process_information = decided_job.processing_information
        machine_evaluate = torch.zeros(self.gantt_machine[0]).to(self.device)
        candidate_machine = torch.zeros(self.gantt_machine[0]).to(self.device)
        count = 0
        for index in range(self.gantt_machine[0]):

            candidate_machine[count] = index + 1
            # np.array([1, 2, 3], dtype=np.int64)
            machine_evaluate[count] = torch.max(torch.tensor([
                self.machines[index].machine_finishing_time[self.machines[index].point - 1], decided_job.A,
                decided_job.operation_finishing_time[decided_job.count - 1]]).to(self.device))
            count += 1
        ##选择最小的开始时间，或者第一个最小开始时间的机器
        decided_machine = candidate_machine[torch.argmin(machine_evaluate[:count])].int()
        return decided_operation, decided_machine

    # 创建下一工序的加工进程
    def execute_rules(self, env, rule, machines):  # 执行规则后需要更新环境 参数t, i,j,tij，然后计算下一状态特征值  如果有新工件到需要更新工件列表
        decided_job, decided_operation, decided_machine = 0, 0, 0


        UC_job = self.get_uncompleted_jobs()
        # OP = self.get_OP(jobs)
        if rule == 0:
            if len(UC_job)!=0:
                decided_job_num = UC_job[0]
                # decided_job = decided_job_num.number
                decided_operation, decided_machine = self.choose_machine(decided_job_num)

        return decided_job_num, decided_operation, decided_machine

    def observations(self,UC_job,reward,rew_list,ep_num):
        if self.activite:
            # UC_job = self.get_new_uncompleted_jobs()
            UC = []
            for job in UC_job:
                UC.append(job.number)
            # print('af', UC)

            ##计算每个工件的blocking time
            bolcking_T, sign_pos = self.estimated_blocking_time(UC_job,self.env.now)
            ##获取每个工件的duedate与当前时间的差值，以及标志位
            TD = self.time_difference(UC_job,self.env.now)

            for i,job in enumerate(UC_job):
                self.obs[i][:self.stage] = job.process_time
                self.obs[i][self.args.max_stage] = TD[i]
                self.obs[i][self.args.max_stage + 1] = bolcking_T[i]
                self.obs[i][self.args.max_stage + 2] = sign_pos[i]
                # self.obs[i][self.args.max_stage + 3] = job.original##获得当前工件的来自新工件还是原来的
                self.obs[i][self.args.max_stage + 3] = reward[0]
                self.obs[i][self.args.max_stage + 4] = reward[1]
                self.obs[i][self.args.max_stage + 5] = rew_list[i][0]
                self.obs[i][self.args.max_stage + 6] = rew_list[i][1]
                # 以下是PPO两个
                if ep_num%2 == 0:
                    self.obs[i][self.args.max_stage + 7] = 0.7
                    self.obs[i][self.args.max_stage + 8] = 0.3
                else:
                    self.obs[i][self.args.max_stage + 7] = 0.3
                    self.obs[i][self.args.max_stage + 8] = 0.7
            self.mask *= 0
            self.mask[:len(UC_job)] += 1

        else:
            self.obs = np.zeros((self.args.max_job, self.args.max_stage + 9))
            self.mask *= 0

        return self.obs, self.mask

    '''这里的reward是向量'''
    def reward(self,job,machine_num):
        ##计算abs（tardiness）
        reward = np.zeros(2)
        obj = np.zeros(2)
        job_finish_time = np.zeros((self.stage), dtype=int)
        energy_reward = 0
        for j in range(self.stage):
            id_ = np.argmin(self.machine_release_time[j])
            if j == 0:
                idle_time = self.env.now - self.machine_release_time[j][machine_num-1]
                job_finish_time[j] = max(
                    self.env.now + job.process_time[j]*self.factor[j],
                    min(self.machine_release_time[j + 1]))
                blocking = job_finish_time[j] - self.env.now + job.process_time[j]*self.factor[j]
                # block_time = job_finish_time[j] - (self.env.now() + job.process_time[j])

            elif j == self.stage - 1:
                job_finish_time[j] = job_finish_time[j - 1] + job.process_time[j]*self.factor[j]
                blocking = 0
                idle_time = job_finish_time[j - 1] - self.machine_release_time[j][id_]
            else:
                job_finish_time[j] = max(
                    job_finish_time[j - 1] + job.process_time[j]*self.factor[j],
                    min(self.machine_release_time[j + 1]))
                idle_time = job_finish_time[j - 1] - self.machine_release_time[j][id_]
                blocking = job_finish_time[j] - self.env.now + job.process_time[j] * self.factor[j]
            e = job.process_time[j] * self.factor[j] * self.energy_fac[j] + blocking * self.energy_fac[
                j] * self.args.blocking_index + idle_time * self.energy_fac[j] * self.args.idle_index
            energy_reward += e
        if max(0, job_finish_time[-1] - job.D) == 0:
            reward[0] = 1
        else:

            reward[0] = 1/max(0, job_finish_time[-1] - job.D)
        reward[1] = 1/energy_reward
        obj[0] = max(0, job_finish_time[-1] - job.D)
        obj[1] = energy_reward
        return  reward,obj

    def time_difference(self,UC_job,current):
        TD = []
        for job in UC_job:
            TD.append(job.D- current)
        return TD

    def estimated_blocking_time(self,UC_job,current_time):
        blocking = []
        sign_pos = []
        for job in UC_job:
            temp_blocking = 0
            job_finish_time = np.zeros((self.stage), dtype=int)
            for j in range(self.stage):

                if j == 0:

                    job_finish_time[j] = max(
                        current_time + job.process_time[j]*self.factor[j],
                        min(self.machine_release_time[j + 1]))
                    block_time = job_finish_time[j] - (current_time + job.process_time[j]*self.factor[j])
                elif j == self.stage - 1:
                    job_finish_time[j] = job_finish_time[j - 1] + job.process_time[j]*self.factor[j]
                    block_time = 0
                else:
                    job_finish_time[j] = max(
                        job_finish_time[j - 1] + job.process_time[j]*self.factor[j],
                        min(self.machine_release_time[j + 1]))

                    block_time = job_finish_time[j] - (job_finish_time[j - 1] + job.process_time[j]*self.factor[j])
                temp_blocking += block_time
            blocking.append(temp_blocking)
            if job.D - job_finish_time[-1] >= 0:

                sign_pos.append(1)
            else:
                sign_pos.append(-1)
        return blocking, sign_pos


    def calculate_reward(self, input, next_input):
        if next_input[6] < input[6]:
            return 1
        else:
            if next_input[6] > input[6]:
                return -1
            else:
                if next_input[5] < input[5]:
                    return 1
                else:
                    if next_input[5] > input[5]:
                        return -1
                    else:
                        if next_input[0] > input[0]:
                            return 1
                        else:
                            if next_input[0] > input[0] * 0.95:
                                return 0
                            else:
                                return -1

    def job_distribution(self,lam, add_job):
        exp_distribution = np.around(np.random.exponential(lam, size=add_job)).astype(int)  # 50 个参数为 lam 的指数分布随机数
        #exp_distribution = np.concatenate(([0], exp_distribution))
        return exp_distribution

    def creat_job(self, DDT, all_process_time, job_number,number,time,original):

        ni = 1
        processing_information = all_process_time[0]


        job = Job(self.env, processing_information, DDT,time , job_number ,self.device,all_process_time,original)
        return job

    def new_job(self, DDT, max_operation_time, job_number,time):
        processing_information = np.zeros(self.gantt_machine[0])

        all_process_time = np.random.randint(1, max_operation_time,size=self.stage)
        ##继续产生其他阶段的加工时间

        for i in range(self.gantt_machine[0]):
            # print('all', all_process_time[0])
            processing_information[i] = all_process_time[0]

        job = Job(self.env, processing_information, DDT, time, job_number, self.device, all_process_time)
        return job

    def caculate_blocking(self,machine_number,T):

        blocking_time = max(min(self.machine_release_time[1]) - (self.machines[machine_number].machine_starting_time[self.machines[machine_number].point] + T),0)
        return blocking_time


