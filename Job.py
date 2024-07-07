import numpy as np
import simpy
import torch

'''工序的进程有先后，当第一个进程结束后创建第二个进程'''

class Job:
    def __init__(self, env, processing_information, DDT, A, job_number,device,process_time,original):
        self.number = job_number
        # self.order = number
        self.device=device
        #print('-----工件%d到达-----:%d' % (self.number,env.now))
        self.job_state = False
        self.env = env
        self.processing_information = processing_information
        self.ni = 1
        self.count = 0  # 指的是已经进入加工进程或者结束进程的个数
        self.construct_count = 0
        self.DDT = DDT
        self.A = A
        self.D = self.calculate_D()
        self.operation_machine = np.zeros(self.ni)
        self.operation_starting_time = torch.zeros(self.ni).to(self.device)
        self.operation_finishing_time = torch.zeros(self.ni).to(self.device)
        self.operation_process = [0 for _ in range(self.ni)]
        self.operation_state = env.event()
        self.finished = 0
        self.tard = 0
        self.process_time = process_time
        self.original = original

    def calculate_D(self):

        return np.round(self.A + self.processing_information*self.DDT)

    # 记录每一个工序进程
    def update_operation_process(self, j, process):
        self.operation_process[j - 1] = process

    def update_construct_count(self):
        self.construct_count -= 1

    # 当确定要加工这个工序时创建该工序进程
    # 先建立的进程先开始
    def create_operation_process(self, jobshop, machine_number,factor ):  # previous_operation是一个进程
        # 当该工件的前一工序加工完成后才可以进入下一工序的加工
        self.construct_count += 1  # 通过与count的差值判断创建的工件是否开始加工
        jobshop.machines[machine_number - 1].job_to_be_machined = self.number
        # print('***创建工件%d,工序%d***:%d' % (self.number, self.construct_count,self.env.now))
        try:
            if self.count != 0:
                yield self.operation_state
                self.operation_state = self.env.event()
            with jobshop.machines[machine_number - 1].machine.request() as request:
                yield request
                # 代表正式开始加工
                jobshop.machines[machine_number - 1].job_to_be_machined = 0
                self.count += 1  # 通过count来监控工作完成进度
                self.operation_machine[self.count - 1] = machine_number
                self.operation_starting_time[self.count - 1] = torch.tensor(self.env.now,dtype=torch.int)
                jobshop.machines[machine_number - 1].update_machine_starting_time(self.env.now)
                jobshop.machines[machine_number - 1].update_machine_processing_job(self.number)
                tij = self.processing_information*factor[jobshop.num][0]
                blocking_time = jobshop.caculate_blocking(machine_number - 1,tij)
                self.operation_finishing_time[self.count - 1] = self.env.now + tij + blocking_time
                jobshop.machines[machine_number - 1].update_machine_finishing_time(self.env.now + tij + blocking_time)
                ##更新后续阶段的机器的开始结束时间
                self.update_jobshop_process_states(jobshop,machine_number,self.env.now,factor)

                yield self.env.timeout(tij + blocking_time)
                #print('---工件%d,工序%d加工完成,加工机器%d---:%d' % (self.number, self.count,machine_number,self.env.now))
                self.operation_state.succeed()  # 激活，可以加工下一工序
                # 当所有工序加工完成后，该工件的加工完成
                if self.count == self.ni:
                    # self.job_state=True
                    self.processing()
        except simpy.Interrupt as inter:
            self.construct_count = self.count
            # print('~~~取消工件%d,工序%d进程~~~:%d'% (self.number, self.construct_count,self.env.now))

    def processing(self):
        self.tard = self.env.now - self.D
        self.finished = 1
        #print('000000-----工件%d加工完成-----00000:%s    %s     %s' % (self.number,self.env.now,self.D,self.A))
        self.operation_state = self.env.event()
        self.job_state = True

    def calculate_mean_tij(self, j):  # 计算该工序的平均加工时间
        # tij = 0
        # operation_processing_information = self.processing_information[j - 1]
        # count=0
        # for index, item in enumerate(operation_processing_information):
        #     if item!=0:
        #         tij += operation_processing_information[index]
        #         count+=1
        tij_ave = self.processing_information
        return tij_ave


    def update_jobshop_process_states(self,jobshop,machine_number, current_time,factor):
        ##计算后面的机器选择和blocking时间
        # 更新
        job_finish_time = np.zeros((jobshop.stage), dtype=int)
        for j in range(jobshop.stage):
            id_ = np.argmin(jobshop.machine_release_time[j])

            if j == 0:
                id_ = machine_number-1
                # print('machine number',machine_number)
                # print('jobshop.machine_release_time[j]',len(jobshop.machine_release_time[j]))
                jobshop.machine_release_time[j][id_] = max(
                    current_time + self.process_time[j]*factor[jobshop.num][j],
                    min(jobshop.machine_release_time[j + 1]))
                job_finish_time[j] = jobshop.machine_release_time[j][id_]
                block_time = job_finish_time[j] - (current_time + self.process_time[j]*factor[jobshop.num][j])
                start_time = current_time
            elif j == jobshop.stage - 1:

                jobshop.machine_release_time[j][id_] = job_finish_time[j-1] + self.process_time[j]*factor[jobshop.num][j]
                job_finish_time[j] = jobshop.machine_release_time[j][id_]
                block_time = 0
                start_time = job_finish_time[j-1]
            else:

                job_finish_time[j] = max(jobshop.machine_release_time[j][id_], job_finish_time[j]) + \
                                     self.process_time[j]*factor[jobshop.num][j]
                jobshop.machine_release_time[j][id_] = max(
                    job_finish_time[j-1] + self.process_time[j]*factor[jobshop.num][j],
                    min(jobshop.machine_release_time[j + 1]))
                job_finish_time[j] = jobshop.machine_release_time[j][id_]
                block_time = job_finish_time[j] - (job_finish_time[j-1] + self.process_time[j]*factor[jobshop.num][j])
                start_time = job_finish_time[j - 1]
            ##找到当前机器的加工工序的个数
            point = len(jobshop.machine_joblist[j][id_])
            jobshop.machines_start_finish_blocking_time[j][id_][point * 3] = start_time
            jobshop.machines_start_finish_blocking_time[j][id_][point * 3 + 1] = start_time + self.process_time[j]*factor[jobshop.num][j]
            jobshop.machines_start_finish_blocking_time[j][id_][point * 3 + 2] = block_time
            jobshop.machine_joblist[j][id_].append(self.number)

    def reset_job(self):
        self.job_state = False
        self.count = 0  # 指的是已经进入加工进程或者结束进程的个数
        self.construct_count = 0
        self.operation_machine = np.zeros(self.ni)
        self.operation_starting_time = torch.zeros(self.ni)
        self.operation_finishing_time = torch.zeros(self.ni)
        self.operation_process = [0 for _ in range(self.ni)]
        self.operation_state =self. env.event()
        self.finished = 0
        self.tard = 0



