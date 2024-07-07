import numpy
import simpy
# import torch
import numpy as np

class Machine:
    def __init__(self, env,device):
        self.env = env
        self.machine = simpy.Resource(self.env, 1)
        self.device=device
        # #每一个机器是一个进程
        # self.machine_state=env.process(self.running())
        self.machine_starting_time = np.zeros(700)
        self.machine_finishing_time = np.zeros(700)
        self.machine_processing_job = np.zeros(700)

        # self.machine_starting_time = torch.zeros(700).to(device)
        # self.machine_finishing_time = torch.zeros(700).to(device)
        # self.machine_processing_job = torch.zeros(700).to(device)
        self.job_to_be_machined=0  # 机器上是否有待加工工件,值为待加工工件编号
        self.point=0


    def update_machine_starting_time(self, t):
        self.machine_starting_time[self.point]= t

        # self.machine_starting_time[self.point] = torch.tensor(t, dtype=torch.int)

    def update_machine_finishing_time(self, t):
        self.machine_finishing_time[self.point]= t
        self.point += 1

    def update_machine_processing_job(self, i):
        self.machine_processing_job[self.point]=i

    def reset_machine(self):
        self.machine_starting_time = np.zeros(700)
        self.machine_finishing_time = np.zeros(700)
        self.machine_processing_job = np.zeros(700)
        self.job_to_be_machined=0  # 机器上是否有待加工工件,值为待加工工件编号
        self.point=0




    # 机器运行过程中需要更新机器状态
    # def running(self):
    #     while True:
    #         with self.machine.request() as request:
    #             self.machine_starting_time.append(self.env.now)
    #             yield request
    #             self.machine_finishing_time.append(self.env.now)
