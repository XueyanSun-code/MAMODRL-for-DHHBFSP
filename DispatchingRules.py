import numpy as np
# import torch
Penalty=1e-6
from meta_Initialization import Initializations


class Rules:
    def __init__(self,device,args,machine_on_stage,fac_factor, energy_fac):
        self.device=device
        self.args = args
        self.machine_on_stage = machine_on_stage
        self.time_f = fac_factor
        self.energy_fac = energy_fac


    """EDD earlist due date"""
    def EDD(self, UC_job):
        '''选择工件'''

        ##对当前的工件进行排序，选择 due date 最小的
        DD = []
        for job in UC_job:
            DD.append(job.D)
        decided_job_num = np.argmin(DD)

        return decided_job_num

    """MDD"""
    def MDD(self, UC_job,time):
        DD = []
        for job in UC_job:
            # t = int(time)
            # print(job.D)
            # print(sum(job.process_time))
            x = max(job.D, sum(job.process_time)+time)
            DD.append(x)
        decided_job_num = np.argmin(DD)

        return decided_job_num


    """minimum slack  time"""
    def MST(self, UC_job,time):
        DD = []
        for job in UC_job:
            x = job.D-(sum(job.process_time)+time)
            DD.append(x)
        decided_job_num = np.argmin(DD)

        return decided_job_num

    """largest process time"""

    def LPT(self, UC_job):
        DD = []
        for job in UC_job:
            x = sum(job.process_time)
            DD.append(x)
        decided_job_num = np.argmax(DD)

        return decided_job_num

    """smallest process time"""
    def SPT(self, UC_job):
        DD = []
        for job in UC_job:
            x = sum(job.process_time)
            DD.append(x)
        decided_job_num = np.argmin(DD)

        return decided_job_num





