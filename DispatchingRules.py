import numpy as np
# import torch
Penalty=1e-6
from meta_Initialization import Initializations
from Pdjaya_2023 import PdJaya
from MOEAD2023 import MOEAD

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

    def Pdjaya2023(self,UC_job,shop_id):
        ##将UC_job的工件加工时间和due date
        numjobs = len(UC_job)
        # numstage = 4 ##本文的环境设置中，阶段为4
        process_time_UC = []
        for i in range(numjobs):
            process_time_UC.append(UC_job[i].process_time)


        order = list(range(1, numjobs + 1))
        polulation_size = 10
        process_time_ = np.array(process_time_UC)
        HGA = PdJaya(polulation_size,0.1,numjobs, self.args.max_stage, self.machine_on_stage[shop_id], process_time_,self.time_f[shop_id],self.energy_fac[shop_id])
        rounds = HGA.PdJaya_running(shop_id)
        ##根据多个可能排序，找出一个较好的.或随机，或根据方向
        # rand = np.random.randint(len(HGA.PF))
        # print('PF',HGA.PF)
        sequence = HGA.PF[0]
        # print(sequence)
        decided_job_num = sequence[0]
        return decided_job_num-1

    def MOEAD2023(self,UC_job,shop_id):
        ##将UC_job的工件加工时间和due date
        numjobs = len(UC_job)
        if numjobs == 1:
            decided_job_num = numjobs
        else:
            # numstage = 4 ##本文的环境设置中，阶段为4
            process_time_UC = []
            for i in range(numjobs):
                process_time_UC.append(UC_job[i].process_time)
    
            order = list(range(1, numjobs + 1))
            polulation_size = 10
            process_time_ = np.array(process_time_UC)
            HGA = MOEAD(polulation_size,0.1,numjobs, self.args.max_stage, self.machine_on_stage[shop_id], process_time_,self.time_f[shop_id],self.energy_fac[shop_id])
            rounds = HGA.MOEAD_running(shop_id)
            ##根据多个可能排序，找出一个较好的.或随机，或根据方向
            rand = np.random.randint(len(HGA.PF))
            sequence = HGA.PF[rand]
            decided_job_num = sequence[0]
        return decided_job_num-1




