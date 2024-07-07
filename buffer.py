import random
import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, capacity,device):
        self.capacity = capacity
        self.device=device
        self.buffer = torch.zeros(self.capacity,16).to(self.device)
        self.count=0

    def add(self, s0, a, r, s1):
        data=torch.cat((s0,a,r,s1)).to(self.device)
        self.buffer[self.count%self.capacity]=data
        self.count += 1


    def sample(self, batch_size):
        if self.count>self.capacity:
            sample= random.sample(range(self.capacity), batch_size)
        else:
            sample=random.sample(range(self.count),batch_size)
        data=torch.zeros(batch_size,16).to(self.device)
        for index,item in enumerate(sample):
            data[index]=self.buffer[item]
        return data[:,:7],data[:,7].long(),data[:,8],data[:,9:]

    # def size(self):
    #     return len(self.buffer)