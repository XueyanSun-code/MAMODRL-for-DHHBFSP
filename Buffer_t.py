import random
import torch
import numpy as np

'''size=600'''
class Buffer(object):
    """
    A buffer for storing trajectories experienced by a agent interacting
    with the environment.
    """
    def __init__(self, args, size, device, gamma=0.99, lam=0.97):
        self.obs_buf = np.zeros((size,args.max_job,args.max_operation*3+2))
        self.obs_mask_buf=np.zeros((size,args.max_job))
        self.act_buf = np.zeros(size)
        self.rew_buf = np.zeros(size)
        self.don_buf = np.zeros(size)
        self.ret_buf = np.zeros(size)
        self.adv_buf = np.zeros(size)
        self.v_buf = np.zeros(size)
        self.prob_a_buf=np.zeros(size)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.max_size = 0, size
        self.device = device
        # self.mini_batch=np.zeros((160,128))

    def add(self, obs, obs_mask, act, rew, don, v,prob_a):
        num=len(obs)
        for i in range(num):
            self.obs_buf[self.ptr] = obs[i]
            self.obs_mask_buf[self.ptr]=obs_mask[i]
            self.act_buf[self.ptr] = act[i]
            self.rew_buf[self.ptr] = rew[i]
            self.don_buf[self.ptr] = don[i]
            self.v_buf[self.ptr] = v[i]
            self.prob_a_buf[self.ptr]=prob_a[i]
            self.ptr += 1

    '''running_del才是计算vf_loss的baseline'''
    def finish_path(self):
        previous_v = 0
        td_target = 0
        running_adv = 0
        # for step in reversed(range(self.ptr)):
        #     running_ret= running_ret *self.gamma *(1 - self.don_buf[step]) + self.rew_buf[step]
        #     self.ret_buf[step]=running_ret
        #     self.adv_buf[step]=self.ret_buf[step]-self.v_buf[step]
        # self.adv_buf=(self.adv_buf-self.adv_buf.mean())/self.adv_buf.std()
        for t in reversed(range(self.ptr)):
            td_target=self.rew_buf[t] + self.gamma * (1 - self.don_buf[t]) * previous_v
            self.ret_buf[t]=td_target
            running_del = td_target- self.v_buf[t]
            running_adv = running_del + self.gamma * self.lam * (1 - self.don_buf[t]) * running_adv
            previous_v = self.v_buf[t]
            self.adv_buf[t]=running_adv



    #只取Ptr size的数据
    def get(self):
        data=dict(obs=self.obs_buf[:self.ptr],
                    obs_mask=self.obs_mask_buf[:self.ptr],
                    act=self.act_buf[:self.ptr],
                    ret=self.ret_buf[:self.ptr],
                    adv=self.adv_buf[:self.ptr],
                    v=self.v_buf[:self.ptr],
                    prob_a=self.prob_a_buf[:self.ptr])

        self.ptr = 0
        # self.mini_batch[:][:]=0
        return data

