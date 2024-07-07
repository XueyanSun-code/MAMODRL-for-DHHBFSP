import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Buffer_t import Buffer
from NetWork_t import PtrNet1,PtrNet2,ActorCritic
from torch.utils.tensorboard import SummaryWriter
# from memory_profiler import profile
import numpy as np

class Agent(object):
    """
    An implementation of the Proximal Policy Optimization (PPO) (by clipping) agent,
    with early stopping based on approximate KL.
    """

    def __init__(self,
                 args,
                 writer=None,
                 steps=0,
                 gamma=0.99,
                 lam=0.97,
                 sample_size=2000,
                 mini_batch=512,
                 train_policy_iters=80,
                 train_vf_iters=80, #batch设置为256 的话 大概训练20次
                 clip_param=0.05,
                 target_kl=0.5,
                 policy_lr=1e-3,
                 vf_lr=1e-3,
                 eval_mode=False,
                 ):
        self.args = args
        self.writer=writer
        self.mini_batch=mini_batch
        self.device = args.device
        self.steps = steps
        self.gamma = gamma
        self.lam = lam
        self.sample_size = sample_size
        self.train_policy_iters = train_policy_iters
        self.train_vf_iters = train_vf_iters
        self.clip_param = clip_param
        self.target_kl = target_kl
        self.policy_lr = policy_lr
        self.vf_lr = vf_lr
        self.eval_mode = eval_mode

        # Main network
        # self.policy = PtrNet1(self.args).to(self.device)
        # self.vf = PtrNet2(self.args).to(self.device)
        self.ACmodel=ActorCritic(args).to(self.device)
        self.old_ACmodel = ActorCritic(args).to(self.device)

        # Create optimizers
        # self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.policy_lr) #修改为阶梯下降
        # self.act_lr_scheduler=optim.lr_scheduler.StepLR(self.policy_optimizer,step_size=3e3,gamma=0.96)
        #
        # self.vf_optimizer = optim.Adam(self.vf.parameters(), lr=self.vf_lr) #修改为阶梯下降
        # self.vf_lr_scheduler = optim.lr_scheduler.StepLR(self.vf_optimizer, step_size=5e3, gamma=0.96)
        self.AC_optimizer=optim.Adam(self.ACmodel.parameters(), lr=self.policy_lr)
        self.AC_lr_scheduler = optim.lr_scheduler.StepLR(self.AC_optimizer, step_size=5e3, gamma=0.96)
        # Experience buffer
        self.buffer = Buffer(self.args,self.sample_size+500, self.device, self.gamma, self.lam)
        self.env = SubEnv(self.args, self.buffer, file_root=None)

        self.train_step=0
        # self.test_step=0

    def compute_vf_loss(self, obs,mask, ret,v_old):
        # Prediction V(s)
        v = self.ACmodel(obs,mask,qf=True)

        # Value loss
        clip_v = v_old + torch.clamp(v - v_old, -self.clip_param, self.clip_param)
        vf_loss = torch.max(F.mse_loss(v, ret), F.mse_loss(clip_v, ret))
        return vf_loss

    def compute_policy_loss(self, obs,mask, act, adv, log_pi_old):
        action_logprobs,dist_entropy = self.ACmodel(obs,mask,action=act)
        # log_pi=torch.log(torch.gather(prob_a,dim=1,index=act.unsqueeze(1).long())).squeeze(1)
        # Policy loss
        ratio = torch.exp(action_logprobs - log_pi_old)
        clip_adv = torch.clamp(ratio, 1. - self.clip_param, 1. + self.clip_param) * adv
        policy_loss = -torch.min(ratio * adv, clip_adv)

        # A sample estimate for KL-divergence, easy to compute
        # approx_kl = (log_pi_old - action_logprobs).mean()
        entropy=dist_entropy
        return policy_loss, entropy

    # def show_actnet_weights(self,step_num):
    #     for name1, param1 in self.ACmodel.policy.named_parameters():
    #         # self.writer.add_histogram(tag='policy' + name1, values=param1, global_step=step_num)
    #         self.writer.add_histogram(tag='policy' + name1 + '_grad', values=param1.grad, global_step=step_num)

    def show_model(self,step_num):
        for name2, param2 in self.ACmodel.named_parameters():
            # self.writer.add_histogram(tag='qf' + name2, values=param2, global_step=step_num)
            self.writer.add_histogram(tag=name2 + '_grad', values=param2.grad, global_step=step_num)

    # @profile
    def train_model(self):
        # mini_batch = torch.Tensor(torch.Tensor(self.buffer.get_mini_batch())).long().to(self.device)
        batch = self.buffer.get()
        obs =torch.Tensor(batch['obs']).to(self.device)
        act = torch.Tensor(batch['act']).to(self.device)
        ret = torch.Tensor(batch['ret']).to(self.device)
        adv = torch.Tensor(batch['adv']).to(self.device)
        prob_a=torch.Tensor(batch['prob_a']).to(self.device)
        mask=torch.Tensor(batch['obs_mask']).to(self.device)


        #出现inf的原因是 选出了概率为0的动作
        log_pi_old = prob_a
        # v_old = self.vf(obs,mask)
        # v_old = v_old.detach()

        # Train policy with multiple steps of gradient descent
        # for i in range(int(self.sample_size*self.train_policy_iters/self.mini_batch)):

        # Train value with multiple steps of gradient descent
        v_old = self.old_ACmodel.qf(obs,mask)
        v_old = v_old.detach()
        # for i in range(int(self.sample_size * self.train_policy_iters / self.mini_batch)):
        for i in range(self.train_policy_iters):
            # minibatch_ind = torch.Tensor(np.random.choice(self.sample_size, self.mini_batch, replace=False)).long()
            policy_loss, entropy= self.compute_policy_loss(obs, mask, act, adv, log_pi_old)
            vf_loss = self.compute_vf_loss(obs, mask,ret,v_old)
            self.writer.add_scalar('policy_loss',policy_loss.mean(),self.train_step)
            self.writer.add_scalar('vf_loss',vf_loss,self.train_step)
            self.writer.add_scalar('entropy',entropy.mean(),self.train_step)
            loss=policy_loss+0.5*vf_loss-0.01*entropy
            self.AC_optimizer.zero_grad()
            loss.mean().backward()
            self.AC_optimizer.step()
            self.show_model(self.train_step)
            self.writer.add_scalar('loss', vf_loss, self.train_step)
            self.train_step+=1

        self.old_ACmodel.load_state_dict(self.ACmodel.state_dict())



    def run(self,eval,i=None):
        # env=SubEnv(self.args,self.buffer,file_root=data_root)
        self.env.reset(eval,i)
        obs=self.env.state
        mask=self.env.mask

        # Keep interacting until agent reaches a terminal state.
        while not (self.env.done): #有问题 需要在思考
            if self.eval_mode:
                action = self.ACmodel(torch.Tensor(obs).unsqueeze(0).to(self.device),torch.Tensor(mask).unsqueeze(0).to(self.device),train=False) #action采用Greedy获取
                action = action.detach().cpu().numpy()
                self.env.step(action)
            else:
                # self.steps += 1
                # Collect experience (s, a, r, s') using some policy
                action,logprobs= self.old_ACmodel(torch.Tensor(obs).unsqueeze(0).to(self.device),torch.Tensor(mask).unsqueeze(0).to(self.device))  #action采用采样分布获取
                action = action.detach().cpu().numpy()
                self.env.step(action)

                # Add experience to buffer
                v = self.old_ACmodel(torch.Tensor(obs).unsqueeze(0).to(self.device),torch.Tensor(mask).unsqueeze(0).to(self.device),qf=True)
                v=v.detach().cpu().numpy()
                self.env.add_v(v)
                self.env.add_prob_a(logprobs.detach().cpu().numpy())
                if self.env.done:
                    self.env.put_in_buffer()

                # Start training when the number of experience is equal to sample size
                if self.buffer.ptr >= self.sample_size:
                    self.buffer.finish_path()
                    self.train_model()
                    # self.steps = 0
            obs = self.env.state
            mask=self.env.mask

        return self.env.get_tardness()

