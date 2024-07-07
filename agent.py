import os

import torch as T
from networks import ActorNetwork_1, CriticNetwork, CriticNetwork_AC,CriticNetwork_dqn,CriticNetwork_dqn2019
from NetWork_t import PtrNet1,PtrNet2,ActorCritic
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class Agent:
    def __init__(self, args, agent_idx, chkpt_dir,
                    alpha=0.01, gamma=0.95, tau=0.01):
        self.gamma = gamma
        self.tau = tau
        # self.n_actions = n_actions
        self.agent_name = 'agent_%s' % agent_idx
        self.args = args
        self.activite = True##表示当前的agent的状态可以进行工件选择，没有正在加工的工件
        self.actor = ActorNetwork_1(args,alpha,
                                  chkpt_dir=chkpt_dir, name=self.agent_name+'_actor')
        self.critic = CriticNetwork(args,alpha,
                            chkpt_dir=chkpt_dir, name=self.agent_name+'_critic')
        self.target_actor = ActorNetwork_1(args,alpha,
                                  chkpt_dir=chkpt_dir, name=self.agent_name+'_target_actor')
        self.target_critic = CriticNetwork(args,alpha,
                            chkpt_dir=chkpt_dir, name=self.agent_name+'_target_critic')

        # self.ACmodel = ActorCritic(args).to(self.device)
        # self.old_ACmodel = ActorCritic(args).to(self.device)

        T.autograd.set_detect_anomaly(True)

        self.update_network_parameters(tau=1)

    def choose_action(self, obs,mask):
        action, logprobs = self.actor(T.Tensor(obs).unsqueeze(0).to(self.args.device),
                                            T.Tensor(mask).unsqueeze(0).to(self.args.device))  # action采用采样分布获取
        action = action.detach().cpu().numpy()


        return action

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                    (1-tau)*target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                    (1-tau)*target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

##更改2019Yang的论文
class Agent_DDQN:
    def __init__(self,  args, agent_idx, chkpt_dir,alpha=0.01, gamma=0.95, tau=0.01):
        # self.model = model
        # self.is_train = is_train
        self.gamma = gamma
        self.tau = tau
        self.epsilon_decay = 0.995
        # self.n_actions = n_actions
        self.agent_name = 'agent_%s' % agent_idx
        self.args = args
        
        self.epsilon = args.epsilon
        self.epsilon_decay = args.epsilon_decay

        self.weight_num = args.weight_num
        # self.trans_mem = deque()
        # self.trans = namedtuple('trans', ['s', 'a', 's_', 'r', 'd'])
        # self.priority_mem = deque()
        self.critic = CriticNetwork_dqn2019(args, alpha,
                                    chkpt_dir=chkpt_dir, name=self.agent_name + '_critic')
        self.target_critic = CriticNetwork_dqn2019(args, alpha,
                                           chkpt_dir=chkpt_dir, name=self.agent_name + '_target_critic')



        self.keep_preference = None


    def choose_action(self, obs, mask, preference=None):
        # random pick a preference if it is not specified
        if preference is None:
            if self.keep_preference is None:
                # preference = torch.from_numpy(
                # 	np.random.dirichlet(np.ones(self.model.reward_size))).type(FloatTensor)
                self.keep_preference = T.randn(self.args.reward_size)
                self.keep_preference = (T.abs(self.keep_preference) / \
                                        T.norm(self.keep_preference, p=1))
            preference = self.keep_preference


        states = obs.flatten()

        _, Q = self.critic.forward(T.Tensor(states).unsqueeze(0).to(self.args.device),T.Tensor(mask).unsqueeze(0).to(self.args.device),
                    T.Tensor(preference).unsqueeze(0).to(self.args.device))


        if np.random.rand() <= self.epsilon:
            return np.random.randint(len(Q))
        else:
            action = Q.max(0)[1].cpu().numpy()
            action = action[0]

        return action

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau


        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                    (1-tau)*target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

    def save_models(self):

        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):

        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()



class Agent_dqn:
    def __init__(self, args, agent_idx, chkpt_dir,
                    alpha=0.01, gamma=0.95, tau=0.01):
        self.gamma = gamma
        self.tau = tau
        self.epsilon_decay = 0.995
        # self.n_actions = n_actions
        self.agent_name = 'agent_%s' % agent_idx
        self.args = args
        self.activite = True##表示当前的agent的状态可以进行工件选择，没有正在加工的工件

        self.critic = CriticNetwork_dqn(args,alpha,
                            chkpt_dir=chkpt_dir, name=self.agent_name+'_critic')

        self.target_critic = CriticNetwork_dqn(args,alpha,
                            chkpt_dir=chkpt_dir, name=self.agent_name+'_target_critic')


        T.autograd.set_detect_anomaly(True)

        self.update_network_parameters(tau=1)

    def choose_action(self, obs,mask,epsilon):
        prob = self.critic(T.Tensor(obs).unsqueeze(0).to(self.args.device),
                                            T.Tensor(mask).unsqueeze(0).to(self.args.device))

        if np.random.rand() <= epsilon:
            return np.random.randint(len(prob))
        else:

            action = T.argmax(prob, dim=1).long()
            return action.detach().cpu().numpy()[0]



    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                    (1-tau)*target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

    def save_models(self):

        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):

        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()



class Agent_td3:
    def __init__(self, args, agent_idx, chkpt_dir,
                    alpha=0.01, gamma=0.95, tau=0.01):
        self.gamma = gamma
        self.tau = tau
        # self.n_actions = n_actions
        self.agent_name = 'agent_%s' % agent_idx
        self.args = args
        self.activite = True##表示当前的agent的状态可以进行工件选择，没有正在加工的工件
        self.actor = ActorNetwork_1(args,alpha,
                                  chkpt_dir=chkpt_dir, name=self.agent_name+'_actor')
        self.critic1 = CriticNetwork(args,alpha,
                            chkpt_dir=chkpt_dir, name=self.agent_name+'_critic1')
        self.critic2 = CriticNetwork(args, alpha,
                                    chkpt_dir=chkpt_dir, name=self.agent_name + '_critic2')
        self.target_actor = ActorNetwork_1(args,alpha,
                                  chkpt_dir=chkpt_dir, name=self.agent_name+'_target_actor')
        self.target_critic1 = CriticNetwork(args,alpha,
                            chkpt_dir=chkpt_dir, name=self.agent_name+'_target_critic1')
        self.target_critic2 = CriticNetwork(args, alpha,
                                           chkpt_dir=chkpt_dir, name=self.agent_name + '_target_critic2')


        T.autograd.set_detect_anomaly(True)

        self.update_network_parameters(tau=1)

    def choose_action(self, obs,mask):
        action, logprobs = self.actor(T.Tensor(obs).unsqueeze(0).to(self.args.device),
                                            T.Tensor(mask).unsqueeze(0).to(self.args.device))  # action采用采样分布获取
        action = action.detach().cpu().numpy()


        return action

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                    (1-tau)*target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

        target_critic1_params = self.target_critic1.named_parameters()
        critic1_params = self.critic1.named_parameters()

        target_critic1_state_dict = dict(target_critic1_params)
        critic1_state_dict = dict(critic1_params)
        for name in critic1_state_dict:
            critic1_state_dict[name] = tau*critic1_state_dict[name].clone() + \
                    (1-tau)*target_critic1_state_dict[name].clone()

        self.target_critic1.load_state_dict(critic1_state_dict)
        
        target_critic2_params = self.target_critic2.named_parameters()
        critic2_params = self.critic2.named_parameters()

        target_critic2_state_dict = dict(target_critic2_params)
        critic2_state_dict = dict(critic2_params)
        for name in critic2_state_dict:
            critic2_state_dict[name] = tau * critic2_state_dict[name].clone() + \
                                       (1 - tau) * target_critic2_state_dict[name].clone()

        self.target_critic2.load_state_dict(critic2_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()




class Agent_AC:
    def __init__(self, args, agent_idx, chkpt_dir,
                    alpha=0.01, gamma=0.95, tau=0.01):
        self.gamma = gamma
        self.tau = tau
        # self.n_actions = n_actions
        self.agent_name = 'agent_%s' % agent_idx
        self.args = args
        self.activite = True##表示当前的agent的状态可以进行工件选择，没有正在加工的工件
        self.actor = ActorNetwork_1(args,alpha,
                                  chkpt_dir=chkpt_dir, name=self.agent_name+'_actor')
        self.critic = CriticNetwork_AC(args,alpha,
                            chkpt_dir=chkpt_dir, name=self.agent_name+'_critic')


        # self.ACmodel = ActorCritic(args).to(self.device)
        # self.old_ACmodel = ActorCritic(args).to(self.device)

        T.autograd.set_detect_anomaly(True)

        # self.update_network_parameters(tau=1)

    def choose_action(self, obs,mask):
        action, logprobs = self.actor(T.Tensor(obs).unsqueeze(0).to(self.args.device),
                                            T.Tensor(mask).unsqueeze(0).to(self.args.device))  # action采用采样分布获取
        action = action.detach().cpu().numpy()


        return action

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                    (1-tau)*target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                    (1-tau)*target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()


class Agent_ppo(object):
    """
    An implementation of the Proximal Policy Optimization (PPO) (by clipping) agent,
    with early stopping based on approximate KL.
    """

    def __init__(self,
                 args,
                 chkpt_dir, name,
                 writer=None,
                 steps=0,
                 gamma=0.99,
                 lam=0.97,
                 sample_size=2000,
                 mini_batch=512,

                 train_vf_iters=40, #batch设置为256 的话 大概训练20次
                 clip_param=0.05,
                 target_kl=0.5,
                 policy_lr=1e-5,
                 vf_lr=1e-3,
                 eval_mode=False,

                 ):
        self.args = args
        self.ckpt_path1 = os.path.join(chkpt_dir, name+'_0_ACmodel')
        self.ckpt_path2 = os.path.join(chkpt_dir, name + '_1_ACmodel')
        self.writer=writer
        self.mini_batch=mini_batch
        self.device = args.device
        self.steps = steps
        self.gamma = gamma
        self.lam = lam
        self.sample_size = sample_size
        self.train_policy_iters = args.iters
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
        self.sec_ACmodel = ActorCritic(args).to(self.device)

        # Create optimizers
        # self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.policy_lr) #修改为阶梯下降
        # self.act_lr_scheduler=optim.lr_scheduler.StepLR(self.policy_optimizer,step_size=3e3,gamma=0.96)
        #
        # self.vf_optimizer = optim.Adam(self.vf.parameters(), lr=self.vf_lr) #修改为阶梯下降
        # self.vf_lr_scheduler = optim.lr_scheduler.StepLR(self.vf_optimizer, step_size=5e3, gamma=0.96)
        self.AC_optimizer=optim.Adam(self.ACmodel.parameters(), lr=self.policy_lr)
        self.AC_lr_scheduler = optim.lr_scheduler.StepLR(self.AC_optimizer, step_size=5e3, gamma=0.96)
        self.sec_AC_optimizer = optim.Adam(self.sec_ACmodel.parameters(), lr=self.policy_lr)
        self.sec_AC_lr_scheduler = optim.lr_scheduler.StepLR(self.sec_AC_optimizer, step_size=5e3, gamma=0.96)
        # Experience buffer
        # self.buffer = Buffer(self.args,self.sample_size+500, self.device, self.gamma, self.lam)
        # self.env = SubEnv(self.args, self.buffer, file_root=None)

        self.train_step=0
        # self.test_step=0

    def choose_action(self, obs,mask,ep_num):
        # if ep_num%2 == 0:
        action, logprobs = self.ACmodel(T.Tensor(obs).unsqueeze(0).to(self.args.device),
                                            T.Tensor(mask).unsqueeze(0).to(self.args.device))  # action采用采样分布获取

        v = self.ACmodel(T.Tensor(obs).unsqueeze(0).to(self.device), T.Tensor(mask).unsqueeze(0).to(self.device),
                             qf=True)
        # else:
        #     action, logprobs = self.sec_ACmodel(T.Tensor(obs).unsqueeze(0).to(self.args.device),
        #                                     T.Tensor(mask).unsqueeze(0).to(self.args.device))  # action采用采样分布获取
        #
        #     v = self.sec_ACmodel(T.Tensor(obs).unsqueeze(0).to(self.device), T.Tensor(mask).unsqueeze(0).to(self.device),
        #                      qf=True)
        action = action.detach().cpu().numpy()
        v = v.detach().cpu().numpy()
        logprobs.detach().cpu().numpy()
        return action,logprobs,v


    def compute_vf_loss(self, obs,mask, ret,v_old):
        # Prediction V(s)
        v = self.ACmodel(obs,mask,qf=True)

        # Value loss
        clip_v = v_old + T.clamp(v - v_old, -self.clip_param, self.clip_param)
        vf_loss = T.zeros(2).to(self.args.device)
        vf_loss[0] = T.max(F.mse_loss(v[:][0], ret[:][0]), F.mse_loss(clip_v[:][0], ret[:][0]))
        vf_loss[1] = T.max(F.mse_loss(v[:][1], ret[:][1]), F.mse_loss(clip_v[:][1], ret[:][1]))
        return vf_loss

    def compute_policy_loss(self, obs,mask, act, adv, log_pi_old):
        action_logprobs,dist_entropy = self.ACmodel(obs,mask,action=act)
        # log_pi=torch.log(torch.gather(prob_a,dim=1,index=act.unsqueeze(1).long())).squeeze(1)
        # Policy loss
        ratio = T.exp(action_logprobs - log_pi_old)
        clip_adv1 = T.clamp(ratio, 1. - self.clip_param, 1. + self.clip_param)
        ratio_ = T.min(ratio , clip_adv1)
        # clip_adv = clip_adv1.expand(2,len(clip_adv1))#* adv
        # clip_adv = T.transpose(clip_adv,dim0=1,dim1=0)* adv
        ratio_ = ratio.expand(2, len(ratio_))  # * adv
        ratio_adv = -T.transpose(ratio_, dim0=1, dim1=0) * adv
        policy_loss = -ratio_adv

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
    def save_models(self):
        T.save(self.ACmodel.state_dict(), self.ckpt_path1)
        T.save(self.sec_ACmodel.state_dict(), self.ckpt_path2)

    def load_models(self):
        self.ACmodel.load_state_dict(T.load(self.ckpt_path1))
        self.sec_ACmodel.load_state_dict(T.load(self.ckpt_path2))





