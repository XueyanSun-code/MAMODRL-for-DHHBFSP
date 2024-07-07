import torch as T
import torch.nn.functional as F
from agent import Agent_DDQN
import torch.nn as nn
import copy
import numpy as np
from torch.autograd import Variable
FloatTensor =  T.FloatTensor
LongTensor = T.LongTensor
ByteTensor = T.ByteTensor

class MADQN:
    def __init__(self, args, n_agents, n_actions, chkpt_dir='tmp/madqn/'):
        self.agents = []
        self.args = args
        self.weight_num = args.weight_num
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.eval_critic_loss_list = [[], [], []]
        # self.eval_actorloss_list = [[], [], []]
        self.training_num = np.zeros(2)
        #chkpt_dir += scenario
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent_DDQN(args, agent_idx, chkpt_dir))

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()


    def choose_action(self, raw_obs,mask,shop_ID):

        for agent_idx, agent in enumerate(self.agents):
            if agent_idx == shop_ID:##只在当前的车间进行工件选择
                action = agent.choose_action(raw_obs,mask)
                # action = action[0]

        return action

    def list_to_vector(self,actor_mask):
        mask_old = actor_mask[0]
        for mm in actor_mask[1:]:
            mask_old = np.concatenate([mask_old, mm])
        return mask_old

    def nontmlinds(self, terminal_batch):

        inds = []
        for i, x in enumerate(terminal_batch):
            if x == False:
                inds.append(i)
        inds = T.LongTensor(inds).to(self.args.device)
        return inds


    def learn(self, memory,preference=None):
        # if not memory.ready():
        #     return
        for agent_id, agent in enumerate(self.agents):
            if not memory.ready(agent_id):
                return
            self.training_num[agent_id] += 1
            print("agent_id=%d,第%d次训练" % (agent_id, self.training_num[agent_id]))
            actor_states,actor_mask,actor_new_states,actor_new_mask,actions, rewards, dones = memory.sample_buffer(agent_id)

            device = self.agents[0].critic.device
            actor_states = T.tensor(actor_states, dtype=T.float).to(device)
            actor_mask = T.tensor(actor_mask, dtype=T.float).to(device)
            actor_new_states = T.tensor(actor_new_states, dtype=T.float).to(device)
            actor_new_mask = T.tensor(actor_new_mask, dtype=T.float).to(device)
            # states = T.tensor(states, dtype=T.float).to(device)
            # all_action = T.tensor(all_action, dtype=T.float).to(device)
            actions = T.LongTensor(actions).squeeze().to(device)
            rewards = T.tensor(rewards,dtype=T.float).to(device)
            # states_ = T.tensor(states_, dtype=T.float).to(device)
            dones = T.tensor(dones).to(device)

            batchify = lambda x:list(x) * self.weight_num
            # ##扩展函数
            # batchify(actor_states)
            state_batch = actor_states.repeat(self.weight_num,1)
            action_batch = actions.repeat(1,self.weight_num).T
            reward_batch = rewards.repeat(self.weight_num,1)
            next_state_batch = actor_new_states.repeat(self.weight_num,1)
            mask_batch = actor_mask.repeat(self.weight_num,1)
            new_mask_batch = actor_new_mask.repeat(self.weight_num,1)
            terminal_batch = dones.repeat(1,self.weight_num)

            if preference is None:
                w_batch = np.random.randn(self.weight_num, self.args.reward_size)
                w_batch = np.abs(w_batch) / \
                          np.linalg.norm(w_batch, ord=1, axis=1, keepdims=True)
                w_batch = T.from_numpy(w_batch.repeat(self.args.batch_size, axis=0)).type(FloatTensor)
            else:
                w_batch = preference.cpu().numpy()
                w_batch = np.expand_dims(w_batch, axis=0)
                w_batch = np.abs(w_batch) / \
                          np.linalg.norm(w_batch, ord=1, axis=1, keepdims=True)
                w_batch = T.from_numpy(w_batch.repeat(self.args.batch_size, axis=0)).type(FloatTensor)

            __, Q = agent.critic.forward(state_batch, mask_batch,
                    w_batch)

            _, DQ = agent.target_critic.forward(next_state_batch, new_mask_batch,
                    w_batch)
            _, act = agent.critic.forward(next_state_batch, new_mask_batch,
                    w_batch)[1].max(1)
            HQ = DQ.gather(1, act.unsqueeze(dim=1)).squeeze()

            w_reward_batch = T.bmm(w_batch.unsqueeze(1),
                                       reward_batch.unsqueeze(2)
                                       ).squeeze()

            nontmlmask = self.nontmlinds(terminal_batch.squeeze())
            with T.no_grad():
                Tau_Q = Variable(T.zeros(self.args.batch_size * self.weight_num).type(FloatTensor))
                Tau_Q[nontmlmask] = self.args.gamma * HQ[nontmlmask]
                Tau_Q += Variable(w_reward_batch)

            critic_loss = F.smooth_l1_loss(Q.gather(1, action_batch), Tau_Q.unsqueeze(dim=1))

            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)

            agent.critic.optimizer.step()

            agent.update_network_parameters()
