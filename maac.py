import torch as T
import torch.nn.functional as F
from agent import Agent_AC
import torch.nn as nn
import copy
import numpy as np

class MAA2C:
    def __init__(self, args, n_agents, n_actions, chkpt_dir='tmp/maac/'):
        self.agents = []
        self.args = args
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.eval_critic_loss_list = [[], [], []]
        self.eval_actorloss_list = [[], [], []]
        #chkpt_dir += scenario
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent_AC(args, agent_idx, chkpt_dir))

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
                action = action[0]

        return action

    def list_to_vector(self,actor_mask):
        mask_old = actor_mask[0]
        for mm in actor_mask[1:]:
            mask_old = np.concatenate([mask_old, mm])
        return mask_old

    def learn(self, memory):
        # if not memory.ready():
        #     return
        for agent_id, agent in enumerate(self.agents):
            if not memory.ready(agent_id):
                return
            actor_states,actor_mask,actor_new_states,actor_new_mask,actions, rewards, dones = memory.sample_buffer(agent_id)

            device = self.agents[0].actor.device
            actor_states = T.tensor(actor_states, dtype=T.float).to(device)
            actor_mask = T.tensor(actor_mask, dtype=T.float).to(device)
            actor_new_states = T.tensor(actor_new_states, dtype=T.float).to(device)
            actor_new_mask = T.tensor(actor_new_mask, dtype=T.float).to(device)
            # states = T.tensor(states, dtype=T.float).to(device)
            # all_action = T.tensor(all_action, dtype=T.float).to(device)
            actions = T.tensor(actions, dtype=T.float).to(device)
            rewards = T.tensor(rewards,dtype=T.float).to(device)
            # states_ = T.tensor(states_, dtype=T.float).to(device)
            dones = T.tensor(dones).to(device)
            #
            # new_actions = copy.copy(all_action)
            # new_mu_actions = copy.copy(all_action)
            # old_actions = all_action


            # new_mask = copy.deepcopy(actor_mask)
            #     new_states = T.tensor(actor_new_states[shop_id[i]][i],
            #                          dtype=T.float).unsqueeze(0).to(device)
            #     mask_ = T.tensor(actor_new_mask[shop_id[i]][i],
            #                           dtype=T.float).unsqueeze(0).to(device)
            #计算新状态对应的目标网络的动作

            # new_actions[i][shop_id[i]] = new_ac[0]

            # all_agents_new_actions.append(new_ac)
            #计算原来actor的state在当下网络的新动作
            # mu_states = T.tensor(actor_states[shop_id[i]][i],
            #                      dtype=T.float).unsqueeze(0).to(device)
            # mu_mask = T.tensor(actor_mask[shop_id[i]][i],
            #                      dtype=T.float).unsqueeze(0).to(device)
            action_logprobs,dist_entropy = agent.actor.forward(actor_states,actor_mask,action=actions)


            actor_states = actor_states.reshape(self.args.batch_size,self.args.max_job*self.args.input_size)
            actor_new_states = actor_new_states.reshape(self.args.batch_size, self.args.max_job * self.args.input_size)
            # Q_pi = agent.critic.forward(actor_states,ac).flatten()
            critic_value = agent.critic.forward(actor_states).flatten()
            critic_value_ = agent.critic.forward(actor_new_states).flatten()
           # critic_value_[dones[:,0]] = 0.0

            #critic_value = copy.deepcopy(critic_value1)
            target = rewards + agent.gamma**critic_value_
            ##这里采用了advantage
            adv = target - critic_value

            actor_loss = (action_logprobs*adv).mean()
            critic_loss = F.mse_loss(critic_value, target)
            print('critic loss for agent'+ str(agent_id), critic_loss.detach().numpy())
            print('actor loss for agent'+ str(agent_id), actor_loss.detach().numpy())

            # save testint result...
            self.eval_critic_loss_list[agent_id].append(critic_loss.detach().numpy())
            self.eval_actorloss_list[agent_id].append(actor_loss.detach().numpy())

            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)

            agent.actor.optimizer.step()

            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)

            agent.critic.optimizer.step()



            # agent.update_network_parameters()
