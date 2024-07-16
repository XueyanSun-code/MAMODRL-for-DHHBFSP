import numpy as np
import torch
from torch import FloatTensor


class MultiAgentReplayBuffer:
    def __init__(self, max_size, args,
                 n_actions, batch_size):
        self.mem_size = max_size
        self.args = args
        self.mem_cntr = 0
        self.n_agents = args.shop_num
        # self.actor_dims = actor_dims
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.reward_memory = np.zeros((self.mem_size, args.shop_num))
        self.terminal_memory = np.zeros((self.mem_size, args.shop_num), dtype=bool)
        # self.all_actions = np.zeros((self.mem_size, args.shop_num))
        self.agent_store_index = np.zeros(args.shop_num, dtype=int)

        self.init_actor_memory()

    def init_actor_memory(self):
        self.actor_state_memory = []
        self.actor_mask_memory = []
        self.actor_new_state_memory = []
        self.actor_new_mask_memory = []
        self.actor_action_memory = []
        self.actor_reward_mamory = []
        self.actor_terminal = []


        for i in range(self.n_agents):
            self.actor_state_memory.append(
                np.zeros((self.mem_size, self.args.max_job*self.args.input_size)))
            self.actor_mask_memory.append(
                np.zeros((self.mem_size, self.args.max_job)))
            self.actor_new_state_memory.append(
                np.zeros((self.mem_size, self.args.max_job*self.args.input_size)))
            self.actor_new_mask_memory.append(
                np.zeros((self.mem_size, self.args.max_job)))
            self.actor_action_memory.append(
                np.zeros((self.mem_size, self.n_actions)))
            self.actor_reward_mamory.append(np.zeros((self.mem_size, 2)))
            self.actor_terminal.append(np.zeros(self.mem_size, dtype=bool))
            # self.actor_store_index.append(np.zeros((self.mem_size)))

    def store_transition(self, raw_obs, mask, action, reward,
                         raw_obs_, mask_, done, shop_id):

        self.actor_state_memory[shop_id][self.agent_store_index[shop_id]] = raw_obs
        self.actor_mask_memory[shop_id][self.agent_store_index[shop_id]] = mask
        self.actor_new_state_memory[shop_id][self.agent_store_index[shop_id]] = raw_obs_
        self.actor_new_mask_memory[shop_id][self.agent_store_index[shop_id]] = mask_
        self.actor_action_memory[shop_id][self.agent_store_index[shop_id]] = action
        # self.actor_store_index[agent_idx][self.agent_store_index[agent_idx]] = index
        self.agent_store_index[shop_id] += 1

        self.actor_reward_mamory[shop_id][self.agent_store_index] = reward
        self.actor_terminal[shop_id][self.agent_store_index] = done
        # self.all_actions[index] = action
        self.mem_cntr += 1

    def reward_update(self, rew):
        for i in range(self.n_agents):
            self.actor_reward_mamory[i][-len(rew[i]):] = rew[i]

    def flatt(self, state):
        flat_state = np.array([])
        for obs in state:
            flat_state = np.concatenate([flat_state, obs])
        return flat_state

    def sample_buffer(self, shop_id):

        max_mem = min(self.agent_store_index[shop_id], self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)


        actor_states = self.actor_state_memory[shop_id][batch]
        actor_mask = self.actor_mask_memory[shop_id][batch]
        actor_new_states = self.actor_new_state_memory[shop_id][batch]
        actor_new_mask = self.actor_new_mask_memory[shop_id][batch]
        actions = self.actor_action_memory[shop_id][batch]
        actor_reward = self.actor_reward_mamory[shop_id][batch]
        actor_terminal = self.actor_terminal[shop_id][batch]
        actor_store_in = []



        return actor_states, actor_mask, actor_new_states, actor_new_mask, actions, actor_reward, actor_terminal

    def ready(self, agent_id):
        if self.agent_store_index[agent_id] >= self.batch_size:
            return True
