import numpy as np


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

        # self.flat_state_memory = np.zeros((self.mem_size, args.max_job*args.shop_num*(args.max_stage + 3)))
        # self.flat_new_state_memory = np.zeros((self.mem_size, args.max_job*args.shop_num*(args.max_stage + 3)))
        # self.state_memory = np.zeros((self.mem_size, args.max_job*args.shop_num, args.max_stage + 3))
        # self.new_state_memory = np.zeros((self.mem_size, args.max_job * args.shop_num, args.max_stage + 3))
        self.reward_memory = np.zeros((self.mem_size, args.shop_num))
        self.terminal_memory = np.zeros((self.mem_size, args.shop_num), dtype=bool)
        # self.all_actions = np.zeros((self.mem_size, args.shop_num))
        self.agent_store_index = np.zeros(args.shop_num, dtype=int)
        # self.decision_shop_id = np.zeros(self.mem_size,dtype=int)##记录当前做决策的shop_id
        self.init_actor_memory()

    def init_actor_memory(self):
        self.actor_state_memory = []
        self.actor_mask_memory = []
        self.actor_new_state_memory = []
        self.actor_new_mask_memory = []
        self.actor_action_memory = []
        self.actor_reward_mamory = []
        self.actor_terminal = []
        # self.actor_store_index = []##因为输入全为0时actor不存储，为了与state保持一直，记录每次的存储序号

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

    def memorize(self, state, action, next_state, reward, terminal):
        self.trans_mem.append(self.trans(
            torch.from_numpy(state).type(FloatTensor),  # state
            action,  # action
            torch.from_numpy(next_state).type(FloatTensor),  # next state
            torch.from_numpy(reward).type(FloatTensor),  # reward
            terminal))  # terminal

        # randomly produce a preference for calculating priority
        # preference = self.keep_preference
        preference = torch.randn(self.model.reward_size)
        preference = (torch.abs(preference) / torch.norm(preference, p=1)).type(FloatTensor)
        state = torch.from_numpy(state).type(FloatTensor)

        _, q = self.model(Variable(state.unsqueeze(0), volatile=True),
                          Variable(preference.unsqueeze(0), volatile=True))

        q = q[0, action].data
        wq = preference.dot(q)

        wr = preference.dot(torch.from_numpy(reward).type(FloatTensor))
        if not terminal:
            next_state = torch.from_numpy(next_state).type(FloatTensor)
            hq, _ = self.model(Variable(next_state.unsqueeze(0), volatile=True),
                               Variable(preference.unsqueeze(0), volatile=True))
            hq = hq.data[0]
            whq = preference.dot(hq)
            p = abs(wr + self.gamma * whq - wq)
        else:
            self.keep_preference = None
            if self.epsilon_decay:
                self.epsilon -= self.epsilon_delta
            p = abs(wr - wq)
        p += 1e-5

        self.priority_mem.append(
            p
        )
        if len(self.trans_mem) > self.mem_size:
            self.trans_mem.popleft()
            self.priority_mem.popleft()


    def store_transition(self, raw_obs, mask, action, reward,
                         raw_obs_, mask_, done, shop_id):
        # this introduces a bug: if we fill up the memory capacity and then
        # zero out our actor memory, the critic will still have memories to access
        # while the actor will have nothing but zeros to sample. Obviously
        # not what we intend.
        # In reality, there's no problem with just using the same index
        # for both the actor and critic states. I'm not sure why I thought
        # this was necessary in the first place. Sorry for the confusion!

        # if self.mem_cntr % self.mem_size == 0 and self.mem_cntr > 0:
        #    self.init_actor_memory()

        # index = self.mem_cntr % self.mem_size

        self.actor_state_memory[shop_id][self.agent_store_index[shop_id]] = raw_obs
        self.actor_mask_memory[shop_id][self.agent_store_index[shop_id]] = mask
        self.actor_new_state_memory[shop_id][self.agent_store_index[shop_id]] = raw_obs_
        self.actor_new_mask_memory[shop_id][self.agent_store_index[shop_id]] = mask_
        self.actor_action_memory[shop_id][self.agent_store_index[shop_id]] = action
        # self.actor_store_index[agent_idx][self.agent_store_index[agent_idx]] = index
        self.agent_store_index[shop_id] += 1
        ##把state打平再存储
        # self.decision_shop_id[index] = shop_id
        # self.state_memory[index] = state
        # self.new_state_memory[index] = state_
        # flat_state = self.flatt(raw_obs)
        # flat_state_ = self.flatt(raw_obs_)
        # self.flat_state_memory[index] = flat_state
        # self.flat_new_state_memory[index] = flat_state_
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

        # states = self.flat_state_memory[batch]
        # rewards = self.reward_memory[batch]
        # states_ = self.flat_new_state_memory[batch]
        # terminal = self.terminal_memory[batch]
        # all_action = self.all_actions[batch]
        # shop_id = self.decision_shop_id[batch]

        actor_states = self.actor_state_memory[shop_id][batch]
        actor_mask = self.actor_mask_memory[shop_id][batch]
        actor_new_states = self.actor_new_state_memory[shop_id][batch]
        actor_new_mask = self.actor_new_mask_memory[shop_id][batch]
        actions = self.actor_action_memory[shop_id][batch]
        actor_reward = self.actor_reward_mamory[shop_id][batch]
        actor_terminal = self.actor_terminal[shop_id][batch]
        actor_store_in = []

        # for agent_idx in range(self.n_agents):
        #     actor_states.append()
        #     actor_mask.append()
        #     actor_new_states.append()
        #     actor_new_mask.append()
        #     actions.append()
        #
        # actor_store_in.append(self.actor_store_index[agent_idx][batch])

        return actor_states, actor_mask, actor_new_states, actor_new_mask, actions, actor_reward, actor_terminal

    def ready(self, agent_id):
        if self.agent_store_index[agent_id] >= self.batch_size:
            return True
