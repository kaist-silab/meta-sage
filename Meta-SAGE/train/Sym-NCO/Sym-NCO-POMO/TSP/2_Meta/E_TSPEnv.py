import torch
import pickle

from TSPEnv import TSPEnv, get_random_problems, augment_xy_data_by_8_fold


class E_TSPEnv(TSPEnv):

    # def __init__(self, **model_params):
    #     super().__init__(**model_params)

    def load_problems_by_index(self, start_index, batch_size, aug_factor=1):
        self.batch_size = batch_size

        if not self.FLAG__use_saved_problems:
            self.problems = get_random_problems(batch_size, self.problem_size)
        else:
            self.saved_index = start_index
            self.problems = self.saved_problems[self.saved_index:self.saved_index+batch_size]
            self.saved_index += batch_size

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                self.problems = augment_xy_data_by_8_fold(self.problems)
                # shape: (8*batch, problem, 2)
            else:
                raise NotImplementedError

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

    def step(self, selected):
        state, reward, done = super().step(selected)
        state.first_node = self.selected_node_list[:, :, 0]
        # shape: (batch, pomo)
        return state, reward, done

    def modify_pomo_size_for_eas(self, new_pomo_size):
        self.pomo_size = new_pomo_size
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

    def use_pkl_saved_problems(self, filename, num_problems, index_begin=0):
        self.FLAG__use_saved_problems = True
        # print(filename)
        with open(filename, 'rb') as pickle_file:
            data = pickle.load(pickle_file)
        partial_data = list(data[i] for i in range(index_begin, index_begin+num_problems))

        self.saved_problems = torch.tensor(partial_data)
        self.saved_index = 0

    def copy_problems(self, old_env):
        self.batch_size = old_env.batch_size
        self.problems = old_env.problems

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

    def reset_by_repeating_bs_env(self, bs_env, repeat):
        self.selected_count = bs_env.selected_count
        self.current_node = bs_env.current_node.repeat_interleave(repeat, dim=1)
        # shape: (batch, pomo)
        self.selected_node_list = bs_env.selected_node_list.repeat_interleave(repeat, dim=1)
        # shape: (batch, pomo, 0~)

        # STEP STATE
        self.step_state.current_node = self.current_node
        # shape: (batch, pomo)
        self.step_state.ninf_mask = bs_env.step_state.ninf_mask.repeat_interleave(repeat, dim=1)
        # shape: (batch, pomo, node)

    def reset_by_gathering_rollout_env(self, rollout_env, gathering_index):
        self.selected_count = rollout_env.selected_count
        self.current_node = rollout_env.current_node.gather(dim=1, index=gathering_index)
        # shape: (batch, pomo)
        exp_gathering_index = gathering_index[:, :, None].expand(-1, -1, self.selected_count)
        self.selected_node_list = rollout_env.selected_node_list.gather(dim=1, index=exp_gathering_index)
        # shape: (batch, pomo, 0~)

        # STEP STATE
        self.step_state.current_node = self.current_node
        # shape: (batch, pomo)
        exp_gathering_index = gathering_index[:, :, None].expand(-1, -1, self.problem_size)
        self.step_state.ninf_mask = rollout_env.step_state.ninf_mask.gather(dim=1, index=exp_gathering_index)
        # shape: (batch, pomo, problem)

    def merge(self, other_env):
        self.current_node = torch.cat((self.current_node, other_env.current_node), dim=1)
        # shape: (batch, pomo1 + pomo2)
        self.selected_node_list = torch.cat((self.selected_node_list, other_env.selected_node_list), dim=1)
        # shape: (batch, pomo1 + pomo2, 0~)

        # STEP STATE
        self.step_state.current_node = self.current_node
        # shape: (batch, pomo1 + pomo2)
        self.step_state.ninf_mask = torch.cat((self.step_state.ninf_mask, other_env.step_state.ninf_mask), dim=1)
        # shape: (batch, pomo1 + pomo2, problem)
