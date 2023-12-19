import numpy as np
import torch
import pickle
import os 
from CVRPEnv import CVRPEnv, get_random_problems, augment_xy_data_by_8_fold

class E_CVRPEnv(CVRPEnv):

    # def __init__(self, **model_params):
    #     super().__init__(**model_params)

    def load_problems_by_index(self, start_index, batch_size, aug_factor=1):
        self.batch_size = batch_size

        if not self.FLAG__use_saved_problems:
            depot_xy, node_xy, node_demand = get_random_problems(batch_size, self.problem_size)
        else:
            self.saved_index = start_index
            depot_xy = self.saved_depot_xy[self.saved_index:self.saved_index+batch_size]
            node_xy = self.saved_node_xy[self.saved_index:self.saved_index+batch_size]
            node_demand = self.saved_node_demand[self.saved_index:self.saved_index+batch_size]
            self.saved_index += batch_size

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = augment_xy_data_by_8_fold(depot_xy, aug_factor)
                node_xy = augment_xy_data_by_8_fold(node_xy, aug_factor)
                node_demand = node_demand.repeat(8, 1)
            elif aug_factor == 3:
                self.batch_size = self.batch_size * 3
                depot_xy = augment_xy_data_by_8_fold(depot_xy, aug_factor)
                node_xy = augment_xy_data_by_8_fold(node_xy, aug_factor)
                node_demand = node_demand.repeat(3, 1)
            elif aug_factor == 4:
                self.batch_size = self.batch_size * 4
                depot_xy = augment_xy_data_by_8_fold(depot_xy, aug_factor)
                node_xy = augment_xy_data_by_8_fold(node_xy, aug_factor)
                node_demand = node_demand.repeat(4, 1)
            else:
                raise NotImplementedError

        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        # shape: (batch, problem+1, 2)
        depot_demand = torch.zeros(size=(self.batch_size, 1))
        # shape: (batch, 1)
        # print(depot_demand.shape)
        # print(node_demand.shape)
        # print(node_demand.shape)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        # shape: (batch, problem+1)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

    def modify_pomo_size_for_eas(self, new_pomo_size):
        
        self.pomo_size = new_pomo_size
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)
        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

    def read_instance_vrp(self, filename):
        self.FLAG__use_saved_problems = True
        file = open(filename, "r")
        lines = [ll.strip() for ll in file]
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("DIMENSION"):
                dimension = int(line.split(':')[1])
            elif line.startswith("CAPACITY"):
                capacity = int(line.split(':')[1])
            elif line.startswith('NODE_COORD_SECTION'):
                locations = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=int)
                i = i + dimension
            elif line.startswith('DEMAND_SECTION'):
                demand = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=int)
                i = i + dimension

            i += 1
        # print(locations.shape)
        # print(locations[:, 1:])
        
        locations = locations[:, 1:]
        max_val = max(locations[:,0].max(), locations[:,1].max())
        locations = locations.astype(float)
        locations[:,0] = locations[:,0] / max_val
        locations[:,1] = locations[:,1] / max_val
        
        # print("x axis max : {}".format(locations[:,0].max()))
        # print("y axis max : {}".format(locations[:,1].max()))
        # locations[:,0] = locations[:,0] / locations[:,1].max()
        # locations[:,1] = locations[:,1] / locations[:,1].max()
        
        demand = demand[:,1:]
        # print(demand[1:].reshape((1, -1)))
        # capacities = [capacity]
        # depot = locations[0]
        
    #     depot = np.array([locations[0]], dtype=np.float).tolist()
    #     node_xy = np.array([locations[1:]], dtype=np.float).tolist()
    #     demand_lst = np.array(demand[1:].reshape(1, -1), dtype=np.float).tolist()
        
    #     capacity_lst = np.array([capacity], dtype=np.float).tolist()
    #     # print(capacity_lst)
    #     # print(demand_lst)
        
    #     dataset = list(zip(
    #     depot,  # Depot location
    #     node_xy,  # Node locations
    #     demand_lst,  # Demand, uniform integer 1 ... 9
    #     capacity_lst  # Capacity, same for whole dataset
    # ))
    #     save_dataset(dataset, "../0_test_data_set/X-n200-k36")

    #     return list(zip(
    #     np.random.uniform(size=(dataset_size, 2)).tolist(),  # Depot location
    #     np.random.uniform(size=(dataset_size, vrp_size, 2)).tolist(),  # Node locations
    #     np.random.randint(1, 10, size=(dataset_size, vrp_size)).tolist(),  # Demand, uniform integer 1 ... 9
    #     np.full(dataset_size, CAPACITIES[vrp_size]).tolist()  # Capacity, same for whole dataset
    # ))



        # print(locations[1:].tolist())
        # dataset = list(zip(depot.astype(np.float).reshape((1, -1)).tolist(),locations[1:].astype(np.float).reshape((1, -1, 2)).tolist(),demand[1:].reshape((1, -1)).tolist(),np.array(capacities,dtype=np.float).tolist()))
        
        # with open((filename[:-4]+".pkl"), 'wb') as f:
        #     pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
        
        self.saved_depot_xy = torch.tensor(locations[0], dtype=torch.float)[None,None,:]
        self.saved_node_xy = torch.tensor(locations[1:], dtype=torch.float)[None, :,:]
        # print(self.saved_depot_xy.shape)
        # print(self.saved_node_xy.shape)
        self.saved_node_demand = torch.tensor(demand[1:].reshape((1, -1)), dtype=torch.float) / torch.tensor(capacity, dtype=torch.float)
        self.saved_index = 0
        print(max_val)
        return (self.saved_node_xy.shape[1], self.saved_node_xy.shape[1], max_val)
    def use_pkl_saved_problems(self, filename, num_problems, index_begin=0):
        self.FLAG__use_saved_problems = True

        with open(filename, 'rb') as pickle_file:
            data = pickle.load(pickle_file)
       
        depot_data = list(data[i][0] for i in range(index_begin, index_begin+num_problems))
        self.saved_depot_xy = torch.tensor(depot_data)[:, None, :]
        # shape: (batch, 1, 2)

        node_data = list(data[i][1] for i in range(index_begin, index_begin+num_problems))
        self.saved_node_xy = torch.tensor(node_data)
        # shape: (batch, problem, 2)
        
        demand_data = list(data[i][2] for i in range(index_begin, index_begin+num_problems))
        
        capacity_data = list(data[i][3] for i in range(index_begin, index_begin+num_problems))
        # print(capacity_data)
        capacity_tensor = torch.tensor(capacity_data, dtype=torch.float)
        self.saved_node_demand = torch.tensor(demand_data, dtype=torch.float)/capacity_tensor[:, None]
        # shape: (batch, problem)

        self.saved_index = 0
        
        #return (self.saved_node_xy.shape[1], self.saved_node_xy.shape[1])

    def copy_problems(self, old_env):
        self.batch_size = old_env.batch_size
        self.depot_node_xy = old_env.depot_node_xy
        self.depot_node_demand = old_env.depot_node_demand

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

    def reset_by_repeating_bs_env(self, bs_env, repeat):
        self.selected_count = bs_env.selected_count
        self.current_node = bs_env.current_node.repeat_interleave(repeat, dim=1)
        # shape: (batch, pomo)
        self.selected_node_list = bs_env.selected_node_list.repeat_interleave(repeat, dim=1)
        # shape: (batch, pomo, 0~)

        self.at_the_depot = bs_env.at_the_depot.repeat_interleave(repeat, dim=1)
        # shape: (batch, pomo)
        self.load = bs_env.load.repeat_interleave(repeat, dim=1)
        # shape: (batch, pomo)
        self.visited_ninf_flag = bs_env.visited_ninf_flag.repeat_interleave(repeat, dim=1)
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = bs_env.ninf_mask.repeat_interleave(repeat, dim=1)
        # shape: (batch, pomo, problem+1)
        self.finished = bs_env.finished.repeat_interleave(repeat, dim=1)
        # shape: (batch, pomo)

    def reset_by_gathering_rollout_env(self, rollout_env, gathering_index):
        self.selected_count = rollout_env.selected_count
        self.current_node = rollout_env.current_node.gather(dim=1, index=gathering_index)
        # shape: (batch, pomo)
        exp_gathering_index = gathering_index[:, :, None].expand(-1, -1, self.selected_count)
        self.selected_node_list = rollout_env.selected_node_list.gather(dim=1, index=exp_gathering_index)
        # shape: (batch, pomo, 0~)

        self.at_the_depot = rollout_env.at_the_depot.gather(dim=1, index=gathering_index)
        # shape: (batch, pomo)
        self.load = rollout_env.load.gather(dim=1, index=gathering_index)
        # shape: (batch, pomo)
        exp_gathering_index = gathering_index[:, :, None].expand(-1, -1, self.problem_size+1)
        self.visited_ninf_flag = rollout_env.visited_ninf_flag.gather(dim=1, index=exp_gathering_index)
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = rollout_env.ninf_mask.gather(dim=1, index=exp_gathering_index)
        # shape: (batch, pomo, problem+1)
        self.finished = rollout_env.finished.gather(dim=1, index=gathering_index)
        # shape: (batch, pomo)

    def merge(self, other_env):

        self.current_node = torch.cat((self.current_node, other_env.current_node), dim=1)
        # shape: (batch, pomo1 + pomo2)
        self.selected_node_list = torch.cat((self.selected_node_list, other_env.selected_node_list), dim=1)
        # shape: (batch, pomo1 + pomo2, 0~)

        self.at_the_depot = torch.cat((self.at_the_depot, other_env.at_the_depot), dim=1)
        # shape: (batch, pomo1 + pomo2)
        self.load = torch.cat((self.load, other_env.load), dim=1)
        # shape: (batch, pomo1 + pomo2)
        self.visited_ninf_flag = torch.cat((self.visited_ninf_flag, other_env.visited_ninf_flag), dim=1)
        # shape: (batch, pomo1 + pomo2, problem+1)
        self.ninf_mask = torch.cat((self.ninf_mask, other_env.ninf_mask), dim=1)
        # shape: (batch, pomo1 + pomo2, problem+1)
        self.finished = torch.cat((self.finished, other_env.finished), dim=1)
        # shape: (batch, pomo1 + pomo2)

def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def save_dataset(dataset, filename):

    filedir = os.path.split(filename)[0]

    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)