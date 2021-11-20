from torch.utils.data import Dataset
import torch
import os
import pickle
from vrp.state_cvrptw import StateCVRPTW
from utils.beam_search import beam_search


class CVRPTW(object):

    # NAME = 'cvrptw'  # Capacitated Vehicle Routing Problem with Time Windows

    DUE = 100.0
    SERVICE_TIME = 1.0
    VEHICLE_CAPACITY = 1.0
    # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    @staticmethod
    def get_costs(dataset, pi, time):
        batch_size, graph_size = dataset['demand'].size()
        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_pi = pi.data.sort(1)[0]
        # print(pi,time)
        # Sorting it should give all zeros at front and then 1...n
        assert (
            torch.arange(1, graph_size + 1, out=pi.data.new()).view(1, -1).expand(batch_size, graph_size) ==
            sorted_pi[:, -graph_size:]
        ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

        # Visiting depot resets capacity so we add demand = -capacity (we make sure it does not become negative)
        demand_with_depot = torch.cat(
            (
                torch.full_like(dataset['demand'][:, :1], -CVRPTW.VEHICLE_CAPACITY),
                dataset['demand']
            ),
            1
        )
        d = demand_with_depot.gather(1, pi)

        used_cap = torch.zeros_like(dataset['demand'][:, 0])
        for i in range(pi.size(1)):
            used_cap += d[:, i]  # This will reset/make capacity negative if i == 0, e.g. depot visited
            # Cannot use less than 0
            used_cap[used_cap < 0] = 0
            assert (used_cap <= CVRPTW.VEHICLE_CAPACITY + 1e-5).all(), "Used more than capacity"

        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))
        # # 以0为分隔符分割tour数组 e.g. [0, 1, 1, 0, 1, 3, 6, 3, 0, 1] -> [[1,1],[1,3,6,3],[1]]
        # pos = np.array([np.where(b > 0)[0] for b in pi])
        # split = np.array([np.where(np.diff(p) != 1)[0] + 1 for p in pos]).tolist()
        # temp = np.array(pi[:, pos][0], dtype=np.int64).tolist()
        # arr = [np.split(b, c) for (b, c) in zip(temp, split)]

        # 一个tour中，不同车辆之间行驶距离的标准差和超时惩罚
        stdv = []
        total_time_p = []

        tw = torch.cat((dataset['due'][:, None], dataset['time_window'][:, :, 1]), 1)  # 加上了due_time的时间窗

        for i in range(pi.size(0)):
            dis = []
            l = 0
            time_b = torch.tensor(0.0).cuda()
            last_t = torch.tensor(0.0)
            for j in range(pi.size(1)):
                t = time[i, j] + dataset['service_time'][i]  # 服务完成的时刻
                flag = 0
                if j == pi.size(1)-1 and pi[i, j] != 0:  # tour最后一位不为0
                    r = j+1
                    d_ = (d[i, (l + 1):r] - d[i, l:(r - 1)]).norm(p=2, dim=1).sum(0) + (d[i, l] - dataset['depot'][i, :]).norm(
                        p=2, dim=0) + (d[i, r - 1] - dataset['depot'][i, :]).norm(p=2, dim=0)
                    dis.append(d_)
                    t += (d[i, j] - dataset['depot'][i, :]).norm(p=2, dim=0)
                    flag = 1

                elif pi[i, j] == 0:  # 遇到0，即该车返回时
                    r = j
                    _d = (d[i, (l+1):r] - d[i, l:(r-1)]).norm(p=2, dim=1).sum(0) + (d[i, l] - dataset['depot'][i, :]).norm(p=2, dim=0) + (d[i, r-1] - dataset['depot'][i, :]).norm(p=2, dim=0)
                    dis.append(_d)
                    l = r+1

                    t = last_t + (d[i, j-1] - dataset['depot'][i, :]).norm(p=2, dim=0)

                # 罚时
                tempp = abs(t - tw[0, pi[i, j]]) if flag == 0 else abs(t - tw[i, 0])
                time_b += tempp
                last_t = t

            total_time_p.append(time_b)
            dis = torch.stack(dis, 0).std()
            stdv.append(dis)


        stdv = torch.stack(stdv, 0)
        total_time_p = torch.stack(total_time_p, 0)

        # # total_time.shape:[1024]
        # total_time = ((d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
        #     + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
        #     + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)
        #     + CVRPTW.SERVICE_TIME * graph_size)  # Last to depot, will be 0 if depot is last
        # assert (total_time < CVRPTW.DUE + 1e-5).all(), "Total time out"

        stdv = stdv.cuda()
        total_time_p = total_time_p.cuda()
        # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
        return (
               (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1).cuda()
               + (d[:, 0] - dataset['depot']).norm(p=2, dim=1).cuda()  # Depot to first
               + (d[:, -1] - dataset['depot']).norm(p=2, dim=1).cuda()  # Last to depot, will be 0 if depot is last
               + stdv
               + 0.001 * total_time_p
        ), None
    @staticmethod
    def make_dataset(*args, **kwargs):
        return VRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateCVRPTW.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = CVRPTW.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


def make_instance(args):
    depot, loc, demand, capacity, time_window, due, service_time, *args = args
    grid_size = 1
    if len(args) > 0:
        depot_types, customer_types, grid_size = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
        'demand': torch.tensor(demand, dtype=torch.float) / capacity,
        'depot': torch.tensor(depot, dtype=torch.float) / grid_size,
        'time_window': torch.tensor(time_window, dtype=torch.float),
        'due': torch.tensor(due, dtype=torch.float),
        'service_time': torch.tensor(service_time, dtype=torch.float)
    }


class VRPDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0):
        super(VRPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = [make_instance(args) for args in data[offset:offset+num_samples]]

        else:

            # From VRP with RL paper https://arxiv.org/abs/1802.04240
            CAPACITIES = {
                10: 20.,
                25: 40.,
                50: 40.,
                100: 50.
            }
            self.data = [{
                'loc': torch.FloatTensor(size, 2).uniform_(0, 1),
                # Uniform 1 - 9, scaled by capacities
                'demand': (torch.FloatTensor(size).uniform_(0, 9).int() + 1).float() / CAPACITIES[size],
                'depot': torch.FloatTensor(2).uniform_(0, 1),
                'time_window':torch.cat((torch.FloatTensor(size, 1).uniform_(0, size/2), torch.FloatTensor(size, 1).uniform_(size/2+1, size)), -1),
                'due':torch.tensor(100.0),
                'service_time':torch.tensor(1.0)
            }for _ in range(num_samples)
        ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
