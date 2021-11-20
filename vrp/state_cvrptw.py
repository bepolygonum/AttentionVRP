import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter


class StateCVRPTW(NamedTuple):
    # Fixed input
    coords: torch.Tensor  # Depot + loc
    demand: torch.Tensor

    time_window: torch.Tensor  # 每个客户的最早开始时间,每个客户的截止时间

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the coords and demands tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    prev_a: torch.Tensor
    used_capacity: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step

    cur_time: torch.Tensor  # 当前时间

    DUE: torch.Tensor
    SERVICE_TIME: torch.Tensor
    VEHICLE_CAPACITY = 1.0  # Hardcoded

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.demand.size(-1))

    @property
    def dist(self):
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):  # If tensor, idx all tensors by this tensor:
            return self._replace(
                ids=self.ids[key],
                prev_a=self.prev_a[key],
                used_capacity=self.used_capacity[key],
                visited_=self.visited_[key],
                lengths=self.lengths[key],
                cur_coord=self.cur_coord[key],
                cur_time=self.cur_time[key],
                time_window=self.time_window[key]
            )
        return super(StateCVRPTW, self).__getitem__(key)

    # Warning: cannot override len of NamedTuple, len should be number of fields, not batch size
    # def __len__(self):
    #     return len(self.used_capacity)

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):

        depot = input['depot']
        loc = input['loc']
        demand = input['demand']
        tw = input['time_window']
        due = input['due']
        service_time = input['service_time']

        batch_size, n_loc, _ = loc.size()
        return StateCVRPTW(
            coords=torch.cat((depot[:, None, :], loc), -2),
            demand=demand,
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            used_capacity=demand.new_zeros(batch_size, 1),
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently
                torch.zeros(
                    batch_size, 1, n_loc + 1,
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord=input['depot'][:, None, :],  # Add step dimension
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),  # Vector with length num_steps
            cur_time=torch.zeros(batch_size, 1, device=loc.device),
            time_window=tw,
            DUE=due,
            SERVICE_TIME=service_time

        )

    def get_final_cost(self):

        assert self.all_finished()

        return self.lengths + (self.coords[self.ids, 0, :] - self.cur_coord).norm(p=2, dim=-1)

    def update(self, selected):

        assert self.i.size(0) == 1, "Can only update if state represents single step"

        # Update the state
        selected = selected[:, None]  # Add dimension for step
        prev_a = selected

        n_loc = self.demand.size(-1)  # Excludes depot

        # Add the length
        cur_coord = self.coords[self.ids, selected]


        # cur_coord = self.coords.gather(
        #     1,
        #     selected[:, None].expand(selected.size(0), 1, self.coords.size(-1))
        # )[:, 0, :]
        lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # (batch_dim, 1)

        # print("prev_a, size:",prev_a.shape)
        # print("prev_a[:, 0], size:",prev_a[:, 0])
        # print("cur_time,size:",self.cur_time[self.ids].shape)
        # print("cur_time[ids],size:", self.cur_time[self.ids].shape)
        # print("size:",((cur_coord - self.cur_coord).norm(p=2, dim=-1)+torch.full(self.cur_time.shape, self.SERVICE_TIME)).shape)

        # 到达该节点的时间
        # cur_time.shape:[val_batch_size,1]
        cur_time = self.cur_time+(cur_coord - self.cur_coord).norm(p=2, dim=-1)

        # 若是早到了需要等待
        for t in range(cur_time.size(0)):
            if prev_a[t, 0] != 0 and cur_time[t, 0] - self.time_window[t, prev_a[t, 0]-1, 0] < 0:
                cur_time[t, 0] = self.time_window[t, prev_a[t, 0]-1, 0]

            if t != 0 and prev_a[t, 0] != 0:
                cur_time[t, 0] += self.SERVICE_TIME[t-1]  # 加上上一个结点的服务时间
            elif prev_a[t, 0] == 0:
                cur_time[t, 0] = 0

        #  cur_time += torch.full(self.cur_time.shape, self.SERVICE_TIME)  # (batch_dim, 1)

        # for x in range(prev_a.size(0)):
        #     if prev_a[x, 0] == 0:
        #         cur_time[x, 0] = 0

        # Not selected_demand is demand of first node (by clamp) so incorrect for nodes that visit depot!
        # selected_demand = self.demand.gather(-1, torch.clamp(prev_a - 1, 0, n_loc - 1))
        selected_demand = self.demand[self.ids, torch.clamp(prev_a - 1, 0, n_loc - 1)]

        # Increase capacity if depot is not visited, otherwise set to 0
        # used_capacity = torch.where(selected == 0, 0, self.used_capacity + selected_demand)
        used_capacity = (self.used_capacity + selected_demand) * (prev_a != 0).float()

        if self.visited_.dtype == torch.uint8:
            # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            # This works, will not set anything if prev_a -1 == -1 (depot)
            visited_ = mask_long_scatter(self.visited_, prev_a - 1)
        return self._replace(
            prev_a=prev_a, used_capacity=used_capacity, visited_=visited_,
            lengths=lengths, cur_coord=cur_coord, i=self.i + 1, cur_time=cur_time
        )

    def all_finished(self):
        return self.i.item() >= self.demand.size(-1) and self.visited.all()

    def get_finished(self):
        return self.visited.sum(-1) == self.visited.size(-1)

    def get_current_node(self):
        return self.prev_a

    def get_cur_time(self):
        return self.cur_time

    def get_mask(self):
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited and
        remaining capacity. 0 = feasible, 1 = infeasible
        Forbids to visit depot twice in a row, unless all nodes have been visited
        :return:
        """

        if self.visited_.dtype == torch.uint8:
            visited_loc = self.visited_[:, :, 1:]
        else:
            visited_loc = mask_long2bool(self.visited_, n=self.demand.size(-1))

        # Nodes that cannot be visited are already visited or too much demand to be served now
        # For demand steps_dim is inserted by indexing with id, for used_capacity insert node dim for broadcasting

        # 或已经超过用户最晚服务时间 超过则为1
        due_time = (self.cur_time[self.ids] +
                    (self.cur_coord[self.ids]-self.coords[self.ids, 1:]).norm(p=2, dim=-1)
                    > self.time_window[self.ids, :, 1]).byte()

        mask_loc = (
            visited_loc |
            (self.demand[self.ids, :] + self.used_capacity[:, :, None] > self.VEHICLE_CAPACITY).byte() |
            due_time
        )

        # Cannot visit the depot if just visited and still unserved nodes
        mask_depot = (self.prev_a == 0) & ((mask_loc == 0).int().sum(-1) > 0)

        return torch.cat((mask_depot[:, :, None], mask_loc.type(torch.bool)), -1)

    def construct_solutions(self, actions):
        return actions
