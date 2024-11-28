import copy
import datetime

import numpy as np
import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter


class StateCVRP(NamedTuple):
    # Fixed input
    coords: torch.Tensor  # Depot + loc
    demand: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the coords and demands tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows
    #veh: torch.Tensor
    # State
    # prev_a: torch.Tensor
    used_capacity: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    #lengths: torch.Tensor
    #travel_times: torch.Tensor
    cur_coord: torch.Tensor
    n_vehicles: int
    # finished: torch.Tensor
    #current_point: torch.Tensor
    i: torch.Tensor  # Keeps track of step
    current_vehicle: torch.Tensor
    vehicle_capacity : torch.Tensor
    # VEHICLE_DRIVING_RANGE_WITH_MAX_CARGO = [1.2, 1.2, 1.2]
    VEHICLE_DRIVING_RANGE_WITH_MAX_CARGO = [1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2]
    # VEHICLE_DRIVING_RANGE_WITHOUT_CARGO = [2.5, 2.5, 2.5]
    VEHICLE_DRIVING_RANGE_WITHOUT_CARGO = [2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5]
    VEHICLE_SPEED_PER_MIN = 0.02
    CHARGE_TIME_BUCKET = 40
    CHARGE_AMOUNT_BUCKET = 1

    # VEHICLE_CAPACITY = [1., 1., 1.]  # Hardcoded
    # VEHICLE_CAPACITY = [1., 1., 1., 1., 1., 1., 1., 1.]  # Hardcoded

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
        assert torch.is_tensor(key) or isinstance(key, slice)  # If tensor, idx all tensors by this tensor:
        return self._replace(
            ids=self.ids[key],
            # prev_a=self.prev_a[key],
            used_capacity=self.used_capacity[key],
            visited_=self.visited_[key],
            cur_coord=self.cur_coord[key],
        )

    # Warning: cannot override len of NamedTuple, len should be number of fields, not batch size
    # def __len__(self):
    #     return len(self.used_capacity)

    @staticmethod
    def initialize(input, n_vehicles, visited_dtype=torch.uint8):

        depot = input['depot']
        loc = input['loc']
        demand = input['demand']
        batch_size, n_loc, _ = loc.size()

        vehicle_capacity = torch.ones(n_vehicles, dtype=torch.float, device=loc.device)
        return StateCVRP(
            coords=torch.cat((depot[:, 0:1, :], loc), -2),
            n_vehicles=n_vehicles,
            demand=demand,
            vehicle_capacity=vehicle_capacity,
            current_vehicle=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            # finished=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            # prev_a=torch.zeros(batch_size, n_vehicles, dtype=torch.long, device=loc.device),
            used_capacity=demand.new_zeros(batch_size, n_vehicles, dtype=torch.float, device=loc.device),
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently
                torch.zeros(
                    batch_size, 1, n_loc+1,
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            cur_coord=input['depot'][:, 0:1, :].expand(batch_size, n_vehicles, -1),
            # Add step dimension
            i=torch.zeros(1, dtype=torch.int64, device=loc.device)  # Vector with length num_steps
        )

    def get_socs(self):
        return self.soc

    def get_final_cost(self):

        assert self.all_finished()

        return self.lengths + (self.coords[self.ids, 0, :] - self.cur_coord).norm(p=2, dim=-1)

    def all_finished(self):
        result = torch.all(self.finished == 1)
        return result
        return self.i.item() >= self.demand.size(-1) and self.visited.all()

    def get_finished(self):
        return self.visited.sum(-1) == self.visited.size(-1)

    def get_current_node(self):
        return self.prev_a

    def get_p2v_mask(self):
        demands = self.demand[:, None, :].expand(self.demand.size(0), self.n_vehicles, self.demand.size(1))
        exceeds_cap = demands + (self.used_capacity[:, :, None].expand_as(demands))
        b = torch.tensor(self.vehicle_capacity, device=self.demand.device)[None, :, None].expand(self.demand.size(0),
                                                                                                 self.n_vehicles,
                                                                                                 self.demand.size(1))
        res = (exceeds_cap > b)
        return res

    def update_state_bs(self, used_cap):
        self.used_capacity.data = used_cap.data
        return self._replace(
            used_capacity=self.used_capacity
        )

    def assign_p2v(self, veh, p_demand):
        # used_cap = self.used_capacity[veh] + p_demand
        self.used_capacity.scatter_add_(1, veh, p_demand)
        return self._replace(
            used_capacity=self.used_capacity
        )

    def make_cap_zero(self, selected_vehicle, mask):
        used_cap = self.used_capacity.clone()
        used_cap[torch.arange(used_cap.size(0)), selected_vehicle] = 0
        self.used_capacity[mask, :] = used_cap[mask, :]
        return self._replace(
            used_capacity=self.used_capacity
        )

    # def assign_p2v(self, veh, p_demand):
    #     # used_cap = self.used_capacity[veh] + p_demand
    #     self.used_capacity.scatter_add_(1, veh, p_demand)
    #     return self._replace(
    #         used_capacity=self.used_capacity
    #     )
    def construct_solutions(self, actions):
        return actions
