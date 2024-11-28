from torch.utils.data import Dataset
import torch
import os
import pickle

from problems.vrp.state_cvrp import StateCVRP
from problems.vrp.state_sdvrp import StateSDVRP
from utils.beam_search import beam_search


class CVRP(object):
    NAME = 'cvrp'  # Capacitated Vehicle Routing Problem

    VEHICLE_CAPACITY = 1.  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    @staticmethod
    def get_costs(dataset, state):
        batch_size, graph_size = dataset['demand'].size()
        costs = torch.zeros(batch_size, dtype=torch.float, device=dataset['loc'].device)
        # Check that tours are valid, i.e. contain 0 to n -1
        for batch in range(batch_size):
            for time in state.travel_times[batch]:
                costs[batch] += time
        return costs

    @staticmethod
    def make_dataset(*args, **kwargs):
        return VRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateCVRP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):
        assert model is not None, "Provide model"

        # fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = CVRP.make_state(
            input, n_vehicles=4,  visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


class SDVRP(object):
    NAME = 'sdvrp'  # Split Delivery Vehicle Routing Problem

    VEHICLE_CAPACITY = 1.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    @staticmethod
    def get_costs(dataset, pi):
        batch_size, graph_size = dataset['demand'].size()

        # Each node can be visited multiple times, but we always deliver as much demand as possible
        # We check that at the end all demand has been satisfied
        demands = torch.cat(
            (
                torch.full_like(dataset['demand'][:, :1], -SDVRP.VEHICLE_CAPACITY),
                dataset['demand']
            ),
            1
        )
        rng = torch.arange(batch_size, out=demands.data.new().long())
        used_cap = torch.zeros_like(dataset['demand'][:, 0])
        a_prev = None
        for a in pi.transpose(0, 1):
            assert a_prev is None or (demands[((a_prev == 0) & (a == 0)), :] == 0).all(), \
                "Cannot visit depot twice if any nonzero demand"
            d = torch.min(demands[rng, a], SDVRP.VEHICLE_CAPACITY - used_cap)
            demands[rng, a] -= d
            used_cap += d
            used_cap[a == 0] = 0
            a_prev = a
        assert (demands == 0).all(), "All demand must be satisfied"

        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

        # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
        return (
                       (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
                       + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
                       + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
               ), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return VRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateSDVRP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):
        assert model is not None, "Provide model"
        assert not compress_mask, "SDVRP does not support compression of the mask"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = SDVRP.make_state(input)

        return beam_search(state, beam_size, propose_expansions)


def make_instance(args):
    depot, loc, demand, capacity, *args = args
    grid_size = 1
    depot_tensor = torch.tensor(depot, dtype=torch.float) / grid_size
    if depot_tensor.ndim == 1:
        depot_tensor = depot_tensor[None,:].expand(2, depot_tensor.size(0))
    # if len(args) > 0:
    #     depot_types, customer_types, grid_size = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
        'demand': torch.tensor(demand, dtype=torch.float) / capacity,
        'depot': depot_tensor,
        # 'CSs': torch.tensor(css, dtype=torch.float),
        # 'vehicles': torch.tensor(vehicles, dtype=torch.float)
    }


class VRPDataset(Dataset):
    def generate_gamma_dist(self, dataset_size, vrp_size):
        shape = 4
        scale = 1.0
        gamma_dist = torch.distributions.Gamma(shape, scale)
        gamma_data_x = gamma_dist.sample((dataset_size, vrp_size,))
        my_min, _ = gamma_data_x.min(1)
        my_min = my_min[:, None]
        my_max, _ = gamma_data_x.max(1)
        my_max = my_max[:, None]

        gamma_data_x = (gamma_data_x - my_min) / (my_max - my_min)

        gamma_data_y = gamma_dist.sample((dataset_size, vrp_size,))
        my_min, _ = gamma_data_y.min(1)
        my_min = my_min[:, None]
        my_max, _ = gamma_data_y.max(1)
        my_max = my_max[:, None]
        gamma_data_y = (gamma_data_y - my_min) / (my_max - my_min)
        gamma_data = torch.stack((gamma_data_x, gamma_data_y), -1)

        return gamma_data

    def generate_beta_dist(self, dataset_size, vrp_size):
        alpha, beta = 10,10  # Adjust parameters for skewness towards center
        gamma_dist = torch.distributions.Beta(alpha, beta)
        gamma_data_x = gamma_dist.sample((dataset_size, vrp_size,))
        my_min, _ = gamma_data_x.min(1)
        my_min = my_min[:, None]
        my_max, _ = gamma_data_x.max(1)
        my_max = my_max[:, None]

        gamma_data_x = ((gamma_data_x - my_min) / (my_max - my_min))

        gamma_data_y = gamma_dist.sample((dataset_size, vrp_size,))
        my_min, _ = gamma_data_y.min(1)
        my_min = my_min[:, None]
        my_max, _ = gamma_data_y.max(1)
        my_max = my_max[:, None]
        gamma_data_y = (gamma_data_y - my_min) / (my_max - my_min)
        gamma_data = torch.stack((gamma_data_x, gamma_data_y), -1)
        return gamma_data

    def __init__(self, filename=None, size=50, depot_size = 2, num_samples=1000000, offset=0, distribution=None):
        super(VRPDataset, self).__init__()
        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = [make_instance(args) for args in data[offset:offset + num_samples]]

        else:

            # From VRP with RL paper https://arxiv.org/abs/1802.04240
            CAPACITIES = {
                5:10.,
                10: 20.,
                20: 30.,
                50: 40.,
                100: 50.,
                400: 150.,
                1000: 200.,
                2000: 300.,
                5000: 300.
            }
            # locs = self.generate_gamma_dist(num_samples, size)
            locs = self.generate_beta_dist(num_samples, size)

            self.data = [
                {
                    # 'loc': torch.FloatTensor(size, 2).uniform_(0, 1),
                    'loc': locs[i, :],
                    # Uniform 1 - 9, scaled by capacities
                    'demand': (torch.FloatTensor(size).uniform_(0, 9).int() + 1).float() / CAPACITIES[size],
                    # 'depot': torch.FloatTensor(depot_size, 2).uniform_(0, 1),
                    'depot': torch.FloatTensor([[0,1],[1,1]]),
                }
                for i in range(num_samples)
            ]

        self.size = len(self.data)


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
