import pickle

import torch
from torch import nn
import math
from typing import NamedTuple

from nets.graph_encoder_with_mask import GraphWithMaskEncoder
from nets.pre_trained_tsp_model import TSPModel
from utils.tensor_functions import compute_in_batches
from utils.lkh import lkh_solve
from nets.graph_encoder import GraphAttentionEncoder
from torch.nn import DataParallel
from utils.functions import sample_many


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)
        return AttentionModelFixed(
            node_embeddings=self.node_embeddings[key],
            context_node_projected=self.context_node_projected[key],
            glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
            glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
            logit_key=self.logit_key[key]
        )


class AttentionModelFixedP2V(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    vehicle_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)
        return AttentionModelFixedP2V(
            node_embeddings=self.node_embeddings[key],
            context_node_projected=self.context_node_projected[key],
            vehicle_node_projected=self.vehicle_node_projected[key],
            glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
            glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
            logit_key=self.logit_key[key]
        )


class AttentionModel(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None):
        super(AttentionModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0
        self.is_vrp = problem.NAME == 'cvrp'

        self.tanh_clipping = tanh_clipping

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size

        self.init_embed_depot = nn.Linear(2, embedding_dim)
        node_dim = 3
        self.init_embed = nn.Linear(node_dim, embedding_dim)
        self.init_embed_depot2 = nn.Linear(4, embedding_dim)
        self.inner_embed_groups_for_cap = nn.Linear(2 * embedding_dim + 1, embedding_dim, bias=False)
        self.inner_embed_groups = nn.Linear(node_dim, embedding_dim)
        self.inner_embed_nodes = nn.Linear(3, embedding_dim)

        # self.embedder = GraphAttentionEncoder(
        #     n_heads=n_heads,
        #     embed_dim=embedding_dim,
        #     n_layers=self.n_encode_layers,
        #     normalization=normalization
        # )

        self.embedder2 = GraphWithMaskEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )

        self.project_node_embeddings_p2v = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_vehicle_embedding = nn.Linear(2 * embedding_dim + 1, embedding_dim, bias=False)
        self.project_step_context_p2v = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_node_query_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_node_query_context2 = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_vehicle_key_value_context = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_depot_to_multi_depot = nn.Linear(embedding_dim, 4 * embedding_dim, bias=False)
        self.tsp_model = TSPModel()
        self.project_out_p2v = nn.Linear(embedding_dim, embedding_dim, bias=False)
        assert embedding_dim % n_heads == 0
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_out_node_to_veh = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.number_of_depots = 0
        self.k = 0
        self.tsp_solver = "AM"

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def set_k_value(self, k):
        self.k = k

    def set_tsp_solver(self, tsp_solver):
        self.tsp_solver = tsp_solver

    def forward(self, input, return_pi=False, lkh_enables=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """

        locs = torch.cat((input["depot"], input["loc"]), dim=1)
        self.number_of_depots = input["depot"].size(1)
        if self.number_of_depots == 2:
            locs1 = torch.cat((input["depot"][:, 0:1, :], input["loc"]), dim=1)
            locs2 = torch.cat((input["depot"][:, 1:2, :], input["loc"]), dim=1)

            distances1 = self.euclidean_distance3(locs1)
            distances2 = self.euclidean_distance3(locs2)
            distances = torch.cat((distances1[:, None, :, :], distances2[:, None, :, :]), 1)
            all_demands = torch.cat(
                (torch.zeros(input["loc"].size(0), self.number_of_depots, dtype=torch.bool, device=input['loc'].device),
                 input['demand']), -1)
            x = self.gen_data(locs, all_demands, 5)
            init_embed2 = self._init_embed2(x)
            embeddings = self.embedder2(init_embed2, None, None)
            depot1 = embeddings[:, 0, :]
            depot2 = embeddings[:, 1, :]
            embeddings = embeddings[:, 1:, :]
            init_depot_embed = torch.stack([depot1, depot2], 1)

        if self.number_of_depots == 3:
            locs1 = torch.cat((input["depot"][:, 0:1, :], input["loc"]), dim=1)
            locs2 = torch.cat((input["depot"][:, 1:2, :], input["loc"]), dim=1)
            locs3 = torch.cat((input["depot"][:, 2:3, :], input["loc"]), dim=1)

            distances1 = self.euclidean_distance3(locs1)
            distances2 = self.euclidean_distance3(locs2)
            distances3 = self.euclidean_distance3(locs3)
            distances = torch.cat((distances1[:, None, :, :], distances2[:, None, :, :], distances3[:, None, :, :]), 1)
            all_demands = torch.cat(
                (torch.zeros(input["loc"].size(0), self.number_of_depots, dtype=torch.bool, device=input['loc'].device),
                 input['demand']), -1)
            x = self.gen_data(locs, all_demands, 5)
            init_embed2 = self._init_embed2(x)
            embeddings = self.embedder2(init_embed2, None, None)
            depot1 = embeddings[:, 0, :]
            depot2 = embeddings[:, 1, :]
            depot3 = embeddings[:, 2, :]
            embeddings = embeddings[:, 2:, :]
            init_depot_embed = torch.stack([depot1, depot2, depot3], 1)

        if self.number_of_depots == 4:
            locs1 = torch.cat((input["depot"][:, 0:1, :], input["loc"]), dim=1)
            locs2 = torch.cat((input["depot"][:, 1:2, :], input["loc"]), dim=1)
            locs3 = torch.cat((input["depot"][:, 2:3, :], input["loc"]), dim=1)
            locs4 = torch.cat((input["depot"][:, 3:4, :], input["loc"]), dim=1)

            distances1 = self.euclidean_distance3(locs1)
            distances2 = self.euclidean_distance3(locs2)
            distances3 = self.euclidean_distance3(locs3)
            distances4 = self.euclidean_distance3(locs4)
            distances = torch.cat((distances1[:, None, :, :], distances2[:, None, :, :], distances3[:, None, :, :],
                                   distances4[:, None, :, :]), 1)

            all_demands = torch.cat(
                (torch.zeros(input["loc"].size(0), self.number_of_depots, dtype=torch.bool, device=input['loc'].device),
                 input['demand']), -1)

            x = self.gen_data(locs, all_demands, 5)
            init_embed2 = self._init_embed2(x)
            embeddings = self.embedder2(init_embed2, None, None)
            depot1 = embeddings[:, 0, :]
            depot2 = embeddings[:, 1, :]
            depot3 = embeddings[:, 2, :]
            depot4 = embeddings[:, 3, :]
            embeddings = embeddings[:, 3:, :]
            init_depot_embed = torch.stack([depot1, depot2, depot3, depot4], 1)

        state, seq2, vehicle_assigned_nodes, vehicle_index, seq, out_log_3, vehicle_number = self.decoder(
            input, embeddings, init_depot_embed, distances)

        with torch.no_grad():
            new_locs = list()
            new_mask = list()
            for i in range(self.number_of_depots):
                locs = torch.cat((input["depot"][:, i:i + 1, :], input["loc"]), dim=1)
                locs = locs[:, None, :, :].expand(locs.size(0), vehicle_number + 10, locs.size(1), locs.size(2))
                max_assigned_number = torch.max(vehicle_index.view(-1, 1))
                current_assigned = vehicle_assigned_nodes[:, i, :, 0:max_assigned_number]
                ssss = current_assigned.unsqueeze(-1).expand(current_assigned.size(0), current_assigned.size(1),
                                                             current_assigned.size(2), 2)
                new_locs.append(torch.gather(locs, 2, ssss))
                new_mask.append(current_assigned == 0)
            if self.number_of_depots == 2:
                new_locs2 = torch.stack((new_locs[0], new_locs[1]), 2)
                new_locs2 = new_locs2.reshape(new_locs2.size(0), new_locs2.size(1) * new_locs2.size(2),
                                              new_locs2.size(3), new_locs2.size(4))
                new_locs2 = new_locs2.reshape(new_locs2.size(0) * new_locs2.size(1), new_locs2.size(2),
                                              new_locs2.size(3))

            if self.number_of_depots == 3:
                new_locs2 = torch.stack((new_locs[0], new_locs[1], new_locs[2]), 2)
                new_locs2 = new_locs2.reshape(new_locs2.size(0), new_locs2.size(1) * new_locs2.size(2),
                                              new_locs2.size(3), new_locs2.size(4))
                new_locs2 = new_locs2.reshape(new_locs2.size(0) * new_locs2.size(1), new_locs2.size(2),
                                              new_locs2.size(3))

            if self.number_of_depots == 4:
                new_locs2 = torch.stack((new_locs[0], new_locs[1], new_locs[2], new_locs[3]), 2)
                new_locs2 = new_locs2.reshape(new_locs2.size(0), new_locs2.size(1) * new_locs2.size(2),
                                              new_locs2.size(3), new_locs2.size(4))
                new_locs2 = new_locs2.reshape(new_locs2.size(0) * new_locs2.size(1), new_locs2.size(2),
                                              new_locs2.size(3))

        if self.tsp_solver == "AM":
            final_cost, _ = self.tsp_model.eval(new_locs2)
            final_cost = final_cost.reshape(input["loc"].size(0), -1)
            final_cost = torch.sum(final_cost, -1)
        elif self.tsp_solver == "LKH":
            all_batch_groups = list()
            for co in range(new_locs2.size(0)):
                if not torch.equal(new_locs2[co][0], new_locs2[co][1]):
                    all_batch_groups.append(new_locs2[co])
            opts = {

            }
            final_cost, duration = self.cvrp_lkh_eval(all_batch_groups, opts)
        else:
            raise "TSP solver has not been set"

        out4 = out_log_3.sum(1)
        return final_cost, out4, None

    def _inner2(self, input, embeddings):
        my_final_costs, out4, _ = self.forward(input)
        return my_final_costs, out4

    def cvrp_lkh_eval(self, dataset, opts):
        import time
        start = time.time()
        sum_time = time.time() - start
        costs, duration = lkh_solve(opts, dataset, 0)
        my_cost = torch.tensor(costs).sum()
        sum_time += duration
        print('Total duration:', sum_time)
        return my_cost, sum_time

    def sum_cost(self, costs, n_tsps_per_route):
        assert len(costs) == n_tsps_per_route
        if not isinstance(costs, torch.Tensor):
            costs = torch.tensor(costs)
        ret = []
        start = 0
        for n in n_tsps_per_route:
            ret.append(costs[start: start + n].sum())
            start += n
        return torch.stack(ret)

    def gen_data(self, coors, demand, k_sparse, cvrplib=False):
        shift_coors = coors - coors[:, 0:1, :]
        _x, _y = shift_coors[:, :, 0], shift_coors[:, :, 1]
        r = torch.sqrt(_x ** 2 + _y ** 2)
        theta = torch.atan2(_y, _x)
        x = torch.stack((r, theta, demand), -1)
        return x

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):
        n_head = 8
        batch_size, tour_count, embed_dim = query.size()
        key_size = val_size = embed_dim // n_head
        glimpse_Q = query.view(batch_size, tour_count, n_head, 1, key_size).permute(2, 0, 1, 3, 4)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))

        # self.compatibility = compatibility.clone()

        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            # compatibility[my_assigned_mask[None, :, None, None, :].clone().expand_as(compatibility)] = -math.inf
            compatibility[mask[None, :, None, None, :].clone().expand_as(compatibility)] = -math.inf

        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)
        glimpse = self.project_out_p2v(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, tour_count, 1, n_head * val_size))

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        final_Q = glimpse
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        # if self.tanh_clipping > 0:
        #    logits = torch.tanh(logits) * self.tanh_clipping

        return logits

    def _init_embed(self, input):

        if self.is_vrp or self.is_orienteering or self.is_pctsp:
            if self.is_vrp:
                features = ('demand',)
            elif self.is_orienteering:
                features = ('prize',)
            else:
                assert self.is_pctsp
                features = ('deterministic_prize', 'penalty')
            return torch.cat(
                (
                    self.init_embed_depot(input['depot']),
                    self.init_embed(torch.cat((
                        input['loc'],
                        *(input[feat][:, :, None] for feat in features)
                    ), -1))
                ),
                1
            )
        # TSP
        return self.init_embed(input)

    def _init_embed2(self, input):
        return self.init_embed(input)

    def _init_embed_depot2(self, input):
        return self.init_embed_depot2(input)

    def sample_many(self, input, batch_rep=1, iter_rep=1):
        """
        :param input: (batch_size, graph_size, node_dim) input node features
        :return:
        """
        # Bit ugly but we need to pass the embeddings as well.
        # Making a tuple will not work with the problem.get_cost function
        return sample_many(
            lambda input: self._inner2(*input),  # Need to unpack tuple into arguments
            lambda input, pi: self.problem.get_costs(input[0], pi),  # Don't need embeddings as input to get_costs
            (input, self._init_embed(input)),  # Pack input with embeddings (additional input)
            batch_rep, iter_rep
        )

    def _select_node(self, probs, mask):

        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(
                -1)).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)

            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected

    def _precompute(self, embeddings, num_steps=1):

        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def decoder(self, input, embeddings, init_depot_embed, distances):

        batch_size = embeddings.size(0)

        vehicle_to_group_map = torch.zeros(batch_size, self.number_of_depots, dtype=torch.int64,
                                           device=input['loc'].device)
        if self.number_of_depots == 2:
            vehicle_to_group_map[:, 1] = 1

        if self.number_of_depots == 3:
            vehicle_to_group_map[:, 1] = 1
            vehicle_to_group_map[:, 2] = 2

        if self.number_of_depots == 4:
            vehicle_to_group_map[:, 1] = 1
            vehicle_to_group_map[:, 2] = 2
            vehicle_to_group_map[:, 3] = 3

        max_vehicle_count = torch.ones(batch_size, dtype=torch.int64, device=input['loc'].device) * (
                    self.number_of_depots - 1)

        customer_num = input['demand'].size(1)
        all_demands = torch.cat((torch.zeros(batch_size, 1, dtype=torch.bool, device=input['loc'].device),
                                 input['demand']), -1)
        accumulated_demand = torch.sum(all_demands, -1)
        vehicle_req_number = (torch.ceil(torch.sum(all_demands, -1))).to(torch.int32) + 1
        vehicle_number = int(torch.max(torch.ceil(accumulated_demand), -1)[0]) + self.number_of_depots
        active_group_mask = ~(torch.arange(vehicle_number, device=embeddings.device)[None, :].expand(embeddings.size(0),
                                                                                                     vehicle_number) < vehicle_req_number.unsqueeze(
            -1))
        total_demand = torch.sum(all_demands, -1)
        assigned_demand = torch.zeros_like(total_demand, device=input['loc'].device)
        outputs3 = []
        sequences = []
        sequences2 = []
        opened_group = torch.zeros(batch_size, self.number_of_depots, dtype=torch.bool, device=input['loc'].device)
        can_open_new = torch.ones(batch_size, dtype=torch.bool, device=input['loc'].device)
        total_number_of_groups = (torch.ceil(total_demand)).to(torch.int32) + self.number_of_depots
        total_opened_groups = torch.zeros(batch_size, dtype=torch.int64, device=input['loc'].device)
        depot_node_mask = torch.ones(batch_size, self.number_of_depots, 1, dtype=torch.bool, device=input['loc'].device)
        assigned_mask = torch.zeros(batch_size, customer_num + 1, dtype=torch.bool, device=input['loc'].device)
        assigned_node_list = torch.zeros(batch_size, customer_num + 1, dtype=torch.bool, device=input['loc'].device)
        wasted_cap = torch.zeros(batch_size, dtype=all_demands.dtype, device=input['loc'].device)
        closed_group_count = torch.zeros(batch_size, dtype=all_demands.dtype, device=input['loc'].device)
        vehicles_parcels2 = torch.zeros(batch_size, self.number_of_depots, customer_num + vehicle_number + 1,
                                        dtype=torch.int64,
                                        device=input['loc'].device)
        vehicle_assigned_nodes = torch.zeros(batch_size, self.number_of_depots, vehicle_number + self.number_of_depots,
                                             customer_num + self.number_of_depots + 1, dtype=torch.int64,
                                             device=input['loc'].device)
        vehicle_index = torch.ones(batch_size, self.number_of_depots, vehicle_number + self.number_of_depots, 1,
                                   dtype=torch.int64,
                                   device=input['loc'].device)
        state = self.problem.make_state(input, self.number_of_depots)
        node_embeddings = embeddings[:, 1:, :]
        node_embeddings = node_embeddings[:, None, :, :].expand(node_embeddings.size(0), init_depot_embed.size(1),
                                                                node_embeddings.size(1), node_embeddings.size(-1))
        total_embeddings = torch.cat((init_depot_embed[:, :, None, :], node_embeddings), 2)

        zero_tensor = torch.zeros_like(depot_node_mask, dtype=torch.bool)
        batch_indices = torch.arange(batch_size)
        i = -1
        while not torch.all(assigned_mask[:, 1:].all(dim=1)):
            i += 1
            last_nodes_for_all_groups = vehicles_parcels2[batch_indices, :, i]
            vehicle_embed = torch.gather(total_embeddings, 2,
                                         last_nodes_for_all_groups.clone()[:, :, None, None].contiguous().expand(
                                             batch_size, last_nodes_for_all_groups.size(1),
                                             1, embeddings.size(-1)))[:, :, 0, :]
            vehicle_embed_with_cap = torch.cat(
                (init_depot_embed.clone(), vehicle_embed, state.used_capacity[..., None]), -1)

            vehicle_path_last_embed = self.project_vehicle_embedding(vehicle_embed_with_cap)

            all_masked = torch.all(active_group_mask, dim=-1)
            active_group_mask2 = active_group_mask.clone()
            active_group_mask2[:, 0][all_masked] = False
            active_group_mask = active_group_mask2.clone()

            last_nodes_for_all_groups = vehicles_parcels2[batch_indices, :, i]
            assigned_mask[:, 0] = False
            if assigned_node_list is not None:
                graph_embed = total_embeddings.clone()
                mask_nan = assigned_node_list[:, None, :, None].expand_as(graph_embed)
                graph_embed[mask_nan] = torch.nan
                graph_embed = torch.nanmean(graph_embed, dim=2)
                graph_embed = self.project_step_context_p2v(graph_embed)

            log_p, mask, log_p_car, selected_vehicle = self._get_log_p(total_embeddings, graph_embed,
                                                                       init_depot_embed,
                                                                       state,
                                                                       vehicle_path_last_embed, vehicle_embed,
                                                                       assigned_mask, depot_node_mask,
                                                                       distances, total_demand,
                                                                       active_group_mask, zero_tensor,
                                                                       last_nodes_for_all_groups,
                                                                       batch_indices, wasted_cap,
                                                                       closed_group_count, opened_group,
                                                                       can_open_new, total_number_of_groups)

            is_new_group = depot_node_mask[batch_indices, selected_vehicle, 0]
            total_opened_groups2 = total_opened_groups.clone()
            total_opened_groups2 = total_opened_groups2 + 1
            total_opened_groups[is_new_group] = total_opened_groups2[is_new_group]

            if_group_number_is_less = total_opened_groups < total_number_of_groups
            can_open_new = if_group_number_is_less.clone()

            opened_group[batch_indices, selected_vehicle] = True

            depot_node_mask[batch_indices, selected_vehicle, 0] = False
            selected_parcel_number = self._select_node_p2v(log_p.exp()[:, 0, :], mask)
            demands = all_demands[batch_indices, selected_parcel_number].unsqueeze(-1)
            assigned_demand = assigned_demand + demands.squeeze(-1)
            state.assign_p2v(selected_vehicle[:, None], demands)
            # outputs.append(log_p[:, 0, :])
            sequences.append(selected_parcel_number)
            new_logits = log_p[batch_indices, 0, selected_parcel_number]
            # outputs2.append(log_p_car)
            sequences2.append(selected_vehicle)
            log_vehicle = log_p_car[batch_indices, selected_vehicle]
            outputs3.append(new_logits + log_vehicle)

            vehicles_parcels2[batch_indices, :, i + 1] = vehicles_parcels2[batch_indices, :, i]
            vehicles_parcels2[batch_indices, selected_vehicle, i + 1] = selected_parcel_number

            assigned_node_list1 = assigned_node_list.clone()
            assigned_node_list1[batch_indices, selected_parcel_number] = True
            assigned_node_list1[batch_indices, 0] = False
            assigned_node_list = assigned_node_list1.clone()

            assigned_mask[batch_indices, selected_parcel_number] = True
            mask2 = (selected_parcel_number == 0)
            total_remained_cap = 1 - state.used_capacity[batch_indices, selected_vehicle]
            if_is_near_zero = total_remained_cap < 1e-6
            total_remained_cap[if_is_near_zero] = 0
            wasted_cap2 = wasted_cap.clone()
            wasted_cap2 = wasted_cap2 + total_remained_cap
            wasted_cap[mask2] = wasted_cap2[mask2]

            closed_group_count2 = closed_group_count.clone()
            closed_group_count2 = closed_group_count2 + 1
            closed_group_count[mask2] = closed_group_count2[mask2]

            state.make_cap_zero(selected_vehicle, mask2)

            depot_node_mask2 = depot_node_mask.clone()
            depot_node_mask2[batch_indices, selected_vehicle, 0] = True
            depot_node_mask[mask2] = depot_node_mask2[mask2]

            opened_group2 = opened_group.clone()
            opened_group2[batch_indices, selected_vehicle] = False
            opened_group[mask2] = opened_group2[mask2]

            current_group_index = vehicle_to_group_map[batch_indices, selected_vehicle]
            current_index = vehicle_index[batch_indices, selected_vehicle, current_group_index].squeeze()
            vehicle_assigned_nodes[
                batch_indices, selected_vehicle, current_group_index, current_index] = selected_parcel_number
            vehicle_index[batch_indices, selected_vehicle, current_group_index] = vehicle_index[
                                                                                      batch_indices, selected_vehicle, current_group_index] + 1

            max_vehicle_count2 = max_vehicle_count.clone()
            max_vehicle_count2 = max_vehicle_count2 + 1
            # if_less = max_vehicle_count2 < overall_number_of_vehicles
            # max_vehicle_count[mask2&if_less] = max_vehicle_count2[mask2&if_less]
            max_vehicle_count[mask2] = max_vehicle_count2[mask2]
            vehicle_to_group_map2 = vehicle_to_group_map.clone()
            vehicle_to_group_map2[batch_indices, selected_vehicle] = max_vehicle_count
            vehicle_to_group_map[mask2, :] = vehicle_to_group_map2[mask2, :]

        return state, torch.stack(sequences, 1), vehicle_assigned_nodes, vehicle_index, \
               torch.stack(sequences2, 1), torch.stack(outputs3, 1), vehicle_number

    def euclidean_distance3(self, all_locs):
        ss = all_locs.unsqueeze(2)
        ss2 = all_locs.unsqueeze(1)
        distance = torch.norm(ss2 - ss, dim=-1)
        return distance

    def _get_top_k_embeddings_and_nodes4(self, embeddings, distances, assigned_nodes, k,
                                         last_nodes_for_all_groups, group_used_cap, init_depot_embed):
        if self.number_of_depots == 2:
            current_distances1 = torch.gather(distances[:, 0, :, :], 1,
                                              last_nodes_for_all_groups[:, 0:1, None].clone().expand(
                                                  last_nodes_for_all_groups.size(0), 1, distances.size(-1)))
            current_distances2 = torch.gather(distances[:, 1, :, :], 1,
                                              last_nodes_for_all_groups[:, 1:2, None].clone().expand(
                                                  last_nodes_for_all_groups.size(0), 1, distances.size(-1)))
            current_distances = torch.cat(
                (current_distances1, current_distances2), 1)

        elif self.number_of_depots == 3:
            current_distances1 = torch.gather(distances[:, 0, :, :], 1,
                                              last_nodes_for_all_groups[:, 0:1, None].clone().expand(
                                                  last_nodes_for_all_groups.size(0), 1, distances.size(-1)))
            current_distances2 = torch.gather(distances[:, 1, :, :], 1,
                                              last_nodes_for_all_groups[:, 1:2, None].clone().expand(
                                                  last_nodes_for_all_groups.size(0), 1, distances.size(-1)))
            current_distances3 = torch.gather(distances[:, 2, :, :], 1,
                                              last_nodes_for_all_groups[:, 2:3, None].clone().expand(
                                                  last_nodes_for_all_groups.size(0), 1, distances.size(-1)))
            current_distances = torch.cat(
                (current_distances1, current_distances2, current_distances3), 1)

        elif self.number_of_depots == 4:
            current_distances1 = torch.gather(distances[:, 0, :, :], 1,
                                              last_nodes_for_all_groups[:, 0:1, None].clone().expand(
                                                  last_nodes_for_all_groups.size(0), 1, distances.size(-1)))

            current_distances2 = torch.gather(distances[:, 1, :, :], 1,
                                              last_nodes_for_all_groups[:, 1:2, None].clone().expand(
                                                  last_nodes_for_all_groups.size(0), 1, distances.size(-1)))
            current_distances3 = torch.gather(distances[:, 2, :, :], 1,
                                              last_nodes_for_all_groups[:, 2:3, None].clone().expand(
                                                  last_nodes_for_all_groups.size(0), 1, distances.size(-1)))
            current_distances4 = torch.gather(distances[:, 3, :, :], 1,
                                              last_nodes_for_all_groups[:, 3:4, None].clone().expand(
                                                  last_nodes_for_all_groups.size(0), 1, distances.size(-1)))
            current_distances = torch.cat(
                (current_distances1, current_distances2, current_distances3, current_distances4), 1)

        if_dis_is_zero = current_distances == 0
        current_distances[if_dis_is_zero] = 3
        new_assigned_distances = assigned_nodes[:, None, :].clone().expand_as(current_distances)
        current_distances[new_assigned_distances] = current_distances[new_assigned_distances] + 2
        current_distances2 = current_distances.permute(0, 2, 1)
        min_current_distance = torch.min(current_distances2, dim=-1)[0]
        min_current_distance = -1 * min_current_distance
        top_values, top_indices = torch.topk(min_current_distance, k=k, dim=-1)

        new_coors = torch.gather(embeddings, 2,
                                 top_indices[:, None, :, None].contiguous().expand(top_indices.size(0),
                                                                                   embeddings.size(1),
                                                                                   top_indices.size(1),
                                                                                   embeddings.size(-1)))
        group_coors = torch.gather(embeddings, 2,
                                   last_nodes_for_all_groups[:, :, None, None].contiguous().expand(
                                       last_nodes_for_all_groups.size(0),
                                       last_nodes_for_all_groups.size(1), 1, embeddings.size(-1)))

        x2 = torch.cat((init_depot_embed, group_coors[:, :, 0, :], group_used_cap.unsqueeze(-1)), -1)
        x2_embed = self.inner_embed_groups_for_cap(x2)
        return top_indices, x2_embed, new_coors

    def get_local_log(self, total_embeddings, state, assigned_mask=None,
                      distances=None, zero_tensor=None,
                      last_nodes_for_all_groups=None, init_depot_embed=None, opened_group=None, can_open_new=None):

        k_num = self.k
        top_indices, group_encodings, node_encodings = self._get_top_k_embeddings_and_nodes4(total_embeddings,
                                                                                             distances, assigned_mask,
                                                                                             k_num,
                                                                                             last_nodes_for_all_groups,
                                                                                             state.used_capacity,
                                                                                             init_depot_embed)
        assigned_node_mask = torch.gather(assigned_mask, 1, top_indices.clone()
                                          .expand(top_indices.size(0), top_indices.size(-1))
                                          ).view(top_indices.size(0), top_indices.size(-1))

        q1 = self.project_node_query_context(node_encodings)
        k1, v1, logit_k = self.project_vehicle_key_value_context(group_encodings[:, None, :, :]).chunk(3, dim=-1)
        batch_size, num_steps, n_dim, embed_dim = q1.size()
        key_size = val_size = embed_dim // self.n_heads
        glimpse_Q1 = q1.view(batch_size, num_steps, n_dim, self.n_heads, key_size).permute(3, 0, 1, 2, 4)
        glimpse_key1 = self._make_heads_p2v(k1, num_steps)
        glimpse_val1 = self._make_heads_p2v(v1, num_steps)

        capacity_mask1 = state.get_p2v_mask()
        capacity_mask1 = torch.cat((zero_tensor, capacity_mask1), -1).permute(0, 2, 1)
        capacity_mask1 = capacity_mask1 | assigned_mask[..., None].clone().expand_as(capacity_mask1)
        all_masked = torch.all(capacity_mask1, dim=-1)
        capacity_mask2 = capacity_mask1.clone()
        capacity_mask2[:, :, 0][all_masked] = False
        can_open_mask = can_open_new.clone()[..., None].expand_as(opened_group)
        can_open_or_already_active = ~opened_group
        can_open_or_already_active[can_open_mask] = False
        capacity_mask3 = torch.gather(capacity_mask2, 1, (top_indices.clone())[..., None].contiguous()
                                      .expand(top_indices.size(0), top_indices.size(-1), capacity_mask2.size(-1))
                                      ).view(top_indices.size(0), top_indices.size(-1), capacity_mask2.size(-1))

        capacity_mask4 = capacity_mask3 & can_open_or_already_active[:, None, :].expand_as(capacity_mask3)
        compatibility = torch.matmul(glimpse_Q1, glimpse_key1.transpose(-2, -1)) / math.sqrt(glimpse_Q1.size(-1))

        if capacity_mask4 is not None:
            compatibility[capacity_mask4[None, :, None, :, :].clone().expand_as(compatibility)] = -math.inf
        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        ss = torch.softmax(compatibility, dim=-1)
        heads = torch.matmul(ss, glimpse_val1)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        final_Q = self.project_out_node_to_veh(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, n_dim, self.n_heads * val_size))

        logits = torch.matmul(final_Q, logit_k.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))
        if self.number_of_depots == 2:
            l1 = logits[:, 0:1, :, 0]
            l2 = logits[:, 1:2, :, 1]
            car_logits = torch.cat((l1, l2), 1)
        elif self.number_of_depots == 3:
            l1 = logits[:, 0:1, :, 0]
            l2 = logits[:, 1:2, :, 1]
            l3 = logits[:, 2:3, :, 2]
            car_logits = torch.cat((l1, l2, l3), 1)
        elif self.number_of_depots == 4:
            l1 = logits[:, 0:1, :, 0]
            l2 = logits[:, 1:2, :, 1]
            l3 = logits[:, 2:3, :, 2]
            l4 = logits[:, 3:4, :, 3]
            car_logits = torch.cat((l1, l2, l3, l4), 1)

        car_logits[capacity_mask3.permute(0, 2, 1)] = -math.inf
        assigned_node_mask_per_car = assigned_node_mask[:, None, :].expand_as(car_logits)
        car_logits2 = car_logits.clone()
        car_logits2[assigned_node_mask_per_car] = -10
        car_logits = car_logits2.clone()

        final_car_logits = torch.max(car_logits, -1)[0]
        if self.tanh_clipping > 0:
            final_car_logits = torch.tanh(final_car_logits) * self.tanh_clipping

        if can_open_or_already_active is not None:
            active_groups2 = can_open_or_already_active.clone()
            all_masked = torch.all(active_groups2, dim=-1)
            active_groups2[:, 0][all_masked] = False
            final_car_logits[active_groups2.clone()] = -math.inf

        final_log = torch.log_softmax(final_car_logits / self.temp, dim=-1)
        assert not torch.isnan(final_log).any()
        selected_vehicle = self._select_node_p2v_no_mask(final_log.exp())

        all_context = torch.zeros_like(total_embeddings[:, 0, :, :])
        rr = final_Q[torch.arange(final_Q.size(0)), selected_vehicle, :, :]
        all_context.scatter_(1, top_indices[..., None].expand_as(rr), rr)
        return all_context, final_log, selected_vehicle, top_indices

    def _get_log_p(self, total_embeddings, graph_embed, init_depot_embed, state, tour_embed, last_nodes,
                   assigned_mask=None,
                   depot_node_mask=None, distances=None, total_demand=None,
                   active_groups=None,
                   zero_tensor=None, last_nodes_for_all_groups=None,
                   batch_indices=None, wasted_cap=None,
                   closed_group_count=None, opened_group=None, can_open_new=None, total_number_of_groups=None,
                   normalize=True):

        local_embedding, car_logits, selected_vehicle, top_indices = self.get_local_log(total_embeddings, state,
                                                                                        assigned_mask,
                                                                                        distances,
                                                                                        zero_tensor,
                                                                                        last_nodes_for_all_groups,
                                                                                        init_depot_embed,
                                                                                        opened_group, can_open_new)

        current_vehicle_embed = tour_embed[batch_indices, selected_vehicle]
        query = graph_embed[batch_indices, selected_vehicle, :][:, None, :] + current_vehicle_embed[:, None, :]
        embeddings = total_embeddings[batch_indices, selected_vehicle, :]
        new_embeddings = embeddings.clone() + local_embedding
        glimpse_key_fixed, glimpse_val_fixed, logit_K = self.project_node_embeddings_p2v(
            new_embeddings[:, None, :, :]).chunk(3, dim=-1)
        batch_size, num_steps, embed_dim = query.size()

        glimpse_K = self._make_heads_for_key_p2v(glimpse_key_fixed, num_steps)
        glimpse_V = self._make_heads_for_key_p2v(glimpse_val_fixed, num_steps)
        can_be_wasted = total_number_of_groups - total_demand
        not_wasted = can_be_wasted - wasted_cap
        remained_group_count = total_number_of_groups - closed_group_count
        threshold = not_wasted / remained_group_count
        current_capacity_status = 1 - state.used_capacity[batch_indices, selected_vehicle]
        dont_allow_visit_depot = threshold <= current_capacity_status
        depot_mask = depot_node_mask[batch_indices, selected_vehicle] | dont_allow_visit_depot[:, None]

        capacity_mask = state.get_p2v_mask()[batch_indices, selected_vehicle]
        capacity_mask = torch.cat((depot_mask, capacity_mask), -1)
        mask = capacity_mask | assigned_mask
        qqq = active_groups[batch_indices, selected_vehicle]
        mask[qqq[:, None].expand_as(mask)] = True
        all_masked = torch.all(mask, dim=-1)
        mask[:, 0][all_masked] = False
        final_log = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)

        if self.tanh_clipping > 0:
            final_log = torch.tanh(final_log) * self.tanh_clipping
        if self.mask_logits:
            final_log[mask[:, None, :]] = -math.inf

        if normalize:
            final_log = torch.log_softmax(final_log / self.temp, dim=-1)
        assert not torch.isnan(final_log).any()

        return final_log, mask, car_logits, selected_vehicle

    def _select_node_p2v(self, probs, mask):

        assert (probs == probs).all(), "Probs should not contain any nans"
        # entropy = self.entropy2(probs, distances, mask)

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(
                -1)).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)

            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)

        return selected

    def _select_node_p2v_no_mask(self, probs):

        assert (probs == probs).all(), "Probs should not contain any nans"
        # entropy = self.entropy2(probs, distances, mask)
        _, selected = probs.max(1)
        return selected
        if self.decode_type == "greedy":
            _, selected = probs.max(1)

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)
        return selected

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
                .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
                .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )

    def _make_heads_p2v(self, v, num_steps=None):
        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
                .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads,
                        -1)
                .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )

    def _make_heads_for_key_p2v(self, v, num_steps=None):
        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
                .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads,
                        -1)
                .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )
