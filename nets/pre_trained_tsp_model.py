import json
import torch
import os

mp = torch.multiprocessing.get_context('spawn')


class TSPModel:
    def __init__(self):
        self.model, self.args = self.load_model("tsp_solver/pre_trained/tsp_10")
        self.model.set_decode_type(
            "greedy" ,
            temp=1)

    def load_problem(self, name):
        from tsp_solver.problems import TSP
        problem = {
            'tsp': TSP,
        }.get(name, None)
        assert problem is not None, "Currently unsupported problem: {}!".format(name)
        return problem

    def torch_load_cpu(self, load_path):
        return torch.load(load_path, map_location=lambda storage, loc: storage)  # Load on CPU

    def _load_model_file(self, load_path, model):
        """Loads the model with parameters from the file and returns optimizer state dict if it is in the file"""

        # Load the model parameters from a saved state
        load_optimizer_state_dict = None
        print('  [*] Loading model from {}'.format(load_path))

        load_data = torch.load(
            os.path.join(
                os.getcwd(),
                load_path
            ), map_location=lambda storage, loc: storage)

        if isinstance(load_data, dict):
            load_optimizer_state_dict = load_data.get('optimizer', None)
            load_model_state_dict = load_data.get('model', load_data)
        else:
            load_model_state_dict = load_data.state_dict()

        state_dict = model.state_dict()

        state_dict.update(load_model_state_dict)

        model.load_state_dict(state_dict)

        return model, load_optimizer_state_dict

    def load_model(self, path, epoch=None):
        from tsp_solver.nets.attention_model import AttentionModel
        if os.path.isfile(path):
            model_filename = path
            path = os.path.dirname(model_filename)
        elif os.path.isdir(path):
            if epoch is None:
                epoch = max(
                    int(os.path.splitext(filename)[0].split("-")[1])
                    for filename in os.listdir(path)
                    if os.path.splitext(filename)[1] == '.pt'
                )
            model_filename = os.path.join(path, 'epoch-{}.pt'.format(epoch))
        else:
            assert False, "{} is not a valid directory or file".format(path)

        args = self.load_args(os.path.join(path, 'args.json'))

        problem = self.load_problem(args['problem'])

        model_class = {
            'attention': AttentionModel
            # 'pointer': PointerNetwork
        }.get(args.get('model', 'attention'), None)
        assert model_class is not None, "Unknown model: {}".format(model_class)

        model = model_class(
            args['embedding_dim'],
            args['hidden_dim'],
            problem,
            n_encode_layers=args['n_encode_layers'],
            mask_inner=True,
            mask_logits=True,
            normalization=args['normalization'],
            tanh_clipping=args['tanh_clipping'],
            checkpoint_encoder=args.get('checkpoint_encoder', False),
            shrink_size=args.get('shrink_size', None)
        )
        # Overwrite model parameters by parameters to load
        load_data = self.torch_load_cpu(model_filename)
        model.load_state_dict({**model.state_dict(), **load_data.get('model', {})})

        model, *_ = self._load_model_file(model_filename, model)

        device = torch.device("cuda:0" if True else "cpu")
        # model.to(device)
        model.eval()  # Put in eval mode

        return model, args

    def load_args(self, filename):
        with open(filename, 'r') as f:
            args = json.load(f)

        # Backwards compatibility
        if 'data_distribution' not in args:
            args['data_distribution'] = None
            probl, *dist = args['problem'].split("_")
            if probl == "op":
                args['problem'] = probl
                args['data_distribution'] = dist[0]
        return args

    def eval(self, input):
        cost, ll = self.model(input)
        return cost, ll
