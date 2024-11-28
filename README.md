# DeepMDV, Learning Global Matching for Multi-depot Vehicle Routing Problems


## Dependencies

* Python>=3.8
* NumPy
* SciPy
* [PyTorch](http://pytorch.org/)>=1.7
* tqdm
* [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger)

## Usage

### Generating data

Training data is generated on the fly. To generate validation and test data (same as used in the paper) for all problems:
```bash
python generate_data.py --problem vrp --name multidepot_vrp -f --seed 1234 --depot_size 3
```

### Evaluation
To evaluate a model, use `eval.py`, which will additionally measure timing and save the results:
```bash
python eval.py data/vrp/vrp100_multidepot3_seed1234.pkl --model pretrained/vrp_3_100/ --decode_strategy greedy  --eval_batch_size 1 -k 50  --tsp_solver AM
```
Using LKH instead of AM for argument --tsp_solver, will replace LKH with AM. If the LKH is not ready, it will first download the LKH and then run it. The trained models are placed in `pretrained` directory. Their name is `vrp_{depot_size}_{graph_size}`. `k` is a parameter defined in the paper and here it is the percentage of the `graph_size`. For example, by setting `k=40` when `graph_size=200`, the number of nearest neighbors selected in the algorithm to generate local context will be 80.  

#### Sampling
To report the best of 1000 sampled solutions, use
```bash
python eval.py data/vrp/vrp100_multidepot2_seed1234.pkl --model pretrained/vrp_2_100/ --decode_strategy sample --width 1000 --eval_batch_size 1 -k 50  --tsp_solver AM  
```

#### Setting the trained model for AM
To select a trained model for AM as tsp solver, you can check `nets/pre_trained_tsp_model.py`. For using TSP trained on 10 nodes put `tsp_solver/pre_trained/tsp_10` in the trained model input at top of the class. For using AM trained with new procedure proposed by the paper, please use `tsp_solver/pre_trained/tsp_new`.
### Training

For training MDVRP instances with 100 nodes and using rollout as REINFORCE baseline:
```bash
python run.py --graph_size 100 --depot_size 3 --baseline rollout  
```

#### Warm start
You can initialize a run using a pretrained model by using the `--load_path` option:
```bash
python run.py --graph_size 100 --depot_size 3 --load_path pretrained/vrp_100/epoch-99.pt
```

The `--load_path` option can also be used to load an earlier run, in which case also the optimizer state will be loaded:
```bash
python run.py --graph_size 100 --depot_size 3 --load_path 'outputs/vrp_100/vrp100_rollout_{datetime}/epoch-0.pt'
```

The `--resume` option can be used instead of the `--load_path` option, which will try to resume the run, e.g. load additionally the baseline state, set the current epoch/step counter and set the random number generator state.

#### Acknowledgements

We thank attention learning to route [https://github.com/wouterkool/attention-learn-to-route] for an easily extendable codebase.