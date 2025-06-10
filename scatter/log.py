from pathlib import Path
import sys
lib_dir = (Path(__file__).parent.parent / 'lib').resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

import os
from argparse import ArgumentParser
from easydict import EasyDict
import pickle

from torch.utils.data import DataLoader

from lib.nas_201_api import NASBench201API as API
from lib.datasets import get_datasets
from lib.models import get_cell_based_tiny_net
from lib.procedures import get_ntk_n


N_SAMPLES = 200
BATCH_SIZE = 72
NUM_BATCHES = 1

parser = ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
parser.add_argument('--use_kcca', action='store_true')
args = parser.parse_args()

api = API('./NAS-Bench-201-v1_0-e61699.pth')
data_paths = {
    "cifar10": "./cifar-10",
    "cifar100": "./cifar-100",
}
train_data, valid_data, xshape, class_num = get_datasets(args.dataset, data_paths[args.dataset], -1)
loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

eigen_measures, losses, accs = list(), list(), list()
while N_SAMPLES:
    print(f'{N_SAMPLES} remaining.')
    arch_index = api.random()
    print(f'Sampled architecture index {arch_index}.')
    config = EasyDict(api.get_net_config(arch_index, args.dataset))
    network = get_cell_based_tiny_net(config).cuda().train()
    try:
        eigen_measure, = get_ntk_n(loader, [network,], recalbn=0, train_mode=True, num_batch=NUM_BATCHES, use_kcca=args.use_kcca)
    except RuntimeError as e:
        print(e)
    else:
        eigen_measures.append(eigen_measure)
        metrics = api.query_by_index(arch_index).get_metrics(dataset=args.dataset, setname='train')
        losses.append(metrics['loss'])
        accs.append(metrics['accuracy'])
        N_SAMPLES -= 1
print('Done. Saving results.')

if not args.use_kcca:
    save_dir = f'./assets/{args.dataset}/condition_number'
else:
    save_dir = f'./assets/{args.dataset}/kernel_cca'
os.makedirs(save_dir, exist_ok=True)

### SAVE MEASURES

with open(f'{save_dir}/eigen_measures.pickle', 'wb') as f:
    pickle.dump(eigen_measures, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(f'{save_dir}/train_losses.pickle', 'wb') as f:
    pickle.dump(losses, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(f'{save_dir}/train_accs.pickle', 'wb') as f:
    pickle.dump(accs, f, protocol=pickle.HIGHEST_PROTOCOL)