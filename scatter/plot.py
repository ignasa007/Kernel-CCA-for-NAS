import os
from argparse import ArgumentParser
import pickle

import matplotlib.pyplot as plt


parser = ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
parser.add_argument('--use_kcca', action='store_true')
args = parser.parse_args()

if not args.use_kcca:
    save_dir = f'./assets/{args.dataset}/condition_number'
else:
    save_dir = f'./assets/{args.dataset}/kernel_cca'
    
### LOAD MEASURES

with open(f'{save_dir}/eigen_measures.pickle', 'rb') as f:
    eigen_measures = pickle.load(f)
with open(f'{save_dir}/train_losses.pickle', 'rb') as f:
    losses = pickle.load(f)
with open(f'{save_dir}/train_accs.pickle', 'rb') as f:
    accs = pickle.load(f)

### PLOT MEASURES

fig, axs = plt.subplots(1, 2, figsize=(6.4*2, 4.8))

axs[0].scatter(eigen_measures, losses, color='blue')
axs[1].scatter(eigen_measures, accs, color='blue')

if args.dataset == 'cifar100' or not args.use_kcca:
    axs[0].set_xscale('log')
    axs[1].set_xscale('log')
if not args.use_kcca:
    axs[0].set_xlabel('Condition Number')
    axs[1].set_xlabel('Condition Number')
else:
    axs[0].set_xlabel('Canonical Correlation')
    axs[1].set_xlabel('Canonical Correlation')

axs[0].set_ylabel('Training Loss')
axs[1].set_ylabel('Training Accuracy')
axs[0].grid()
axs[1].grid()
os.makedirs(save_dir, exist_ok=True)
fig.tight_layout()
plt.savefig(f'{save_dir}/plot.png')