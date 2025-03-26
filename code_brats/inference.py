import os
import argparse
import sys

import torch
from dataloaders.dataset import *
from networks_mu_elbo.net_factory import net_factory, net_factory_hyperbolic
from torch.utils.data import DataLoader
from utils.inference_patch_ensemble import test_all_case
from dataloaders.msd_brats import BraTS

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='MSD_BRATS', help='dataset_name')
parser.add_argument('--root_path', type=str, default='./', help='Name of Dataset') # /data/himashi/Co-Manifolds/data
parser.add_argument('--exp', type=str, default='Co_Manifold', help='exp_name')
parser.add_argument('--model', type=str, default='vnet', help='model_name')
parser.add_argument('--max_iteration', type=int, default=10000, help='maximum iteration to train')
parser.add_argument('--max_samples', type=int, default=62, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=3, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=6, help='batch_size of labeled data per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=77, help='trained samples')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--ce_w', type=float, default=1.0, help='weight to balance ce loss')
parser.add_argument('--dl_w', type=float, default=1.0, help='weight to balance dice loss')
parser.add_argument('--alpha', type=float, default=0.005, help='weight to balance alpha')
parser.add_argument('--beta', type=float, default=0.02, help='weight to balance beta')
parser.add_argument('--clip', type=float, default=1.0, help='max norm size for gradient clipping')
parser.add_argument('--t_m', type=float, default=0.2, help='mask threashold')

# Hyperbolic
parser.add_argument('--prior', type=str, default='WrappedNormal', help='prior distribution',
                    choices=['WrappedNormal', 'RiemannianNormal', 'Normal'])
parser.add_argument('--prior_euc', type=str, default='Normal', help='prior distribution',
                    choices=['WrappedNormal', 'RiemannianNormal', 'Normal'])
parser.add_argument('--posterior', type=str, default='WrappedNormal', help='posterior distribution',
                    choices=['WrappedNormal', 'RiemannianNormal', 'Normal'])
parser.add_argument('--posterior_euc', type=str, default='Normal', help='posterior distribution',
                    choices=['WrappedNormal', 'RiemannianNormal', 'Normal'])
parser.add_argument('--prior-std', type=float, default=1., help='scale stddev by this value (default:1.)')
parser.add_argument('--learn-prior-std', action='store_true', default=False)

parser.add_argument('--latent-dim', type=int, default=2, metavar='L', help='latent dimensionality (default: 10)')
parser.add_argument('--hidden-dim', type=int, default=256, help='number of hidden layers dimensions (default: 100)')
parser.add_argument('--c', type=float, default=1., help='curvature')
parser.add_argument('--manifold', type=str, default='PoincareBall', choices=['Euclidean', 'PoincareBall'])
parser.add_argument('--manifold_euc', type=str, default='Euclidean', choices=['Euclidean', 'PoincareBall'])


args = parser.parse_args()
snapshot_path = args.root_path + "/model/{}_{}_{}_labeled_{}_ce_{}_dl_{}_tm_{}_bs_{}_alpha_{}_beta/{}".format(args.dataset_name, args.exp, args.labelnum, args.ce_w, args.dl_w, args.t_m, args.labeled_bs, args.alpha, args.beta, args.model)
test_save_path = args.root_path + "/model/{}_{}_{}_labeled_{}_ce_{}_dl_{}_tm_{}_bs_{}_alpha_{}_beta/{}_predictions_ensemble/".format(args.dataset_name, args.exp, args.labelnum, args.ce_w, args.dl_w, args.t_m, args.labeled_bs, args.alpha, args.beta, args.model)


num_classes = 4
patch_size = (128, 128, 96)
args.root_path =  '/data/datasets/MSD_BRATS/Task01_BrainTumour/'
args.max_samples = 387
train_data_path = args.root_path

db_test = BraTS(base_dir=train_data_path, split='test', patch_size=patch_size)
testloader = DataLoader(db_test, batch_size=1, num_workers=2, pin_memory=True, shuffle=False)

if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)


def calculate_metric():
    net1 = net_factory_hyperbolic(args, net_type=args.model, in_chns=4, class_num=num_classes - 1, mode="test")
    save_mode_path = os.path.join(snapshot_path, 'best_model_1.pth')
    net1.load_state_dict(torch.load(save_mode_path), strict=False)
    print("init weight from {}".format(save_mode_path))
    net1.eval()

    net2 = net_factory(args, net_type=args.model, in_chns=4, class_num=num_classes - 1, mode="test")
    save_mode_path2 = os.path.join(snapshot_path, 'best_model_2.pth')
    net2.load_state_dict(torch.load(save_mode_path2), strict=False)
    print("init weight from {}".format(save_mode_path2))
    net2.eval()

    avg_metric = test_all_case(net1, net2, testloader, patch_size=patch_size, save_result=True, test_save_path=test_save_path)

    return avg_metric


if __name__ == '__main__':
    metric = calculate_metric()
    print(metric)
