from utils.patch_ensemble import test_all_case_w
import torch.backends.cudnn as cudnn

import argparse
import os
import torch.backends.cudnn as cudnn

from dataloaders.dataset import *
from networks_mu_elbo.net_factory import net_factory, net_factory_hyperbolic


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='PA', help='dataset_name')
parser.add_argument('--root_path', type=str, default='./', help='Name of Dataset')
parser.add_argument('--model_path', type=str, default='./', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='Co_Manifold_CL', help='exp_name')
parser.add_argument('--model', type=str, default='vnet', help='model_name')
parser.add_argument('--detail', type=int, default=1, help='print metrics for every samples?')
parser.add_argument('--nms', type=int, default=0, help='apply NMS post-procssing?')

parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size of labeled data per gpu')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=6, help='trained samples')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--ce_w', type=float, default=0.85, help='weight to balance ce loss')
parser.add_argument('--dl_w', type=float, default=0.85, help='weight to balance dice loss')
parser.add_argument('--t_m', type=float, default=0.3, help='weight to balance mask loss')
parser.add_argument('--alpha', type=float, default=0.005, help='weight to balance alpha')
parser.add_argument('--beta', type=float, default=0.02, help='weight to balance beta')
parser.add_argument('--clip', type=float, default=12.0, help='max norm size for gradient clipping')

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

parser.add_argument('--w1', type=float, default=0.8, help='weight for M1')
parser.add_argument('--w2', type=float, default=0.2, help='weight for M2')

FLAGS = parser.parse_args()

torch.autograd.set_detect_anomaly(True)

if FLAGS.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(FLAGS.seed)
    torch.cuda.manual_seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

snapshot_path = FLAGS.model_path + "/model/{}_{}_{}_labeled_{}_ce_{}_dl_{}_tm_{}_bs_{}_alpha_{}_beta/{}".format(FLAGS.dataset_name, FLAGS.exp, FLAGS.labelnum, FLAGS.ce_w, FLAGS.dl_w, FLAGS.t_m, FLAGS.labeled_bs, FLAGS.alpha, FLAGS.beta, FLAGS.model)

test_save_path = FLAGS.model_path + "/model/{}_{}_{}_labeled_{}_ce_{}_dl_{}_tm_{}_bs_{}_alpha_{}_beta/{}/{}_predictions_ensemble_w1_{}_w2_{}/".format(FLAGS.dataset_name, FLAGS.exp, FLAGS.labelnum, FLAGS.ce_w, FLAGS.dl_w, FLAGS.t_m, FLAGS.labeled_bs, FLAGS.alpha, FLAGS.beta, FLAGS.model,  FLAGS.model, FLAGS.w1, FLAGS.w2)


num_classes = 2

patch_size = (96, 96, 96)
FLAGS.root_path = '../data/Pancreas'
#../data/Pancreas/Pancreas_h5
with open(FLAGS.root_path + '/test.list', 'r') as f:
    image_list = f.readlines()
image_list = ["../data/Pancreas/Pancreas_h5/" + item.replace('\n', '') + "_norm.h5" for item in image_list]

if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
print(test_save_path)


def calculate_metric():
    net_1 = net_factory_hyperbolic(FLAGS, net_type='vnet', in_chns=1, class_num=num_classes - 1, mode="test")
    save_mode_path_1 = os.path.join(snapshot_path, 'best_model_1.pth'.format(FLAGS.model))
    net_1.load_state_dict(torch.load(save_mode_path_1), strict=False)
    print("init weight from {}".format(save_mode_path_1))
    net_1.eval()

    net_2 = net_factory(FLAGS, net_type='vnet', in_chns=1, class_num=num_classes - 1, mode="test")
    save_mode_path_2 = os.path.join(snapshot_path, 'best_model_2.pth'.format(FLAGS.model))
    net_2.load_state_dict(torch.load(save_mode_path_2), strict=False)
    print("init weight from {}".format(save_mode_path_2))
    net_2.eval()

    avg_metric = test_all_case_w(FLAGS.model, 1, net_1, net_2, image_list, num_classes=num_classes,
                               patch_size=patch_size, stride_xy=8, stride_z=8,
                               save_result=True, test_save_path=test_save_path,
                               metric_detail=FLAGS.detail, nms=FLAGS.nms, w1=FLAGS.w1, w2=FLAGS.w2)

    return avg_metric


if __name__ == '__main__':
    metric = calculate_metric()
    print(metric)
