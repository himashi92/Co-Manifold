import argparse
import logging
import os
import sys
import gc
import geoopt
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable

from networks_mu_elbo.critic import Discriminator
from dataloaders.dataset import *
from networks_mu_elbo.net_factory import net_factory, net_factory_hyperbolic
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from utils.visualize import save_plots_all
from utils import ramps, losses, test_patch
from utils.embedding_loss_new import discrepancy_loss_torch_center
from utils.losses import loss_hyp_uncertainty, gen_loss, disc_loss, loss_mask

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='PA', help='dataset_name')
parser.add_argument('--root_path', type=str, default='./', help='Name of Dataset')
parser.add_argument('--model_path', type=str, default='./', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='Co_Manifold_CL')
parser.add_argument('--model', type=str, default='vnet', help='model_name')
parser.add_argument('--max_iteration', type=int, default=15000, help='maximum iteration to train')
parser.add_argument('--max_samples', type=int, default=62, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=3, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=6, help='batch_size of labeled data per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='lr to train')
parser.add_argument('--beta1', type=float, default=0.9, help='first parameter of Adam (default: 0.9)')
parser.add_argument('--beta2', type=float, default=0.999, help='second parameter of Adam (default: 0.900)')
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

parser.add_argument('--consistency', type=float, default=1.0, help='consistency_weight')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')

args = parser.parse_args()

snapshot_path = args.model_path + "/model/{}_{}_{}_labeled_{}_ce_{}_dl_{}_tm_{}_bs_{}_alpha_{}_beta/{}".format(args.dataset_name, args.exp, args.labelnum, args.ce_w, args.dl_w, args.t_m, args.labeled_bs, args.alpha, args.beta, args.model)

num_classes = 2

patch_size = (96, 96, 96)
args.root_path = '../data/Pancreas'
args.max_samples = 62
train_data_path = args.root_path

labeled_bs = args.labeled_bs
max_iterations = args.max_iteration
base_lr = args.base_lr

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


def get_current_consistency_weight(args, epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


if __name__ == "__main__":
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    model_1 = net_factory_hyperbolic(args, net_type=args.model, in_chns=1, class_num=num_classes - 1, mode="train")
    model_2 = net_factory(args, net_type=args.model, in_chns=1, class_num=num_classes - 1, mode="train")
    model_1 = model_1.cuda()
    model_2 = model_2.cuda()

    critic = Discriminator()
    critic = critic.cuda()

    db_train = Pancreas(base_dir=train_data_path,
                       split='train',
                       transform=transforms.Compose([
                           RandomCrop(patch_size),
                           ToTensor(),
                       ]))

    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - labeled_bs)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=12, pin_memory=True, worker_init_fn=worker_init_fn)

    optimizer_1 = optim.SGD(model_1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer_2 = optim.SGD(model_2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    dis_optimizer = torch.optim.AdamW(critic.parameters(), lr=1e-4)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    dice_loss = losses.dice_loss
    iter_num = 0
    best_dice_1 = 0
    best_dice_2 = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    CE = torch.nn.BCELoss()
    MSE = torch.nn.MSELoss()
    iterator = tqdm(range(max_epoch), ncols=70)

    scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_1, T_max=max_epoch)
    scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_2, T_max=max_epoch)

    c_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(dis_optimizer, T_max=max_epoch)

    alpha = args.alpha
    beta = args.beta

    hyp = geoopt.manifolds.PoincareBall(c=1.0, learnable=False)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            labeled_gt_set = label_batch[:labeled_bs, ...]
            labeled_set = volume_batch[:labeled_bs, ...]
            unlabeled_set = volume_batch[labeled_bs:, ...]

            # Train Model 1
            model_1.train()

            outputs1, features1, enc_mu1, z1, z1_E = model_1(volume_batch)
            l_outputs_np_1 = outputs1[:labeled_bs, ...]
            u_outputs_np_1 = outputs1[labeled_bs:, ...]

            mu1, log_sigma1, _ = enc_mu1

            # Train Model 2
            model_2.train()

            outputs2, features2, enc_mu2, z2, z2_H = model_2(volume_batch)
            l_outputs_np_2 = outputs2[:labeled_bs, ...]
            u_outputs_np_2 = outputs2[labeled_bs:, ...]

            mu2, log_sigma2, _ = enc_mu2

            sig_out_1 = torch.sigmoid(outputs1)
            sig_out_2 = torch.sigmoid(outputs2)

            y_all_1 = torch.sigmoid(outputs1)

            y_prob_1 = torch.sigmoid(l_outputs_np_1)
            y_sig_1_u = torch.sigmoid(u_outputs_np_1)

            loss_elbo1 = 0.5 * (1 + log_sigma1 - mu1 ** 2 - torch.exp(log_sigma1))
            loss_seg_1 = args.ce_w * CE(y_prob_1[:, 0, :, :, :], (labeled_gt_set == 1).float())
            loss_seg_dice_1 = args.dl_w * dice_loss(y_prob_1[:, 0, :, :, :], labeled_gt_set == 1)
            loss_sup_1 = loss_seg_1 + loss_seg_dice_1 + abs(loss_elbo1.mean())

            y_all_2 = torch.sigmoid(outputs2)

            y_prob_2 = torch.sigmoid(l_outputs_np_2)
            y_sig_2_u = torch.sigmoid(u_outputs_np_2)

            loss_elbo2 = 0.5 * (1 + log_sigma2 - mu2 ** 2 - torch.exp(log_sigma2))
            loss_seg_2 = args.ce_w * CE(y_prob_2[:, 0, :, :, :], (labeled_gt_set == 1).float())
            loss_seg_dice_2 = args.dl_w * dice_loss(y_prob_2[:, 0, :, :, :], labeled_gt_set == 1)
            loss_sup_2 = loss_seg_2 + loss_seg_dice_2 + abs(loss_elbo2.mean())

            zs_H = z1
            zs_E = z2

            zs_E_H = z2_H
            zs_H_E = z1_E

            zs_H_l = zs_H[:, :labeled_bs]
            zs_E_H_l = zs_E_H[:, :labeled_bs]

            zs_H_u = zs_H[:, labeled_bs:]
            zs_E_H_u = zs_E_H[:, labeled_bs:]

            zs_E_l = zs_E[:, :labeled_bs]
            zs_H_E_l = zs_H_E[:, :labeled_bs]

            zs_E_u = zs_E[:, labeled_bs:]
            zs_H_E_u = zs_H_E[:, labeled_bs:]

            l_angle_1 = discrepancy_loss_torch_center(zs_H_l, zs_E_l)
            u_angle_1 = discrepancy_loss_torch_center(zs_H_u, zs_E_u)
            loss_space_1 = l_angle_1.mean() + u_angle_1.mean()

            l_angle_2 = discrepancy_loss_torch_center(zs_E_H_l, zs_H_E_l)
            u_angle_2 = discrepancy_loss_torch_center(zs_E_H_u, zs_H_E_u)
            loss_space_2 = l_angle_2.mean() + u_angle_2.mean()

            loss_space = 0.5 * (loss_space_1 + loss_space_2)

            critic_segs_1 = torch.sigmoid(critic(y_all_2))
            loss_mask_1 = loss_mask(y_all_2, y_all_1, critic_segs_1, args.t_m)
            loss_mask_2 = loss_hyp_uncertainty(outputs1, y_all_1, y_all_2, args.t_m)

            iter_num = iter_num + 1

            target_real_1 = torch.ones_like(labeled_gt_set.unsqueeze(1))
            target_real_1.cuda()
            target_fake_1 = torch.zeros_like(label_batch.unsqueeze(1))
            target_fake_1.cuda()

            g_critic_segs_1_1 = torch.sigmoid(critic(y_all_1))
            g_critic_segs_1_2 = torch.sigmoid(critic(labeled_gt_set.unsqueeze(1).float()))
            target_real_g_1 = torch.ones_like(label_batch.unsqueeze(1))
            target_real_g_1.cuda()
            loss_adversarial_gen_1 = gen_loss(g_critic_segs_1_1, target_real_g_1)
            loss_adversarial_1 = disc_loss(g_critic_segs_1_1, g_critic_segs_1_2, target_fake_1, target_real_1)

            consistency_weight = get_current_consistency_weight(args, iter_num // 150)

            loss_unsup_1 = loss_mask_1 - alpha * loss_space + beta * loss_adversarial_gen_1
            loss_1 = loss_sup_1 + consistency_weight * loss_unsup_1

            optimizer_1.zero_grad()
            loss_1.backward(retain_graph=True)

            logging.info(
                'M1 iteration %d : loss : %03f, loss_sup: %03f,loss_space: %03f, loss_mask: %03f, loss_elbo: %03f, best_dice_1: %03f' % (
                    iter_num, loss_1, loss_sup_1, loss_space, loss_mask_1, loss_elbo1.mean(), best_dice_1))

            writer.add_scalar('Labeled_loss1/loss_seg_dice', loss_seg_dice_1, iter_num)
            writer.add_scalar('Co_loss1/mask_loss', loss_mask_1, iter_num)

            g_critic_segs_2_1 = torch.sigmoid(critic(y_all_2))
            g_critic_segs_2_2 = torch.sigmoid(critic(labeled_gt_set.unsqueeze(1).float()))
            loss_adversarial_gen_2 = gen_loss(g_critic_segs_2_1, target_real_g_1)
            loss_adversarial_2 = disc_loss(g_critic_segs_2_1, g_critic_segs_2_2, target_fake_1, target_real_1)

            loss_unsup_2 = loss_mask_2 - alpha * loss_space + beta * loss_adversarial_gen_2
            loss_2 = loss_sup_2 + consistency_weight * loss_unsup_2

            optimizer_2.zero_grad()
            loss_2.backward()

            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model_1.parameters(), args.clip)
                torch.nn.utils.clip_grad_norm_(model_2.parameters(), args.clip)

            optimizer_1.step()
            optimizer_2.step()

            logging.info(
                'M2 iteration %d : loss : %03f, loss_sup: %03f, loss_space: %03f, loss_mask: %03f, loss_elbo: %03f, best_dice_2: %03f' % (
                    iter_num, loss_2, loss_sup_2, loss_space, loss_mask_2, loss_elbo2.mean(), best_dice_2))

            writer.add_scalar('Labeled_loss2/loss_seg_dice', loss_seg_dice_2, iter_num)
            writer.add_scalar('Co_loss/space_loss', loss_space, iter_num)
            writer.add_scalar('Co_loss2/mask_loss', loss_mask_2, iter_num)

            del loss_1, loss_2, loss_sup_1, loss_sup_2, loss_unsup_1, loss_unsup_2, loss_mask_2, loss_seg_1, loss_seg_2, loss_seg_dice_1, loss_seg_dice_2
            gc.collect()
            torch.cuda.empty_cache()

            del g_critic_segs_1_1, g_critic_segs_2_1, g_critic_segs_1_2, g_critic_segs_2_2, y_prob_1, y_all_1, y_sig_1_u, y_prob_2, y_all_2, y_sig_2_u
            gc.collect()
            torch.cuda.empty_cache()

            # Train Discriminator 1
            loss_adversarial_1 = loss_adversarial_1.clone().detach().requires_grad_(True)
            loss_adversarial_2 = loss_adversarial_2.clone().detach().requires_grad_(True)

            dis_optimizer.zero_grad()

            critic_loss_1 = 1.0 * (loss_adversarial_1 + loss_adversarial_2)

            writer.add_scalar('loss/loss_critic1', critic_loss_1, iter_num)
            critic_loss_1.backward()
            dis_optimizer.step()

            if scheduler_1 is not None:
                scheduler_1.step()
            if scheduler_2 is not None:
                scheduler_2.step()
            if c_scheduler is not None:
                c_scheduler.step()

            if iter_num >= 700 and iter_num % 150 == 0:
                model_1.eval()
                dice_sample_1 = test_patch.var_all_case(model_1, num_classes=num_classes,
                                                                 patch_size=patch_size,
                                                                 stride_xy=16, stride_z=16,
                                                                 dataset_name='Pancreas_CT')
                if dice_sample_1 > best_dice_1:
                    best_dice_1 = dice_sample_1
                    save_best_path = os.path.join(snapshot_path, 'best_model_1.pth'.format(args.model))
                    torch.save(model_1.state_dict(), save_best_path)

                    logging.info("save best model to {}".format(save_best_path))

                    save_best_path_c1 = os.path.join(snapshot_path, 'best_critic_1.pth')
                    torch.save(critic.state_dict(), save_best_path_c1)

                writer.add_scalar('Var_dice1/Dice', dice_sample_1, iter_num)
                writer.add_scalar('Var_dice1/Best_dice', best_dice_1, iter_num)
                model_1.train()

                model_2.eval()
                dice_sample_2 = test_patch.var_all_case(model_2, num_classes=num_classes, patch_size=patch_size,
                                                        stride_xy=16, stride_z=16, dataset_name='Pancreas_CT')
                if dice_sample_2 > best_dice_2:
                    best_dice_2 = dice_sample_2
                    save_best_path = os.path.join(snapshot_path, 'best_model_2.pth'.format(args.model))
                    torch.save(model_2.state_dict(), save_best_path)

                    logging.info("save best model to {}".format(save_best_path))

                    save_best_path_c2 = os.path.join(snapshot_path, 'best_critic_2.pth')
                    torch.save(critic.state_dict(), save_best_path_c2)

                writer.add_scalar('Var_dice2/Dice', dice_sample_2, iter_num)
                writer.add_scalar('Var_dice2/Best_dice', best_dice_2, iter_num)
                model_2.train()

            if iter_num >= max_iterations:
                save_mode_path_1 = os.path.join(snapshot_path, 'm1_iter_' + str(iter_num) + '.pth')
                torch.save(model_1.state_dict(), save_mode_path_1)
                logging.info("save model 1 to {}".format(save_mode_path_1))

                save_mode_path_2 = os.path.join(snapshot_path, 'm2_iter_' + str(iter_num) + '.pth')
                torch.save(model_2.state_dict(), save_mode_path_2)
                logging.info("save model 2 to {}".format(save_mode_path_2))

                save_critic_path = os.path.join(snapshot_path, 'c_iter_' + str(iter_num) + '.pth')
                torch.save(critic.state_dict(), save_critic_path)
                logging.info("save critic to {}".format(save_critic_path))
                break

        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
