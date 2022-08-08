import argparse
import gc
import logging
import os
import sys
sys.path.append("..")
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from sgan.data.loader import data_loader
from sgan.losses import gan_g_loss, gan_d_loss, l2_loss, graph_loss
from pathlib import Path

from sgan.models import TrajectoryGenerator, TrajectoryDiscriminator
from sgan.SocialSTGCNN import  social_stgcnn
from sgan.utils import int_tuple, bool_flag, get_total_norm
from sgan.utils import relative_to_abs, get_dset_path
from sgan.metrics import check_accuracy
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'

# Dataset options
parser.add_argument('--dataset_name', default='zara1', type=str)
parser.add_argument('--delim', default='\t')
parser.add_argument('--loader_num_workers', default=2, type=int)
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--pred_len', default=12, type=int)
parser.add_argument('--skip', default=1, type=int)

# Optimization
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_iterations', default=0, type=int)
parser.add_argument('--num_epochs', default=200, type=int)

# Model Options
parser.add_argument('--embedding_dim', default=16, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--dropout', default=0, type=float)
parser.add_argument('--batch_norm', default=0, type=bool_flag)
parser.add_argument('--mlp_dim', default=64, type=int)

# Generator Options
parser.add_argument('--encoder_h_dim_g', default=32, type=int)
parser.add_argument('--decoder_h_dim_g', default=32, type=int)
parser.add_argument('--noise_dim', default=(8,), type=int_tuple)
parser.add_argument('--noise_type', default='gaussian')
parser.add_argument('--noise_mix_type', default='global')
parser.add_argument('--clipping_threshold_g', default=1.5, type=float)
parser.add_argument('--g_learning_rate', default=1e-3, type=float)
parser.add_argument('--g_steps', default=1, type=int)

# Pooling Options
parser.add_argument('--pooling_type', default='pool_net', type=str)
parser.add_argument('--pool_every_timestep', default=0, type=bool_flag)
# Pool Net Option
parser.add_argument('--bottleneck_dim', default=32, type=int)

# Social Pooling Options
parser.add_argument('--neighborhood_size', default=2.0, type=float)
parser.add_argument('--grid_size', default=8, type=int)

# Discriminator Options
parser.add_argument('--d_type', default='local', type=str)
parser.add_argument('--encoder_h_dim_d', default=64, type=int)
parser.add_argument('--d_learning_rate', default=1e-3, type=float)
parser.add_argument('--d_steps', default=2, type=int)
parser.add_argument('--clipping_threshold_d', default=0, type=float)

# Loss Options
parser.add_argument('--l2_loss_weight', default=1, type=float)
parser.add_argument('--best_k', default=1, type=int)

# Output
parser.add_argument('--output_dir', default=os.getcwd())
parser.add_argument('--print_every', default=50, type=int)
parser.add_argument('--checkpoint_every', default=10, type=int)
parser.add_argument('--checkpoint_name', default="0")
parser.add_argument('--checkpoint_start_from', default=None)
parser.add_argument('--restore_from_checkpoint', default=1, type=int)
parser.add_argument('--num_samples_check', default=5000, type=int)

# Misc
parser.add_argument('--use_gpu', default=1, type=int)
parser.add_argument('--timing', default=0, type=int)
parser.add_argument('--gpu_num', default="0", type=str)
parser.add_argument('--logging_file', default="SGAN-20-VP-20-8", type=str)
#Social STGNN
parser.add_argument('--social_stgcnn', default=False, help='Use Social STGCNN', action='store_true')
parser.add_argument('--input_size', type=int, default=2)
parser.add_argument('--output_size', type=int, default=5)
parser.add_argument('--n_stgcnn', type=int, default=1,help='Number of ST-GCNN layers')
parser.add_argument('--n_txpcnn', type=int, default=5, help='Number of TXPCNN layers')
parser.add_argument('--kernel_size', type=int, default=3)
parser.add_argument('--use_lrschd', action="store_true", default=True,
                    help='Use lr rate scheduler')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate')
parser.add_argument('--lr_sh_rate', type=int, default=150,
                    help='number of steps to drop the lr')
parser.add_argument('--clip_grad', type=float, default=None,
                    help='gadient clipping')



args = parser.parse_args()


args = parser.parse_args()
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'

args.checkpoint_name = args.logging_file
if args.social_stgcnn:
    args.logging_file = "social-stgcnn" + "-" + args.dataset_name + "-" + str(args.pred_len)
    args.checkpoint_name = args.logging_file
logging_file = args.logging_file
if not os.path.exists(f"./logs/{args.dataset_name}/{args.logging_file}"):
    if not os.path.exists("./logs"):
      os.mkdir("./logs")
    Path(f"./logs/{args.dataset_name}/{args.logging_file}").mkdir(parents=True)
logger = logging.getLogger()
logging.basicConfig(filename=f"./logs/{args.dataset_name}/{args.logging_file}/{args.logging_file}.log", filemode="a", format=FORMAT)
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
logger.addHandler(console)
logger.info(args.logging_file)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)


def get_dtypes(args):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if args.use_gpu == 1:
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype

def train_sgan(train_dset, train_loader, val_loader):
    long_dtype, float_dtype = get_dtypes(args)

    iterations_per_epoch = len(train_dset) / args.batch_size / args.d_steps
    if args.num_epochs:
        args.num_iterations = int(iterations_per_epoch * args.num_epochs)

    logger.info(
        'There are {} iterations per epoch'.format(iterations_per_epoch)
    )

    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,  # dimension of embedding layer of input
        encoder_h_dim=args.encoder_h_dim_g,  # dimension of hidden layer in the encoder
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)

    generator.apply(init_weights)
    generator.type(float_dtype).train()
    logger.info('Here is the generator:')
    logger.info(generator)

    discriminator = TrajectoryDiscriminator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        h_dim=args.encoder_h_dim_d,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_norm=args.batch_norm,
        d_type=args.d_type)

    discriminator.apply(init_weights)
    discriminator.type(float_dtype).train()
    logger.info('Here is the discriminator:')
    logger.info(discriminator)

    print("Generator_params", sum(p.numel() for p in generator.parameters() if p.requires_grad))
    print("Discriminator_params",sum(p.numel() for p in discriminator.parameters() if p.requires_grad))




    g_loss_fn = gan_g_loss
    d_loss_fn = gan_d_loss

    optimizer_g = optim.Adam(generator.parameters(), lr=args.g_learning_rate)
    optimizer_d = optim.Adam(
        discriminator.parameters(), lr=args.d_learning_rate
    )

    # Maybe restore from checkpoint
    restore_path = None
    if args.checkpoint_start_from is not None:
        restore_path = args.checkpoint_start_from
    elif args.restore_from_checkpoint == 1:
        restore_path = os.path.join(
            args.output_dir, "logs", args.checkpoint_name, f'%s_with_model.pt' % args.checkpoint_name
        )

    if restore_path is not None and os.path.isfile(restore_path):
        logger.info('Restoring from checkpoint {}'.format(restore_path))
        checkpoint = torch.load(restore_path)
        generator.load_state_dict(checkpoint['g_state'])
        discriminator.load_state_dict(checkpoint['d_state'])
        optimizer_g.load_state_dict(checkpoint['g_optim_state'])
        optimizer_d.load_state_dict(checkpoint['d_optim_state'])
        t = checkpoint['counters']['t']
        epoch = checkpoint['counters']['epoch']
        checkpoint['restore_ts'].append(t)
    else:
        # Starting from scratch, so initialize checkpoint data structure
        t, epoch = 0, 0
        checkpoint = {
            'args': args.__dict__,
            'G_losses': defaultdict(list),
            'D_losses': defaultdict(list),
            'losses_ts': [],
            'metrics_val': defaultdict(list),
            'metrics_train': defaultdict(list),
            'sample_ts': [],
            'restore_ts': [],
            'norm_g': [],
            'norm_d': [],
            'counters': {
                't': None,
                'epoch': None,
            },
            'g_state': None,
            'g_optim_state': None,
            'd_state': None,
            'd_optim_state': None,
            'g_best_state': None,
            'd_best_state': None,
            'best_t': None,
            'g_best_nl_state': None,
            'd_best_state_nl': None,
            'best_t_nl': None,
        }
    t0 = None
    log_dir = f"./logs/{args.dataset_name}/{logging_file}"
    writer = SummaryWriter(log_dir=log_dir)
    while t < args.num_iterations:
        gc.collect()
        d_steps_left = args.d_steps
        g_steps_left = args.g_steps
        epoch += 1
        logger.info('Starting epoch {}'.format(epoch))
        for batch in train_loader:
            if args.timing == 1:
                torch.cuda.synchronize()
                t1 = time.time()

            # Decide whether to use the batch for stepping on discriminator or
            # generator; an iteration consists of args.d_steps steps on the
            # discriminator followed by args.g_steps steps on the generator.
            if d_steps_left > 0:
                step_type = 'd'
                losses_d = discriminator_step(args, batch, generator,
                                              discriminator, d_loss_fn,
                                              optimizer_d)
                checkpoint['norm_d'].append(
                    get_total_norm(discriminator.parameters()))
                d_steps_left -= 1
            elif g_steps_left > 0:
                step_type = 'g'
                losses_g = generator_step(args, batch, generator,
                                          discriminator, g_loss_fn,
                                          optimizer_g)
                checkpoint['norm_g'].append(
                    get_total_norm(generator.parameters())
                )
                g_steps_left -= 1

            if args.timing == 1:
                torch.cuda.synchronize()
                t2 = time.time()
                logger.info('{} step took {}'.format(step_type, t2 - t1))

            # Skip the rest if we are not at the end of an iteration
            if d_steps_left > 0 or g_steps_left > 0:
                continue

            if args.timing == 1:
                if t0 is not None:
                    logger.info('Interation {} took {}'.format(
                        t - 1, time.time() - t0
                    ))
                t0 = time.time()

            # Maybe save loss
            if t % args.print_every == 0:
                logger.info('t = {} / {}'.format(t + 1, args.num_iterations))
                for k, v in sorted(losses_d.items()):
                    logger.info('  [D] {}: {:.3f}'.format(k, v))
                    checkpoint['D_losses'][k].append(v)
                for k, v in sorted(losses_g.items()):
                    logger.info('  [G] {}: {:.3f}'.format(k, v))
                    checkpoint['G_losses'][k].append(v)
                checkpoint['losses_ts'].append(t)

            # Maybe save a checkpoint
            if t > 0 and t % args.checkpoint_every == 0:
                checkpoint['counters']['t'] = t
                checkpoint['counters']['epoch'] = epoch
                checkpoint['sample_ts'].append(t)

                # Check stats on the validation set
                logger.info('Checking stats on val ...')
                metrics_val = check_accuracy(
                    args, val_loader, generator, discriminator, d_loss_fn
                )
                logger.info('Checking stats on train ...')
                metrics_train = check_accuracy(
                    args, train_loader, generator, discriminator,
                    d_loss_fn, limit=True
                )

                writer.add_scalar("VarietyLoss/Train", metrics_train["g_l2_loss_rel"], epoch)
                writer.add_scalar("D_loss/Train", metrics_train["d_loss"], epoch)
                writer.add_scalar("FDE/Train", metrics_train["fde"], epoch)
                writer.add_scalar("ADE/Train", metrics_train["ade"], epoch)

                writer.add_scalar("VarietyLoss/Val", metrics_val["g_l2_loss_rel"], epoch)
                writer.add_scalar("D_loss/Val", metrics_val["d_loss"], epoch)
                writer.add_scalar("FDE/Val", metrics_val["fde"], epoch)
                writer.add_scalar("ADE/Val", metrics_val["ade"], epoch)

                for k, v in sorted(metrics_val.items()):
                    logger.info('  [val] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_val'][k].append(v)
                for k, v in sorted(metrics_train.items()):
                    logger.info('  [train] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_train'][k].append(v)

                min_ade = min(checkpoint['metrics_val']['ade'])
                min_ade_nl = min(checkpoint['metrics_val']['ade_nl'])

                if metrics_val['ade'] == min_ade:
                    logger.info('New low for avg_disp_error')
                    checkpoint['best_t'] = t
                    checkpoint['g_best_state'] = generator.state_dict()
                    checkpoint['d_best_state'] = discriminator.state_dict()

                if metrics_val['ade_nl'] == min_ade_nl:
                    logger.info('New low for avg_disp_error_nl')
                    checkpoint['best_t_nl'] = t
                    checkpoint['g_best_nl_state'] = generator.state_dict()
                    checkpoint['d_best_nl_state'] = discriminator.state_dict()

                # Save another checkpoint with model weights and
                # optimizer state
                checkpoint['g_state'] = generator.state_dict()
                checkpoint['g_optim_state'] = optimizer_g.state_dict()
                checkpoint['d_state'] = discriminator.state_dict()
                checkpoint['d_optim_state'] = optimizer_d.state_dict()
                checkpoint_path = os.path.join(
                    args.output_dir, "logs", args.dataset_name, args.checkpoint_name,
                    f'%s_with_model.pt' % args.checkpoint_name
                )
                logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                torch.save(checkpoint, checkpoint_path)
                logger.info('Done.')

                # Save a checkpoint with no model weights by making a shallow
                # copy of the checkpoint excluding some items
                checkpoint_path = os.path.join(
                    args.output_dir, "logs", args.dataset_name, args.checkpoint_name,
                    f'%s_no_model.pt' % args.checkpoint_name)
                logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                key_blacklist = [
                    'g_state', 'd_state', 'g_best_state', 'g_best_nl_state',
                    'g_optim_state', 'd_optim_state', 'd_best_state',
                    'd_best_nl_state'
                ]
                small_checkpoint = {}
                for k, v in checkpoint.items():
                    if k not in key_blacklist:
                        small_checkpoint[k] = v
                torch.save(small_checkpoint, checkpoint_path)
                logger.info('Done.')

            t += 1
            d_steps_left = args.d_steps
            g_steps_left = args.g_steps
            if t >= args.num_iterations:
                break

def train_stgcnn(train_loader, val_loader):

    model = social_stgcnn(n_stgcnn=args.n_stgcnn, n_txpcnn=args.n_txpcnn,
                          output_feat=args.output_size, seq_len=args.obs_len,
                          kernel_size=args.kernel_size, pred_seq_len=args.pred_len).cuda()
    print("Social STGCNN Params", sum(p.numel() for p in model.parameters() if p.requires_grad))


    print(model)
    model.train()
    loss_batch = 0
    batch_count = 0
    is_fst_loss = True
    metrics = {'train_loss': [], 'val_loss': []}
    constant_metrics = {'min_val_epoch': -1, 'min_val_loss': 9999999999999999}
    loader_len = len(train_loader)
    turn_point = int(loader_len / args.batch_size) * args.batch_size + loader_len % args.batch_size - 1
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    checkpoint_dir = os.path.join(
        args.output_dir, "logs", args.dataset_name, args.checkpoint_name,"val_best.pt"
    )
    if args.use_lrschd:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_sh_rate, gamma=0.2)

    log_dir = f"./logs/{args.dataset_name}/{logging_file}"
    writer = SummaryWriter(log_dir=log_dir)
    logger.info('Training started ...')
    for epoch in range(args.num_epochs):
        for cnt, batch in enumerate(train_loader):
            batch_count += 1

            # Get data
            batch = [tensor.cuda() for tensor in batch]
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
            loss_mask, V_obs, A_obs, V_tr, A_tr = batch

            optimizer.zero_grad()


            V_obs_tmp = V_obs.permute(0, 3, 1, 2)

            V_pred, _ = model(V_obs_tmp, A_obs.squeeze())

            V_pred = V_pred.permute(0, 2, 3, 1)

            V_tr = V_tr.squeeze()
            A_tr = A_tr.squeeze()
            V_pred = V_pred.squeeze()

            if batch_count % args.batch_size != 0 and cnt != turn_point:
                l = graph_loss(V_pred, V_tr)
                if is_fst_loss:
                    loss = l
                    is_fst_loss = False
                else:
                    loss += l

            else:
                loss = loss / args.batch_size
                is_fst_loss = True
                loss.backward()

                if args.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

                optimizer.step()
                # Metrics
                loss_batch += loss.item()
                logging.info(f'TRAIN: Epoch: {epoch} Loss : {loss_batch / batch_count}')

        metrics['train_loss'].append(loss_batch / batch_count)
        loss_train = loss_batch / batch_count

        #Evaluate model each epoch
        model.eval()
        loss_batch = 0
        batch_count = 0
        is_fst_loss = True
        loader_len = len(val_loader)
        turn_point = int(loader_len / args.batch_size) * args.batch_size + loader_len % args.batch_size - 1

        for cnt, batch in enumerate(val_loader):
            batch_count += 1

            # Get data
            batch = [tensor.cuda() for tensor in batch]
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
            loss_mask, V_obs, A_obs, V_tr, A_tr = batch

            V_obs_tmp = V_obs.permute(0, 3, 1, 2)

            V_pred, _ = model(V_obs_tmp, A_obs.squeeze())

            V_pred = V_pred.permute(0, 2, 3, 1)

            V_tr = V_tr.squeeze()
            A_tr = A_tr.squeeze()
            V_pred = V_pred.squeeze()

            if batch_count % args.batch_size != 0 and cnt != turn_point:
                l = graph_loss(V_pred, V_tr)
                if is_fst_loss:
                    loss = l
                    is_fst_loss = False
                else:
                    loss += l

            else:
                loss = loss / args.batch_size
                is_fst_loss = True
                # Metrics
                loss_batch += loss.item()
                logging.info(f'VALD: Epoch: {epoch} Loss : {loss_batch / batch_count}')

        metrics['val_loss'].append(loss_batch / batch_count)

        if metrics['val_loss'][-1] < constant_metrics['min_val_loss']:
            constant_metrics['min_val_loss'] = metrics['val_loss'][-1]
            constant_metrics['min_val_epoch'] = epoch
            torch.save(model.state_dict(), checkpoint_dir)  # OK
        loss_val =  loss_batch / batch_count

        #perform logging
        writer.add_scalar("Loss/Train", loss_train, epoch)
        writer.add_scalar("Loss/Val", loss_val, epoch)
        if args.use_lrschd:
            scheduler.step()

        logger.info('*' * 30)
        logger.info(f'Epoch: {args.logging_file} : {epoch}')
        for k, v in metrics.items():
            if len(v) > 0:
                logger.info(f"{k} {v[-1]}")

        logger.info(constant_metrics)
        logger.info('*' * 30)



def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    train_path = get_dset_path(args.dataset_name, 'train')
    val_path = get_dset_path(args.dataset_name, 'val')


    logger.info("Initializing train dataset")
    train_dset, train_loader = data_loader(args, train_path)

    logger.info("Initializing val dataset")
    _, val_loader = data_loader(args, val_path)

    if args.social_stgcnn:
        train_stgcnn(train_loader, val_loader)
    else:
        train_sgan(train_dset, train_loader, val_loader)


def discriminator_step(
    args, batch, generator, discriminator, d_loss_fn, optimizer_d
):
    batch = [tensor.cuda() for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
     loss_mask, seq_start_end) = batch
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)


    generator_out = generator(obs_traj, obs_traj_rel, seq_start_end)

    pred_traj_fake_rel = generator_out
    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

    traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
    traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

    scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
    scores_real = discriminator(traj_real, traj_real_rel, seq_start_end)

    # Compute loss with optional gradient penalty
    data_loss = d_loss_fn(scores_real, scores_fake)
    losses['D_data_loss'] = data_loss.item()
    loss += data_loss
    losses['D_total_loss'] = loss.item()

    optimizer_d.zero_grad()
    loss.backward()
    if args.clipping_threshold_d > 0:
        nn.utils.clip_grad_norm_(discriminator.parameters(),
                                 args.clipping_threshold_d)
    optimizer_d.step()

    return losses


def generator_step(
    args: None, batch: tuple, generator, discriminator, g_loss_fn: gan_d_loss, optimizer_g: torch.optim
):
    batch = [tensor.cuda() for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
     loss_mask, seq_start_end) = batch
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)
    g_l2_loss_rel = []

    loss_mask = loss_mask[:, args.obs_len:]

    for _ in range(args.best_k):
        generator_out = generator(obs_traj, obs_traj_rel, seq_start_end)

        pred_traj_fake_rel = generator_out
        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

        if args.l2_loss_weight > 0:
            g_l2_loss_rel.append(args.l2_loss_weight * l2_loss(
                pred_traj_fake_rel,
                pred_traj_gt_rel,
                loss_mask,
                mode='raw'))

    g_l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
    if args.l2_loss_weight > 0:
        g_l2_loss_rel = torch.stack(g_l2_loss_rel, dim=1)
        for start, end in seq_start_end.data:
            _g_l2_loss_rel = g_l2_loss_rel[start:end]
            _g_l2_loss_rel = torch.sum(_g_l2_loss_rel, dim=0)
            _g_l2_loss_rel = torch.min(_g_l2_loss_rel) / torch.sum(
                loss_mask[start:end])
            g_l2_loss_sum_rel += _g_l2_loss_rel
        losses['G_l2_loss_rel'] = g_l2_loss_sum_rel.item()
        loss += g_l2_loss_sum_rel

    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

    scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
    discriminator_loss = g_loss_fn(scores_fake)

    loss += discriminator_loss
    losses['G_discriminator_loss'] = discriminator_loss.item()
    losses['G_total_loss'] = loss.item()

    optimizer_g.zero_grad()
    loss.backward()
    if args.clipping_threshold_g > 0:
        nn.utils.clip_grad_norm_(
            generator.parameters(), args.clipping_threshold_g
        )
    optimizer_g.step()

    return losses




if __name__ == '__main__':

    main(args)
