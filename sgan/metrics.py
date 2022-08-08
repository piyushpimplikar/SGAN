import torch
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs
from sgan.losses import l2_loss
import numpy as np
import math
def check_accuracy(
    args, loader, generator, discriminator, d_loss_fn, limit=False
):
    d_losses = []
    metrics = {}
    g_l2_losses_abs, g_l2_losses_rel = ([],) * 2
    disp_error, disp_error_l, disp_error_nl = ([],) * 3
    f_disp_error, f_disp_error_l, f_disp_error_nl = ([],) * 3
    total_traj, total_traj_l, total_traj_nl = 0, 0, 0
    loss_mask_sum = 0
    generator.eval()
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch
            linear_ped = 1 - non_linear_ped
            loss_mask = loss_mask[:, args.obs_len:]

            pred_traj_fake_rel = generator(
                obs_traj, obs_traj_rel, seq_start_end
            )
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

            g_l2_loss_abs, g_l2_loss_rel = cal_l2_losses(
                pred_traj_gt, pred_traj_gt_rel, pred_traj_fake,
                pred_traj_fake_rel, loss_mask
            )
            ade, ade_l, ade_nl = cal_ade(
                pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
            )

            fde, fde_l, fde_nl = cal_fde(
                pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
            )

            traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
            traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
            traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
            traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

            scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
            scores_real = discriminator(traj_real, traj_real_rel, seq_start_end)

            d_loss = d_loss_fn(scores_real, scores_fake)
            d_losses.append(d_loss.item())

            g_l2_losses_abs.append(g_l2_loss_abs.item())
            g_l2_losses_rel.append(g_l2_loss_rel.item())
            disp_error.append(ade.item())
            disp_error_l.append(ade_l.item())
            disp_error_nl.append(ade_nl.item())
            f_disp_error.append(fde.item())
            f_disp_error_l.append(fde_l.item())
            f_disp_error_nl.append(fde_nl.item())

            loss_mask_sum += torch.numel(loss_mask.data)
            total_traj += pred_traj_gt.size(1)
            total_traj_l += torch.sum(linear_ped).item()
            total_traj_nl += torch.sum(non_linear_ped).item()
            if limit and total_traj >= args.num_samples_check:
                break

    metrics['d_loss'] = sum(d_losses) / len(d_losses)
    metrics['g_l2_loss_abs'] = sum(g_l2_losses_abs) / loss_mask_sum
    metrics['g_l2_loss_rel'] = sum(g_l2_losses_rel) / loss_mask_sum

    metrics['ade'] = sum(disp_error) / (total_traj * args.pred_len)
    metrics['fde'] = sum(f_disp_error) / total_traj
    if total_traj_l != 0:
        metrics['ade_l'] = sum(disp_error_l) / (total_traj_l * args.pred_len)
        metrics['fde_l'] = sum(f_disp_error_l) / total_traj_l
    else:
        metrics['ade_l'] = 0
        metrics['fde_l'] = 0
    if total_traj_nl != 0:
        metrics['ade_nl'] = sum(disp_error_nl) / (
            total_traj_nl * args.pred_len)
        metrics['fde_nl'] = sum(f_disp_error_nl) / total_traj_nl
    else:
        metrics['ade_nl'] = 0
        metrics['fde_nl'] = 0

    generator.train()
    return metrics


def cal_l2_losses(
    pred_traj_gt, pred_traj_gt_rel, pred_traj_fake, pred_traj_fake_rel,
    loss_mask
):
    g_l2_loss_abs = l2_loss(
        pred_traj_fake, pred_traj_gt, loss_mask, mode='sum'
    )
    g_l2_loss_rel = l2_loss(
        pred_traj_fake_rel, pred_traj_gt_rel, loss_mask, mode='sum'
    )
    return g_l2_loss_abs, g_l2_loss_rel


def cal_ade(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped):
    ade = displacement_error(pred_traj_fake, pred_traj_gt)
    ade_l = displacement_error(pred_traj_fake, pred_traj_gt, linear_ped)
    ade_nl = displacement_error(pred_traj_fake, pred_traj_gt, non_linear_ped)
    return ade, ade_l, ade_nl


def cal_fde(
    pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
):
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])
    fde_l = final_displacement_error(
        pred_traj_fake[-1], pred_traj_gt[-1], linear_ped
    )
    fde_nl = final_displacement_error(
        pred_traj_fake[-1], pred_traj_gt[-1], non_linear_ped
    )
    return fde, fde_l, fde_nl


def ade(predAll, targetAll, count_):
    All = len(predAll)
    sum_all = 0
    for s in range(All):
        pred = np.swapaxes(predAll[s][:, :count_[s], :], 0, 1)
        target = np.swapaxes(targetAll[s][:, :count_[s], :], 0, 1)

        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0
        for i in range(N):
            for t in range(T):
                sum_ += math.sqrt((pred[i, t, 0] - target[i, t, 0]) ** 2 + (pred[i, t, 1] - target[i, t, 1]) ** 2)
        sum_all += sum_ / (N * T)

    return sum_all / All


def fde(predAll, targetAll, count_):
    All = len(predAll)
    sum_all = 0
    for s in range(All):
        pred = np.swapaxes(predAll[s][:, :count_[s], :], 0, 1)
        target = np.swapaxes(targetAll[s][:, :count_[s], :], 0, 1)
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0
        for i in range(N):
            for t in range(T - 1, T):
                sum_ += math.sqrt((pred[i, t, 0] - target[i, t, 0]) ** 2 + (pred[i, t, 1] - target[i, t, 1]) ** 2)
        sum_all += sum_ / (N)

    return sum_all / All


def seq_to_nodes(seq_):
    max_nodes = seq_.shape[1]  # number of pedestrians in the graph
    seq_ = seq_.squeeze()
    seq_len = seq_.shape[2]

    V = np.zeros((seq_len, max_nodes, 2))
    for s in range(seq_len):
        step_ = seq_[:, :, s]
        for h in range(len(step_)):
            V[s, h, :] = step_[h]

    return V.squeeze()


def nodes_rel_to_nodes_abs(nodes, init_node):
    nodes_ = np.zeros_like(nodes)
    for s in range(nodes.shape[0]):
        for ped in range(nodes.shape[1]):
            nodes_[s, ped, :] = np.sum(nodes[:s + 1, ped, :], axis=0) + init_node[ped, :]

    return nodes_.squeeze()


def closer_to_zero(current, new_v):
    dec = min([(abs(current), current), (abs(new_v), new_v)])[1]
    if dec != current:
        return True
    else:
        return False


def bivariate_loss(V_pred, V_trgt):
    # mux, muy, sx, sy, corr
    # assert V_pred.shape == V_trgt.shape
    normx = V_trgt[:, :, 0] - V_pred[:, :, 0]
    normy = V_trgt[:, :, 1] - V_pred[:, :, 1]

    sx = torch.exp(V_pred[:, :, 2])  # sx
    sy = torch.exp(V_pred[:, :, 3])  # sy
    corr = torch.tanh(V_pred[:, :, 4])  # corr

    sxsy = sx * sy

    z = (normx / sx) ** 2 + (normy / sy) ** 2 - 2 * ((corr * normx * normy) / sxsy)
    negRho = 1 - corr ** 2

    # Numerator
    result = torch.exp(-z / (2 * negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon))
    result = torch.mean(result)

    return result
