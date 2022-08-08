import torch

from torch.utils.data import DataLoader
import pickle
import glob
import torch.distributions.multivariate_normal as torchdist
from sgan.utils import *
from sgan.data.loader import data_loader
from sgan.metrics import seq_to_nodes, nodes_rel_to_nodes_abs, ade, fde
from sgan.SocialSTGCNN import  social_stgcnn
import copy


def test(model, loader_test, KSTEPS=20):
    model.eval()
    ade_bigls = []
    fde_bigls = []
    raw_data_dict = {}
    step = 0
    for batch in loader_test:
        step += 1
        # Get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
        loss_mask, V_obs, A_obs, V_tr, A_tr = batch

        num_of_objs = obs_traj_rel.shape[1]

        # Forward
        # V_obs = batch,seq,node,feat
        # V_obs_tmp = batch,feat,seq,node
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)

        V_pred, _ = model(V_obs_tmp, A_obs.squeeze())
        # print(V_pred.shape)
        # torch.Size([1, 5, 12, 2])
        # torch.Size([12, 2, 5])
        V_pred = V_pred.permute(0, 2, 3, 1)
        # torch.Size([1, 12, 2, 5])>>seq,node,feat
        # V_pred= torch.rand_like(V_tr).cuda()

        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()
        num_of_objs = obs_traj_rel.shape[1]
        V_pred, V_tr = V_pred[:, :num_of_objs, :], V_tr[:, :num_of_objs, :]
        # print(V_pred.shape)

        # For now I have my bi-variate parameters
        # normx =  V_pred[:,:,0:1]
        # normy =  V_pred[:,:,1:2]
        sx = torch.exp(V_pred[:, :, 2])  # sx
        sy = torch.exp(V_pred[:, :, 3])  # sy
        corr = torch.tanh(V_pred[:, :, 4])  # corr

        cov = torch.zeros(V_pred.shape[0], V_pred.shape[1], 2, 2).cuda()
        cov[:, :, 0, 0] = sx * sx
        cov[:, :, 0, 1] = corr * sx * sy
        cov[:, :, 1, 0] = corr * sx * sy
        cov[:, :, 1, 1] = sy * sy
        mean = V_pred[:, :, 0:2]

        mvnormal = torchdist.MultivariateNormal(mean, cov)

        ### Rel to abs
        ##obs_traj.shape = torch.Size([1, 6, 2, 8]) Batch, Ped ID, x|y, Seq Len

        # Now sample 20 samples
        ade_ls = {}
        fde_ls = {}
        V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())
        V_x_rel_to_abs = nodes_rel_to_nodes_abs(V_obs.data.cpu().numpy().squeeze().copy(),
                                                V_x[0, :, :].copy())

        V_y = seq_to_nodes(pred_traj_gt.data.cpu().numpy().copy())
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_tr.data.cpu().numpy().squeeze().copy(),
                                                V_x[-1, :, :].copy())

        raw_data_dict[step] = {}
        raw_data_dict[step]['obs'] = copy.deepcopy(V_x_rel_to_abs)
        raw_data_dict[step]['trgt'] = copy.deepcopy(V_y_rel_to_abs)
        raw_data_dict[step]['pred'] = []

        for n in range(num_of_objs):
            ade_ls[n] = []
            fde_ls[n] = []

        for k in range(KSTEPS):

            V_pred = mvnormal.sample()

            # V_pred = seq_to_nodes(pred_traj_gt.data.numpy().copy())
            V_pred_rel_to_abs = nodes_rel_to_nodes_abs(V_pred.data.cpu().numpy().squeeze().copy(),
                                                       V_x[-1, :, :].copy())
            raw_data_dict[step]['pred'].append(copy.deepcopy(V_pred_rel_to_abs))

            # print(V_pred_rel_to_abs.shape) #(12, 3, 2) = seq, ped, location
            for n in range(num_of_objs):
                pred = []
                target = []
                obsrvs = []
                number_of = []
                pred.append(V_pred_rel_to_abs[:, n:n + 1, :])
                target.append(V_y_rel_to_abs[:, n:n + 1, :])
                obsrvs.append(V_x_rel_to_abs[:, n:n + 1, :])
                number_of.append(1)

                ade_ls[n].append(ade(pred, target, number_of))
                fde_ls[n].append(fde(pred, target, number_of))

        for n in range(num_of_objs):
            ade_bigls.append(min(ade_ls[n]))
            fde_bigls.append(min(fde_ls[n]))

    ade_ = sum(ade_bigls) / len(ade_bigls)
    fde_ = sum(fde_bigls) / len(fde_bigls)
    return ade_, fde_, raw_data_dict


paths = ['D:/Summer Sem 22/Seminar Deep Learning in Computer Vision/Code/Condor_logs/logs/*social-stgcnn*']
KSTEPS = 20

print("*" * 50)
print('Number of samples:', KSTEPS)
print("*" * 50)


def main():
    for feta in range(len(paths)):
        ade_ls = []
        fde_ls = []
        path = paths[feta]
        exps = glob.glob(path)
        print('Model being tested are:', exps)

        for exp_path in exps:
            exp_path = exp_path.replace("\\", "/")
            print("*" * 50)
            print("Evaluating model:", exp_path)

            model_path = exp_path + '/val_best.pth'
            args_path = exp_path + '/args.pkl'
            with open(args_path, 'rb') as f:
                args = pickle.load(f)

            stats = exp_path + '/constant_metrics.pkl'
            with open(stats, 'rb') as f:
                cm = pickle.load(f)
            print("Stats:", cm)

            # Data prep
            obs_seq_len = args.obs_seq_len
            pred_seq_len = args.pred_seq_len

            test_path = './datasets/' + args.dataset + '/test'
            setattr(args,"social_stgcnn", True)
            setattr(args, "obs_len", obs_seq_len)
            setattr(args, "pred_len", pred_seq_len)
            setattr(args, "skip", 1)
            setattr(args, "delim", "\t")
            setattr(args, "loader_num_workers", 2)
            test_dset, test_loader = data_loader(args, test_path)


            # Defining the model
            model = social_stgcnn(n_stgcnn=args.n_stgcnn, n_txpcnn=args.n_txpcnn,
                                  output_feat=args.output_size, seq_len=args.obs_seq_len,
                                  kernel_size=args.kernel_size, pred_seq_len=args.pred_seq_len).cuda()
            model.load_state_dict(torch.load(model_path))

            ade_ = 999999
            fde_ = 999999
            print("Testing ....")
            ad, fd, raw_data_dic_ = test(model, test_loader)
            ade_ = min(ade_, ad)
            fde_ = min(fde_, fd)
            ade_ls.append(ade_)
            fde_ls.append(fde_)
            print("ADE:", ade_, " FDE:", fde_)

        print("*" * 50)

        print("Avg ADE:", sum(ade_ls) / 5)
        print("Avg FDE:", sum(fde_ls) / 5)


if __name__ == '__main__':
    main()
