from torch.utils.data import DataLoader

from sgan.data.trajectories import TrajectoryDataset, seq_collate


def data_loader(args, path):
    # obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
    # non_linear_ped_list, loss_mask_list
    #non linear ped -> if np.polynomial squared error of (degree-1) term is greater than threshold. Why coefficient of degree-1 is taken? to introduce non linearity
    dset = TrajectoryDataset(
        args,
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim)

    if args.social_stgcnn:
        loader = DataLoader(
            dset,
            batch_size=1,
            shuffle=True,
            num_workers=args.loader_num_workers)
    else:
        loader = DataLoader(
            dset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.loader_num_workers,
            collate_fn=seq_collate)

    return dset, loader
