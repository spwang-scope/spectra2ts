from data_loader import Dataset_Custom
from torch.utils.data import DataLoader

def data_provider(args, flag):

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size

    if args.data_dir is None:
        raise ValueError("data_dir does not exist")
    if args.data_filename is None:
        raise ValueError("data_filename does not exist")

    data_set = Dataset_Custom(
        args = args,
        root_path=args.data_dir,
        data_path=args.data_filename,
        flag=flag,
        size=[args.seq_len, args.pred_len],
        target=args.target
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
