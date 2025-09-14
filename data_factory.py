from data_loader import Dataset_Custom
from torch.utils.data import DataLoader

def data_provider(args, flag):

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    batch_size = args.batch_size

    if args.root_path is None:
        raise ValueError("root_path does not exist")
    if args.data_path is None:
        raise ValueError("data_path does not exist")

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
    else:
        shuffle_flag = True
        drop_last = True

    # Handle scaler passing for TSLib standard
    if flag == 'train':
        data_set = Dataset_Custom(
            args = args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.pred_len],
            target=args.target
        )
        # Store scaler for test dataset
        args._scaler = data_set.scaler
    else:
        # Use scaler from training dataset
        scaler = getattr(args, '_scaler', None)
        data_set = Dataset_Custom(
            args = args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.pred_len],
            target=args.target,
            scaler=scaler
        )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
