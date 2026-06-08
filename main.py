import os
import torch
from torch.utils.data import DataLoader
import argparse
from utils.LossFunction import BCEDiceLoss
from datasets.LEVIR_CD import LEVIRCDDataset
from utils.train import train
from model.stnet import STNet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path',          type=str,   default='./data/LEVIR-CD')
    parser.add_argument('--model_name',            type=str,   default=None)
    parser.add_argument('--cuda',                  type=str,   default='0')
    parser.add_argument('--batch_size',            type=int,   default=1)
    parser.add_argument('--num_epochs',            type=int,   default=200)
    parser.add_argument('--lr',                    type=float, default=1e-3)
    parser.add_argument('--num_classes',           type=int,   default=2)
    parser.add_argument('--miou_max',              type=float, default=0.7)
    parser.add_argument('--lr_scheduler',          type=int,   default=3)
    parser.add_argument('--lr_scheduler_gamma',    type=float, default=0.99)
    parser.add_argument('--warmup',                type=int,   default=1)
    parser.add_argument('--checkpoint_step',       type=int,   default=10)
    parser.add_argument('--validation_step',       type=int,   default=1)
    parser.add_argument('--summary_path',          type=str,   default='./summary')
    parser.add_argument('--save_checkpoint_path',  type=str,   default='./checkpoints')
    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.summary_path, args.model_name ,args.save_checkpoint_path)):
        os.makedirs(os.path.join(args.summary_path, args.model_name ,args.save_checkpoint_path))

    os.environ['CUDA_VISIBLE_DEVICES']=args.cuda
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset_train = LEVIRCDDataset(args.dataset_path, mode='train')
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size,shuffle=True, num_workers=2, drop_last=True)
    dataset_val = LEVIRCDDataset(args.dataset_path, mode='val')
    dataloader_val = DataLoader(dataset_val, batch_size=1,shuffle=True,num_workers=4)

    model = STNet(in_chans=3, num_classes=2, vss_depth=2, vssm_path_size=4)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=1e-4)

    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_scheduler, gamma=args.lr_scheduler_gamma)
    criterion = BCEDiceLoss()

    train(args,
            model,
            optimizer,
            criterion,
            dataloader_train,
            dataloader_val,
            exp_lr_scheduler
            )

if __name__ == '__main__':
    main()