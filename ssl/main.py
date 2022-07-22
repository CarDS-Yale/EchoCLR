import os
import shutil

import pandas as pd
import torch
import torchvision

from dataset import EchoDataset
from losses import NT_Xent
from model import SimCLR


def main(args):
    # Create output directory and clean out if already exists
    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    model_dir = os.path.join(args.out_dir, args.model_name)

    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.mkdir(model_dir)

    history = pd.DataFrame({'epoch': [], 'loss': []})
    history.to_csv(os.path.join(model_dir, 'history.csv'), index=False)

    device = 'cuda:0'

    set_seed(0)

    train_dataset = EchoDataset(data_dir=args.data_dir, split='train', clip_len=args.clip_len, sampling_rate=args.sampling_rate)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.n_gpu*args.batch_size, shuffle=True, num_workers=12, worker_init_fn=seed_worker, drop_last=True)

    encoder = torchvision.models.video.r3d_18(pretrained=False)
    model = SimCLR(encoder=encoder, projection_dim=args.projection_dim, n_features=encoder.fc.in_features, frame_reordering=args.frame_reordering)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.n_gpu))).to(device)
    else:
        model = model.to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_fxn = NT_Xent(args.batch_size, args.temperature, world_size=args.n_gpu)
    if self.frame_reordering:
        cls_loss_fxn = torch.nn.CrossEntropyLoss()

    for epoch in range(1, args.num_epochs + 1):
        running_loss = 0.
        pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch}')

        for i, batch in pbar:
            if self.frame_reordering:
                x_i, x_j, t_i, t_j = batch
                x_i = x_i.to(device)
                x_j = x_j.to(device)
                t_i = t_i.to(device)
                t_j = t_j.to(device)

                h_i, h_j, z_i, z_j, t_hat_i, t_hat_j = model.forward(x_i, x_j)

                loss = loss_fxn(z_i, z_j) + cls_loss_fxn(torch.cat([t_hat_i, t_hat_j]), torch.cat([t_i, t_j]))
            else:
                x_i, x_j = batch
                x_i = x_i.to(device)
                x_j = x_j.to(device)

                h_i, h_j, z_i, z_j = model.forward(x_i, x_j)

                loss = loss_fxn(z_i, z_j)

            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            pbar.set_postfix({'loss': running_loss / (i + 1)})

        current_metrics = pd.DataFrame({'epoch': [epoch], 'loss': [running_loss / (i + 1)]})
        current_metrics.to_csv(os.path.join(model_dir, 'history.csv'), mode='a', header=False, index=False)

        if epoch % args.save_freq == 0:
            torch.save({'weights': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(), 'optimizer': optimizer.state_dict()},
                       os.path.join(model_dir, f'chkpt_epoch-{epoch}.pt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/gih5/mounts/nfs_echo_yale/031522_echo_avs_preprocessed')
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    
    parser.add_argument('--frame_reordering', action='store_true', default=False)

    parser.add_argument('--batch_size', type=int, default=196)
    parser.add_argument('--temperature', type=float, default=0.05)
    parser.add_argument('--projection_dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--clip_len', type=int, default=4)
    parser.add_argument('--sampling_rate', type=int, default=1)

    parser.add_argument('--save_freq', type=int, default=20)
    