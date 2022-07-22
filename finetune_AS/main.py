import tqdm
import torchvision
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, matthews_corrcoef, confusion_matrix
from sklearn.utils import compute_class_weight
import random
import torch
import os
import pandas as pd
import numpy as np
import cv2
import argparse
import shutil

from utils import train, train_kd, validate, evaluate, seed_worker, set_seed
from dataset import EchoDataset

def main(args):
    MODEL_NAME = args.model_name
    MODEL_NAME += f'_frac-{args.frac}' if args.frac != 1.0 else ''
    MODEL_NAME += '_pretr' if not args.rand_init else '_rand'
    MODEL_NAME += f'_ssl-{args.ssl.split("/")[-1]}' if args.ssl != '' else ''
    MODEL_NAME += '_aug' if args.augment else ''
    MODEL_NAME += f'_clip-len-{args.clip_len}-stride-{args.sampling_rate}'
    MODEL_NAME += f'_num-clips-{args.num_clips}'
    MODEL_NAME += '_cw' if args.use_class_weights else ''
    MODEL_NAME += f'_lr-{args.lr}'
    MODEL_NAME += f'_{args.max_epochs}ep'
    MODEL_NAME += f'_patience-{args.patience}' if args.patience != 1e4 else ''
    MODEL_NAME += f'_bs-{args.batch_size}'
    MODEL_NAME += f'_ls-{args.label_smoothing}' if args.label_smoothing != 0. else ''
    MODEL_NAME += f'_drp-0.25' if args.dropout_fc else ''
    MODEL_NAME += f'_weight-avg' if args.weight_averaging else ''
    MODEL_NAME += f'_TTA-{args.n_TTA}' if args.n_TTA > 0 else ''
    MODEL_NAME += f'_seed-{args.seed}' if args.seed != 0 else ''

    # Create output directory for model (and delete if already exists)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    model_dir = os.path.join(args.output_dir, MODEL_NAME)
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.mkdir(model_dir)

    # Set all seeds for reproducibility
    set_seed(args.seed)

    # Set device
    device = torch.device('cuda:0')

    # Create model
    if args.model_name == '3dresnet18':
        model = torchvision.models.video.r3d_18(pretrained=not args.rand_init)

        if args.dropout_fc:
            model.fc = torch.nn.Sequential(torch.nn.Linear(512, 1), torch.nn.Dropout(0.25))

            if args.lpft != '':
                weights = dict(torch.load(args.lpft, map_location='cpu')['weights'])

                model.fc[0].weight.data = weights['fc.0.weight']
                model.fc[0].bias.data = weights['fc.0.bias']

        else:
            model.fc = torch.nn.Linear(512, 1)
            if args.lpft != '':
                weights = dict(torch.load(args.lpft, map_location='cpu')['weights'])

                model.fc.weight.data = weights['fc.weight']
                model.fc.bias.data = weights['fc.bias']

        if args.ssl != '':
            checkpoints = [f for f in os.listdir(args.ssl) if f.endswith('.pt')]
            idx = np.argmax([int(f.split('.')[0].split('-')[1]) for f in checkpoints])
            weights = torch.load(os.path.join(args.ssl, checkpoints[idx]), map_location='cpu')['weights']
            weights = {k.replace('encoder.', ''):v for k, v in weights.items() if 'encoder' in k}

            model.load_state_dict(weights, strict=False)
    
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.n_gpu))).to(device)
    else:
        model = model.to(device)

    # Create datasets
    train_dataset    = EchoDataset(data_dir=args.data_dir, split='train', clip_len=args.clip_len, sampling_rate=args.sampling_rate, num_clips=args.num_clips, augment=args.augment, frac=args.frac, kinetics=(args.ssl == '') and (not args.rand_init))
    val_dataset      = EchoDataset(data_dir=args.data_dir, split='val', clip_len=args.clip_len, sampling_rate=args.sampling_rate, num_clips=args.num_clips, kinetics=(args.ssl == '') and (not args.rand_init))
    test_dataset     = EchoDataset(data_dir=args.data_dir, split='test', clip_len=args.clip_len, sampling_rate=args.sampling_rate, num_clips=args.num_clips, n_TTA=args.n_TTA, kinetics=(args.ssl == '') and (not args.rand_init))
    ext_test_dataset = EchoDataset(data_dir=args.data_dir, split='ext_test', clip_len=args.clip_len, sampling_rate=args.sampling_rate, num_clips=args.num_clips, n_TTA=args.n_TTA, kinetics=(args.ssl == '') and (not args.rand_init))

    # Create loaders
    train_loader    = torch.utils.data.DataLoader(train_dataset, batch_size=args.n_gpu*args.batch_size, shuffle=True, num_workers=8, worker_init_fn=seed_worker)
    val_loader      = torch.utils.data.DataLoader(val_dataset, batch_size=args.n_gpu*args.batch_size, shuffle=False, num_workers=4)
    test_loader     = torch.utils.data.DataLoader(test_dataset, batch_size=args.n_gpu*args.batch_size, shuffle=False, num_workers=8)
    ext_test_loader = torch.utils.data.DataLoader(ext_test_dataset, batch_size=args.n_gpu*args.batch_size, shuffle=False, num_workers=8)

    # Create csv documenting training history
    history = pd.DataFrame(columns=['epoch', 'phase', 'loss', 'auroc', 'aupr', 'acc', 'b_acc', 'mcc', 'precision', 'recall', 'f1'])
    history.to_csv(os.path.join(model_dir, 'history.csv'), index=False)

    # Set class weights
    if args.use_class_weights:
        class_weights = compute_class_weight(class_weight='balanced', classes=np.sort(np.unique((train_dataset.label_df['av_stenosis'] == 'Severe').values)), y=(train_dataset.label_df['av_stenosis'] == 'Severe').values)
        
        print('Class weights:', class_weights)
        
        pos_weight = (class_weights / class_weights.min())[1]

        print('Normalized positive class weight:', pos_weight)

        loss_fxn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]).to(device))
        eval_loss_fxn = torch.nn.BCELoss(weight=torch.Tensor([pos_weight]).to(device))
    else:
        loss_fxn = torch.nn.BCEWithLogitsLoss()
        eval_loss_fxn = torch.nn.BCELoss()

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.eval_only == '':
        # Train with early stopping
        epoch = 1
        early_stopping_dict = {'best_loss': 1e8, 'epochs_no_improve': 0}
        best_model_wts = None
        while epoch <= args.max_epochs and early_stopping_dict['epochs_no_improve'] <= args.patience:
            if args.kd != '':
                history = train_kd(student=student, teacher=teacher, device=device, cls_loss_fxn=loss_fxn, kd_loss_fxn=kd_loss_fxn, optimizer=optimizer, data_loader=train_loader, history=history, epoch=epoch, model_dir=model_dir, plax_weight=args.plax_weight, plax_weight_agg=args.plax_weight_agg)
                history, early_stopping_dict, best_model_wts = validate(model=student, device=device, loss_fxn=eval_loss_fxn, optimizer=optimizer, data_loader=val_loader, history=history, epoch=epoch, model_dir=model_dir, early_stopping_dict=early_stopping_dict, best_model_wts=best_model_wts, plax_weight=args.plax_weight, plax_weight_agg=args.plax_weight_agg)
            else:
                history = train(model=model, device=device, loss_fxn=loss_fxn, optimizer=optimizer, data_loader=train_loader, history=history, epoch=epoch, model_dir=model_dir, plax_weight=args.plax_weight, plax_weight_agg=args.plax_weight_agg)
                history, early_stopping_dict, best_model_wts = validate(model=model, device=device, loss_fxn=eval_loss_fxn, optimizer=optimizer, data_loader=val_loader, history=history, epoch=epoch, model_dir=model_dir, early_stopping_dict=early_stopping_dict, best_model_wts=best_model_wts, plax_weight=args.plax_weight, plax_weight_agg=args.plax_weight_agg)

            epoch += 1
        
        if args.weight_averaging:
            checkpoints = [torch.load(os.path.join(model_dir, f), map_location='cpu')['weights'] for f in os.listdir(model_dir) if f.endswith('.pt')]
            for key in best_model_wts:
                best_model_wts[key] = torch.stack([chkpt[key] for chkpt in checkpoints], dim=0).sum(dim=0) / len(checkpoints)
    else:
        history = pd.read_csv(os.path.join(args.eval_only, 'history.csv'))
        checkpoints = [f for f in os.listdir(args.eval_only) if f.endswith('.pt')]
        idx = np.argmax([int(f.split('.')[0].split('-')[1]) for f in checkpoints])
        best_model_wts = torch.load(os.path.join(args.eval_only, checkpoints[idx]), map_location='cpu')['weights']

    # Evaluate on test set
    evaluate(model=model, device=device, loss_fxn=eval_loss_fxn, data_loader=test_loader, split='test', classes=test_dataset.CLASSES, history=history, model_dir=model_dir, weights=best_model_wts, plax_weight=args.plax_weight, plax_weight_agg=args.plax_weight_agg, n_TTA=args.n_TTA)

    # Evaluate on external test set
    evaluate(model=model, device=device, loss_fxn=eval_loss_fxn, data_loader=ext_test_loader, split='ext_test', classes=ext_test_dataset.CLASSES, history=history, model_dir=model_dir, weights=best_model_wts, plax_weight=args.plax_weight, plax_weight_agg=args.plax_weight_agg, n_TTA=args.n_TTA)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/gih5/mounts/nfs_echo_yale/031522_echo_avs_preprocessed')
    parser.add_argument('--output_dir', type=str, default='/home/gih5/echo_avs/binary_results_v3')
    parser.add_argument('--clip_len', type=int, default=16)
    parser.add_argument('--sampling_rate', type=int, default=1)
    parser.add_argument('--num_clips', type=int, default=4)
    
    parser.add_argument('--model_name', type=str, default='3dresnet18')
    parser.add_argument('--rand_init', action='store_true', default=False)
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=1e4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--label_smoothing', type=float, default=0.)
    parser.add_argument('--use_class_weights', action='store_true', default=False)
    parser.add_argument('--dropout_fc', action='store_true', default=False)
    parser.add_argument('--augment', action='store_true', default=False)
    parser.add_argument('--lpft', type=str, default='')

    parser.add_argument('--ssl', type=str, default='')
    parser.add_argument('--frac', type=float, default=1.0)

    parser.add_argument('--weight_averaging', action='store_true', default=False)
    parser.add_argument('--n_TTA', type=int, default=0)

    parser.add_argument('--eval_only', type=str, default='')

    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    print(args)

    main(args)