import os
import random

import cv2
import numpy as np
import pandas as pd
import torch

from scipy.ndimage import rotate

from utils import load_video

class EchoDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, clip_len=16, sampling_rate=1, num_clips=4, augment=False, frac=1.0, kinetics=False, N_TTA=0):
        assert split in ['train', 'val', 'test', 'ext_test'], "split must be one of ['train', 'val', 'test', 'ext_test']"

        self.split = split
        self.clip_len = clip_len
        self.num_clips = num_clips
        self.sampling_rate = sampling_rate
        self.augment = augment
        self.kinetics = kinetics

        self.video_dir = os.path.join(data_dir, 'videos')
        self.label_df = pd.read_csv(os.path.join(data_dir, self.split + '.csv'))

        if frac != 1.0:
            study_ids = np.sort(self.label_df['acc_num'].unique())

            self.label_df = self.label_df[self.label_df['acc_num'].isin(np.random.choice(study_ids, size=int(frac*study_ids.size), replace=False))]

            print('Num studies:', int(frac*study_ids.size))
        self.label_df['label'] = self.label_df['av_stenosis'].apply(lambda x: 1 if x == 'Severe' else 0)
        
        print(self.label_df['label'].value_counts())

        self.CLASSES = ['None', 'Severe']

        # Kinetics-400 mean and std
        self.mean = np.array([0.43216, 0.394666, 0.37645])
        self.std = np.array([0.22803, 0.22145, 0.216989])

    def _sample_frames(self, x):
        if self.split == 'train':
            if x.shape[0] > self.clip_len*self.sampling_rate:
                start_idx = np.random.choice(x.shape[0]-self.clip_len*self.sampling_rate, size=1)[0]
                x = x[start_idx:(start_idx+self.clip_len*self.sampling_rate):self.sampling_rate]
                x = np.transpose(x, (3, 0, 1, 2))
            else:
                x = x[::self.sampling_rate]
                x = np.pad(x, ((0, self.clip_len-x.shape[0]), (0, 0), (0, 0), (0, 0)), mode='constant')
                x = np.transpose(x, (3, 0, 1, 2))
        else:
            if x.shape[0] >= self.clip_len*self.sampling_rate + self.num_clips:
                start_indices = np.arange(0, x.shape[0]-self.clip_len*self.sampling_rate, (x.shape[0]-self.clip_len*self.sampling_rate) // self.num_clips)[:self.num_clips]
                x = np.stack([x[start_idx:(start_idx+self.clip_len*self.sampling_rate):self.sampling_rate] for start_idx in start_indices], axis=0)
                x = np.transpose(x, (4, 0, 1, 2, 3))
            elif x.shape[0] > self.clip_len*self.sampling_rate:
                x = x[::self.sampling_rate]
                x = x[:self.clip_len]
                x = np.stack([x] * self.num_clips, axis=0)
                x = np.transpose(x, (4, 0, 1, 2, 3))
            else:
                x = x[::self.sampling_rate]
                x = np.pad(x, ((0, self.clip_len-x.shape[0]), (0, 0), (0, 0), (0, 0)), mode='constant')
                x = np.stack([x] * self.num_clips, axis=0)
                x = np.transpose(x, (4, 0, 1, 2, 3))

        return x

    def _augment(self, x):
        pad = 8

        l, h, w, c = x.shape
        temp = np.zeros((l, h + 2 * pad, w + 2 * pad, c), dtype=x.dtype)
        temp[:, pad:-pad, pad:-pad, :] = x
        i, j = np.random.randint(0, 2 * pad, 2)
        x = temp[:, i:(i + h), j:(j + w), :]

        if random.uniform(0, 1) > 0.5:
            # H flip
            x = np.stack([cv2.flip(frame, 1) for frame in x], axis=0)

        if random.uniform(0, 1) > 0.5:
            # Rotation
            angle = np.random.choice(np.arange(-10, 11), size=1)[0]

            x = np.stack([rotate(frame, angle, reshape=False) for frame in x], axis=0)

        return x

    def __len__(self):
        return self.label_df.shape[0]

    def __getitem__(self, idx):
        plax_prob, fname, acc_num, _, video_num, label = self.label_df.iloc[idx, :]

        x = load_video(os.path.join(self.video_dir, fname))

        if self.augment:
            x = self._augment(x)

        x = self._sample_frames(x)

        x = (x - x.min()) / (x.max() - x.min())

        if self.kinetics:
            if self.split == 'train':
                x -= self.mean.reshape(3, 1, 1, 1)
                x /= self.std.reshape(3, 1, 1, 1)
            else:
                x -= self.mean.reshape(3, 1, 1, 1, 1)
                x /= self.std.reshape(3, 1, 1, 1, 1)

        y = np.array([label])

        return {'x': torch.from_numpy(x).float(), 'y': torch.from_numpy(y).float(), 'acc_num': acc_num, 'video_num': video_num, 'plax_prob': plax_prob}
