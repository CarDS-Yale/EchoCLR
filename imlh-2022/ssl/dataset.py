import itertools
import os
import random

import cv2
import numpy as np
import pandas as pd

import torch
import tqdm

from scipy.ndimage import rotate

from utils import load_video

class EchoDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, clip_len=16, sampling_rate=1, multi_instance=False, frame_reordering=False, n=None):
        assert split in ['train', 'val', 'test', 'ext_test'], "split must be one of ['train', 'val', 'test', 'ext_test']"

        self.split = split
        self.clip_len = clip_len
        self.sampling_rate = sampling_rate
        self.multi_instance = multi_instance
        self.frame_reordering = frame_reordering

        assert not ((not multi_instance) and frame_reordering), "frame_reordering can only be enabled when multi_instance is enabled"

        self.video_dir = os.path.join(data_dir, 'videos')
        self.label_df = pd.read_csv(os.path.join(data_dir, self.split + '.csv'))

        if n is not None:
            self.label_df = self.label_df.iloc[:n, :]

        self.study_ids = np.sort(self.label_df['acc_num'].unique())

        if self.multi_instance:
            self.fnames_i = []
            self.fnames_j = []
            for study_id in tqdm.tqdm(self.study_ids):
                fnames = self.label_df[self.label_df['acc_num'] == study_id]['fpath'].values.tolist()

                if len(fnames) == 1:
                    self.fnames_i.append(fnames[0])
                    self.fnames_j.append(fnames[0])
                else:
                    for fname_pair in itertools.combinations(fnames, 2):
                        self.fnames_i.append(fname_pair[0])
                        self.fnames_j.append(fname_pair[1])

            if self.frame_reordering:
                self.temporal_orderings = [_ for _ in itertools.permutations(np.arange(self.clip_len))]

    def _sample_frames(self, x):
        if x.shape[0] > self.clip_len*self.sampling_rate:
            start_idx = np.random.choice(x.shape[0]-self.clip_len*self.sampling_rate, size=1)[0]
            x = x[start_idx:(start_idx+self.clip_len*self.sampling_rate):self.sampling_rate]
        else:
            x = x[::self.sampling_rate]
            x = np.pad(x, ((0, self.clip_len-x.shape[0]), (0, 0), (0, 0), (0, 0)), mode='constant')

        return x

    def _augment(self, x):
        pad = 8

        l, h, w, c = x.shape
        temp = np.zeros((l, h + 2 * pad, w + 2 * pad, c), dtype=x.dtype)
        temp[:, pad:-pad, pad:-pad, :] = x  # pylint: disable=E1130
        i, j = np.random.randint(0, 2 * pad, 2)
        x = temp[:, i:(i + h), j:(j + w), :]

        if random.uniform(0, 1) > 0.5:
            # H flip
            x = np.stack([cv2.flip(frame, 1) for frame in x], axis=0)

        if random.uniform(0, 1) > 0.5:
            # Rotation
            angle = np.random.choice(np.arange(-10, 11), size=1)[0]

            x = np.stack([rotate(frame, angle, reshape=False) for frame in x], axis=0)

        if self.frame_reordering:
            # Frame re-ordering
            reordering_label = np.random.choice(len(self.temporal_orderings), size=1)[0]
            reordering = self.temporal_orderings[reordering_label]
            x = x[reordering, :, :, :]

            return x, reordering_label
        else:
            return x

    def __len__(self):
        if self.multi_instance:
            return len(self.fnames_i)
        else:
            return self.label_df.shape[0]

    def __getitem__(self, idx):
        if self.multi_instance:
            x_i = load_video(os.path.join(self.video_dir, self.fnames_i[idx]))
            x_j = load_video(os.path.join(self.video_dir, self.fnames_j[idx]))

            # Sample frames to form clip from each "view"
            x_i = self._sample_frames(x_i)
            x_j = self._sample_frames(x_j)

            # Augment each view and obtain frame ordering label
            if self.frame_reordering:
                x_i, reordering_i = self._augment(x_i)
                x_j, reordering_j = self._augment(x_j)
                reordering_i = np.array(reordering_i)
                reordering_j = np.array(reordering_j)
            else:
                x_i = self._augment(x_i)
                x_j = self._augment(x_j)
        else:
            plax_prob, fname, acc_num, label, video_num = self.label_df.iloc[idx, :]
            x = load_video(os.path.join(self.video_dir, fname))

            x = self._sample_frames(x)

            x_i = self._augment(x)
            x_j = self._augment(x)

        # Min-max normalize and swap axes for PyTorch
        x_i = (x_i - x_i.min()) / (x_i.max() - x_i.min())
        x_j = (x_j - x_j.min()) / (x_j.max() - x_j.min())

        x_i = np.transpose(x_i, (3, 0, 1, 2))
        x_j = np.transpose(x_j, (3, 0, 1, 2))

        if self.frame_reordering:
            return torch.from_numpy(x_i).float(), torch.from_numpy(x_j).float(), torch.from_numpy(reordering_i).long(), torch.from_numpy(reordering_j).long()
        else:
            return torch.from_numpy(x_i).float(), torch.from_numpy(x_j).float()
