import os
import torch
import torch.utils.data as data
import numpy as np
import random
import csv


def transform(feature, max_frames):
    num_frames, _ = feature.shape
    if num_frames > max_frames:
        start_idx = random.choice(range(num_frames - max_frames))
        new_feature = feature[start_idx:start_idx + max_frames, :]
    else:
        new_feature = np.zeros([max_frames, feature.shape[1]])
        new_feature[0:num_frames, :] = feature
    return torch.Tensor(new_feature)


class VideoDataset(data.Dataset):
    def __init__(self, root, label, transform=None, score_type='pcs'):
        videos = []
        with open(label) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                video_id = row['number']
                score = float(row[score_type])
                videos.append((video_id + '.npy', score))
        self.root = root
        self.videos = videos
        self.transform = lambda x: transform(x, 300)

    def __getitem__(self, index):
        filename, score = self.videos[index]
        feature = np.load(os.path.join(self.root, filename))
        if self.transform is not None:
            feature = self.transform(feature)
        return feature, torch.Tensor([score])

    def __len__(self):
        return len(self.videos)
