from dataloader import VideoDataset, transform
from model import Scoring
import torch
import torch.utils.data as data
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', help="model name")
parser.add_argument('--root', help="data root")
parser.add_argument('--score', help="which score type to test")
args = parser.parse_args()


def test_shuffle():
    testset = VideoDataset(root=args.root,
                           label="./data/test_dataset.csv", transform=transform, score_type=args.score)
    test_loader = torch.utils.data.DataLoader(testset,
                                             batch_size=64, shuffle=False, num_workers=0)

    scoring = Scoring(feature_size=4096)
    if torch.cuda.is_available():
        scoring.cuda()
    scoring.load_state_dict(torch.load("./models/" + args.name + ".pt"))
    scoring.eval()
    val_pred = []
    val_sample = 0
    val_loss = 0
    val_truth = []
    for j, (features, scores) in enumerate(test_loader):
        val_truth.append(scores.numpy())
        if torch.cuda.is_available():
            features = features.cuda()
            scores = scores.cuda()
        regression, _ = scoring(features)
        val_pred.append(regression.data.cpu().numpy())
        regr_loss = scoring.loss(regression, scores)
        val_loss += (regr_loss.data.item()) * scores.shape[0]
        val_sample += scores.shape[0]
    val_truth = np.concatenate(val_truth)
    val_pred = np.concatenate(val_pred)
    for i in range(val_truth.shape[0]):
        print('True: ' + str(val_truth[i]) + '\t' + "Pred: " + str(val_pred[i]) + '\t' + 'Diff: ' + str(
            val_truth[i] - val_pred[i]) + '\n')


test_shuffle()
