from dataloader import VideoDataset, transform
from model import Scoring
import torch
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from scipy.stats import spearmanr as sr
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', help="model name")
parser.add_argument('--root', help="data root")
parser.add_argument('--score', help="which score type to train")
args = parser.parse_args()


def print_logs(log_str):
    log.write(log_str)
    print(log_str)


def train_shuffle(min_mse=200, max_corr=0):
    round_max_spea = 0
    round_min_mse = 200
    trainset = VideoDataset(root=args.root,
                            label="./data/train_dataset.csv", transform=transform,
                            score_type=args.score)
    train_loader = torch.utils.data.DataLoader(trainset,
                                              batch_size=128, shuffle=True, num_workers=0)
    testset = VideoDataset(root=args.root,
                           label="./data/test_dataset.csv", transform=transform,
                           score_type=args.score)
    test_loader = torch.utils.data.DataLoader(testset,
                                             batch_size=64, shuffle=False, num_workers=0)

    scoring = Scoring(feature_size=4096)
    if torch.cuda.is_available():
        scoring.cuda()
    optimizer = optim.Adam(params=scoring.parameters(), lr=1e-4)
    for epoch in range(120):
        print_logs("Epoch: " + str(epoch))
        total_regr_loss = 0
        total_sample = 0
        for i, (features, scores) in enumerate(train_loader):
            if torch.cuda.is_available():
                features = Variable(features).cuda()
                scores = Variable(scores).cuda()
            logits, penal = scoring(features)
            if penal is None:
                regr_loss = scoring.loss(logits, scores)
            else:
                regr_loss = scoring.loss(logits, scores) + penal
            optimizer.zero_grad()
            regr_loss.backward()
            optimizer.step()
            total_regr_loss += regr_loss.data.item() * scores.shape[0]
            total_sample += scores.shape[0]
        print_logs("Classification Loss: " + str(total_regr_loss / total_sample))

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
        val_sr, _ = sr(val_truth, val_pred)
        if val_loss / val_sample < min_mse:
            torch.save(scoring.state_dict(), './models/' + args.name + '.pt')
        torch.save(scoring.state_dict(), './models/' + args.name + '.pt')
        min_mse = min(min_mse, val_loss / val_sample)
        max_corr = max(max_corr, val_sr)
        round_min_mse = min(round_min_mse, val_loss / val_sample)
        round_max_spea = max(val_sr, round_max_spea)
        print_logs("Val Loss: %.2f Correlation: %.2f Min Val Loss: %.2f Max Correlation: %.2f\n" %
                   (val_loss / val_sample, val_sr, min_mse, max_corr))
        scoring.train()
    print_logs('MSE: %.2f spearman: %.2f' % (round_min_mse, round_max_spea))


log = open('./logs/' + args.name + '.txt', 'w')
min_mse = 200
max_corr = 0
train_shuffle(min_mse, max_corr)
log.close()
