import cv2
import sys
import os
from os.path import join
from glob import glob
import numpy as np
from c3dmodel import *
import torch
import argparse

BATCH_SIZE = 32

parser = argparse.ArgumentParser()
parser.add_argument('--root', help="videos data root")
parser.add_argument('--out', help="dir for output")
parser.add_argument('-f', help="number of frames per clip", default=16)
parser.add_argument('-l', help="specify output layer", default=FC6_LAYER)
args = parser.parse_args()


def extract_frames(video, frame_dir):
    if os.path.isdir(frame_dir):
        print("[Warning] frame_dir={} does exist. Will overwrite".format(frame_dir))
    else:
        os.makedirs(frame_dir)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print("[Error] video={} can not be opened.".format(video))
        sys.exit(-6)

    ret, frame = cap.read()
    frame_num = 0
    while ret:
        frame_file = os.path.join(
                frame_dir,
                '{0:06d}.jpg'.format(frame_num)
                )
        cv2.imwrite(frame_file, frame)
        ret, frame = cap.read()
        frame_num += 1

    return frame_num - 1


def delete_frames(frame_dir):
    for filename in os.listdir(frame_dir):
        file_path = os.path.join(frame_dir, filename)
        try:
            os.remove(file_path)
        except Exception as e:
            print('Failed to delete file %s. Reason: %s' % (file_path, e))
    try:
        os.rmdir(frame_dir)
    except Exception as e:
        print('Failed to delete dir %s. Reason: %s' % (frame_dir, e))

def get_clips(frame_dir, num_frames_per_clip):
    overlap = int(num_frames_per_clip / 2)
    clips = sorted(glob(join(frame_dir, '*.jpg')))
    clips = np.array([cv2.resize(cv2.cvtColor(cv2.imread(frame), cv2.COLOR_BGR2RGB), (200, 112)) for frame in clips])
    clips = clips[:, :, 44:44 + 112, :]
    clips = np.array([clips[i : i + num_frames_per_clip] for i in range(0, len(clips), overlap) if len(clips[i : i + num_frames_per_clip]) == num_frames_per_clip])
    clips = clips.transpose(0, 4, 1, 2, 3)
    clips = np.float32(clips)
    return torch.from_numpy(clips)


def extractC3D1(video_file, out_layer_type=FC6_LAYER, num_frames_per_clip=16):
    video_id, _ = os.path.splitext(os.path.basename(video_file))

    tmp_dir = './tmp'
    print("Frames extraction started")
    frame_dir = os.path.join(tmp_dir, video_id)
    frames = extract_frames(video_file, frame_dir)
    print("Frames extraction finished, extracted {} frames".format(frames))

    print("Clips extraction started")
    clips = get_clips(frame_dir, num_frames_per_clip)
    if torch.cuda.is_available():
        clips = clips.cuda()
    print("Clips extraction finished")

    model = C3D()
    model.load_state_dict(torch.load('c3d.pickle'))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    print("Features extraction started")
    features = []
    for i in range(0, len(clips), BATCH_SIZE):
        print("Batch:", str(i + 1))
        feats = model(torch.from_numpy(np.expand_dims(clips[i:BATCH_SIZE], axis=0)), out_layer_type)
        batch_features = feats.data.cpu()
        features.append(batch_features)

    features = torch.cat(features, 0)
    features = features.numpy()
    print("Features extraction finished, extracted {} features".format(features.shape[0]))

    np.save(os.path.join(args.out, video_id), features)

    print("Deleting frames")
    delete_frames(frame_dir)


def extractC3D(videos_path, out_layer_type=FC6_LAYER, num_frames_per_clip=16):
    videos = [join(videos_path, f) for f in os.listdir(videos_path) if not f.startswith('.')]
    for video in videos:
        print("############## PROCESSING VIDEO {} ##############".format(video))
        video_path = os.path.join(videos_path, video)
        print("Video path:", video_path)
        extractC3D1(video_path, out_layer_type, num_frames_per_clip)


if __name__ == '__main__':
    extractC3D(args.root, args.l, args.f)