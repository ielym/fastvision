import cv2
import numpy as np
import random
import math

def sampleFramesByIndex(cap, idxs):
    '''
    :param cap: cv2.VideoCapture
    :param idxs: list, sampling idxs : [0, 2, 4, 6, 8]
    :return:
    '''

    ret, frame = cap.read()

    frames = []

    total = len(idxs)
    already = 0
    idx = 0
    while ret and already < total:
        if idx in idxs:
            frames.append(np.expand_dims(frame, 0))
            already += 1
        idx += 1
        ret, frame = cap.read()

    return np.concatenate(frames, axis=0)

def randomSampling(cap, frames=1):
    '''
    :param cap: cv2.VideoCapture
    :param nums: sample frame numbers
    :return:

    sample from a video by random frame idx
    '''
    TOTAL_FRAMES = math.floor(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if TOTAL_FRAMES < frames:
        raise Exception("randomSampling TOTAL_FRAMES less than nums")

    sample_idxs = random.sample(range(TOTAL_FRAMES), frames)
    sample_idxs.sort()

    frames = sampleFramesByIndex(cap, sample_idxs)

    return frames

def averageSampling(cap, frames=1):
    '''
    :param cap: cv2.VideoCapture
    :param nums: sample frame numbers
    :return:

    sample from a video by same step
    '''
    TOTAL_FRAMES = math.floor(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if TOTAL_FRAMES < frames:
        raise Exception("randomSampling TOTAL_FRAMES less than nums")

    step = TOTAL_FRAMES // frames
    sample_idxs = list(range(0, step * frames, step))
    sample_idxs.sort()

    frames = sampleFramesByIndex(cap, sample_idxs)
    return frames

def randomClipSampling(cap, clips=1, frames_per_clip=1):
    '''
    :param cap: cv2.VideoCapture
    :param nums: sample frame numbers
    :return:

    sample from a video, first clip whole video into anverage {nums} frames, then sample a frame from each clip randomly.
    '''

    TOTAL_FRAMES = math.floor(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if TOTAL_FRAMES < clips:
        raise Exception("randomClipSampling clips less than frames")

    step = TOTAL_FRAMES // clips
    sample_idxs = list(range(0, step * clips, step))
    sample_idxs.append(TOTAL_FRAMES)

    frames_idxs = []
    for i in range(1, len(sample_idxs)):
        start_idx = sample_idxs[i - 1]
        end_idx = sample_idxs[i]
        if (end_idx - start_idx + 1) < frames_per_clip:
            raise Exception("randomClipSampling frames_per_clip less than clip frames")

        frames_idxs.extend(random.sample(range(start_idx, end_idx), 1))
    frames_idxs.sort()

    frames = sampleFramesByIndex(cap, frames_idxs)
    return frames