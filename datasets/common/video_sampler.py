import cv2
import numpy as np
import random
import math

def countRealFrames(cap):
    '''
    :param cap:
    :return:
    Under some unknown situations, cap.get(cv2.CAP_PROP_FRAME_COUNT) != real total frames
    '''
    cnt = 0
    ret, _ = cap.read()
    while ret:
        cnt += 1
        ret, _ = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return cnt

def sampleFramesByIndex(cap, idxs):
    '''
    :param cap: cv2.VideoCapture
    :param idxs: list, sampling idxs : [0, 2, 4, 6, 8]
    :return:
    '''

    frames = []

    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(np.expand_dims(frame, 0))

    return np.concatenate(frames, axis=0)

def consecutiveSampling(cap, frames=64, TOTAL_FRAMES=None):
    '''
    :param cap: cv2.VideoCapture
    :param nums: sample frame numbers
    :return:

    sample consecutive ${frames} frames from a video
    '''
    TOTAL_FRAMES = math.floor(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not TOTAL_FRAMES else TOTAL_FRAMES
    if TOTAL_FRAMES < frames:
        raise Exception("consecutiveSampling TOTAL_FRAMES less than nums")

    start_idx = random.randint(0, TOTAL_FRAMES - frames)
    end_idx = start_idx + frames
    sample_idxs = list(range(start_idx, end_idx))
    sample_idxs.sort()

    sampled_frames = sampleFramesByIndex(cap, sample_idxs)

    if len(sampled_frames) != frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        real_frames = countRealFrames(cap)
        sampled_frames = consecutiveSampling(cap, frames, real_frames)

    return sampled_frames

def randomSampling(cap, frames=1, TOTAL_FRAMES=None):
    '''
    :param cap: cv2.VideoCapture
    :param nums: sample frame numbers
    :return:

    sample from a video by random frame idx
    '''
    TOTAL_FRAMES = math.floor(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not TOTAL_FRAMES else TOTAL_FRAMES

    if TOTAL_FRAMES < frames:
        raise Exception("randomSampling TOTAL_FRAMES less than nums")

    sample_idxs = random.sample(range(TOTAL_FRAMES), frames)
    sample_idxs.sort()

    sampled_frames = sampleFramesByIndex(cap, sample_idxs)

    if len(sampled_frames) != frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        real_frames = countRealFrames(cap)
        sampled_frames = randomSampling(cap, frames, real_frames)

    return sampled_frames

def averageSampling(cap, frames=1, TOTAL_FRAMES=None):
    '''
    :param cap: cv2.VideoCapture
    :param nums: sample frame numbers
    :return:

    sample from a video by same step
    '''
    TOTAL_FRAMES = math.floor(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not TOTAL_FRAMES else TOTAL_FRAMES

    if TOTAL_FRAMES < frames:
        raise Exception("randomSampling TOTAL_FRAMES less than nums")

    step = TOTAL_FRAMES // frames
    sample_idxs = list(range(0, step * frames, step))
    sample_idxs.sort()

    sampled_frames = sampleFramesByIndex(cap, sample_idxs)

    if len(sampled_frames) != frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        real_frames = countRealFrames(cap)
        sampled_frames = averageSampling(cap, frames, real_frames)

    return sampled_frames

def randomClipSampling(cap, clips=1, frames_per_clip=1, TOTAL_FRAMES=None):
    '''
    :param cap: cv2.VideoCapture
    :param nums: sample frame numbers
    :return:

    sample from a video, first clip whole video into average {nums} frames, then sample a frame from each clip randomly.
    '''
    TOTAL_FRAMES = math.floor(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not TOTAL_FRAMES else TOTAL_FRAMES

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

    sampled_frames = sampleFramesByIndex(cap, frames_idxs)

    if len(sampled_frames) != clips * frames_per_clip:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        real_frames = countRealFrames(cap)
        sampled_frames = randomClipSampling(cap, clips, frames_per_clip, real_frames)

    return sampled_frames