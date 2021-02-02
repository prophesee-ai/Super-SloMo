"""
script to upsample your videos
"""
from __future__ import absolute_import

import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import cv2
from skvideo.io import FFmpegWriter

from video_stream import VideoStream
from slowmo_warp import SlowMoWarp
from viz import draw_arrows
from utils import grab_videos

from torchvision.utils import make_grid
from tqdm import tqdm


def show_slowmo(last_frame, frame, flow_fw, flow_bw, interp, fps):
    """SlowMo visualization

    Args:
        last_frame: prev rgb frame (h,w,3)
        current_frame: current rgb frame (h,w,3)
        flow_fw: flow forward (1,h,w,2)
        flow_bw: flow backward (1,h,w,2)
        interp: last_frame + interpolated frames
        fps: current frame-rate
    """
    def viz_flow(frame, flow):
        flow1 = flow.data.cpu().numpy()
        frame0 = draw_arrows(frame, flow1[0], step=4, flow_unit="pixels")
        return frame0
    color = (0, 0, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    height, width = interp[0].shape[:2]
    virtual_fps = fps * len(interp)

    viz_flow_fw = viz_flow(last_frame.copy(), flow_fw)
    viz_flow_bw = viz_flow(frame.copy(), flow_bw)
    for j, item in enumerate(interp):
        img = item.copy()

        img = cv2.putText(img, "orig fps: " + str(fps), (10, height - 90), font, 1.0, color, 2)
        img = cv2.putText(img, "virtual fps: " + str(virtual_fps), (10, height - 60), font, 1.0, color, 2,)
        img = cv2.putText(img, "#" + str(j), (10, height - 30), font, 1.0, color, 2)

        vizu = np.concatenate([viz_flow_fw[None], viz_flow_bw[None], img[None]])

        vizu = torch.from_numpy(vizu).permute(0, 3, 1, 2).contiguous()
        vizu = make_grid(vizu, nrow=2).permute(1, 2, 0).contiguous().numpy()

        cv2.imshow("result", vizu)
        key = cv2.waitKey(0)
        if key == 27:
            return 0
    return 1


def slowmo_video(
    video_filename,
    out_name="",
    video_fps=240,
    height=-1,
    width=-1,
    sf=-1,
    seek_frame=0,
    max_frames=-1,
    lambda_flow=0.5,
    cuda=True,
    viz=False,
):
    """SlowMo Interpolates video

    It produces another .mp4 video + .npy file for timestamps.

    Args:
        video_filename: file path
        out_name: out file path
        video_fps: video frame rate
        height: desired height
        width: desired width
        sf: desired frame-rate scale-factor (if -1 it interpolates based on maximum optical flow)
        seek_frame: seek in video before interpolating
        max_frames: maximum number of frames to interpolate
        lambda_flow: when interpolating with maximum flow, we multiply the maximum flow by this
        factor to compute the actual number of frames.
        cuda: use cuda
        viz: visualize the flow and interpolated frames
    """

    if os.path.isdir(video_filename):
        filenames = grab_videos(video_filename)
        random.shuffle(filenames)
        video_filename = filenames[0]

    print("Video filename: ", video_filename)
    print("Out Video: ", out_name)

    stream = VideoStream(
        video_filename,
        height,
        width,
        seek_frame=seek_frame,
        max_frames=max_frames,
        random_start=False,
        rgb=True)
    height, width = stream.height, stream.width

    filename = 'model_zoo/SuperSloMo.ckpt'
    slomo = SlowMoWarp(height, width, filename, lambda_flow=lambda_flow, cuda=cuda)

    fps = video_fps

    delta_t = 1.0 / fps
    delta_t_us = delta_t * 1e6

    print("fps: ", fps)
    print("Total length: ", len(stream))
    timestamps = []

    last_frame = None
    first_write = True

    num_video = 0

    if out_name:
        video_writer = FFmpegWriter(out_name)

    if viz:
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)

    last_ts = 0
    for i, frame in enumerate(tqdm(stream)):
        ts = i * delta_t

        if last_frame is not None:

            t_start = last_ts
            t_end = ts

            with torch.no_grad():
                out = slomo.forward(last_frame, frame, sf=sf)

            interp = [last_frame] + out["interpolated"]
            dt = (t_end - t_start) / len(interp)
            interp_ts = np.linspace(t_start, t_end - dt, len(interp))

            if out["sf"] == 0:
                print("skipping here, flow too small")
                continue

            if out_name:
                for item in interp:
                    video_writer.writeFrame(item[..., ::-1])

            timestamps.append(interp_ts)

            if viz:
                key = show_slowmo(last_frame, frame, *out['flow'], interp, fps)
                if key == 0:
                    break

            last_ts = ts

        last_frame = frame.copy()

    if viz:
        cv2.destroyWindow("result")

    if out_name:
        video_writer.close()
        timestamps_out = np.concatenate(timestamps)
        np.save(os.path.splitext(out_name)[0] + "_ts.npy", timestamps_out)


def rewrite_folder(idir, odir, fps=240, height=480, width=640, sf=-1, max_frames=100, ext='.mp4', viz=False):
    """Applies SlowMotion on a directory of videos

    Args:
        idir: input directory
        odir: output directory
        fps: input frame-rate
        height: desired height
        width: desired width
        sf: scale-factor
        max_frames: maximum number of frames to interpolate
        ext: look for files with this extension
        viz: visualize the Result
    """
    wsf = str(sf) if sf > 0 else "async"
    print('frame_rate factor: ', wsf)
    filenames = grab_videos(idir)
    for item in filenames:
        otem, _ = os.path.splitext(os.path.basename(item))
        otem = os.path.join(odir, otem + ext)
        if os.path.exists(otem):
            continue
        slowmo_video(item, otem, fps, height, width, sf, viz=viz, max_frames=max_frames)


if __name__ == "__main__":
    import fire
    fire.Fire()
