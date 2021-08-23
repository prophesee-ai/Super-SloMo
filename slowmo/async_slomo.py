"""
script to upsample your videos
"""
from __future__ import absolute_import

import os
import urllib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from skvideo.io import FFmpegWriter
from PIL import Image
from slowmo.video_stream import VideoStream
from slowmo.slowmo_warp import SlowMoWarp
from slowmo.viz import draw_arrows
from slowmo.utils import grab_videos

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
        frame0 = draw_arrows(frame, flow1[0], step=16, flow_unit="pixels")
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
        img = cv2.putText(img, "virtual fps: " + str(virtual_fps), (10, height - 60), font, 1.0, color, 2, )
        img = cv2.putText(img, "#" + str(j), (10, height - 30), font, 1.0, color, 2)

        vizu = np.concatenate([viz_flow_fw[None], viz_flow_bw[None], img[None]])

        vizu = torch.from_numpy(vizu).permute(0, 3, 1, 2).contiguous()
        vizu = make_grid(vizu, nrow=2).permute(1, 2, 0).contiguous().numpy()

        cv2.imshow("result", vizu)
        key = cv2.waitKey(5)
        if key == 27:
            return 0
    return 1


def main_video(
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
        checkpoint='SuperSloMo.ckpt',
        crf=1
):
    """SlowMo Interpolates video
    It produces another .mp4 video + .npy file for timestamps.
    Args:
        video_filename: file path or list of ordered frame images
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
        checkpoint: if not provided will download it
    """

    print("Out Video: ", out_name)

    if isinstance(video_filename, list):
        # video_filename is a list of images
        print("First image of the list: ", video_filename[0])
        im = Image.open(video_filename[0])
        width, height = im.size
        stream = video_filename
    else:
        print("Video filename: ", video_filename)
        stream = VideoStream(
            video_filename,
            height,
            width,
            seek_frame=seek_frame,
            max_frames=max_frames,
            random_start=False,
            rgb=True)
        height, width = stream.height, stream.width

    slomo = SlowMoWarp(height, width, checkpoint, lambda_flow=lambda_flow, cuda=cuda)

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
        video_writer = FFmpegWriter(out_name, outputdict={
            '-vcodec': 'libx264',  # use the h.264 codec
            '-crf': str(crf),  # set the constant rate factor to 0, which is lossless
            #'-preset': 'veryslow'  # the slower the better compression, in princple, try
            # other options see https://trac.ffmpeg.org/wiki/Encode/H.264
        })

    last_ts = 0
    for i, frame in enumerate(tqdm(stream)):
        if isinstance(frame, str):
            frame = cv2.imread(frame)[:, :, ::-1]
            assert frame.shape[2] == 3

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
                    video_writer.writeFrame(item)

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


def main(
        input_path,
        output_path,
        video_fps=240,
        height=-1,
        width=-1,
        sf=-1,
        seek_frame=0,
        max_frames=-1,
        lambda_flow=0.5,
        cuda=True,
        viz=False,
        checkpoint='SuperSloMo.ckpt',
        rewrite=True):
    """Same Documentation, just with additional input directory"""
    main_fun = lambda x, y: main_video(x, y, video_fps, height, width, sf, seek_frame, max_frames, lambda_flow, cuda, viz,
                                       checkpoint)
    wsf = str(sf) if sf > 0 else "asynchronous"
    print('Interpolation frame_rate factor: ', wsf)
    if os.path.isdir(input_path):
        assert os.path.isdir(output_path)
        filenames = grab_videos(input_path)
        for item in filenames:
            ext = os.path.splitext(item)[1]
            otem, _ = os.path.splitext(os.path.basename(item))
            otem = os.path.join(output_path, otem + ext)
            if os.path.exists(otem) and not rewrite:
                continue
            main_fun(item, otem)
    else:
        main_fun(input_path, output_path)




if __name__ == "__main__":
    import fire
    fire.Fire(main)
