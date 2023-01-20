# Super-SloMo [![MIT Licence](https://badges.frapsoft.com/os/mit/mit.svg?v=103)](https://opensource.org/licenses/mit-license.php)
This is an addon to https://github.com/avinashpaliwal/Super-SloMo for Event-Based simulation. The code is standalone and updated to latest pytorch, you do not need to install avinashpaliwal code.

## Pretrained model
So far you can use the same model trained on [adobe240fps dataset by avinashpaliwal](https://drive.google.com/open?id=1IvobLDbRiBgZr3ryCRrWL8xDbMZ-KnpF).

## Asynchronous Video Converter
This addon allows to create a video with motion-dependant interpolation. This means it will produce more or less frames dependently on maximum optical flow between 2 frames.

Be sure the repo is in your PYTHONPATH. For example:
```bash
export PYTHONPATH=$PYTHONPATH:path\to\repo\Super-SloMo/:
```

Now, you can run the main tool for one video:
```bash
python async_slomo.py path\to\video.mp4 path\to\output.mp4 --sf -1 --checkpoint path\to\checkpoint.ckpt --video_fps M --lambda_flow 0.5 --viz 1
```
Or for an entire Folder:
```bash
python async_slomo.py input_path\to\ output_path\to\ --sf -1 --checkpoint path\to\checkpoint.ckpt --video_fps M
```

After running the script, you should see 2 files per video:
- output.mp4
- output_ts.npy

This can then be used by our event-based simulator available in the [Professional Offer](https://www.prophesee.ai/metavision-intelligence-professional-request) of [Metavision Intelligence](https://www.prophesee.ai/metavision-intelligence).
