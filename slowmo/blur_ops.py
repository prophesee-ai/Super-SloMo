"""
blur operations
"""
import torch
import math
import numba
from numba import cuda


def blur_from_flow(img, flow, bilinear=True, pad_reflect=True):
    """
    img: (H,W) tensor (we will complexify later)
    flow: (H,W,2) tensor
    """
    height, width = img.shape
    out = torch.zeros_like(img)
    device = img.device
    on_gpu = 'cuda' in str(device)

    args = [img, out, flow]
    sizes = (height, width)
    if on_gpu:
        block_dim = (16, 16)
        grid_dim = tuple(int(math.ceil(a/b)) for a, b in zip(sizes, block_dim))
        cu_args = [cuda.as_cuda_array(v) for v in args]
        _cuda_kernel_blur_from_flow[grid_dim, block_dim](*cu_args, bilinear, pad_reflect, 2)
    else:
        args = [v.numpy() for v in args]
        _cpu_kernel_blur_from_flow(*args, bilinear, pad_reflect, 2)
    return out


@numba.jit
def _cpu_kernel_blur_from_flow(img, out, flow, bilinear, pad_reflect, f=2):
    height, width = img.shape
    for y in range(height):
        for x in range(width):
            u, v = flow[y, x, 0], flow[y, x, 1]
            mag = math.ceil(math.sqrt(u**2 + v**2))

            n = max(1, int(f*mag))

            value = img[y, x] / n
            for i in range(1, n):
                ux = x - u / n * i
                uy = y - v / n * i

                # we need a padding strategy
                # (reflect for instance if x > width => x' = width-x)
                if ux < 0 or ux >= width or uy < 0 or uy >= height:
                    if pad_reflect:
                        if ux < 0:
                            ux = abs(ux)
                        elif ux >= width:
                            ux = 2*(width)-ux-1

                        if uy < 0:
                            uy = abs(uy)
                        elif uy >= height:
                            uy = 2*(height)-uy-1
                    else:
                        continue

                if bilinear:
                    for lim_y in (math.floor(uy), math.ceil(uy)):
                        for lim_x in (math.floor(ux), math.ceil(ux)):
                            lim_y = int(lim_y)
                            lim_x = int(lim_x)
                            if lim_y >= height or lim_x >= width:
                                if pad_reflect:
                                    if lim_x < 0:
                                        lim_x = abs(lim_x)
                                    elif lim_x >= width:
                                        lim_x = 2*(width)-lim_x-1

                                    if lim_y < 0:
                                        lim_y = abs(lim_y)
                                    elif lim_y >= height:
                                        lim_y = 2*(height)-lim_y-1
                                else:
                                    continue
                            weight = (1-abs(lim_x-ux)) * (1-abs(lim_y-uy))
                            value += img[lim_y, lim_x] * weight / n
                else:
                    lim_y = round(uy)
                    lim_x = round(ux)
                    value += img[lim_y, lim_x] / n

            out[y, x] = value


@cuda.jit()
def _cuda_kernel_blur_from_flow(img, out, flow, bilinear, pad_reflect, f=2):
    y, x = cuda.grid(2)
    height, width = img.shape
    if y < height and x < width:
        """
        go in reverse of the flow line & select 4 neighbors with
        bilinear pooling
        """
        u, v = flow[y, x, 0], flow[y, x, 1]
        mag = math.ceil(math.sqrt(u**2 + v**2))

        n = int(f*mag)
        n = max(1, n)

        value = img[y, x] / n
        for i in range(1, n):
            ux = x - u / n * i
            uy = y - v / n * i

            # we need a padding strategy
            # (reflect for instance if x > width => x' = width-x)
            if ux < 0 or ux >= width or uy < 0 or uy >= height:
                if pad_reflect:
                    if ux < 0:
                        ux = abs(ux)
                    elif ux >= width:
                        ux = 2*width-1-ux

                    if uy < 0:
                        uy = abs(uy)
                    elif uy >= height:
                        uy = 2*height-1-uy
                else:
                    continue

            if bilinear:
                for lim_y in (math.floor(uy), math.ceil(uy)):
                    for lim_x in (math.floor(ux), math.ceil(ux)):
                        lim_y = int(lim_y)
                        lim_x = int(lim_x)
                        if lim_y >= height or lim_x >= width:
                            if pad_reflect:
                                if lim_x < 0:
                                    lim_x = abs(lim_x)
                                elif ux >= width:
                                    lim_x = 2*width-1-lim_x

                                if lim_y < 0:
                                    lim_y = abs(lim_y)
                                elif lim_y >= height:
                                    lim_y = 2*height-1-lim_y
                            else:
                                continue
                        weight = (1-abs(lim_x-ux)) * (1-abs(lim_y-uy))
                        value += img[lim_y, lim_x] * weight / n
            else:
                lim_y = round(uy)
                lim_x = round(ux)
                value += img[lim_y, lim_x] / n

        out[y, x] = value
