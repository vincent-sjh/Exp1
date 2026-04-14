import numpy as np
from PIL import Image


def imread(path):
    im = Image.open(path)
    if im.mode == "RGBA":
        im = im.convert("RGB")
    elif im.mode != "RGB":
        im = im.convert("RGB")
    arr = np.asarray(im, dtype=np.float64) / 255.0
    return arr


def imread_rgba(path):
    im = Image.open(path)
    if im.mode == "RGBA":
        arr = np.asarray(im, dtype=np.float64)
        rgb = arr[..., :3] / 255.0
        alpha = arr[..., 3] / 255.0
        return rgb, alpha
    rgb = np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0
    return rgb, None


def imread_mask01(path):
    im = Image.open(path)
    arr = np.asarray(im, dtype=np.float64)
    if arr.ndim == 2:
        m = arr / 255.0
    elif im.mode == "RGBA":
        rgb = arr[..., :3]
        m = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]) / 255.0
        m = m * (arr[..., 3] / 255.0)
    else:
        m = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        m = m / 255.0
    return m.astype(np.float64)


def imread_removal_mask01(path):
    # 返回“保留权重”[0,1]；物体移除里删除量 remove = 1 - m
    im = Image.open(path)
    arr = np.asarray(im, dtype=np.float64)
    if arr.ndim == 2:
        return (arr / 255.0).astype(np.float64)
    if im.mode == "RGBA":
        rgb = arr[..., :3]
        a = arr[..., 3] / 255.0
        lum = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]) / 255.0
        # 全透明当保留；有 alpha 时用 (1-a)+a*亮度
        return (1.0 - a + a * lum).astype(np.float64)
    m = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return (m / 255.0).astype(np.float64)


def imwrite(path, img):
    x = np.asarray(img)
    if x.dtype != np.uint8:
        x = np.clip(x * 255.0, 0.0, 255.0).round().astype(np.uint8)
    Image.fromarray(x, mode="RGB").save(path)


def imwrite_rgba(path, rgb, alpha=None):
    rgb_u8 = np.clip(rgb * 255.0, 0.0, 255.0).round().astype(np.uint8)
    if alpha is None:
        Image.fromarray(rgb_u8, mode="RGB").save(path)
    else:
        a = np.clip(alpha * 255.0, 0.0, 255.0).round().astype(np.uint8)
        rgba = np.concatenate([rgb_u8, a[..., None]], axis=-1)
        Image.fromarray(rgba, mode="RGBA").save(path)
