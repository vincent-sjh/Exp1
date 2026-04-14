import os

import numpy as np

# 5 点二项式核，归一化后相当于 Burt–Adelson 那套
_K5 = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float64) / 16.0


def _reflect_pad(img, top, bottom, left, right):
    return np.pad(img, ((top, bottom), (left, right), (0, 0)), mode="reflect")


def _conv_rows(img, k):
    r = k.shape[0] // 2
    padded = _reflect_pad(img, r, r, 0, 0)
    h, w, c = img.shape
    out = np.zeros((h, w, c), dtype=np.float64)
    i = 0
    for ki in k:
        out = out + ki * padded[i : i + h, :, :]
        i = i + 1
    return out


def _conv_cols(img, k):
    r = k.shape[0] // 2
    padded = _reflect_pad(img, 0, 0, r, r)
    h, w, c = img.shape
    out = np.zeros((h, w, c), dtype=np.float64)
    i = 0
    for ki in k:
        out = out + ki * padded[:, i : i + w, :]
        i = i + 1
    return out


def gaussian_blur(img):
    x = np.asarray(img, dtype=np.float64)
    x = _conv_rows(x, _K5)
    x = _conv_cols(x, _K5)
    return x


def pyramid_reduce(img):
    b = gaussian_blur(img)
    return b[::2, ::2, :]


def pyramid_expand(img, out_h, out_w):
    # 偶数格插零再模糊，乘 4 与 reduce 配套
    h, w, c = img.shape
    up = np.zeros((out_h, out_w, c), dtype=np.float64)
    up[::2, ::2, :] = img
    up = gaussian_blur(up) * 4.0
    return up


def gaussian_pyramid(img, levels):
    g = [np.asarray(img, dtype=np.float64)]
    for _ in range(levels - 1):
        g.append(pyramid_reduce(g[-1]))
    return g


def laplacian_pyramid_from_gaussian(G):
    L = []
    i = 0
    while i < len(G) - 1:
        gh, gw, _ = G[i].shape
        exp = pyramid_expand(G[i + 1], gh, gw)
        L.append(G[i] - exp)
        i = i + 1
    return L, G[-1]


def laplacian_pyramid(img, levels):
    G = gaussian_pyramid(img, levels)
    return laplacian_pyramid_from_gaussian(G)


def save_gaussian_pyramid_pngs(G, out_dir):
    from io_image import imwrite

    os.makedirs(out_dir, exist_ok=True)
    i = 0
    for g in G:
        path = os.path.join(out_dir, "gaussian_level_{:02d}.png".format(i))
        imwrite(path, np.clip(np.asarray(g, dtype=np.float64), 0.0, 1.0))
        i = i + 1


def save_laplacian_pyramid_pngs(L, out_dir):
    from io_image import imwrite

    os.makedirs(out_dir, exist_ok=True)
    i = 0
    for lev in L:
        x = np.asarray(lev, dtype=np.float64)
        lo = float(x.min())
        hi = float(x.max())
        if hi - lo < 1e-12:
            vis = np.zeros_like(x)
        else:
            vis = (x - lo) / (hi - lo)
        path = os.path.join(out_dir, "laplacian_level_{:02d}.png".format(i))
        imwrite(path, vis)
        i = i + 1


def reconstruct_from_laplacian(L, top_g):
    g = top_g
    i = len(L) - 1
    while i >= 0:
        li = L[i]
        h, w, _ = li.shape
        g = li + pyramid_expand(g, h, w)
        i = i - 1
    return g


def blend_laplacian(img_a, img_b, mask_01, levels):
    # 多频带融合：每一层用掩膜的高斯金字塔当权重
    if img_a.shape != img_b.shape:
        raise ValueError("两张图尺寸要一样")
    h, w, _ = img_a.shape
    if mask_01.shape != (h, w):
        raise ValueError("掩膜尺寸要和图像一致")

    GA = gaussian_pyramid(img_a, levels)
    GB = gaussian_pyramid(img_b, levels)
    m3 = mask_01[..., None].astype(np.float64)
    GM = gaussian_pyramid(m3, levels)

    LA, _ = laplacian_pyramid_from_gaussian(GA)
    LB, _ = laplacian_pyramid_from_gaussian(GB)

    LS = []
    j = 0
    while j < len(LA):
        la = LA[j]
        lb = LB[j]
        gm = GM[j]
        LS.append(la * gm + lb * (1.0 - gm))
        j = j + 1

    gma = GM[-1]
    G0_top = GA[-1] * gma + GB[-1] * (1.0 - gma)
    return reconstruct_from_laplacian(LS, G0_top)
