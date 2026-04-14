import numpy as np

ENERGY_GRADIENT = "gradient"
ENERGY_SALIENCY = "saliency"


def _luminance(rgb):
    x = np.asarray(rgb, dtype=np.float64)
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def energy_gradient_magnitude(rgb):
    L = _luminance(rgb)
    gx = np.zeros_like(L)
    gy = np.zeros_like(L)
    gx[:, 1:-1] = 0.5 * (L[:, 2:] - L[:, :-2])
    gx[:, 0] = L[:, 1] - L[:, 0]
    gx[:, -1] = L[:, -1] - L[:, -2]
    gy[1:-1, :] = 0.5 * (L[2:, :] - L[:-2, :])
    gy[0, :] = L[1, :] - L[0, :]
    gy[-1, :] = L[-1, :] - L[-2, :]
    return np.sqrt(gx * gx + gy * gy + 1e-12)


def energy_saliency_spectral_residual(rgb):
    L = _luminance(rgb)
    F = np.fft.fft2(L)
    A = np.abs(F) + 1e-8
    P = np.angle(F)
    log_A = np.log(A)
    # 对 log 幅度做 3x3 均值（谱残差显著性）
    Pd = np.pad(log_A, 1, mode="reflect")
    avg = (
        Pd[0:-2, 0:-2]
        + Pd[0:-2, 1:-1]
        + Pd[0:-2, 2:]
        + Pd[1:-1, 0:-2]
        + Pd[1:-1, 1:-1]
        + Pd[1:-1, 2:]
        + Pd[2:, 0:-2]
        + Pd[2:, 1:-1]
        + Pd[2:, 2:]
    ) / 9.0
    R = log_A - avg
    smap = np.fft.ifft2(np.exp(R + 1j * P))
    out = np.abs(smap) ** 2
    return np.maximum(out, 1e-12)


def energy_map(rgb, mode):
    if mode == ENERGY_GRADIENT:
        return energy_gradient_magnitude(rgb)
    if mode == ENERGY_SALIENCY:
        return energy_saliency_spectral_residual(rgb)
    raise ValueError("能量模式只能是 gradient 或 saliency")


def cumulative_map_vertical(energy):
    h, w = energy.shape
    M = np.empty_like(energy, dtype=np.float64)
    M[0] = energy[0]
    r = 1
    while r < h:
        prev = M[r - 1]
        left = np.roll(prev, 1)
        right = np.roll(prev, -1)
        left[0] = prev[0]
        right[-1] = prev[-1]
        M[r] = energy[r] + np.minimum(np.minimum(left, prev), right)
        r = r + 1
    return M


def backtrack_vertical_seam(M):
    h, w = M.shape
    seam = np.zeros(h, dtype=np.int32)
    seam[-1] = int(np.argmin(M[-1]))
    r = h - 2
    while r >= 0:
        c = seam[r + 1]
        c0 = max(0, c - 1)
        c1 = min(w - 1, c + 1)
        seam[r] = c0 + int(np.argmin(M[r, c0 : c1 + 1]))
        r = r - 1
    return seam


def remove_vertical_seam(rgb, seam):
    h, w, c = rgb.shape
    out = np.zeros((h, w - 1, c), dtype=np.float64)
    for r in range(h):
        col = int(seam[r])
        out[r, :, :] = np.delete(rgb[r], col, axis=0)
    return out


def remove_vertical_seam_2d(mask, seam):
    h, w = mask.shape
    out = np.zeros((h, w - 1), dtype=mask.dtype)
    for r in range(h):
        col = int(seam[r])
        out[r, :] = np.delete(mask[r], col, axis=0)
    return out


def vertical_seam_step(rgb, energy_scale=None, energy_mode=ENERGY_GRADIENT, energy_bias=None):
    e = energy_map(rgb, energy_mode)
    if energy_scale is not None:
        e = e * energy_scale
    if energy_bias is not None:
        e = e + energy_bias
    M = cumulative_map_vertical(e)
    seam = backtrack_vertical_seam(M)
    return remove_vertical_seam(rgb, seam), seam


def transpose_rgb(rgb):
    return np.transpose(rgb, (1, 0, 2))


def horizontal_seam_step(rgb, energy_scale=None, energy_mode=ENERGY_GRADIENT, energy_bias=None):
    t = transpose_rgb(rgb)
    if energy_scale is not None:
        scale_t = energy_scale.transpose()
    else:
        scale_t = None
    if energy_bias is not None:
        bias_t = energy_bias.transpose()
    else:
        bias_t = None
    t2, seam = vertical_seam_step(t, scale_t, energy_mode, bias_t)
    return transpose_rgb(t2), seam


def resize_by_seam_counts(rgb, vertical_seams, horizontal_seams, energy_mode=ENERGY_GRADIENT):
    if vertical_seams < 0 or vertical_seams > rgb.shape[1]:
        raise ValueError("竖直接缝数量不对")
    if horizontal_seams < 0 or horizontal_seams > rgb.shape[0]:
        raise ValueError("水平接缝数量不对")

    x = np.asarray(rgb, dtype=np.float64).copy()
    for _ in range(vertical_seams):
        x, _ = vertical_seam_step(x, None, energy_mode, None)
    for _ in range(horizontal_seams):
        x, _ = horizontal_seam_step(x, None, energy_mode, None)
    return x


def object_removal_by_seams(
    rgb,
    mask_remove,
    max_seams=2000,
    min_mask_sum=1.0,
    vertical_before_horizontal=1,
    mask_negative_pull=180.0,
    energy_mode=ENERGY_GRADIENT,
):
    # remove=1-m；能量乘 mod 再在删除区加负能量，让缝尽量走掩膜里
    n = vertical_before_horizontal
    if n < 1:
        n = 1
    cycle = n + 1

    x = np.asarray(rgb, dtype=np.float64).copy()
    m = np.asarray(mask_remove, dtype=np.float64)
    if m.shape != x.shape[:2]:
        raise ValueError("掩膜和图像宽高不一致")
    seams = 0
    step = 0
    eps = 0.001
    gamma = 5.0
    bg_boost = 0.85
    pull = float(mask_negative_pull)

    while seams < max_seams:
        remove = np.clip(1.0 - m, 0.0, 1.0)
        if float(np.sum(remove)) <= min_mask_sum:
            break
        if not np.any(remove > 0.08):
            break
        mod = eps + (1.0 - eps) * np.power(1.0 - remove, gamma)
        mod = mod * (1.0 + bg_boost * (1.0 - remove))
        mod = np.clip(mod, 1e-6, 12.0).astype(np.float64)
        energy_bias = (-pull * remove).astype(np.float64)
        want_vertical = (step % cycle) < n

        if want_vertical:
            if x.shape[1] >= 2:
                x, seam = vertical_seam_step(x, mod, energy_mode, energy_bias)
                m = remove_vertical_seam_2d(m, seam)
            elif x.shape[0] >= 2:
                x, seam = horizontal_seam_step(x, mod, energy_mode, energy_bias)
                m = remove_vertical_seam_2d(m.T, seam).T
            else:
                break
        else:
            if x.shape[0] >= 2:
                x, seam = horizontal_seam_step(x, mod, energy_mode, energy_bias)
                m = remove_vertical_seam_2d(m.T, seam).T
            elif x.shape[1] >= 2:
                x, seam = vertical_seam_step(x, mod, energy_mode, energy_bias)
                m = remove_vertical_seam_2d(m, seam)
            else:
                break
        seams = seams + 1
        step = step + 1
    return x
