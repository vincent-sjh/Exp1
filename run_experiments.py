import argparse
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from io_image import imread, imread_mask01, imread_removal_mask01, imread_rgba, imwrite, imwrite_rgba  # noqa: E402
from pyramid import (  # noqa: E402
    blend_laplacian,
    gaussian_pyramid,
    laplacian_pyramid_from_gaussian,
    reconstruct_from_laplacian,
    save_gaussian_pyramid_pngs,
    save_laplacian_pyramid_pngs,
)
from seam_carving import (  # noqa: E402
    ENERGY_GRADIENT,
    ENERGY_SALIENCY,
    object_removal_by_seams,
    resize_by_seam_counts,
)

OUT = os.path.join(ROOT, "output")
_RULER = 64


def ensure_out():
    os.makedirs(OUT, exist_ok=True)


def _energy_label(energy):
    if energy == ENERGY_GRADIENT:
        return "梯度 (gradient)"
    if energy == ENERGY_SALIENCY:
        return "显著性 (saliency)"
    return str(energy)


def _report_start(title):
    print()
    print("=" * _RULER)
    print("  {}".format(title))
    print("-" * _RULER)


def _report_line(label, value):
    if label:
        print("  · {}：{}".format(label, value))
    else:
        print("    {}".format(value))


def _report_end():
    print("=" * _RULER)
    print("  ✅ 本任务已完成")


def task_pyramid_reconstruction(levels=6):
    ensure_out()
    inp = os.path.join(ROOT, "pictures/Image_Pyramid/Image_Reconstruction/building.jpg")
    img = imread(inp)
    G = gaussian_pyramid(img, levels)
    laplacian_levels, top_gaussian = laplacian_pyramid_from_gaussian(G)
    gauss_dir = os.path.join(OUT, "pyramid_gaussian")
    lap_dir = os.path.join(OUT, "pyramid_laplacian")
    save_gaussian_pyramid_pngs(G, gauss_dir)
    save_laplacian_pyramid_pngs(laplacian_levels, lap_dir)
    recon = reconstruct_from_laplacian(laplacian_levels, top_gaussian)
    err = float(np.mean((img - recon) ** 2))
    imwrite(os.path.join(OUT, "pyramid_reconstruction.png"), recon)
    recon_path = os.path.join(OUT, "pyramid_reconstruction.png")
    h, w = img.shape[0], img.shape[1]
    _report_start("【1】金字塔重建（Laplacian 重建误差）")
    _report_line("输入", inp)
    _report_line("金字塔层数", str(levels))
    _report_line("原图尺寸", "{} × {}（高 × 宽）".format(h, w))
    _report_line("重建 MSE", "{:.6e}（越小越好）".format(err))
    print("-" * _RULER)
    _report_line("重建图", recon_path)
    _report_line("高斯金字塔", gauss_dir + os.sep)
    _report_line("拉普拉斯金字塔", lap_dir + os.sep)
    _report_end()


def task_pyramid_blend(levels=6):
    ensure_out()
    apple = os.path.join(ROOT, "pictures/Image_Pyramid/Image_Blending/apple.jpg")
    orange = os.path.join(ROOT, "pictures/Image_Pyramid/Image_Blending/orange.jpg")
    maskp = os.path.join(ROOT, "pictures/Image_Pyramid/Image_Blending/mask.png")
    a = imread(apple)
    b = imread(orange)
    m = imread_mask01(maskp)
    blended = blend_laplacian(a, b, m, levels)
    out_path = os.path.join(OUT, "pyramid_blend.png")
    imwrite(out_path, blended)
    ha, wa = a.shape[0], a.shape[1]
    _report_start("【2】金字塔融合（多分辨率混合）")
    _report_line("图 A", apple)
    _report_line("图 B", orange)
    _report_line("融合掩膜", maskp)
    _report_line("金字塔层数", str(levels))
    _report_line("尺寸", "{} × {}（高 × 宽）".format(ha, wa))
    print("-" * _RULER)
    _report_line("输出", out_path)
    _report_end()


def task_seam_resize(vertical_seams, horizontal_seams, energy):
    ensure_out()
    inp = os.path.join(ROOT, "pictures/Seam_Carving/Image_Resizing/Rider.png")
    rgb, _ = imread_rgba(inp)
    out = resize_by_seam_counts(rgb, vertical_seams, horizontal_seams, energy)
    outp = os.path.join(OUT, "seam_resize.png")
    imwrite_rgba(outp, out, None)
    hi, wi = rgb.shape[0], rgb.shape[1]
    ho, wo = out.shape[0], out.shape[1]
    _report_start("【3】接缝缩放（Seam Carving）")
    _report_line("输入", inp)
    _report_line("删除竖直接缝", str(vertical_seams))
    _report_line("删除水平接缝", str(horizontal_seams))
    _report_line("能量", _energy_label(energy))
    print("-" * _RULER)
    _report_line("尺寸变化", "{}×{} → {}×{}（高×宽）".format(hi, wi, ho, wo))
    _report_line("输出", outp)
    _report_end()


DEFAULT_REMOVAL_MASK = os.path.join(
    ROOT, "pictures/Seam_Carving/Object_Removal/only_mask.png"
)


def task_seam_remove(mask_path, max_seams, removal_vh_ratio, energy):
    ensure_out()
    inp = os.path.join(ROOT, "pictures/Seam_Carving/Object_Removal/Couple.png")
    rgb = imread(inp)
    if mask_path:
        resolved = mask_path
    else:
        resolved = DEFAULT_REMOVAL_MASK
    if not os.path.isfile(resolved):
        print("", file=sys.stderr)
        print("❌ 找不到掩膜：{}".format(resolved), file=sys.stderr)
        print("  请准备掩膜文件，或使用参数 --mask 指定路径（黑=删，白=留）", file=sys.stderr)
        print("", file=sys.stderr)
        sys.exit(1)

    m = imread_removal_mask01(resolved)
    if m.shape != rgb.shape[:2]:
        raise SystemExit("掩膜尺寸 {} 与图像 {} 不一致".format(m.shape, rgb.shape[:2]))
    out = object_removal_by_seams(
        rgb,
        m,
        max_seams,
        vertical_before_horizontal=removal_vh_ratio,
        energy_mode=energy,
    )
    outp = os.path.join(OUT, "seam_object_removal.png")
    imwrite(outp, out)
    hi, wi = rgb.shape[0], rgb.shape[1]
    ho, wo = out.shape[0], out.shape[1]
    _report_start("【4】物体移除（基于接缝）")
    _report_line("输入图像", inp)
    _report_line("掩膜文件", resolved)
    _report_line("最多接缝数", str(max_seams))
    _report_line("竖:横 节奏", "每轮 {} 条竖缝 + 1 条横缝".format(removal_vh_ratio))
    _report_line("能量", _energy_label(energy))
    print("-" * _RULER)
    _report_line("尺寸变化", "{}×{} → {}×{}（高×宽）".format(hi, wi, ho, wo))
    _report_line("输出", outp)
    _report_end()


def main():
    p = argparse.ArgumentParser(description="DIP EXP1 RUNNER")
    p.add_argument(
        "task",
        choices=["all", "pyramid_recon", "pyramid_blend", "seam_resize", "seam_remove"],
    )
    p.add_argument("--levels", type=int, default=6, help="金字塔层数")
    p.add_argument("--vertical_seams", type=int, default=50, help="删掉的竖缝条数")
    p.add_argument("--horizontal_seams", type=int, default=0, help="删除的水平缝条数")
    p.add_argument(
        "--energy",
        type=str,
        choices=[ENERGY_GRADIENT, ENERGY_SALIENCY],
        default=ENERGY_SALIENCY,
        help="接缝能量计算方式:gradient=亮度梯度模长,saliency=谱残差显著性",
    )
    p.add_argument(
        "--mask",
        type=str,
        default=os.path.join(ROOT, "pictures/Seam_Carving/Object_Removal/mask.png"),
        help="mask文件路径",
    )
    p.add_argument("--max_seams", type=int, default=69, help="物体移除最多删多少条接缝")
    p.add_argument(
        "--removal_vh_ratio",
        type=int,
        default=40,
        metavar="N",
        help="物体移除时,每轮先删 N 条竖接缝再删1条横接缝",
    )
    args = p.parse_args()

    if args.task == "all":
        print()
        print("=" * _RULER)
        print("  ✅ DIP 实验 · 顺序执行四个子任务（依次 4 项）")
        print("=" * _RULER)
        print()

    if args.task in ("all", "pyramid_recon"):
        task_pyramid_reconstruction(args.levels)
    if args.task in ("all", "pyramid_blend"):
        task_pyramid_blend(args.levels)
    if args.task in ("all", "seam_resize"):
        task_seam_resize(args.vertical_seams, args.horizontal_seams, args.energy)
    if args.task in ("all", "seam_remove"):
        task_seam_remove(args.mask, args.max_seams, args.removal_vh_ratio, args.energy)

    if args.task == "all":
        print()
        print("─" * _RULER)
        print("  ✅ 全部完成 · 结果目录")
        _report_line("路径", OUT)
        print("─" * _RULER)
        print()


if __name__ == "__main__":
    main()
