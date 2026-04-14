import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

from io_image import imread, imread_removal_mask01, imwrite


def main():
    p = argparse.ArgumentParser(description="检查掩膜和原图是否对齐")
    p.add_argument(
        "--couple",
        default=os.path.join(ROOT, "pictures/Seam_Carving/Object_Removal/Couple.png"),
    )
    p.add_argument(
        "--mask",
        default=os.path.join(ROOT, "pictures/Seam_Carving/Object_Removal/mask.png"),
    )
    p.add_argument(
        "--output",
        default=os.path.join(ROOT, "output/couple_mask_overlay.png"),
    )
    p.add_argument(
        "--strength",
        type=float,
        default=0.5,
        help="红色最大透明度，0~1",
    )
    args = p.parse_args()

    img = imread(args.couple)
    keep = imread_removal_mask01(args.mask)
    if keep.shape != img.shape[:2]:
        print(
            "尺寸不一致: 图 {} 掩膜 {}".format(img.shape[:2], keep.shape),
            file=sys.stderr,
        )
        sys.exit(1)

    remove = np.clip(1.0 - keep, 0.0, 1.0)
    strength = float(np.clip(args.strength, 0.0, 1.0))
    alpha = remove[..., None] * strength
    alpha = alpha.astype(np.float64)
    red = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    out = img * (1.0 - alpha) + red * alpha

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir == "":
        out_dir = "."
    os.makedirs(out_dir, exist_ok=True)
    imwrite(args.output, out)
    print("已写入 {}".format(args.output))


if __name__ == "__main__":
    main()
