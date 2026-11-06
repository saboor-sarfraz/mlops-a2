#!/usr/bin/env python3
"""
Visual exploratory data analysis for a two-class image dataset.

Example:
  python visual_eda.py "C:\\path\\to\\dataset" --classes "real art data" "ai images"

Outputs PNG figures suitable for a report, plus a text file with short interpretations.
Requires: Pillow, matplotlib, numpy
"""
from pathlib import Path
import argparse
import random
import textwrap
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff",
    ".webp", ".ico", ".jfif", ".avif", ".heic"
}

# Written interpretations (Task 1.3). They are tied to this dataset's audit numbers
# (see dataset_audit/dataset_summary.csv) and replace the generic template captions.
WRITTEN_INTERPRETATIONS = [
    "Figure 1 - Sample images by class: Each row shows 12 random images from one class (the class "
    "name is printed at the left of the row). Both classes mix portraits, animals, landscapes and "
    "illustration-like pictures, so subject matter alone does not separate them, although the AI row "
    "leans towards polished, highly stylised scenes. Image shapes vary a lot between portrait, "
    "landscape and square, so the pipeline must decide how to handle aspect ratio.",
    "Figure 2 - Width distribution by class: Both classes are right-skewed. RealArt has a median width "
    "of 940 px and a long tail up to 8495 px, while AiArtData has a median of 1024 px, a tighter spread "
    "and a maximum of 6000 px. Most images are under about 2000 px wide, so resizing to a small fixed "
    "input size will almost always shrink them, and the few very wide images will not matter.",
    "Figure 3 - Height distribution by class: The pattern mirrors width: RealArt peaks at lower heights "
    "(median 720 px) with outliers up to 7091 px, whereas AiArtData is more bell-shaped around 500 to "
    "1000 px (median 800 px) and stops near 4001 px. The two classes overlap heavily, which is good "
    "because image size alone should not be a strong clue to the label.",
    "Figure 4 - File size distribution by class: Both histograms are strongly right-skewed with most "
    "files below about 500 KB; the median is about 145 KB for RealArt and 200 KB for AiArtData, and the "
    "largest files are 10.2 MB and 5.6 MB. Because file size and resolution differ slightly between the "
    "classes, a model could learn these shortcuts instead of real content, so all images are resized and "
    "re-encoded to the same shape before training.",
]


def scan_class(folder):
    rows = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        try:
            size = path.stat().st_size
            with Image.open(path) as im:
                width, height = im.size
                im.load()
                preview = ImageOps.exif_transpose(im).convert("RGB").copy()
            rows.append({
                "path": path, "size": size, "width": width,
                "height": height, "preview": preview
            })
        except Exception as exc:
            print(f"Skipping unreadable image: {path} ({exc})")
    return rows

def format_bytes(n):
    units = ["B", "KB", "MB", "GB"]
    value = float(n)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", help="Parent folder containing class subfolders")
    parser.add_argument("--classes", nargs="*", default=None,
                        help="Class folder names; by default all immediate subfolders")
    parser.add_argument("--samples", type=int, default=12,
                        help="Number of sample images to show per class (default 12)")
    parser.add_argument("--output", default="visual_eda",
                        help="Output folder for figures (default visual_eda)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.dataset_root).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Dataset root does not exist: {root}")
    if args.classes:
        folders = [root / name for name in args.classes]
        missing = [str(p) for p in folders if not p.is_dir()]
        if missing:
            raise SystemExit("Class folder(s) not found: " + ", ".join(missing))
    else:
        folders = sorted(p for p in root.iterdir() if p.is_dir())
    if not folders:
        raise SystemExit("No class subfolders found.")
    out.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    all_rows = {}
    for folder in folders:
        all_rows[folder.name] = scan_class(folder)
        print(f"{folder.name}: {len(all_rows[folder.name])} readable images")

    # Figure 1: sample image grid per class
    n_classes = len(folders)
    samples = max(1, args.samples)
    fig, axes = plt.subplots(
        n_classes, samples, figsize=(samples * 1.65, n_classes * 2.15),
        squeeze=False
    )
    for r, folder in enumerate(folders):
        rows = all_rows[folder.name]
        chosen = random.sample(rows, min(samples, len(rows))) if rows else []
        for c in range(samples):
            ax = axes[r][c]
            ax.axis("off")
            if c < len(chosen):
                ax.imshow(chosen[c]["preview"])
                if c == 0:
                    # axis('off') hides set_ylabel, so draw the class name as plain text
                    # in the left margin of the row (this was missing in the first run).
                    ax.text(-0.08, 0.5, folder.name, transform=ax.transAxes,
                            rotation=90, ha="right", va="center",
                            fontsize=11, fontweight="bold")
            elif c == 0:
                ax.text(0.5, 0.5, "No readable images", ha="center", va="center")
        # class label is in the left margin via ylabel
    fig.suptitle("Representative Image Samples by Class", fontsize=15)
    fig.tight_layout(rect=[0.03, 0, 1, 0.95])
    sample_path = out / "figure_1_sample_images.png"
    fig.savefig(sample_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Histograms: width, height, and file size by class.
    metrics = [
        ("width", "Image width (pixels)", "Width"),
        ("height", "Image height (pixels)", "Height"),
        ("size", "File size (bytes)", "File size"),
    ]
    interpretation = []
    interpretation.append(
        "Figure 1 — Sample images by class: The grid displays randomly selected readable images "
        "from each class to allow a visual check of image content, variety, and obvious quality issues. "
        "Use it to describe visible patterns only; do not claim that the classes are separable from appearance alone."
    )

    for idx, (key, xlabel, label) in enumerate(metrics, start=2):
        fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 4.2), squeeze=False)
        for col, folder in enumerate(folders):
            ax = axes[0][col]
            vals = [row[key] for row in all_rows[folder.name]]
            if vals:
                if key == "size":
                    # Bytes are strongly skewed often; plot in KB for readability.
                    plot_vals = np.array(vals, dtype=float) / 1024
                    ax.hist(plot_vals, bins=30, edgecolor="black", alpha=0.8)
                    ax.set_xlabel("File size (KB)")
                else:
                    ax.hist(vals, bins=30, edgecolor="black", alpha=0.8)
                    ax.set_xlabel(xlabel)
                ax.set_ylabel("Number of images")
                ax.set_title(f"{folder.name} (n={len(vals)})")
                ax.grid(axis="y", alpha=0.25)
            else:
                ax.text(0.5, 0.5, "No readable images", ha="center", va="center")
                ax.set_title(folder.name)
        fig.suptitle(f"Distribution of {label} by Class", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        filename = f"figure_{idx}_{key}_histograms.png"
        fig.savefig(out / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)

        if key == "size":
            unit_note = "File size is shown in kilobytes (KB) for readability."
        else:
            unit_note = f"{label} is measured in pixels."
        interpretation.append(
            f"Figure {idx} — {label} distribution by class: The histograms show how {label.lower()} "
            f"values are distributed within each class, making spread, skew, and potential outliers easier to inspect. "
            f"{unit_note} Compare the two panels to see whether the classes differ in this property; "
            f"the plot alone does not establish why any difference exists."
        )

    interpretation = WRITTEN_INTERPRETATIONS
    notes = [
        "VISUAL EDA: FIGURE CAPTIONS AND INTERPRETATIONS",
        "",
        *[f"{i+1}. {line}" for i, line in enumerate(interpretation)],
        "",
        "Method note: Only images successfully opened and decoded by Pillow are included in the visual grid and histograms. "
        "Sample images are selected randomly using a fixed seed (42 by default) for reproducibility.",
        f"Dataset root: {root}",
        "Classes: " + ", ".join(folder.name for folder in folders),
        "",
        "Figures saved:",
        "  figure_1_sample_images.png",
        "  figure_2_width_histograms.png",
        "  figure_3_height_histograms.png",
        "  figure_4_size_histograms.png",
    ]
    (out / "figure_interpretations.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print("\nFigures saved to:", out)
    for p in sorted(out.glob("*.png")):
        print(" -", p.name)
    print(" - figure_interpretations.txt")

if __name__ == "__main__":
    main()
