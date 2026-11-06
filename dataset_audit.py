#!/usr/bin/env python3
"""
Dataset audit for two class folders, e.g.:
dataset/
    real_art_data/
    ai_images/

Creates:
  - image_audit.csv: one row per discovered image
  - class_counts.csv: image counts per class
  - dataset_summary.csv: min/median/max dimensions and file sizes per class + overall
  - duplicate_files.csv: all files belonging to exact-duplicate groups
  - audit_report.txt: corrupt/duplicate totals and run details
"""
from pathlib import Path
import argparse
import csv
import hashlib
import statistics
from collections import Counter, defaultdict

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Pillow is required. Install it with: python -m pip install Pillow")

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff",
    ".webp", ".ico", ".jfif", ".avif", ".heic"
}

def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()

def median_or_blank(values):
    return statistics.median(values) if values else ""

def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser(description="Audit image dataset folders.")
    parser.add_argument(
        "dataset_root",
        help="Root folder containing one subfolder per class, e.g. ./dataset"
    )
    parser.add_argument(
        "--classes", nargs="*", default=None,
        help="Optional class folder names. Default: use all immediate subfolders."
    )
    parser.add_argument(
        "--output", default="dataset_audit",
        help="Output directory for audit reports (default: dataset_audit)"
    )
    args = parser.parse_args()

    root = Path(args.dataset_root).expanduser().resolve()
    out_dir = Path(args.output).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Dataset root is not a directory: {root}")

    if args.classes:
        class_dirs = []
        for name in args.classes:
            folder = root / name
            if not folder.is_dir():
                raise SystemExit(f"Class folder not found: {folder}")
            class_dirs.append(folder)
    else:
        class_dirs = sorted(p for p in root.iterdir() if p.is_dir())

    if not class_dirs:
        raise SystemExit("No class subfolders found. Pass the parent folder or use --classes.")

    out_dir.mkdir(parents=True, exist_ok=True)
    image_rows = []
    class_counts = Counter()
    hash_to_rows = defaultdict(list)
    metric_values = defaultdict(lambda: {"width": [], "height": [], "file_size_bytes": []})
    corrupt_rows = []

    for class_dir in class_dirs:
        class_name = class_dir.name
        paths = sorted(
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        for path in paths:
            try:
                size = path.stat().st_size
            except OSError:
                size = ""

            row = {
                "class": class_name,
                "relative_path": str(path.relative_to(root)),
                "file_name": path.name,
                "file_size_bytes": size,
                "width": "",
                "height": "",
                "colour_mode": "",
                "format": "",
                "sha256": "",
                "status": "ok",
                "error": "",
            }

            try:
                # verify() detects many malformed/truncated image files.
                with Image.open(path) as im:
                    row["width"], row["height"] = im.size
                    row["colour_mode"] = im.mode
                    row["format"] = im.format or ""
                    im.verify()
                # Reopen and force decoding so truncated/corrupt pixel data is more likely to be caught.
                with Image.open(path) as im:
                    im.load()
            except Exception as exc:
                row["status"] = "corrupt_or_unreadable"
                row["error"] = f"{type(exc).__name__}: {exc}"
                corrupt_rows.append(row.copy())

            try:
                row["sha256"] = sha256_file(path)
            except Exception as exc:
                row["error"] = (row["error"] + " | " if row["error"] else "") + f"Hash error: {exc}"
                if row["status"] == "ok":
                    row["status"] = "read_error"

            image_rows.append(row)
            class_counts[class_name] += 1

            if row["sha256"]:
                hash_to_rows[row["sha256"]].append(row)

            if row["status"] == "ok":
                for metric in ("width", "height", "file_size_bytes"):
                    if row[metric] != "":
                        metric_values[class_name][metric].append(int(row[metric]))
                        metric_values["OVERALL"][metric].append(int(row[metric]))

    image_fields = [
        "class", "relative_path", "file_name", "file_size_bytes",
        "width", "height", "colour_mode", "format", "sha256", "status", "error"
    ]
    write_csv(out_dir / "image_audit.csv", image_rows, image_fields)

    count_rows = [
        {"class": name, "image_count": class_counts[name]}
        for name in sorted(class_counts)
    ]
    count_rows.append({"class": "OVERALL", "image_count": sum(class_counts.values())})
    write_csv(out_dir / "class_counts.csv", count_rows, ["class", "image_count"])

    summary_rows = []
    for class_name in [p.name for p in class_dirs] + ["OVERALL"]:
        values = metric_values[class_name]
        result = {"class": class_name}
        for metric in ("width", "height", "file_size_bytes"):
            vals = values[metric]
            result[f"{metric}_min"] = min(vals) if vals else ""
            result[f"{metric}_median"] = median_or_blank(vals)
            result[f"{metric}_max"] = max(vals) if vals else ""
            result[f"{metric}_valid_count"] = len(vals)
        summary_rows.append(result)
    summary_fields = ["class"]
    for metric in ("width", "height", "file_size_bytes"):
        summary_fields += [f"{metric}_min", f"{metric}_median", f"{metric}_max", f"{metric}_valid_count"]
    write_csv(out_dir / "dataset_summary.csv", summary_rows, summary_fields)

    duplicate_groups = {h: rows for h, rows in hash_to_rows.items() if len(rows) > 1}
    duplicate_rows = []
    for digest, rows in sorted(duplicate_groups.items()):
        for row in rows:
            duplicate_rows.append({
                "sha256": digest,
                "duplicate_group_size": len(rows),
                "class": row["class"],
                "relative_path": row["relative_path"],
                "file_size_bytes": row["file_size_bytes"],
                "status": row["status"],
            })
    write_csv(
        out_dir / "duplicate_files.csv",
        duplicate_rows,
        ["sha256", "duplicate_group_size", "class", "relative_path", "file_size_bytes", "status"]
    )

    total_images = len(image_rows)
    corrupt_count = sum(1 for r in image_rows if r["status"] == "corrupt_or_unreadable")
    read_error_count = sum(1 for r in image_rows if r["status"] == "read_error")
    duplicate_groups_count = len(duplicate_groups)
    duplicate_files_count = sum(len(rows) for rows in duplicate_groups.values())
    extra_duplicate_copies = sum(len(rows) - 1 for rows in duplicate_groups.values())

    report_lines = [
        "DATASET AUDIT REPORT",
        f"Dataset root: {root}",
        f"Classes scanned: {', '.join(p.name for p in class_dirs)}",
        f"Total image files discovered: {total_images}",
        f"Corrupt/unreadable image files: {corrupt_count}",
        f"Files with hash/read errors: {read_error_count}",
        f"Exact duplicate hash groups: {duplicate_groups_count}",
        f"Files in duplicate groups (includes first copy): {duplicate_files_count}",
        f"Extra duplicate copies beyond one per group: {extra_duplicate_copies}",
        "",
        "Files created:",
        "  image_audit.csv",
        "  class_counts.csv",
        "  dataset_summary.csv",
        "  duplicate_files.csv",
        "  audit_report.txt",
        "",
        "Notes:",
        "- Counts include all discovered files with recognized image extensions, including corrupt files.",
        "- Dimension/file-size summaries include only images successfully decoded by Pillow.",
        "- SHA-256 duplicates are byte-for-byte identical files; visually identical images re-encoded differently will not match.",
    ]
    (out_dir / "audit_report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\n".join(report_lines))
    print(f"\nOutput folder: {out_dir}")

if __name__ == "__main__":
    main()
