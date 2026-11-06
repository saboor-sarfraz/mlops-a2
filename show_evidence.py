
import csv
from pathlib import Path

REPORT_DIR = Path("dataset_audit")

def read_csv(filename):
    with open(REPORT_DIR / filename, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def heading(text):
    print("\n" + "=" * 76)
    print(text)
    print("=" * 76)

# 1. Image counts per class
heading("EVIDENCE 1: IMAGE COUNTS PER CLASS")
for row in read_csv("class_counts.csv"):
    print(f"{row['class']:<35} {row['image_count']:>10} images")

# 2. Min, median, max statistics
heading("EVIDENCE 2: IMAGE DIMENSIONS AND FILE SIZE STATISTICS")
for row in read_csv("dataset_summary.csv"):
    print(f"\nClass: {row['class']}")
    for metric, label in [
        ("width", "Width (pixels)"),
        ("height", "Height (pixels)"),
        ("file_size_bytes", "File size (bytes)")
    ]:
        print(
            f"  {label:<22} "
            f"Min: {row[metric + '_min']:>12}  "
            f"Median: {row[metric + '_median']:>12}  "
            f"Max: {row[metric + '_max']:>12}"
        )

# 3. Corrupt files and duplicate counts
heading("EVIDENCE 3: CORRUPT FILES AND EXACT DUPLICATES")
report_path = REPORT_DIR / "audit_report.txt"
print(report_path.read_text(encoding="utf-8"))

# 4. List duplicate files, if any
duplicates = read_csv("duplicate_files.csv")
heading("EVIDENCE 4: EXACT DUPLICATE FILES")
if duplicates:
    for row in duplicates:
        print(
            f"Group size: {row['duplicate_group_size']} | "
            f"Class: {row['class']} | "
            f"File: {row['relative_path']}"
        )
else:
    print("No exact duplicate files detected.")

print("\nEnd of dataset audit evidence.")
