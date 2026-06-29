import os
import sys
import csv
import time
import argparse

import cv2

# If Puspin_v1.py is NOT in the same folder as this script, uncomment and edit:
# sys.path.append(r"F:\path\to\folder\containing\Puspin_v1")

from Puspin_v1 import detect

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

def find_images(root_folder):
    """Walk root_folder recursively, yield (full_path, category, filename)."""
    for dirpath, _, filenames in os.walk(root_folder):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMG_EXTS:
                full_path = os.path.join(dirpath, fname)
                category = os.path.basename(dirpath)
                yield full_path, category, fname


def run_batch(dataset_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    csv_path = os.path.join(output_folder, "results.csv")

    images = list(find_images(dataset_folder))
    total = len(images)
    if total == 0:
        print(f"No images found under: {dataset_folder}")
        return

    print(f"Found {total} image(s). Starting batch test...\n")

    rows = []
    errors = []
    risk_counts = {"High": 0, "Low": 0}
    start_time = time.time()

    for i, (full_path, category, fname) in enumerate(images, 1):
        print(f"[{i}/{total}] {category}/{fname} ... ", end="", flush=True)
        t0 = time.time()
        try:
            img = cv2.imread(full_path)
            if img is None:
                raise ValueError("cv2.imread returned None (unreadable/corrupt file)")

            result = detect(img)
            elapsed = time.time() - t0

            risk_counts[result.risk_level] = risk_counts.get(result.risk_level, 0) + 1

            rows.append({
                "filename": fname,
                "category": category,
                "full_path": full_path,
                "risk_level": result.risk_level,
                "redness_score": f"{result.redness_score * 100:.2f}%",
                "bald_area_ratio": f"{result.bald_area_ratio * 100:.2f}%",
                "lesion_circularity": result.lesion_circularity,
                "affected_pixel_count": result.affected_pixel_count,
                "stat_base_redness": result.metrics.get("stat_base_redness"),
                "stat_std_redness": result.metrics.get("stat_std_redness"),
                "processing_time_sec": round(elapsed, 3),
                "error": "",
            })

            # Save overlay image, mirroring category subfolder
            out_subfolder = os.path.join(output_folder, "overlays", category)
            os.makedirs(out_subfolder, exist_ok=True)
            out_path = os.path.join(out_subfolder, fname)
            cv2.imwrite(out_path, result.overlay_image)

            print(f"done ({result.risk_level}, {elapsed:.2f}s)")

        except Exception as e:
            elapsed = time.time() - t0
            errors.append((full_path, str(e)))
            rows.append({
                "filename": fname,
                "category": category,
                "full_path": full_path,
                "risk_level": "",
                "redness_score": "",
                "bald_area_ratio": "",
                "lesion_circularity": "",
                "affected_pixel_count": "",
                "stat_base_redness": "",
                "stat_std_redness": "",
                "processing_time_sec": round(elapsed, 3),
                "error": str(e),
            })
            print(f"ERROR: {e}")

    # Write CSV
    fieldnames = [
        "filename", "category", "full_path", "risk_level", "redness_score",
        "bald_area_ratio", "lesion_circularity", "affected_pixel_count",
        "stat_base_redness", "stat_std_redness", "processing_time_sec", "error",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total_time = time.time() - start_time

    # Summary
    print("\n" + "=" * 50)
    print("BATCH TEST SUMMARY")
    print("=" * 50)
    print(f"Total images:     {total}")
    print(f"Successful:       {total - len(errors)}")
    print(f"Errors:           {len(errors)}")
    for level, count in risk_counts.items():
        print(f"  Risk={level}: {count}")
    print(f"Total time:       {total_time:.2f}s")
    print(f"Avg time/image:   {total_time / total:.2f}s")
    print(f"\nResults CSV:      {csv_path}")
    print(f"Overlay images:   {os.path.join(output_folder, 'overlays')}")

    if errors:
        print("\nFiles that failed:")
        for path, err in errors:
            print(f"  - {path}: {err}")


def main():
    parser = argparse.ArgumentParser(description="Batch-test Puspin_v1.detect() over a folder of images.")
    parser.add_argument("dataset_folder", help="Folder containing images (searched recursively)")
    parser.add_argument("--output", default="batch_results", help="Folder to write results.csv and overlays into")
    args = parser.parse_args()

    run_batch(args.dataset_folder, args.output)


if __name__ == "__main__":
    main()