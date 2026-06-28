"""
manual_review.py

Lets you quickly review overlay images one-by-one and rate whether the
highlighted area correctly marks the real affected area, recording your
judgment to a CSV.

CONTROLS (while the image window is focused):
    Y -> Correct     (highlight matches the real affected area well)
    N -> Incorrect   (highlight is in the wrong place / missed it entirely)
    P -> Partial      (highlight is close, but too small / too large / off)
    S -> Skip          (not sure right now, revisit later)
    Q -> Quit and save progress (safe to resume anytime)

USAGE:
    python manual_review.py "batch_results\\overlays"
    python manual_review.py "batch_results\\overlays" --output review_log.csv

RESUMING:
    If the output CSV already exists, images already reviewed in it are
    skipped automatically, so you can stop partway through and continue
    later without redoing work.
"""

import os
import csv
import argparse

import cv2

IMAGE_EXTS = {".png"}

KEY_MAP = {
    ord('y'): "Correct",
    ord('Y'): "Correct",
    ord('n'): "Incorrect",
    ord('N'): "Incorrect",
    ord('p'): "Partial",
    ord('P'): "Partial",
    ord('s'): "Skipped",
    ord('S'): "Skipped",
}
QUIT_KEYS = {ord('q'), ord('Q'), 27}  # 27 = ESC


def find_images(root_folder):
    """Walk root_folder recursively, yield (full_path, category, filename)."""
    for dirpath, _, filenames in os.walk(root_folder):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in IMAGE_EXTS:
                category = os.path.basename(dirpath)
                yield os.path.join(dirpath, fname), category, fname


def load_existing_reviews(csv_path):
    reviewed = set()
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                reviewed.add(row["full_path"])
    return reviewed


def append_review(csv_path, row, write_header):
    fieldnames = ["filename", "category", "full_path", "rating", "notes"]
    mode = "a" if os.path.exists(csv_path) else "w"
    with open(csv_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def print_summary(csv_path):
    if not os.path.exists(csv_path):
        return
    counts = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            counts[row["rating"]] = counts.get(row["rating"], 0) + 1
    total = sum(counts.values())
    if total == 0:
        return
    print("\nReview tally so far:")
    for rating, count in counts.items():
        pct = count / total * 100
        print(f"  {rating}: {count} ({pct:.1f}%)")


def run_review(overlays_folder, csv_path):
    images = list(find_images(overlays_folder))
    if not images:
        print(f"No images found under: {overlays_folder}")
        return

    already_reviewed = load_existing_reviews(csv_path)
    write_header = not os.path.exists(csv_path)

    todo = [img for img in images if img[0] not in already_reviewed]
    print(f"Total images: {len(images)} | Already reviewed: {len(already_reviewed)} | Remaining: {len(todo)}\n")

    if not todo:
        print("Nothing left to review!")
        print_summary(csv_path)
        return

    print("Controls: [Y]=Correct  [N]=Incorrect  [P]=Partial  [S]=Skip  [Q]=Quit & save\n")

    cv2.namedWindow("Manual Review", cv2.WINDOW_NORMAL)

    for i, (full_path, category, fname) in enumerate(todo, 1):
        img = cv2.imread(full_path)
        if img is None:
            print(f"[{i}/{len(todo)}] {category}/{fname} ... could not load, skipping")
            continue

        cv2.setWindowTitle("Manual Review", f"[{i}/{len(todo)}] {category}/{fname}  (Y/N/P/S, Q to quit)")
        cv2.imshow("Manual Review", img)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in KEY_MAP:
                rating = KEY_MAP[key]
                append_review(csv_path, {
                    "filename": fname, "category": category, "full_path": full_path,
                    "rating": rating, "notes": "",
                }, write_header)
                write_header = False
                print(f"[{i}/{len(todo)}] {category}/{fname} -> {rating}")
                break
            elif key in QUIT_KEYS:
                cv2.destroyAllWindows()
                print(f"\nStopped early. Progress saved to {csv_path}. Run again to resume.")
                print_summary(csv_path)
                return
            # else: ignore unrecognized key, keep waiting

    cv2.destroyAllWindows()
    print(f"\nAll done! Results saved to {csv_path}")
    print_summary(csv_path)


def main():
    parser = argparse.ArgumentParser(description="Manually review overlay images and rate highlight accuracy.")
    parser.add_argument("overlays_folder", help="Folder containing overlay images (e.g. batch_results/overlays)")
    parser.add_argument("--output", default="manual_review_log.csv", help="CSV file to save ratings to")
    args = parser.parse_args()

    run_review(args.overlays_folder, args.output)


if __name__ == "__main__":
    main()