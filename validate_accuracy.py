"""
validate_accuracy.py

Validates Puspin_v1.detect() accuracy against ground-truth labels.

HOW GROUND TRUTH WORKS HERE:
    Your dataset is already organized into folders by condition, e.g.:

        Dataset/
            Healthy/
                img1.png
            Allergy dermatitis/
                img2.png
            Ringworm/
                img3.png

    Since detect() only outputs a binary risk_level ("High" / "Low"),
    this script needs to know which folder names should map to "High"
    (abnormality expected) and which map to "Low" (healthy expected).

    >>> EDIT CATEGORY_LABELS BELOW TO MATCH YOUR ACTUAL FOLDER NAMES <<<

USAGE:
    python validate_accuracy.py "F:\\path\\to\\Dataset"
    python validate_accuracy.py "F:\\path\\to\\Dataset" --output "F:\\path\\to\\results"
"""

import os
import csv
import time
import argparse

import cv2

from Puspin_v1 import detect

IMAGE_EXTS = {".png"}

# ============================================================
# EDIT THIS: map every dataset subfolder name to the expected
# ground-truth label. "High" = abnormality should be detected.
# "Low" = should come back healthy/clear.
# ============================================================
CATEGORY_LABELS = {
    "Normal": "Low",
    "Allergy dermatitis": "High",
    "Feline Acne": "High",
    "Ringworm":  "High",
    "Flea Allergy Dermatitis":  "High",
    "Superficial Pyoderma": "High"

    # add every category folder you actually have...
}


def find_images(root_folder):
    """Walk root_folder recursively, yield (full_path, category, filename)."""
    for dirpath, _, filenames in os.walk(root_folder):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMAGE_EXTS:
                category = os.path.basename(dirpath)
                yield os.path.join(dirpath, fname), category, fname


def run_validation(dataset_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    csv_path = os.path.join(output_folder, "validation_results.csv")

    images = list(find_images(dataset_folder))
    total = len(images)
    if total == 0:
        print(f"No images found under: {dataset_folder}")
        return

    print(f"Found {total} image(s). Starting validation...\n")

    rows = []
    skipped_unknown_category = []
    tp = tn = fp = fn = 0
    start = time.time()

    for i, (full_path, category, fname) in enumerate(images, 1):
        expected = CATEGORY_LABELS.get(category)
        if expected is None:
            skipped_unknown_category.append((category, fname))
            print(f"[{i}/{total}] {category}/{fname} ... SKIPPED (category not in CATEGORY_LABELS)")
            continue

        print(f"[{i}/{total}] {category}/{fname} ... ", end="", flush=True)
        try:
            img = cv2.imread(full_path)
            if img is None:
                raise ValueError("cv2.imread returned None (unreadable/corrupt file)")

            result = detect(img)
            predicted = result.risk_level
            correct = (predicted == expected)

            if expected == "High" and predicted == "High":
                tp += 1
            elif expected == "Low" and predicted == "Low":
                tn += 1
            elif expected == "Low" and predicted == "High":
                fp += 1
            elif expected == "High" and predicted == "Low":
                fn += 1

            rows.append({
                "filename": fname,
                "category": category,
                "expected_label": expected,
                "predicted_label": predicted,
                "correct": correct,
                "redness_score": f"{result.redness_score*100:.2f}%",
                "bald_area_ratio": f"{result.bald_area_ratio*100:.2f}%",
                "lesion_circularity": result.lesion_circularity,
                "affected_pixel_count": result.affected_pixel_count,
                "error": "",
            })
            print(f"{'OK' if correct else 'WRONG'} (expected={expected}, predicted={predicted})")

        except Exception as e:
            rows.append({
                "filename": fname, "category": category, "expected_label": expected,
                "predicted_label": "", "correct": "", "redness_score": "",
                "bald_area_ratio": "", "lesion_circularity": "", "affected_pixel_count": "",
                "error": str(e),
            })
            print(f"ERROR: {e}")

    fieldnames = ["filename", "category", "expected_label", "predicted_label", "correct",
                  "redness_score", "bald_area_ratio", "lesion_circularity",
                  "affected_pixel_count", "error"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total_evaluated = tp + tn + fp + fn
    accuracy = (tp + tn) / total_evaluated if total_evaluated else 0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0   # recall / true positive rate
    specificity = tn / (tn + fp) if (tn + fp) else 0   # true negative rate
    precision = tp / (tp + fp) if (tp + fp) else 0
    f1 = (2 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) else 0

    elapsed = time.time() - start

    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    print(f"Total images evaluated: {total_evaluated}")
    print(f"Skipped (unknown category): {len(skipped_unknown_category)}")
    print()
    print("Confusion Matrix:")
    print(f"  True Positive  (High correctly detected):  {tp}")
    print(f"  True Negative  (Low correctly detected):   {tn}")
    print(f"  False Positive (Low misdetected as High):  {fp}")
    print(f"  False Negative (High misdetected as Low):  {fn}")
    print()
    print(f"Accuracy:     {accuracy:.2%}")
    print(f"Sensitivity:  {sensitivity:.2%}  (catches real abnormalities)")
    print(f"Specificity:  {specificity:.2%}  (correctly clears healthy cases)")
    print(f"Precision:    {precision:.2%}")
    print(f"F1 Score:     {f1:.2%}")
    print(f"\nTotal time: {elapsed:.2f}s")
    print(f"Results CSV: {csv_path}")

    if skipped_unknown_category:
        print("\nWARNING: some images were skipped because their folder name")
        print("isn't listed in CATEGORY_LABELS. Add these to the dict:")
        for c in sorted(set(c for c, _ in skipped_unknown_category)):
            print(f"  - \"{c}\"")


def main():
    parser = argparse.ArgumentParser(description="Validate Puspin_v1.detect() accuracy against ground-truth labels.")
    parser.add_argument("dataset_folder", help="Folder containing labeled images (organized by category subfolders)")
    parser.add_argument("--output", default="validation_results", help="Folder to write validation_results.csv into")
    args = parser.parse_args()

    run_validation(args.dataset_folder, args.output)


if __name__ == "__main__":
    main()
