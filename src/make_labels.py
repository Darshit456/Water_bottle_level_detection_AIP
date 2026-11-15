r"""
make_labels.py
Reads the photo folders and creates labels.csv.

Expected folder structure (path set in config.py):
    D:\AI_Project\Bottle_dataset\Bottle_A\0000gm\photo1.jpg
    D:\AI_Project\Bottle_dataset\Bottle_A\0012gm\photo2.jpg
    D:\AI_Project\Bottle_dataset\Bottle_B\...
First folder = bottle (Bottle_A -> A), second folder = water weight in grams.

Run from the project folder:
    python src/make_labels.py
"""
import re

import pandas as pd

from config import DATA_DIR, LABELS_CSV

# ---------- Settings ----------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
TEST_EVERY_NTH_LEVEL = 5      # every 5th weight level goes to test (about 20%)
MIN_PHOTOS_PER_LEVEL = 3      # warn if a level has fewer photos than this


def parse_bottle_id(folder_name):
    """'Bottle_A' -> 'A'. Any other name is kept as it is."""
    return folder_name.split("_")[-1]


def parse_grams(folder_name):
    """'0012gm' -> 12. Returns None if the folder name has no number."""
    match = re.search(r"(\d+)", folder_name)
    return int(match.group(1)) if match else None


def collect_rows():
    """Walk through the dataset folder and make one row per photo."""
    rows, problems = [], []
    for bottle_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        bottle_id = parse_bottle_id(bottle_dir.name)
        for level_dir in sorted(p for p in bottle_dir.iterdir() if p.is_dir()):
            grams = parse_grams(level_dir.name)
            if grams is None:
                problems.append(f"Skipped folder (no number in name): {level_dir}")
                continue

            files = [p for p in level_dir.iterdir() if p.is_file()]
            images = [p for p in files if p.suffix.lower() in IMAGE_EXTENSIONS]
            others = [p for p in files if p.suffix.lower() not in IMAGE_EXTENSIONS]

            if others:
                problems.append(f"{len(others)} non-image file(s) ignored in {level_dir} "
                                f"(e.g. {others[0].name})")
            if len(images) < MIN_PHOTOS_PER_LEVEL:
                problems.append(f"Only {len(images)} photo(s) in {level_dir}")

            for img in sorted(images):
                rows.append({
                    # path relative to DATA_DIR, so the dataset can be moved
                    "filename": img.relative_to(DATA_DIR).as_posix(),
                    "bottle_id": bottle_id,
                    "water_g": grams,
                })
    return pd.DataFrame(rows), problems


def assign_split(df):
    """
    Split by weight LEVEL, not by single photo.
    All photos of one level go to the same split, otherwise the test
    is too easy (almost identical photos in train and test).
    Empty and full levels always stay in train.
    """
    df["split"] = "train"
    for bottle_id, group in df.groupby("bottle_id"):
        levels = sorted(group["water_g"].unique())
        inner = levels[1:-1]                          # without empty and full
        test_levels = inner[2::TEST_EVERY_NTH_LEVEL]  # spread over the range
        is_test = (df["bottle_id"] == bottle_id) & (df["water_g"].isin(test_levels))
        df.loc[is_test, "split"] = "test"
    return df


def print_summary(df):
    print("\nPer bottle:")
    for bottle_id, group in df.groupby("bottle_id"):
        levels = sorted(group["water_g"].unique())
        gaps = [b - a for a, b in zip(levels, levels[1:])]
        n_test = group.loc[group["split"] == "test", "water_g"].nunique()
        line = (f"  {bottle_id}: {len(group)} photos, {len(levels)} levels "
                f"({levels[0]}-{levels[-1]} g), test levels: {n_test}")
        if gaps:
            biggest = max(gaps)
            start = levels[gaps.index(biggest)]
            line += f", biggest gap: {biggest} g (from {start} g)"
        print(line)


def print_baseline(df):
    """The simplest possible 'model': always predict the training average."""
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    if test.empty:
        print("\nNo test levels yet (need more levels per bottle).")
        return
    mean = train["water_g"].mean()
    mae = (test["water_g"] - mean).abs().mean()
    print(f"\nBaseline: always predict {mean:.0f} g  ->  test MAE = {mae:.1f} g")
    print("Your model must clearly beat this number.")


def main():
    if not DATA_DIR.exists():
        print(f"Dataset folder not found: {DATA_DIR}")
        print("Check DATA_DIR in src/config.py")
        return

    df, problems = collect_rows()
    if df.empty:
        print("No photos found. Check the folder structure.")
        return

    df = assign_split(df)
    df.to_csv(LABELS_CSV, index=False)

    print(f"Saved {len(df)} rows to {LABELS_CSV}")
    print(f"Train photos: {(df['split'] == 'train').sum()}, "
          f"test photos: {(df['split'] == 'test').sum()}")

    if problems:
        print(f"\nWarnings ({len(problems)}):")
        for p in problems:
            print("  -", p)

    print_summary(df)
    print_baseline(df)


if __name__ == "__main__":
    main()
