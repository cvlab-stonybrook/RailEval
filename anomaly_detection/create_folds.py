import os
import argparse
import random
import csv

def collect_files(directory):
    files = []
    for fname in os.listdir(directory):
        if "output" in fname:
            continue
        fpath = os.path.join(directory, fname)
        if os.path.isfile(fpath):
            files.append(fpath)
    return files

def write_combined_csv(normal_list, anormal_list, csv_path):
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['file_path', 'label'])  # header
        for f in normal_list:
            writer.writerow([f, 0])
        for f in anormal_list:
            writer.writerow([f, 1])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--normal_dir', type=str, required=True, help='Directory of normal images')
    parser.add_argument('--anormal_dir', type=str, required=True, help='Directory of anormal images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save CSV files')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for shuffling')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)

    # Collect files
    normal_files = collect_files(args.normal_dir)
    anormal_files = collect_files(args.anormal_dir)

    print(f"Found {len(normal_files)} normal files and {len(anormal_files)} anormal files.")

    # Shuffle
    random.shuffle(normal_files)
    random.shuffle(anormal_files)

    # Split into 4 folds
    def split_folds(file_list):
        fold_size = len(file_list) // 4
        folds = [file_list[i*fold_size:(i+1)*fold_size] for i in range(4)]
        remainder = file_list[4*fold_size:]
        for i, extra in enumerate(remainder):
            folds[i].append(extra)
        return folds

    normal_folds = split_folds(normal_files)
    anormal_folds = split_folds(anormal_files)

    # Create train/val combined CSV per fold
    for fold in range(4):
        normal_val = normal_folds[fold]
        normal_train = [f for i in range(4) if i != fold for f in normal_folds[i]]
        anormal_val = anormal_folds[fold]
        anormal_train = [f for i in range(4) if i != fold for f in anormal_folds[i]]

        train_csv_path = os.path.join(args.output_dir, f'train_{fold+1}.csv')
        val_csv_path = os.path.join(args.output_dir, f'val_{fold+1}.csv')

        write_combined_csv(normal_train, anormal_train, train_csv_path)
        write_combined_csv(normal_val, anormal_val, val_csv_path)

        print(f"Fold {fold+1}: train={len(normal_train)+len(anormal_train)}, val={len(normal_val)+len(anormal_val)}")

if __name__ == '__main__':
    main()
