import argparse
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from anomaly_dataset import AnomalyDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', type=str, required=True, help='CSV file listing image paths and labels for test')
    parser.add_argument('--name', type=str, default='exp', help='Experiment name')
    parser.add_argument('--checkpoints_dir', type=str, default='./ckpt', help='Directory with saved checkpoints')
    parser.add_argument('--results_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--mask', action='store_true', help='Segmentation as mask')
    parser.add_argument('--normal_mask_dir', type=str, default=None, help='Directory containing mask normal images (if used)')
    parser.add_argument('--anormal_mask_dir', type=str, default=None, help='Directory containing mask anormal images (if used)')
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    mask_preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]) if args.mask else None

    # Load dataset
    test_dataset = AnomalyDataset(csv_file=args.datafile, mask=args.mask, normal_mask_dir=args.normal_mask_dir, anormal_mask_dir=args.anormal_mask_dir,
                                   transform=preprocess, mask_transform=mask_preprocess)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Load model
    model = models.resnet18(pretrained=False)
    old_conv = model.conv1
    in_channels = 6 if args.mask else 3
    model.conv1 = nn.Conv2d(in_channels, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                            stride=old_conv.stride, padding=old_conv.padding, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    checkpoint_path = os.path.join(args.checkpoints_dir, f'{args.name}.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    correct, total = 0, 0
    total_loss = 0.0
    results = []

    with torch.no_grad():
        for inputs, labels, names in test_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for i in range(inputs.size(0)):
                img_name = names[i]
                results.append((int(preds[i].item()), int(labels[i].item()), img_name))

    accuracy = correct / total * 100.0
    avg_loss = total_loss / total

    result_file = os.path.join(args.results_dir, f'{args.name}_results.txt')
    with open(result_file, 'w') as f:
        f.write(f'Prediction accuracy: {accuracy:.6f}\n')
        f.write(f'Average loss: {avg_loss:.6f}\n')
        f.write('[Predicted Label, GT Label, Image Name]\n')
        for pred, gt, fname in results:
            f.write(f'{pred}, {gt}, {fname}\n')

    print(f'Results saved to {result_file}')
    print(f'Overall accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    main()
