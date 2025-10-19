import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from anomaly_dataset import AnomalyDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', type=str, required=True, help='CSV file listing image paths and labels')
    parser.add_argument('--name', type=str, default='exp', help='Experiment name')
    parser.add_argument('--checkpoints_dir', type=str, default='./ckpt', help='Directory to save checkpoints')
    parser.add_argument('--mask', action='store_true', help='If use segmentation as mask')
    parser.add_argument('--normal_mask_dir', type=str, default=None, help='Directories containing mask normal images (if used)')
    parser.add_argument('--anormal_mask_dir', type=str, default=None, help='Directories containing mask anormal images (if used)')
    args = parser.parse_args()

    os.makedirs(args.checkpoints_dir, exist_ok=True)

    print("========== Experiment Settings ==========")
    print(f"Experiment name: {args.name}")
    print(f"CSV file: {args.datafile}")
    print(f"Checkpoints dir: {args.checkpoints_dir}")
    print(f"Using mask: {args.mask}")
    print(f"Normal Mask directory: {args.normal_mask_dir}")
    print(f"Anormal Mask directory: {args.anormal_mask_dir}")
    print("=========================================")

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    mask_preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]) if args.mask else None

    train_dataset = AnomalyDataset(csv_file=args.datafile, mask=args.mask, normal_mask_dir=args.normal_mask_dir, anormal_mask_dir=args.anormal_mask_dir,
                                   transform=preprocess, mask_transform=mask_preprocess)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    model = models.resnet18(pretrained=True)
    old_conv = model.conv1
    in_channels = 6 if args.mask else 3
    model.conv1 = nn.Conv2d(in_channels, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                            stride=old_conv.stride, padding=old_conv.padding, bias=False)
    with torch.no_grad():
        model.conv1.weight[:, :3, :, :] = old_conv.weight

    model.fc = nn.Linear(model.fc.in_features, 1)

    # === Debugging info ===
    print("========== Model Configuration ==========")
    print(f"Input channels: {model.conv1.in_channels}")
    print(f"Output channels (first conv): {model.conv1.out_channels}")
    print(f"Final FC input features: {model.fc.in_features}")
    print(f"Final FC output features: {model.fc.out_features}")
    print("=========================================")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")

        scheduler.step()

        save_path = os.path.join(args.checkpoints_dir, f'{args.name}_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), save_path)
        print(f"Saved checkpoint to {save_path}")
if __name__ == '__main__':
    main()

