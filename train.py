import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import EfficientNet_B3_Weights
from PIL import Image
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split

class CarDataset(Dataset):
    def __init__(self, img_paths, labels, transform):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(self.labels[idx])

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def train_model(
    data_dir="/home/a/hecto_AI/data/train",
    num_epochs=80,
    batch_size=8,
    lr=1e-4,
    weight_decay=0.05,
    model_path="efficientnet_b3_finetuned.pth",
    patience=8,
    min_delta=0.001
):
    class_folders = sorted(glob(os.path.join(data_dir, "*")))
    img_paths, labels = [], []
    for class_idx, folder in enumerate(class_folders):
        images = glob(os.path.join(folder, "*.jpg")) + glob(os.path.join(folder, "*.png"))
        img_paths.extend(images)
        labels.extend([class_idx] * len(images))

    # 최신 EfficientNet-B3 weights 객체 불러오기
    weights = EfficientNet_B3_Weights.IMAGENET1K_V1
    transform = weights.transforms()  # 300x300 + normalization 자동 적용

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        img_paths, labels, test_size=0.1, stratify=labels
    )

    train_dataset = CarDataset(train_paths, train_labels, transform)
    val_dataset = CarDataset(val_paths, val_labels, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_b3(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_folders))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler()
    early_stopping = EarlyStopping(patience, min_delta)
    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        print(f"\nEpoch {epoch+1}/{num_epochs} started...")

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.amp.autocast(device_type='cuda'):
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                    val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Elapsed: {elapsed//3600:.0f}h {(elapsed%3600)//60:.0f}m")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            print(f"★ Best model saved to {model_path}")

        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

if __name__ == "__main__":
    train_model()