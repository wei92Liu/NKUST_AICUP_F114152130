import os
import glob
import numpy as np
import nibabel as nib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from monai.networks.nets import resnet
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    ScaleIntensityRanged,
    EnsureTyped,
    DivisiblePadD,
    ResizeD,
)

# =========================================================
# Config
# =========================================================
DATA_ROOT = r"E:\cardiac_segmentation"
IMG_DIR = os.path.join(DATA_ROOT, "training_image")
LBL_DIR = os.path.join(DATA_ROOT, "training_label")

OUT_PATH = os.path.join(DATA_ROOT, "best_class3_classifier.pth")

LR = 1e-4
BATCH_SIZE = 1
EPOCHS = 50
NEG_SUBSAMPLE = 20
RESIZE_SHAPE = (128, 128, 128)  # 避免 OOM 的關鍵

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# 掃描全部病人
# =========================================================
def load_patient_list():
    label_paths = sorted(glob.glob(os.path.join(LBL_DIR, "patient*.nii.gz")))
    img_paths, has_calc = [], []

    print("Scanning labels...")

    for p in label_paths:
        arr = nib.load(p).get_fdata()
        img_path = os.path.join(IMG_DIR, os.path.basename(p).replace("_gt", ""))
        img_paths.append(img_path)
        has_calc.append(1 if np.any(arr == 3) else 0)

    print(f"Total={len(img_paths)}, pos={sum(has_calc)}, neg={len(img_paths)-sum(has_calc)}")
    return img_paths, has_calc


# =========================================================
# 抽取所有正樣本 + 部分負樣本
# =========================================================
def subsample(img_paths, labels, max_neg=20):
    img_paths = np.array(img_paths)
    labels = np.array(labels)

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    np.random.shuffle(neg_idx)
    keep_neg_idx = neg_idx[:max_neg]
    keep_pos_idx = pos_idx

    keep_idx = np.concatenate([keep_pos_idx, keep_neg_idx])
    np.random.shuffle(keep_idx)

    print(f"\nUse pos={len(keep_pos_idx)}, neg={len(keep_neg_idx)}, total={len(keep_idx)}")

    return img_paths[keep_idx].tolist(), labels[keep_idx].tolist()


# =========================================================
# Split
# =========================================================
def stratified_split(paths, labels, val_ratio=0.2):
    paths = np.array(paths)
    labels = np.array(labels)

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)

    n_pos_val = max(1, int(len(pos_idx) * val_ratio))
    n_neg_val = max(1, int(len(neg_idx) * val_ratio))

    val_idx = np.concatenate([pos_idx[:n_pos_val], neg_idx[:n_neg_val]])
    train_idx = np.concatenate([pos_idx[n_pos_val:], neg_idx[n_neg_val:]])

    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)

    return (
        paths[train_idx].tolist(),
        paths[val_idx].tolist(),
        labels[train_idx].tolist(),
        labels[val_idx].tolist(),
    )


# =========================================================
# Dataset with Resize
# =========================================================
class PatientDataset(Dataset):
    def __init__(self, files, labels):
        self.files = files
        self.labels = labels

        self.transform = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=-1000, a_max=2000,
                b_min=0.0, b_max=1.0, clip=True),
            ResizeD(keys=["image"], spatial_size=RESIZE_SHAPE),  # 避免 OOM 的核心
            EnsureTyped(keys=["image"]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = self.transform({"image": self.files[idx]})
        img = data["image"].float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


# =========================================================
# 3D ResNet10
# =========================================================
def get_model():
    model = resnet.resnet10(
        spatial_dims=3,
        n_input_channels=1,
        num_classes=2,
        pretrained=False
    )
    return model.to(device)


# =========================================================
# Train
# =========================================================
def train_classifier():
    # load
    img_paths, has_calc = load_patient_list()
    img_paths, has_calc = subsample(img_paths, has_calc, max_neg=NEG_SUBSAMPLE)

    # split
    train_x, val_x, train_y, val_y = stratified_split(img_paths, has_calc)

    # datasets
    train_ds = PatientDataset(train_x, train_y)
    val_ds   = PatientDataset(val_x, val_y)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = get_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    best_acc = 0
    print("\n===== Train Classifier =====")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        correct = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()

        train_acc = correct / len(train_ds)

        # Validation
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                val_correct += (logits.argmax(1) == labels).sum().item()

        val_acc = val_correct / len(val_ds)

        print(f"[Epoch {epoch:03d}] Loss={total_loss:.4f}, TrainAcc={train_acc:.3f}, ValAcc={val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), OUT_PATH)
            print(f"  *** Saved best classifier (ValAcc={val_acc:.3f})")

    print("\nTraining done.")
    print(f"Best acc = {best_acc:.3f}")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    train_classifier()
