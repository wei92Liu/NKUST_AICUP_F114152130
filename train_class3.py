import warnings
warnings.filterwarnings("ignore")

import os
import glob
import time
import numpy as np
import nibabel as nib

import torch
import torch.nn.functional as F
from contextlib import nullcontext

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    ScaleIntensityRanged,
    RandCropByLabelClassesd,
    RandFlipd,
    RandRotate90d,
    EnsureTyped,
    DivisiblePadD,
)
from monai.data import CacheDataset, DataLoader
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism


# ======================================================
# 基本設定
# ======================================================
DATA_ROOT = r"E:\cardiac_segmentation"
IMAGE_DIR = os.path.join(DATA_ROOT, "training_image")
LABEL_DIR = os.path.join(DATA_ROOT, "training_label")
OUTPUT_DIR = DATA_ROOT

NUM_CLASSES_MODEL = 2  # binary: 0=non-calc, 1=calc
ROI_SIZE = (96, 96, 96)
BATCH_SIZE = 1
NUM_WORKERS = 4

MAX_EPOCHS = 300
PATIENCE = 40  # tiny lesion → 需要耐心

# CE 權重（比 Dice 弱：避免爆炸）
CE_WEIGHTS = torch.tensor([0.2, 3.0], dtype=torch.float32)
LAMBDA_DICE = 1.0
LAMBDA_CE = 0.3

LR = 1e-4
WEIGHT_DECAY = 1e-5

set_determinism(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================
# 掃描全部病人，找到含 calc 的
# ======================================================
def find_positive_cases():
    label_paths = sorted(glob.glob(os.path.join(LABEL_DIR, "patient*.nii.gz")))
    pos, neg = [], []

    print("Scanning labels for class3...")
    for p in label_paths:
        arr = nib.load(p).get_fdata()
        if np.any(arr == 3):
            pos.append(p)
        else:
            neg.append(p)

    print(f"Total cases = {len(label_paths)}")
    print(f"  Positive (class3) = {len(pos)}")
    print(f"  Negative          = {len(neg)}")
    return pos, neg


# ======================================================
# 分割資料：只用 positive
# ======================================================
def get_dataset_splits():
    pos_labels, _ = find_positive_cases()
    pos_labels = sorted(pos_labels)
    pos_images = [
        os.path.join(IMAGE_DIR, os.path.basename(p).replace("_gt", ""))
        for p in pos_labels
    ]

    N = len(pos_labels)
    n_train = max(1, int(0.8 * N))
    n_train = min(n_train, N - 1)

    train = [
        {"image": img, "label": lbl}
        for img, lbl in zip(pos_images[:n_train], pos_labels[:n_train])
    ]
    val = [
        {"image": img, "label": lbl}
        for img, lbl in zip(pos_images[n_train:], pos_labels[n_train:])
    ]

    print(f"Train = {len(train)}, Val = {len(val)}")
    print("Train cases:", [os.path.basename(x["image"]) for x in train])
    print("Val   cases:", [os.path.basename(x["image"]) for x in val])
    return train, val


# ======================================================
# Transform（不做 spacing 保留 original resolution）
# ======================================================
def get_transforms():
    train_t = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1000, a_max=2000,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        # 90% 抽 calc，10% 抽 background patch
        RandCropByLabelClassesd(
            keys=["image", "label"],
            label_key="label",
            spatial_size=ROI_SIZE,
            num_classes=4,
            num_samples=4,
            ratios=[0.1, 0, 0, 0.9],
            allow_smaller=True,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        EnsureTyped(keys=["image", "label"]),
        DivisiblePadD(keys=["image", "label"], k=16),
    ])

    val_t = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1000, a_max=2000,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        EnsureTyped(keys=["image", "label"]),
        DivisiblePadD(keys=["image", "label"], k=16),
    ])

    return train_t, val_t


# ======================================================
# DataLoader
# ======================================================
def get_loaders():
    train_files, val_files = get_dataset_splits()
    train_t, val_t = get_transforms()

    train_ds = CacheDataset(train_files, train_t, cache_rate=1.0, num_workers=NUM_WORKERS)
    val_ds = CacheDataset(val_files, val_t, cache_rate=1.0, num_workers=NUM_WORKERS)

    train = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=NUM_WORKERS, pin_memory=True)
    val = DataLoader(val_ds, batch_size=1, shuffle=False,
                     num_workers=NUM_WORKERS, pin_memory=True)
    return train, val


# ======================================================
# Label 重映射：0/1/2->0, 3->1
# ======================================================
def remap_label(lbl):
    lbl = lbl.long().clone()
    lbl[lbl != 3] = 0
    lbl[lbl == 3] = 1
    return lbl


# ======================================================
# Tiny Lesion Metrics
# ======================================================
def lesion_metrics(pred, label):
    """
    pred, label : shape (B,D,H,W), 0/1

    回傳:
    recall, precision, lesion_f1, score
    """
    pred_np = pred.cpu().numpy()
    lbl_np = label.cpu().numpy()

    # voxel-based
    TP = np.logical_and(pred_np == 1, lbl_np == 1).sum()
    FP = np.logical_and(pred_np == 1, lbl_np == 0).sum()
    FN = np.logical_and(pred_np == 0, lbl_np == 1).sum()

    recall = TP / (TP + FN + 1e-8)
    precision = TP / (TP + FP + 1e-8)

    # lesion-wise F1
    lesion_total = 1 if lbl_np.sum() > 0 else 0
    lesion_tp    = 1 if (pred_np.sum() > 0 and lesion_total == 1) else 0

    lesion_f1 = lesion_tp / lesion_total if lesion_total > 0 else 0.0

    score = (recall + lesion_f1) / 2
    return recall, precision, lesion_f1, score


# ======================================================
# Validate
# ======================================================
def validate(model, loader):
    model.eval()
    recalls, precs, lfs, scores = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = remap_label(batch["label"].to(device))
            labels = labels[:, 0]  # (B,D,H,W)

            logits = sliding_window_inference(
                images, ROI_SIZE, 1, model, overlap=0.25
            )
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)  # (B,D,H,W)

            r, p, lf, s = lesion_metrics(preds, labels)
            recalls.append(r)
            precs.append(p)
            lfs.append(lf)
            scores.append(s)

    return (np.mean(recalls), np.mean(precs), np.mean(lfs), np.mean(scores))


# ======================================================
# Model + Optimizer
# ======================================================
def get_model_and_optim():
    model = SwinUNETR(
        img_size=ROI_SIZE,
        in_channels=1,
        out_channels=NUM_CLASSES_MODEL,
        feature_size=48,
        use_checkpoint=True,
    ).to(device)

    ce_weight = CE_WEIGHTS.to(device)
    loss_fn = DiceCELoss(
        include_background=True,
        to_onehot_y=True,
        softmax=True,
        ce_weight=ce_weight,
        lambda_dice=LAMBDA_DICE,
        lambda_ce=LAMBDA_CE,
    )

    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=MAX_EPOCHS)

    return model, loss_fn, optim, sched


# ======================================================
# main + Early Stopping
# ======================================================
def main():
    train_loader, val_loader = get_loaders()
    model, loss_fn, optim, sched = get_model_and_optim()

    if device.type == "cuda":
        scaler = torch.amp.GradScaler()
        amp_ctx = lambda: torch.amp.autocast("cuda")
    else:
        scaler = None
        amp_ctx = nullcontext

    best_score = -1
    patience = 0
    save_path = os.path.join(OUTPUT_DIR, "best_swinunetr_class3.pth")

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        steps = 0

        print(f"\n===== [class3] Epoch {epoch}/{MAX_EPOCHS} =====")

        # ---------- Train ----------
        for batch in train_loader:
            steps += 1
            imgs = batch["image"].to(device)
            lbls = remap_label(batch["label"].to(device))

            optim.zero_grad()

            with amp_ctx():
                out = model(imgs)
                loss = loss_fn(out, lbls)

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()

            total_loss += loss.item()
            if steps % 10 == 0:
                print(f"  step {steps}, loss={loss.item():.4f}")

        sched.step()
        avg_loss = total_loss / steps
        print(f"Epoch {epoch} Train Loss = {avg_loss:.4f}")

        # ---------- Validation ----------
        recall, prec, lf1, score = validate(model, val_loader)
        print(f"  [Val] Recall={recall:.4f}, Precision={prec:.4f}, LesionF1={lf1:.4f}")
        print(f"  [Val] Score = {score:.4f}")

        # ---------- Early Stopping ----------
        if score > best_score:
            best_score = score
            patience = 0
            torch.save(model.state_dict(), save_path)
            print(f"  *** Saved best class3 model (score={score:.4f})")
        else:
            patience += 1
            print(f"  No improvement. patience {patience}/{PATIENCE}")

        if patience >= PATIENCE:
            print("\n=== Early Stopping Triggered ===")
            print(f"Best Score = {best_score:.4f}")
            break

        print(f"Epoch time: {(time.time()-t0)/60:.2f} min")

    print(f"\nTraining completed. Best Score={best_score:.4f}")
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()
