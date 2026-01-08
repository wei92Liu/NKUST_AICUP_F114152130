import warnings
warnings.filterwarnings("ignore")  # 關閉所有警告

import os
import glob
import time
import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
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
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism


# ======================================================
# 基本設定
# ======================================================
DATA_ROOT = r"E:\cardiac_segmentation"
IMAGE_DIR = os.path.join(DATA_ROOT, "training_image")
LABEL_DIR = os.path.join(DATA_ROOT, "training_label")
OUTPUT_DIR = DATA_ROOT

NUM_CLASSES_MODEL = 3     # 0=背景, 1=myocardium, 2=valve
ROI_SIZE = (128, 128, 128)
BATCH_SIZE = 1
NUM_WORKERS = 4

MAX_EPOCHS = 300
PATIENCE = 30  # Early stopping

CE_WEIGHTS = torch.tensor([0.1, 1.0, 2.0], dtype=torch.float32)

LR = 1e-4
WEIGHT_DECAY = 1e-5

set_determinism(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================
# 分割資料
# ======================================================
def get_dataset_splits():
    images = sorted(glob.glob(os.path.join(IMAGE_DIR, "patient*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(LABEL_DIR, "patient*.nii.gz")))

    assert len(images) == len(labels) > 0

    data = [{"image": img, "label": seg} for img, seg in zip(images, labels)]
    train_files = data[:40]
    val_files = data[40:]

    print(f"Train: {len(train_files)}, Val: {len(val_files)}")
    return train_files, val_files


# ======================================================
# Transform
# ======================================================
def get_transforms():
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1, 1, 1),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1000,
            a_max=2000,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        # 不抽取 class3
        RandCropByLabelClassesd(
            keys=["image", "label"],
            label_key="label",
            spatial_size=ROI_SIZE,
            num_classes=3,   # 只看 0/1/2
            num_samples=2,
            allow_smaller=True,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        EnsureTyped(keys=["image", "label"]),
        DivisiblePadD(keys=["image", "label"], k=16),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1, 1, 1),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1000,
            a_max=2000,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        EnsureTyped(keys=["image", "label"]),
        DivisiblePadD(keys=["image", "label"], k=16),
    ])

    return train_transforms, val_transforms


# ======================================================
# DataLoader
# ======================================================
def get_loaders():
    train_files, val_files = get_dataset_splits()
    train_t, val_t = get_transforms()

    train_ds = CacheDataset(train_files, train_t, cache_rate=1.0, num_workers=NUM_WORKERS)
    val_ds = CacheDataset(val_files, val_t, cache_rate=1.0, num_workers=NUM_WORKERS)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader


# ======================================================
# Label remap (class3 → 0)
# ======================================================
def remap_label_for_class12(labels):
    labels = labels.long()
    labels[labels == 3] = 0
    return labels


# ======================================================
# IoU
# ======================================================
def compute_iou(pred, label, num_classes=3):
    pred_np = pred.cpu().numpy()
    label_np = label.cpu().numpy()

    ious = []
    for c in range(num_classes):
        p = pred_np == c
        l = label_np == c
        inter = np.logical_and(p, l).sum()
        union = np.logical_or(p, l).sum()
        ious.append(np.nan if union == 0 else inter / union)

    return np.array(ious, dtype=np.float32)


# ======================================================
# Validate
# ======================================================
def validate(model, val_loader, dice_metric):
    model.eval()
    dice_metric.reset()

    dice_list = []
    iou_list = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = remap_label_for_class12(batch["label"].to(device))  # (B,1,D,H,W)

            # 推論
            logits = sliding_window_inference(
                images,
                ROI_SIZE,
                sw_batch_size=1,
                predictor=model,
                overlap=0.25,
            )

            probs = torch.softmax(logits, dim=1)             # (B,3,D,H,W)
            preds = torch.argmax(probs, dim=1, keepdim=True) # (B,1,D,H,W)

            preds_idx = preds[:, 0].long()   # (B,D,H,W)
            labels_idx = labels[:, 0].long()

            # one-hot
            preds_oh = F.one_hot(preds_idx, NUM_CLASSES_MODEL).permute(0, 4, 1, 2, 3).float()
            labels_oh = F.one_hot(labels_idx, NUM_CLASSES_MODEL).permute(0, 4, 1, 2, 3).float()

            # ---- Dice ----
            dice_metric(y_pred=preds_oh, y=labels_oh)
            dice_per_class = dice_metric.aggregate().cpu().numpy()
            dice_metric.reset()

            # 修正 (1,3) → (3,)
            if dice_per_class.ndim == 2 and dice_per_class.shape[0] == 1:
                dice_per_class = dice_per_class[0]

            # ---- IoU ----
            iou_per_class = compute_iou(
                preds_idx.unsqueeze(1), labels_idx.unsqueeze(1),
                num_classes=NUM_CLASSES_MODEL
            )

            dice_list.append(dice_per_class)
            iou_list.append(iou_per_class)

    dice_mean = np.nanmean(np.stack(dice_list), axis=0)
    iou_mean  = np.nanmean(np.stack(iou_list), axis=0)

    score_c1 = (dice_mean[1] + iou_mean[1]) / 2
    score_c2 = (dice_mean[2] + iou_mean[2]) / 2
    final_score = (score_c1 + score_c2) / 2

    return dice_mean, iou_mean, final_score


# ======================================================
# Model / Loss / Optimizer
# ======================================================
def get_model_and_optim():
    model = SwinUNETR(
        img_size=ROI_SIZE,
        in_channels=1,
        out_channels=NUM_CLASSES_MODEL,
        feature_size=48,
        use_checkpoint=True,
    ).to(device)

    loss_fn = DiceCELoss(
        include_background=False,
        to_onehot_y=True,
        softmax=True,
        ce_weight=CE_WEIGHTS.to(device),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=MAX_EPOCHS,
    )

    dice_metric = DiceMetric(include_background=True, reduction="none")

    return model, loss_fn, optimizer, scheduler, dice_metric


# ======================================================
# main + Early Stopping
# ======================================================
def main():
    train_loader, val_loader = get_loaders()
    model, loss_fn, optimizer, scheduler, dice_metric = get_model_and_optim()

    if device.type == "cuda":
        scaler = torch.amp.GradScaler()
        amp_ctx = lambda: torch.amp.autocast("cuda")
    else:
        scaler = None
        amp_ctx = nullcontext

    best_score = -1
    patience_cnt = 0
    save_path = os.path.join(OUTPUT_DIR, "best_swinunetr_class12.pth")

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        model.train()
        train_loss = 0
        step = 0

        print(f"\n===== [class1+2] Epoch {epoch}/{MAX_EPOCHS} =====")

        # ---------------- Train ----------------
        for batch in train_loader:
            step += 1
            images = batch["image"].to(device)
            labels = remap_label_for_class12(batch["label"].to(device))  # long

            optimizer.zero_grad()

            with amp_ctx():
                outputs = model(images)
                loss = loss_fn(outputs, labels)

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item()

            if step % 10 == 0:
                print(f"  step {step:4d}, loss = {loss.item():.4f}")

        scheduler.step()

        avg_loss = train_loss / max(step, 1)
        print(f"Epoch {epoch} train loss = {avg_loss:.4f}")

        # ---------------- Validation ----------------
        dice_mean, iou_mean, val_score = validate(model, val_loader, dice_metric)

        print(f"  [Val] Dice = {dice_mean}")
        print(f"  [Val] IoU  = {iou_mean}")
        print(f"  [Val] Final Score = {val_score:.4f}")

        # ---------------- Early Stopping ----------------
        if val_score > best_score:
            best_score = val_score
            patience_cnt = 0
            torch.save(model.state_dict(), save_path)
            print(f"  *** Saved best model (score={val_score:.4f})")
        else:
            patience_cnt += 1
            print(f"  No improvement. patience {patience_cnt}/{PATIENCE}")

        if patience_cnt >= PATIENCE:
            print("\n=== Early Stopping Triggered ===")
            print(f"Best Score = {best_score:.4f}")
            break

        print(f"Epoch time: {(time.time() - t0)/60:.2f} min")

    print(f"\nTraining completed. Best Score = {best_score:.4f}")
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()
