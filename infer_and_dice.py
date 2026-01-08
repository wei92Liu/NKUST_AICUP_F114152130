# infer_and_dice.py
import os, glob, argparse
from typing import Optional, Dict, Tuple
from inspect import signature

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, EnsureTyped, DivisiblePadD, Invertd, SaveImaged, LoadImage,
    ResizeD
)
from monai.data import MetaTensor
from monai.networks.nets import SwinUNETR, resnet
from monai.inferers import sliding_window_inference

torch.backends.cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True


# -----------------------
# 小工具
# -----------------------
def argmax_channel(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=1, keepdim=False)

def onehot_from_label(label: torch.Tensor, num_classes: int) -> torch.Tensor:
    if label.dim() == 5:
        label = label[:, 0]
    return torch.nn.functional.one_hot(label.long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

def compute_dice_per_class(pred: torch.Tensor, gt: torch.Tensor, num_classes: int) -> Dict[int, float]:
    dices: Dict[int, float] = {}
    with torch.no_grad():
        pred_oh = onehot_from_label(pred.unsqueeze(1), num_classes)
        gt_oh   = onehot_from_label(gt.unsqueeze(1),   num_classes)
        for c in range(1, num_classes):
            p, g = pred_oh[:, c], gt_oh[:, c]
            inter = (p * g).sum().item()
            denom = p.sum().item() + g.sum().item()
            dice = (2.0 * inter) / denom if denom > 0 else (1.0 if g.sum().item() == 0 else 0.0)
            dices[c] = dice
    return dices

def _index_labels(labdir: str):
    idx = {}
    lab_paths = sorted(glob.glob(os.path.join(labdir, "*.nii"))) + \
                sorted(glob.glob(os.path.join(labdir, "*.nii.gz")))
    suffixes = ["_label", "_labels", "_seg", "_mask", "_gt"]
    for p in lab_paths:
        s = _stem_nii(p)
        idx[s] = p  # 完整 stem
        for suf in suffixes:
            if s.endswith(suf) and len(s) > len(suf):
                idx[s[:-len(suf)]] = p  # 去尾綴後也建一個索引
    return idx

# -----------------------
# SwinUNETR 建置（相容不同 MONAI 版本）
# -----------------------
def _make_swinunetr(img_size, in_ch, out_ch, device="cuda"):
    sig = signature(SwinUNETR)
    kw = {}
    if "img_size" in sig.parameters:       kw["img_size"] = img_size
    if "in_channels" in sig.parameters:    kw["in_channels"] = in_ch
    if "out_channels" in sig.parameters:   kw["out_channels"] = out_ch
    if "feature_size" in sig.parameters:   kw["feature_size"] = 48
    if "use_checkpoint" in sig.parameters: kw["use_checkpoint"] = True
    if "spatial_dims" in sig.parameters:   kw["spatial_dims"] = 3
    model = SwinUNETR(**kw).to(device).eval()
    return model

def build_class12_model(img_size=(128,128,128), device="cuda"):
    return _make_swinunetr(img_size=img_size, in_ch=1, out_ch=3, device=device)

def build_class3_model(img_size=(96,96,96), device="cuda"):
    return _make_swinunetr(img_size=img_size, in_ch=1, out_ch=2, device=device)


# -----------------------
# 分支前處理（含可還原的 meta）
# -----------------------
def make_transforms(image_path: str):
    base_keys = ["image"]

    t12 = Compose([
        LoadImaged(keys=base_keys, image_only=False),
        EnsureChannelFirstd(keys=base_keys),
        Orientationd(keys=base_keys, axcodes="RAS"),
        EnsureTyped(keys=base_keys, dtype=torch.float32),
        Spacingd(keys=base_keys, pixdim=(1,1,1), mode=("bilinear",)),
        ScaleIntensityRanged(keys=base_keys, a_min=-1000, a_max=2000, b_min=0.0, b_max=1.0, clip=True),
        DivisiblePadD(keys=base_keys, k=16),
    ])
    t3 = Compose([
        LoadImaged(keys=base_keys, image_only=False),
        EnsureChannelFirstd(keys=base_keys),
        Orientationd(keys=base_keys, axcodes="RAS"),
        ScaleIntensityRanged(keys=base_keys, a_min=-1000, a_max=2000, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=base_keys, dtype=torch.float32),
        DivisiblePadD(keys=base_keys, k=16),
    ])

    inv12 = Invertd(keys="pred12", transform=t12, orig_keys="image", meta_keys="pred12_meta",
                    nearest_interp=True, to_tensor=True)
    inv3  = Invertd(keys="pred3",  transform=t3,  orig_keys="image", meta_keys="pred3_meta",
                    nearest_interp=True, to_tensor=True)

    data = {"image": image_path}
    return t12, t3, inv12, inv3, data


# -----------------------
# SW 推論（支援 overlap/mode/FP16）
# -----------------------
@torch.inference_mode()
def run_sw_infer(model, image: torch.Tensor, roi_size, sw_batch=4, device="cuda",
                 use_fp16=False, sw_overlap=0.25, sw_mode="constant"):
    x = image.half() if (use_fp16 and device == "cuda") else image.float()
    with torch.amp.autocast("cuda", enabled=(use_fp16 and device == "cuda")):
        logits = sliding_window_inference(
            inputs=x, roi_size=roi_size, sw_batch_size=sw_batch,
            predictor=lambda t: model(t), overlap=sw_overlap, mode=sw_mode,
        )
    return logits


# -----------------------
# class3 二元分類器（依你的訓練碼設定）
# -----------------------
CLS_RESIZE = (128,128,128)  # 與訓練一致  :contentReference[oaicite:2]{index=2}

def build_class3_classifier(device="cuda"):
    model = resnet.resnet10(
        spatial_dims=3, n_input_channels=1, num_classes=2, pretrained=False
    ).to(device).eval()
    return model

def classifier_transform():
    # 與訓練前處理一致：Load→ChannelFirst→RAS→強度縮放→Resize  :contentReference[oaicite:3]{index=3}
    return Compose([
        LoadImaged(keys=["image"], image_only=False),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=2000, b_min=0.0, b_max=1.0, clip=True),
        ResizeD(keys=["image"], spatial_size=CLS_RESIZE),
        EnsureTyped(keys=["image"], dtype=torch.float32),
    ])

@torch.inference_mode()
def predict_class3_presence(cls_model, img_path: str, device="cuda") -> int:
    t = classifier_transform()
    d = t({"image": img_path})
    x = d["image"].unsqueeze(0).to(device)  # (1,1,D,H,W)
    logits = cls_model(x)
    pred = int(logits.argmax(1).item())  # 1=有 class3, 0=無
    return pred


# -----------------------
# 單筆資料：雙分支推論 + gating + 還原 + Dice
# -----------------------
@torch.inference_mode()
def process_case(
    image_path: str,
    out_path: Optional[str],
    model12, model3,
    roi12=(128,128,128), roi3=(96,96,96),
    swb:int=4, device:str="cuda",
    label_path: Optional[str]=None,
    sw_overlap: float = 0.25, sw_mode: str = "constant",
    use_fp16: bool = False,
    cls_model: Optional[torch.nn.Module] = None,  # 供 gating
) -> Tuple[str, Optional[Dict[int, float]]]:

    # (A) 先做 class12 分支
    t12, t3, inv12, inv3, data = make_transforms(image_path)
    d12 = t12(data.copy())
    img12 = d12["image"].unsqueeze(0).to(device)
    logit12 = run_sw_infer(model12, img12, roi_size=roi12, sw_batch=swb, device=device,
                           use_fp16=use_fp16, sw_overlap=sw_overlap, sw_mode=sw_mode)
    pred12  = argmax_channel(logit12)
    d12["pred12"] = MetaTensor(pred12.float(), meta=d12["image"].meta)
    d12["pred12_meta"] = d12["image"].meta

    # (B) 決定是否需要跑 class3 分支（用分類器 gating）
    need_class3 = True
    if cls_model is not None:
        prob = predict_class3_presence(cls_model, image_path, device=device)
        need_class3 = (prob == 1)

    if need_class3:
        d3 = t3(data.copy())
        img3 = d3["image"].unsqueeze(0).to(device)
        logit3 = run_sw_infer(model3, img3, roi_size=roi3, sw_batch=swb, device=device,
                              use_fp16=use_fp16, sw_overlap=sw_overlap, sw_mode=sw_mode)
        pred3 = argmax_channel(logit3)
        d3["pred3"] = MetaTensor(pred3.float(), meta=d3["image"].meta)
        d3["pred3_meta"] = d3["image"].meta
        d3  = inv3(d3)
        pred3_inv = d3["pred3"].long()
    else:
        # 直接全零（無 class3）
        d3  = t3(data.copy())   # 為了拿 meta
        pred3_inv = torch.zeros_like(d3["image"][:,0].long())  # (1,Z,Y,X)

    # (C) 還原 class12 並融合
    d12 = inv12(d12)

    # 反投影後的張量
    pred12_inv = d12["pred12"].long()   # (1,D,H,W) or (D,H,W)

    # 注意：pred3_inv 已在上面分支內「決定並賦值」好了
    # - 若 need_class3=True： pred3_inv 來自 d3 經 inv3 還原
    # - 若 need_class3=False： pred3_inv 是 zeros_like(...)
    # 因此這裡**不要**再讀 d3["pred3"]，避免 KeyError

    # 補 channel 維
    if pred12_inv.ndim == 3: pred12_inv = pred12_inv.unsqueeze(0)
    if pred3_inv.ndim  == 3: pred3_inv  = pred3_inv.unsqueeze(0)

    # 以 pred12 空間為準
    final_pred  = pred12_inv.clone().long()
    target_size = final_pred.shape[-3:]  # (D,H,W)

    # class3 對齊到相同空間大小（最近鄰，不改 label 值）
    pred3_inv = F.interpolate(pred3_inv.float(), size=target_size, mode="nearest").long()

    # 若 pred3_inv 是概率，先二值化：
    # pred3_inv = (pred3_inv > 0.5).long()

    # 融合：class-3 位置覆蓋成 3
    final_pred[pred3_inv == 1] = 3


    # (D) 存檔
    if out_path is None:
        base = os.path.splitext(os.path.basename(image_path))[0]
        if base.endswith(".nii"):
            base = base[:-4]
        out_path = os.path.join(os.path.dirname(image_path), f"{base}_predict.nii.gz")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    saver = SaveImaged(
        keys="final", meta_keys="final_meta",
        output_dir=os.path.dirname(out_path),
        output_postfix="", output_ext=".nii.gz",
        separate_folder=False, resample=False
    )
    save_dict = {"final": final_pred.float(), "final_meta": d12["image"].meta}
    saver(save_dict)

    # 重新命名為 out_path（避免 SaveImaged 預設命名）
    gen_path = os.path.join(os.path.dirname(out_path),
                            os.path.basename(image_path).replace(".nii.gz", "") + ".nii.gz")
    if os.path.exists(gen_path) and gen_path != out_path:
        try: os.replace(gen_path, out_path)
        except Exception: pass

    # (E) Dice
    dices = None
    if label_path:
        lab, _meta = LoadImage(image_only=False)(label_path)
        lab = torch.as_tensor(lab).unsqueeze(0).long()
        dices = compute_dice_per_class(final_pred, lab, num_classes=4)

    return out_path, dices


# -----------------------
# 分類器評估（在 training_image / training_label 上）
# -----------------------
@torch.inference_mode()
def evaluate_classifier_on_dir(cls_model, imgdir: str, labdir: str, device="cuda"):
    images = sorted(glob.glob(os.path.join(imgdir, "*.nii"))) + \
            sorted(glob.glob(os.path.join(imgdir, "*.nii.gz")))
    print(f"[Classifier] found images: {len(images)} under {imgdir}")

    lab_index = _index_labels(labdir)

    TP=FP=TN=FN=0
    paired = 0
    missed = []
    for img_path in images:
        stem = _stem_nii(img_path)
        lab_path = lab_index.get(stem, None)
        if lab_path is None:
            continue
        paired += 1

        gt, _m = LoadImage(image_only=False)(lab_path)
        gt = torch.as_tensor(gt).long()
        gt_has = int((gt == 3).any().item())

        pr_has = predict_class3_presence(cls_model, img_path, device=device)

        if pr_has == 1 and gt_has == 1: TP += 1
        elif pr_has == 1 and gt_has == 0: FP += 1
        elif pr_has == 0 and gt_has == 0: TN += 1
        elif pr_has == 0 and gt_has == 1:
            FN += 1
            missed.append(os.path.basename(img_path))

    if paired == 0:
        print(f"[Classifier] 0 pairs matched. Check filenames in {imgdir} / {labdir}.")
        # 額外 debug：列出前幾個影像 stem 與 index 的 keys 長相
        print("Example image stems:", [ _stem_nii(p) for p in images[:5] ])
        print("Example label stems:", list(lab_index.keys())[:10])
        return

    total = TP+FP+TN+FN
    acc = (TP+TN)/total if total else 0.0
    prec = TP/(TP+FP) if (TP+FP)>0 else 0.0
    rec  = TP/(TP+FN) if (TP+FN)>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0

    print("\n=== Class3 classifier evaluation on training set ===")
    print(f"Total={total}  TP={TP} FP={FP} TN={TN} FN={FN}")
    print(f"Acc={acc:.3f}  Prec={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")
    if missed:
        print("FN cases (GT has class3 but predicted none):")
        for n in missed[:20]:
            print("  -", n)
        if len(missed) > 20:
            print(f"  ... and {len(missed)-20} more")
            
def _stem_nii(p: str) -> str:
    bn = os.path.basename(p)
    if bn.endswith(".nii.gz"):
        return bn[:-7]
    if bn.endswith(".nii"):
        return bn[:-4]
    return os.path.splitext(bn)[0]

def _find_label_for(img_path: str, labdir: str):
    s = _stem_nii(img_path)
    cands = [
        os.path.join(labdir, s + ".nii.gz"),
        os.path.join(labdir, s + ".nii"),
        os.path.join(labdir, s + "_label.nii.gz"),
        os.path.join(labdir, s + "_label.nii"),
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    return None

# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    # 單檔
    ap.add_argument("--image", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out",   default=None)

    # 目錄
    ap.add_argument("--imgdir", default=None, help="*.nii.gz images")
    ap.add_argument("--labdir", default=None, help="labels (for Dice or classifier eval)")
    ap.add_argument("--outdir", default=None, help="predictions output dir")

    # 權重/設定
    ap.add_argument("--w12", default="best_swinunetr_class12.pth")
    ap.add_argument("--w3",  default="best_swinunetr_class3.pth")
    ap.add_argument("--cls_w", default="best_class3_classifier.pth", help="class3 binary classifier weight")
    ap.add_argument("--roi12", default="128,128,128")
    ap.add_argument("--roi3",  default="96,96,96")
    ap.add_argument("--swb",   type=int, default=4)
    ap.add_argument("--gpu",   type=int, default=0)

    # 速度/品質
    ap.add_argument("--fp16", action="store_true", help="enable FP16 for UNet branches")
    ap.add_argument("--sw_overlap", type=float, default=0.25)
    ap.add_argument("--sw_mode", type=str, default="constant", choices=["constant","gaussian"])

    # 分類器評估與 gating
    ap.add_argument("--eval_class3", action="store_true",
                    help="evaluate class3 classifier on training_image/label before inference")
    ap.add_argument("--use_cls_gate", action="store_true",
                    help="use classifier gating to skip class3 branch when predicted absent")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    roi12 = tuple(map(int, args.roi12.split(",")))
    roi3  = tuple(map(int, args.roi3.split(",")))

    # 構建模型 + 載入權重（UNet 分支）
    model12 = build_class12_model(img_size=roi12, device=device)
    model3  = build_class3_model(img_size=roi3,  device=device)
    sd12 = torch.load(args.w12, map_location=device); sd12 = sd12.get("state_dict", sd12)
    sd3  = torch.load(args.w3,  map_location=device); sd3  = sd3.get("state_dict",  sd3)
    model12.load_state_dict(sd12, strict=False)
    model3.load_state_dict(sd3,  strict=False)
    model12.float(); model3.float()  # 避免 half/float 混用

    # 分類器
    cls_model = None
    if args.eval_class3 or args.use_cls_gate:
        cls_model = build_class3_classifier(device=device)
        ck = torch.load(args.cls_w, map_location=device)
        cls_model.load_state_dict(ck, strict=True)
        cls_model.eval()

    # 先評估分類器（在 training set）
    if args.eval_class3:
        if args.imgdir is None or args.labdir is None:
            raise SystemExit("--eval_class3 需要同時指定 --imgdir 與 --labdir（通常指 training_image / training_label）")
        evaluate_classifier_on_dir(cls_model, args.imgdir, args.labdir, device=device)

    # 單檔模式
    if args.image is not None:
        out_path, dices = process_case(
            image_path=args.image, out_path=args.out,
            model12=model12, model3=model3,
            roi12=roi12, roi3=roi3, swb=args.swb, device=device,
            label_path=args.label,
            sw_overlap=args.sw_overlap, sw_mode=args.sw_mode, use_fp16=args.fp16,
            cls_model=cls_model if args.use_cls_gate else None,
        )
        print(f"[OK] Saved: {out_path}")
        if dices: print("Dice:", {k: f"{v:.4f}" for k, v in dices.items()})
        return

    # 目錄模式
    if args.imgdir is None:
        raise SystemExit("Please specify --image or --imgdir.")

    outdir = args.outdir or os.path.join(os.getcwd(), "predict")
    os.makedirs(outdir, exist_ok=True)

    images = sorted(glob.glob(os.path.join(args.imgdir, "*.nii.gz")))
    if len(images) == 0:
        raise SystemExit(f"No .nii.gz found in: {args.imgdir}")

    sum_dice = {1: 0.0, 2: 0.0, 3: 0.0}
    cnt_dice = {1: 0,   2: 0,   3: 0}
    
    lab_index = _index_labels(args.labdir) if args.labdir else {}

    for img_path in images:
        base = os.path.splitext(os.path.basename(img_path))[0]
        if base.endswith(".nii"): base = base[:-4]
        out_path = os.path.join(outdir, f"{base}_predict.nii.gz")

        # 嘗試找標註（若有，計算 Dice）
        label_path = None
        if args.labdir:
            stem = base
            label_path = lab_index.get(stem, None)
            cand1 = os.path.join(args.labdir, f"{base}.nii.gz")
            cand2 = os.path.join(args.labdir, f"{base}_label.nii.gz")
            if os.path.exists(cand1): label_path = cand1
            elif os.path.exists(cand2): label_path = cand2

        out_path, dices = process_case(
            image_path=img_path, out_path=out_path,
            model12=model12, model3=model3,
            roi12=roi12, roi3=roi3, swb=args.swb, device=device,
            label_path=label_path,
            sw_overlap=args.sw_overlap, sw_mode=args.sw_mode, use_fp16=args.fp16,
            cls_model=cls_model if args.use_cls_gate else None,
        )
        print(f"[OK] Saved: {out_path}")
        if dices:
            print("Dice:", {k: f"{v:.4f}" for k, v in dices.items()})
            for c, v in dices.items():
                sum_dice[c] += v; cnt_dice[c] += 1

    if any(cnt_dice[c] > 0 for c in [1,2,3]):
        print("\n=== Average Dice over labeled cases ===")
        for c in [1,2,3]:
            if cnt_dice[c] > 0:
                print(f"class {c}: {sum_dice[c]/cnt_dice[c]:.4f}  (n={cnt_dice[c]})")


if __name__ == "__main__":
    main()
