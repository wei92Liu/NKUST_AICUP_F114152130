# 類神經網路期末報告 — AICUP 心臟 3D 影像分割（SwinUNETR / MONAI）

隊伍：TEAM_8929
隊員：劉育瑋
Private leaderboard：0.734497 / Rank 139


此 Repo 為 **AICUP 競賽專案**的程式碼整理與實驗紀錄，任務為 3D 心臟 CT（NIfTI `.nii.gz`）多類別分割：

- Class 0：背景（Background）
- Class 1：心肌（Myocardium）
- Class 2：主動脈瓣（Aortic Valve）
- Class 3：鈣化（Calcification, tiny lesion）

因 class3 體積極小且出現比例低，本專案採用「分支式」策略：
- **class1+2 主分割模型**：穩定分割大型結構（訓練時忽略 class3）
- **class3 專用模型**：將 class3 做成二元分割（calc / non-calc）
- **class3 classifier（可選）**：病人層級 has-calc / no-calc 門控，用於降低 class3 假陽性
- **推論融合**：以 class12 結果為底，class3 以「覆寫」方式合併（class3 優先）

硬體環境以 **RTX 3080 Ti（12GB VRAM）** 為主要考量，提供較不易 OOM 的推論參數與建議。

---

## 1. Repo 內容

主要腳本：
- `train_class12.py`：訓練 class1+2 分割（輸出 3 類：bg / 1 / 2，訓練時將 GT 的 class3 視為背景）
- `train_class3.py`：訓練 class3 二元分割（calc vs non-calc）
- `train_class3_classifier.py`：訓練病人層級 class3 classifier（可選）
- `infer_and_dice.py`：推論整合 + 融合輸出（可選評估 Dice / class3 指標）

---

## 2. 資料夾結構（自行準備競賽資料，不包含在 Repo）

建議放置方式（可依自己路徑調整，只要同步修改腳本內路徑或參數）：

```text
race/
  training_image/        # patientXXXX.nii.gz
  training_label/        # patientXXXX.nii.gz (GT segmentation)
  testing_image/         # patientXXXX.nii.gz
  prediction/            # 推論輸出資料夾（會自動建立）
  *.py
```

---

## 3. 環境安裝（Windows + Python 3.10）

### (A) 建議使用 venv（不一定要 Conda）
在專案根目錄（`race/`）開 PowerShell：

```powershell
python -m venv aicup
.\aicup\Scripts\activate
python -m pip install --upgrade pip
```

### (B) 安裝套件（範例）
> PyTorch 請依你的 CUDA 版本安裝對應版本（建議用官方指令）；其餘套件可用 pip。

```powershell
pip install monai nibabel numpy tqdm tensorboard
```

---

## 4. 訓練流程

### Step 1：訓練 class1 + class2（主分割）
```powershell
.\aicup\Scripts\python.exe .\train_class12.py
```

輸出（範例）：
- `best_swinunetr_class12.pth`

### Step 2：訓練 class3（二元 tiny lesion）
```powershell
.\aicup\Scripts\python.exe .\train_class3.py
```

輸出（範例）：
- `best_swinunetr_class3.pth`

### Step 3（可選）：訓練 class3 classifier（病人層級）
```powershell
.\aicup\Scripts\python.exe .\train_class3_classifier.py
```

輸出（範例）：
- `best_class3_classifier.pth`

---

## 5. 推論（testing_image → prediction）

### 推論整批資料（建議）
> **PowerShell 多行換行符號是反引號 `（不是 `^`）**  
> `^` 是 cmd.exe 的語法，在 PowerShell 會被當成一般字元導致參數錯誤。

```powershell
.\aicup\Scripts\python.exe .\infer_and_dice.py `
  --imgdir .\testing_image `
  --outdir .\prediction `
  --w12 .\best_swinunetr_class12.pth `
  --w3  .\best_swinunetr_class3.pth `
  --cls_w .\best_class3_classifier.pth `
  --gpu 0 --fp16 `
  --roi12 128 128 128 `
  --roi3  96 96 96 `
  --swb 4 --sw_overlap 0.25 --sw_mode constant `
  --use_cls_gate
```

### 3080 Ti 仍 OOM 的調整建議
優先嘗試：
- `--swb 4 → 2 → 1`
- `--roi3 96³ → 80³ → 64³`
- `--sw_mode gaussian`（較平滑，速度略慢）

---

## 6. 融合規則（推論端）

融合採用 **class3 優先覆寫**：
1. 先用 class12 模型得到 bg / class1 / class2
2. 再用 class3 模型得到二元 calc mask（0/1）
3. 對於 class3 預測為 1 的 voxel，最終輸出覆寫為 label=3
4. 若開啟 `--use_cls_gate`，classifier 判定「無鈣化」時可忽略 class3 覆寫，以降低假陽性

---

<!-- ## 7. 建議的 .gitignore（避免誤上傳資料與權重）

建立 `.gitignore`，加入：

```gitignore
# datasets
training_image/
training_label/
testing_image/
prediction/
predict/

# checkpoints
*.pth

# python
__pycache__/
*.pyc
.venv/
aicup/
*.log
```

--- -->

## 7. 參考資源
- PyTorch：https://pytorch.org/
- MONAI：https://monai.io/
- OpenAI GPT-5.1：https://openai.com/index/gpt-5-1/

---

## 8. 資料聲明
本 Repo 僅包含程式碼與說明文件，不包含競賽資料集與影像標註；競賽資料請依主辦單位規範使用與保存。
