# ═══════════════════════════════════════
# ファイル名：train.py
# 概要：ResNet-50転移学習でお菓子6種類分類モデルを学習（ぷっちょ版）
# 実行環境：Google Colaboratory（GPU推奨）
# 作成日：2024年
# ═══════════════════════════════════════

# ════════════════════════════════════════════════════
# セクション0：Google Colab環境設定
# ════════════════════════════════════════════════════
# 【Colabで実行】ライブラリインストール
# !pip install japanize-matplotlib -q
# 【Colabで実行】ドライブマウント
# from google.colab import drive
# drive.mount('/content/drive')
# DATA_ROOT = '/content/drive/MyDrive/snack_ai_project/snack_project'

DATA_ROOT = './snack_project'  # ローカル実行用

import torch, random
import numpy as np

# シード固定
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 使用デバイス: {device}")
if torch.cuda.is_available():
    print(f"   GPU名: {torch.cuda.get_device_name(0)}")


# ════════════════════════════════════════════════════
# セクション1：ライブラリのインポート
# ════════════════════════════════════════════════════
import os, time, copy
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
try:
    import japanize_matplotlib
except ImportError:
    print("⚠️ japanize-matplotlib未インストール")
from tqdm import tqdm


# ════════════════════════════════════════════════════
# セクション2：データ準備
# ════════════════════════════════════════════════════
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 学習用データ拡張
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
# 検証用前処理
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

train_dir = os.path.join(DATA_ROOT, "dataset", "train")
val_dir = os.path.join(DATA_ROOT, "dataset", "val")
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                         shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                       shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

class_names = train_dataset.classes
num_classes = len(class_names)
print(f"\n📊 クラス名: {class_names}")
print(f"📊 クラス数: {num_classes}")
print(f"📊 学習データ: {len(train_dataset)}枚")
print(f"📊 検証データ: {len(val_dataset)}枚")


# ════════════════════════════════════════════════════
# セクション3：モデル構築（ResNet-50転移学習）
# ════════════════════════════════════════════════════
def build_model(num_classes):
    """ResNet-50転移学習モデル構築"""
    try:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    except Exception:
        model = models.resnet50(pretrained=True)
    for p in model.parameters():
        p.requires_grad = False
    in_feat = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_feat, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model

model = build_model(num_classes).to(device)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n🧠 ResNet-50ベース")
print(f"   総パラメータ: {total_params:,}")
print(f"   学習対象: {trainable_params:,}")


# ════════════════════════════════════════════════════
# セクション4：学習設定
# ════════════════════════════════════════════════════
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
NUM_EPOCHS = 30
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)


class EarlyStopping:
    """連続patience回val_loss改善なしで学習停止"""
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best = None
        self.counter = 0
        self.early_stop = False
    def __call__(self, val_loss):
        score = -val_loss
        if self.best is None:
            self.best = score
        elif score < self.best + self.delta:
            self.counter += 1
            print(f"   ⚠️ EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best = score
            self.counter = 0

early_stopping = EarlyStopping(patience=5)


# ════════════════════════════════════════════════════
# セクション5：学習実行
# ════════════════════════════════════════════════════
def train_one_epoch(model, loader, criterion, optimizer, device):
    """1エポックの学習"""
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    for inputs, labels in tqdm(loader, desc="  Train", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    """検証"""
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="  Val  ", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return running_loss / total, correct / total, all_preds, all_labels


# 学習ループ
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_val_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())
save_dir = os.path.join(DATA_ROOT, "models")
os.makedirs(save_dir, exist_ok=True)
best_model_path = os.path.join(save_dir, "best_model.pth")

print(f"\n🚀 学習開始（最大 {NUM_EPOCHS} エポック）\n")
start = time.time()

for epoch in range(NUM_EPOCHS):
    es = time.time()
    print(f"━━━━━━━ Epoch {epoch+1}/{NUM_EPOCHS} ━━━━━━━")
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
    scheduler.step()
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    print(f"  Train: loss={train_loss:.4f} acc={train_acc:.4f}")
    print(f"  Val  : loss={val_loss:.4f} acc={val_acc:.4f}")
    print(f"  時間: {time.time()-es:.1f}秒 LR: {scheduler.get_last_lr()[0]:.6f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save({
            "model_state_dict": best_model_wts,
            "class_names": class_names,
            "num_classes": num_classes,
        }, best_model_path)
        print(f"  💾 best_model.pth保存（val_acc={best_val_acc:.4f}）")
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("\n⏹ EarlyStopping終了")
        break

print(f"\n🎉 学習完了！ 総時間: {(time.time()-start)/60:.2f}分")
print(f"   最高検証精度: {best_val_acc:.4f}")
model.load_state_dict(best_model_wts)


# ════════════════════════════════════════════════════
# セクション6：結果可視化
# ════════════════════════════════════════════════════
def plot_history(h):
    """学習曲線プロット"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0,0].plot(h["train_loss"], color="blue"); axes[0,0].set_title("Train Loss")
    axes[0,1].plot(h["val_loss"], color="red"); axes[0,1].set_title("Val Loss")
    axes[1,0].plot(h["train_acc"], color="blue"); axes[1,0].set_title("Train Accuracy")
    axes[1,1].plot(h["val_acc"], color="red"); axes[1,1].set_title("Val Accuracy")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model, loader, class_names, device):
    """混同行列を可視化"""
    en_to_jp = {
        "parinko": "ぱりんこ", "pocky": "ポッキー",
        "kitkat": "キットカット", "puccho": "ぷっちょ",
        "marshmallow": "マシュマロ", "potato_chips": "ポテチ",
    }
    jp_labels = [en_to_jp.get(c, c) for c in class_names]
    _, _, all_preds, all_labels = validate(model, loader, criterion, device)
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
               xticklabels=jp_labels, yticklabels=jp_labels)
    plt.xlabel("予測"); plt.ylabel("正解")
    plt.title("混同行列")
    plt.tight_layout()
    plt.show()
    print("\n📊 分類レポート:")
    print(classification_report(all_labels, all_preds,
                                target_names=jp_labels, digits=4))


def show_misclassified(model, loader, class_names, device, max_show=8):
    """誤分類画像を表示"""
    en_to_jp = {
        "parinko": "ぱりんこ", "pocky": "ポッキー",
        "kitkat": "キットカット", "puccho": "ぷっちょ",
        "marshmallow": "マシュマロ", "potato_chips": "ポテチ",
    }
    model.eval()
    mis_imgs, mis_preds, mis_labels = [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            i_d, l_d = inputs.to(device), labels.to(device)
            outputs = model(i_d)
            _, predicted = outputs.max(1)
            for i in range(inputs.size(0)):
                if predicted[i] != l_d[i]:
                    mis_imgs.append(inputs[i].cpu())
                    mis_preds.append(predicted[i].cpu().item())
                    mis_labels.append(l_d[i].cpu().item())
                if len(mis_imgs) >= max_show:
                    break
            if len(mis_imgs) >= max_show:
                break
    if not mis_imgs:
        print("✅ 誤分類なし！")
        return
    n = len(mis_imgs)
    fig, axes = plt.subplots(1, n, figsize=(3*n, 4))
    if n == 1:
        axes = [axes]
    for i, (img, p, l) in enumerate(zip(mis_imgs, mis_preds, mis_labels)):
        i_np = img.numpy().transpose(1, 2, 0)
        i_np = i_np * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
        i_np = np.clip(i_np, 0, 1)
        axes[i].imshow(i_np)
        axes[i].set_title(f"予測:{en_to_jp[class_names[p]]}\n正解:{en_to_jp[class_names[l]]}",
                        fontsize=9)
        axes[i].axis("off")
    plt.suptitle("誤分類サンプル")
    plt.tight_layout()
    plt.show()


plot_history(history)
plot_confusion_matrix(model, val_loader, class_names, device)
show_misclassified(model, val_loader, class_names, device, max_show=8)


# ════════════════════════════════════════════════════
# セクション7：推論テスト
# ════════════════════════════════════════════════════
def predict_image(image_path, model, class_names, device, top_k=3):
    """1枚画像の推論テスト"""
    model.eval()
    img = Image.open(image_path).convert("RGB")
    img_t = val_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)[0]
        topk_probs, topk_idx = probs.topk(top_k)
    en_to_jp = {
        "parinko": "ぱりんこ", "pocky": "ポッキー",
        "kitkat": "キットカット", "puccho": "ぷっちょ ブドウ味",
        "marshmallow": "マシュマロ", "potato_chips": "ポテチうすしお",
    }
    print(f"\n🔍 推論対象: {image_path}")
    print(f"   Top-{top_k}予測:")
    for r, (p, idx) in enumerate(zip(topk_probs, topk_idx), 1):
        cls_en = class_names[idx]
        cls_jp = en_to_jp.get(cls_en, cls_en)
        print(f"   {r}位: {cls_jp:<15} {p.item()*100:.2f}%")
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"予測: {en_to_jp.get(class_names[topk_idx[0]], '?')} ({topk_probs[0].item()*100:.1f}%)")
    plt.show()


if len(val_dataset) > 0:
    sample_path, _ = val_dataset.samples[0]
    predict_image(sample_path, model, class_names, device, top_k=3)

print("\n✨ train.py全セクション完了！")
