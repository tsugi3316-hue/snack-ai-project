# ═══════════════════════════════════════
# ファイル名：rename_images.py
# 概要：撮影画像を自動リネーム&train/val振り分け（HEIC対応強化版）
# 使い方：python rename_images.py
# 作成日：2024年
# ═══════════════════════════════════════

import os, shutil, random
from pathlib import Path
from PIL import Image

# HEIC対応（iPhone写真用）
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    print("✅ HEIC形式に対応しました（iPhone写真OK）")
except ImportError:
    print("⚠️ pillow-heifが未インストール。HEICが読めない場合: pip install pillow-heif")

# 再現性のためのシード固定
random.seed(42)

# 設定
RAW_DIR = Path("raw_images")
DATASET_DIR = Path("snack_project") / "dataset"
TRAIN_RATIO = 0.8
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".bmp"}

# 日本語/英語フォルダ名対応マッピング（ぷっちょ版）
CLASS_MAPPING = {
    "ぱりんこ": "parinko", "パリンコ": "parinko", "parinko": "parinko",
    "ポッキー": "pocky", "pocky": "pocky", "Pocky": "pocky",
    "キットカット": "kitkat", "kitkat": "kitkat", "KitKat": "kitkat",
    "ぷっちょ": "puccho", "ぷっちょブドウ": "puccho",
    "ぷっちょブドウ味": "puccho", "プッチョ": "puccho", "puccho": "puccho",
    "マシュマロ": "marshmallow", "marshmallow": "marshmallow",
    "ポテチ": "potato_chips", "ポテトチップス": "potato_chips",
    "ポテチうすしお": "potato_chips", "ポテトチップスうすしお": "potato_chips",
    "potato_chips": "potato_chips",
}


def is_image_file(path):
    """画像ファイルか判定"""
    return path.suffix.lower() in VALID_EXTENSIONS


def convert_to_jpg(src, dst):
    """画像をJPG形式に統一変換（エラー耐性強化）"""
    try:
        img = Image.open(src)
        # 画像を完全に読み込む（遅延読込を防ぐ）
        img.load()
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(dst, "JPEG", quality=95)
        return True
    except Exception as e:
        # エラー詳細を表示
        error_msg = str(e)[:50]  # エラーメッセージを短く
        print(f"   ❌ 変換失敗: {src.name} ({src.suffix}) → {error_msg}")
        return False


def count_existing(class_name):
    """既存ファイル数をカウント"""
    count = 0
    for split in ["train", "val"]:
        d = DATASET_DIR / split / class_name
        if d.exists():
            count += len([f for f in d.iterdir()
                         if f.is_file() and not f.name.startswith(".")])
    return count


def setup_raw_folders():
    """raw_images下に各クラスフォルダを作成"""
    RAW_DIR.mkdir(exist_ok=True)
    jp_folders = ["ぱりんこ", "ポッキー", "キットカット",
                  "ぷっちょ", "マシュマロ", "ポテチ"]
    for jp in jp_folders:
        (RAW_DIR / jp).mkdir(exist_ok=True)
        print(f"   ✅ 作成: {RAW_DIR / jp}")
    print("\n💡 各フォルダに写真を入れてから、再度実行してください\n")


def rename_and_split(append_mode=False):
    """画像をリネーム&train/valに振り分け"""
    print("=" * 60)
    print("  画像リネーム&自動仕分け（HEIC対応版）")
    print("=" * 60)
    if not RAW_DIR.exists():
        print(f"\n❌ {RAW_DIR}がありません。先に1)でフォルダ作成してください")
        return

    total = 0
    failed_total = 0
    stats = {}
    for folder in sorted(RAW_DIR.iterdir()):
        if not folder.is_dir():
            continue
        cls = CLASS_MAPPING.get(folder.name)
        if cls is None:
            print(f"\n⚠️ 未対応: {folder.name} → スキップ")
            continue
        print(f"\n📂 処理中: {folder.name} → {cls}")

        # 全ファイル取得
        all_files = sorted([f for f in folder.iterdir() if f.is_file()])
        # 画像のみフィルター
        files = [f for f in all_files if is_image_file(f)]
        # 画像以外のファイルを表示
        non_images = [f for f in all_files if not is_image_file(f)
                     and not f.name.startswith(".")]
        if non_images:
            print(f"   ⚠️ 画像以外のファイルあり（スキップ）:")
            for nf in non_images[:5]:
                print(f"      - {nf.name}")
            if len(non_images) > 5:
                print(f"      （他 {len(non_images)-5} 件）")

        if not files:
            print("   ⚠️ 画像なし")
            continue
        print(f"   📸 検出画像数: {len(files)}枚")
        random.shuffle(files)
        n_train = int(len(files) * TRAIN_RATIO)
        train_files, val_files = files[:n_train], files[n_train:]

        start_idx = count_existing(cls) if append_mode else 0
        if not append_mode:
            for split in ["train", "val"]:
                td = DATASET_DIR / split / cls
                if td.exists():
                    for f in td.iterdir():
                        if f.is_file() and not f.name.startswith("."):
                            f.unlink()

        # train配置
        td = DATASET_DIR / "train" / cls
        td.mkdir(parents=True, exist_ok=True)
        st = 0
        st_failed = 0
        for i, src in enumerate(train_files):
            dst = td / f"{cls}_{start_idx + i + 1:04d}.jpg"
            if convert_to_jpg(src, dst):
                st += 1
            else:
                st_failed += 1

        # val配置
        vd = DATASET_DIR / "val" / cls
        vd.mkdir(parents=True, exist_ok=True)
        sv = 0
        sv_failed = 0
        for i, src in enumerate(val_files):
            dst = vd / f"{cls}_{start_idx + len(train_files) + i + 1:04d}.jpg"
            if convert_to_jpg(src, dst):
                sv += 1
            else:
                sv_failed += 1

        failed = st_failed + sv_failed
        print(f"   ✅ train: {st}枚 / val: {sv}枚 / 失敗: {failed}枚")
        stats[cls] = {"train": st, "val": sv, "failed": failed}
        total += st + sv
        failed_total += failed

    # 結果サマリー
    print("\n" + "=" * 60)
    print("  処理結果サマリー")
    print("=" * 60)
    print(f"\n{'クラス名':<15} {'train':<8} {'val':<8} {'失敗':<8} {'合計':<6}")
    print("-" * 50)
    for cls, s in stats.items():
        print(f"{cls:<15} {s['train']:<8} {s['val']:<8} {s['failed']:<8} {s['train']+s['val']:<6}")
    print(f"\n🎉 合計 {total}枚処理完了！")
    if failed_total > 0:
        print(f"⚠️ 変換失敗: {failed_total}枚（学習データから除外）")


def main():
    print("\n" + "=" * 60)
    print("  📷 画像リネーム&仕分けツール（HEIC対応版）")
    print("=" * 60)
    print("\n1) raw_imagesフォルダ準備")
    print("2) 上書きモード（既存削除&新規振り分け）")
    print("3) 追加モード（既存に追加）")
    print("4) 終了")
    choice = input("\n選択 (1/2/3/4): ").strip()
    if choice == "1":
        setup_raw_folders()
    elif choice == "2":
        c = input("\n⚠️ 既存画像削除されます。続行? (y/n): ").strip().lower()
        if c == "y":
            rename_and_split(False)
        else:
            print("キャンセル")
    elif choice == "3":
        rename_and_split(True)
    elif choice == "4":
        print("終了")
    else:
        print("無効な選択")


if __name__ == "__main__":
    main()
