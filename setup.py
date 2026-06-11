# ═══════════════════════════════════════
# ファイル名：setup.py
# 概要：フォルダ構成と栄養素DB（ぷっちょ版・6種類）を自動生成
# 使い方：python setup.py
# 作成日：2024年
# ═══════════════════════════════════════

import os
import pandas as pd
from pathlib import Path


# 識別する6種類のお菓子
CLASS_NAMES = ["parinko", "pocky", "kitkat", "puccho", "marshmallow", "potato_chips"]


def create_folders():
    """snack_projectフォルダ構成を作成"""
    root = Path("snack_project")
    folders = [
        root / "dataset" / "train",
        root / "dataset" / "val",
        root / "models",
    ]
    for split in ["train", "val"]:
        for cls in CLASS_NAMES:
            folders.append(root / "dataset" / split / cls)

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
        print(f"✅ フォルダ作成: {folder}")

    for split in ["train", "val"]:
        for cls in CLASS_NAMES:
            (root / "dataset" / split / cls / ".gitkeep").touch(exist_ok=True)
    print("\n📁 フォルダ作成完了\n")


def create_nutrition_csv():
    """栄養素データベースを生成"""
    nutrition_data = {
        "snack_name": [
            "ぱりんこ",
            "ポッキーチョコレート",
            "キットカット ミニ",
            "ぷっちょ ブドウ味",
            "マシュマロ",
            "ポテトチップス うすしお",
        ],
        "class_name": [
            "parinko", "pocky", "kitkat", "puccho", "marshmallow", "potato_chips"
        ],
        "kcal_per_100g":    [493,    490,    552,    400,    324,    560],
        "carbs_per_100g":   [68.5,   66.9,   59.5,   89.0,   79.3,   54.0],
        "fat_per_100g":     [22.6,   21.7,   31.0,   3.8,    0.0,    36.0],
        "protein_per_100g": [4.2,    9.0,    7.2,    0.5,    2.1,    5.1],
        "serving_size_g":   [20,     29,     11.6,   25,     10,     60],
        "kcal_per_serving": [99,     142,    64,     100,    32,     336],
    }
    df = pd.DataFrame(nutrition_data)
    csv_path = Path("snack_project") / "nutrition.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ nutrition.csv作成: {csv_path}")
    print("\n📋 内容:")
    print(df.to_string(index=False))


def main():
    print("=" * 60)
    print("  お菓子カロリー推定AIアプリ セットアップ（ぷっちょ版）")
    print("=" * 60)
    create_folders()
    create_nutrition_csv()
    print("\n" + "=" * 60)
    print("  🎉 セットアップ完了！")
    print("=" * 60)


if __name__ == "__main__":
    main()
