# ═══════════════════════════════════════════════════════════════════
# ファイル名：app.py
# 概要      ：お菓子カロリー推定AIアプリ（カメラ撮影 + アップロード対応版）
# 作成日    ：2024年
# 使い方    ：streamlit run app.py
# 入力方法  ：① カメラで撮影  ② ファイルからアップロード
# ═══════════════════════════════════════════════════════════════════

# ────────────────────────────────────────────────
# 1. ライブラリのインポート
# ────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import io

# ────────────────────────────────────────────────
# 2. ページ基本設定
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="お菓子カロリー推定AI",
    page_icon="🍪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ────────────────────────────────────────────────
# 3. カスタムCSS（見た目を整える）
# ────────────────────────────────────────────────
st.markdown("""
<style>
    /* メインタイトル */
    .main-title {
        font-size: 38px;
        font-weight: bold;
        color: #FF6F00;
        text-align: center;
        padding: 10px;
        margin-bottom: 10px;
    }
    /* サブタイトル */
    .sub-title {
        font-size: 16px;
        color: #666;
        text-align: center;
        margin-bottom: 30px;
    }
    /* 結果ボックス */
    .result-box {
        background: linear-gradient(135deg, #FFE082 0%, #FFB74D 100%);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .result-name {
        font-size: 32px;
        font-weight: bold;
        color: #5D4037;
    }
    .result-conf {
        font-size: 22px;
        color: #6D4C41;
        margin-top: 8px;
    }
    /* 栄養テーブルボックス */
    .nutrition-box {
        background-color: #FFF3E0;
        border-left: 6px solid #FF6F00;
        padding: 18px;
        border-radius: 8px;
        margin: 15px 0;
    }
    /* 警告/情報ボックス */
    .mode-banner-prod {
        background-color: #C8E6C9;
        color: #1B5E20;
        padding: 12px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .mode-banner-dummy {
        background-color: #FFF9C4;
        color: #F57F17;
        padding: 12px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────
# 4. 設定値
# ────────────────────────────────────────────────
# モデル・データファイルのパス
MODEL_PATH = Path("snack_project") / "models" / "best_model.pth"
NUTRITION_PATH = Path("snack_project") / "nutrition.csv"

# クラス名 → 日本語表示名
CLASS_JP = {
    "parinko": "ぱりんこ",
    "pocky": "ポッキーチョコレート",
    "kitkat": "キットカット",
    "puccho": "ぷっちょ ブドウ味",
    "marshmallow": "マシュマロ",
    "potato_chips": "ポテトチップス うすしお",
}

# クラス名 → 絵文字
CLASS_EMOJI = {
    "parinko": "🍘",
    "pocky": "🥢",
    "kitkat": "🍫",
    "puccho": "🍇",
    "marshmallow": "🍡",
    "potato_chips": "🥔",
}

# 1日推奨摂取カロリー(成人目安)
DAILY_KCAL = 2000

# ────────────────────────────────────────────────
# 5. 栄養データの読み込み
# ────────────────────────────────────────────────
@st.cache_data
def load_nutrition_db():
    """nutrition.csv を読み込む"""
    if not NUTRITION_PATH.exists():
        return None
    return pd.read_csv(NUTRITION_PATH, encoding="utf-8-sig")

# ────────────────────────────────────────────────
# 6. AIモデルの読み込み
# ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """
    学習済みモデルを読み込む。
    モデルが存在しない場合はダミーモードを返す。
    返り値: (model, class_names, mode)
        mode = "production" or "dummy"
    """
    class_names = list(CLASS_JP.keys())
    
    # モデルファイルが無い → ダミーモード
    if not MODEL_PATH.exists():
        return None, class_names, "dummy"
    
    try:
        # ResNet-50 構造を構築（学習時と同じ）
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        # 学習時の最終層の構造に合わせる
        model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, len(class_names))
        )
        
        # チェックポイント読み込み
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        
        # state_dict 形式の判定
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            # チェックポイントにクラス名が保存されていれば使用
            if "class_names" in checkpoint:
                class_names = checkpoint["class_names"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model, class_names, "production"
    
    except Exception as e:
        st.error(f"❌ モデルの読み込みに失敗しました: {e}")
        return None, class_names, "dummy"

# ────────────────────────────────────────────────
# 7. 画像の前処理
# ────────────────────────────────────────────────
def preprocess_image(image):
    """PIL画像をモデル入力用テンソルに変換"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    if image.mode != "RGB":
        image = image.convert("RGB")
    return transform(image).unsqueeze(0)

# ────────────────────────────────────────────────
# 8. 推論処理
# ────────────────────────────────────────────────
def predict(image, model, class_names, mode):
    """
    画像から各クラスの確信度を計算
    返り値: {クラス名: 確率, ...}
    """
    if mode == "dummy":
        # ダミーモード：Dirichlet分布でランダムに確率を生成
        np.random.seed(None)  # 毎回違う結果
        probs = np.random.dirichlet(np.ones(len(class_names)) * 0.5)
        # 最大確率を 60-95% にスケール（リアル感を出す）
        probs = probs / probs.max() * np.random.uniform(0.6, 0.95)
        probs = probs / probs.sum()
        return {name: float(p) for name, p in zip(class_names, probs)}
    
    # 本番モード：実際にモデルで推論
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0].numpy()
    return {name: float(p) for name, p in zip(class_names, probs)}

# ────────────────────────────────────────────────
# 9. 栄養成分表の表示
# ────────────────────────────────────────────────
def display_nutrition_table(class_en, df):
    """栄養成分表を表示し、該当行を返す"""
    row = df[df["class_name"] == class_en]
    if row.empty:
        st.warning("⚠️ このお菓子の栄養データが登録されていません")
        return None
    row = row.iloc[0]
    
    jp_name = CLASS_JP.get(class_en, class_en)
    emoji = CLASS_EMOJI.get(class_en, "🍪")
    
    st.markdown(f"""
    <div class="nutrition-box">
        <h3 style="margin-top:0;">{emoji} {jp_name} の栄養成分（100gあたり）</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # 4カラムで主要栄養素表示
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🔥 カロリー", f"{row['kcal_per_100g']:.0f} kcal")
    with col2:
        st.metric("🍞 炭水化物", f"{row['carbs_per_100g']:.1f} g")
    with col3:
        st.metric("🧈 脂質", f"{row['fat_per_100g']:.1f} g")
    with col4:
        st.metric("🥩 たんぱく質", f"{row['protein_per_100g']:.1f} g")
    
    # 標準1食分情報
    st.info(
        f"📦 **標準1食分**: {row['serving_size_g']:.1f}g "
        f"≒ **{row['kcal_per_serving']:.0f} kcal**"
    )
    return row

# ────────────────────────────────────────────────
# 10. メインUI
# ────────────────────────────────────────────────
def main():
    # === ヘッダー ===
    st.markdown('<div class="main-title">🍪 お菓子カロリー推定AI 🍪</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-title">画像をアップロード、またはカメラで撮影するだけ！AIがお菓子を判定し、栄養成分とカロリーを表示します</div>',
                unsafe_allow_html=True)
    
    # === サイドバー（説明） ===
    with st.sidebar:
        st.header("📖 使い方")
        st.markdown("""
        1. **入力方法を選ぶ**
           - 📷 カメラで撮影
           - 📁 ファイルをアップロード
        2. **AIが自動判定**
        3. **栄養成分表が表示**
        4. **重さを入力 → カロリー計算**
        """)
        
        st.divider()
        st.header("🍬 対応お菓子")
        for en, jp in CLASS_JP.items():
            emoji = CLASS_EMOJI.get(en, "🍪")
            st.write(f"{emoji} {jp}")
        
        st.divider()
        st.caption("📌 撮影のコツ")
        st.caption("・お菓子を中央に配置")
        st.caption("・明るい場所で撮影")
        st.caption("・1種類ずつが理想")
    
    # === 栄養データ読み込み ===
    nutrition_df = load_nutrition_db()
    if nutrition_df is None:
        st.error(f"❌ {NUTRITION_PATH} が見つかりません。")
        st.info("先に `python setup.py` を実行してください。")
        st.stop()
    
    # === モデル読み込み ===
    model, class_names, mode = load_model()
    
    # === モードバナー ===
    if mode == "production":
        st.markdown(
            '<div class="mode-banner-prod">✅ 本番モードで動作中（学習済みモデル使用）</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="mode-banner-dummy">⚠️ ダミーモードで動作中（モデル未配置のため、ランダム結果）</div>',
            unsafe_allow_html=True
        )
    
    # ═══════════════════════════════════════════
    # ⭐ 画像入力エリア（カメラ撮影 or アップロード）
    # ═══════════════════════════════════════════
    st.markdown("### 📸 画像を入力してください")
    
    # タブで切り替え
    tab1, tab2 = st.tabs(["📷 カメラで撮影", "📁 ファイルからアップロード"])
    
    camera_file = None
    uploaded_file = None
    
    with tab1:
        st.markdown("""
        💡 **使い方**：
        - 「**Take Photo**」ボタンでお菓子を撮影
        - スマホでもPCのWebカメラでも動作します
        - 初回は **カメラへのアクセス許可** が表示されます
        """)
        camera_file = st.camera_input(
            "お菓子にカメラを向けて、Take Photo ボタンを押してください",
            key="camera"
        )
    
    with tab2:
        st.markdown("💡 撮影済みの画像ファイルを選択してください")
        uploaded_file = st.file_uploader(
            "画像ファイルを選択（JPG / PNG / JPEG）",
            type=["jpg", "jpeg", "png"],
            key="uploader"
        )
    
    # どちらかの入力を取得（カメラ優先）
    image_source = camera_file if camera_file else uploaded_file
    
    # ═══════════════════════════════════════════
    # 画像が入力されたら推論実行
    # ═══════════════════════════════════════════
    if image_source is not None:
        try:
            # 画像を開く
            image = Image.open(image_source)
        except Exception as e:
            st.error(f"❌ 画像の読み込みに失敗しました: {e}")
            st.stop()
        
        # 入力画像と判定結果を左右に並べる
        col_img, col_result = st.columns([1, 1])
        
        with col_img:
            st.markdown("#### 📥 入力画像")
            st.image(image, use_container_width=True)
        
        with col_result:
            st.markdown("#### 🤖 AI判定結果")
            
            with st.spinner("判定中..."):
                confidences = predict(image, model, class_names, mode)
            
            # 最も確率が高いクラス
            top_class = max(confidences, key=confidences.get)
            top_conf = confidences[top_class]
            jp_name = CLASS_JP.get(top_class, top_class)
            emoji = CLASS_EMOJI.get(top_class, "🍪")
            
            # 結果ボックス
            st.markdown(f"""
            <div class="result-box">
                <div class="result-name">{emoji} {jp_name}</div>
                <div class="result-conf">確信度: {top_conf*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            # 全クラスの確信度をバーで表示
            st.markdown("**📊 全クラスの確信度**")
            sorted_confs = sorted(confidences.items(), key=lambda x: -x[1])
            for cls, conf in sorted_confs:
                jp = CLASS_JP.get(cls, cls)
                em = CLASS_EMOJI.get(cls, "🍪")
                st.progress(
                    float(conf),
                    text=f"{em} {jp}: {conf*100:.1f}%"
                )
        
        st.divider()
        
        # ═══════════════════════════════════════
        # 栄養成分表の表示
        # ═══════════════════════════════════════
        st.markdown("### 📊 栄養成分情報")
        nutrition_row = display_nutrition_table(top_class, nutrition_df)
        
        if nutrition_row is None:
            st.stop()
        
        st.divider()
        
        # ═══════════════════════════════════════
        # カロリー計算
        # ═══════════════════════════════════════
        st.markdown("### 🧮 食べた量からカロリーを計算")
        
        col_input, col_btn = st.columns([3, 1])
        with col_input:
            weight = st.number_input(
                "食べた重さ（g）を入力してください",
                min_value=0.0,
                max_value=500.0,
                value=float(nutrition_row["serving_size_g"]),
                step=1.0,
                help="標準1食分が初期値として入っています"
            )
        with col_btn:
            st.write("")
            st.write("")
            calc_btn = st.button("🔢 カロリー計算", use_container_width=True, type="primary")
        
        if calc_btn or weight != float(nutrition_row["serving_size_g"]):
            # 実食分のカロリー・栄養素を計算
            kcal = nutrition_row["kcal_per_100g"] * weight / 100
            carbs = nutrition_row["carbs_per_100g"] * weight / 100
            fat = nutrition_row["fat_per_100g"] * weight / 100
            protein = nutrition_row["protein_per_100g"] * weight / 100
            
            st.markdown(f"#### 📈 {weight:.0f}g 食べた場合の栄養")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("🔥 カロリー", f"{kcal:.0f} kcal")
            with c2:
                st.metric("🍞 炭水化物", f"{carbs:.1f} g")
            with c3:
                st.metric("🧈 脂質", f"{fat:.1f} g")
            with c4:
                st.metric("🥩 たんぱく質", f"{protein:.1f} g")
            
            # 1日推奨摂取量との比較
            ratio = kcal / DAILY_KCAL * 100
            st.markdown("#### 📊 1日推奨摂取カロリー（2,000kcal）との比較")
            st.progress(min(ratio/100, 1.0), text=f"1日推奨カロリーの {ratio:.1f}%")
            
            # 警告メッセージ
            if ratio >= 50:
                st.error(f"⚠️ 1日の半分以上のカロリーです！食べ過ぎ注意⚠️")
            elif ratio >= 30:
                st.warning(f"⚠️ 1日の {ratio:.1f}% を占めます。食べ過ぎに注意しましょう")
            elif ratio >= 15:
                st.info(f"💡 おやつとしては標準的な量です（1日の {ratio:.1f}%）")
            else:
                st.success(f"✅ 適量です（1日の {ratio:.1f}%）")
    
    else:
        # 画像が未入力の場合
        st.info("👆 上のタブから **カメラで撮影** または **ファイルをアップロード** してください")
    
    # === フッター ===
    st.divider()
    st.caption("🤖 Powered by ResNet-50 (Transfer Learning) | 開発：杉山達俊 | 2026年6月")

# ────────────────────────────────────────────────
# 11. エントリーポイント
# ────────────────────────────────────────────────
if __name__ == "__main__":
    main()
