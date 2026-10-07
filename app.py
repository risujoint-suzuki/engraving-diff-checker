import cv2
import fitz  # PyMuPDF
import numpy as np
import streamlit as st
from skimage.metrics import structural_similarity as ssim

st.set_page_config(page_title="刻印画像解析システム", layout="wide")


# --- 1. パスワード認証機能 ---
def check_password():
    def password_entered():
        if st.session_state["password"] == "SecretPass123!":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input(
            "パスワードを入力してください:",
            type="password",
            on_change=password_entered,
            key="password",
        )
        return False
    elif not st.session_state["password_correct"]:
        st.text_input(
            "パスワードを入力してください:",
            type="password",
            on_change=password_entered,
            key="password",
        )
        st.error("パスワードが正しくありません")
        return False
    return True


# --- 2. PDF処理ロジック ---
def process_pdf(pdf_bytes):
    """PDFをメモリ上で画像化"""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=300)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.h, pix.w, pix.n
    )
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


# --- 3. ORB特徴点マッチングによる自動位置合わせ ---
def align_images_orb(img_pdf, img_photo, max_features=5000):
    """マーカーなしで2枚の画像の位置・角度・スケールを揃える"""
    gray_pdf = cv2.cvtColor(img_pdf, cv2.COLOR_BGR2GRAY)
    gray_photo = cv2.cvtColor(img_photo, cv2.COLOR_BGR2GRAY)

    # ORB特徴点検出器の初期化
    orb = cv2.ORB_create(nfeatures=max_features)

    # 特徴点と特徴量の抽出
    kp_pdf, des_pdf = orb.detectAndCompute(gray_pdf, None)
    kp_photo, des_photo = orb.detectAndCompute(gray_photo, None)

    if des_pdf is None or des_photo is None:
        return None, "特徴点が検出できませんでした。"

    # マッチング器（BFMatcher）で特徴点を接続
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des_pdf, des_photo)

    # 距離の近い（一致度の高い）上位マッチング点を抽出
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[: int(len(matches) * 0.2)]  # 上位20%を使用

    if len(good_matches) < 4:
        return (
            None,
            "十分な一致点が見つかりませんでした。撮影画像が不鮮明か、角度がつきすぎています。",
        )

    # 特徴点の座標配列を作成
    pts_pdf = np.float32([kp_pdf[m.queryIdx].pt for m in good_matches]).reshape(
        -1, 1, 2
    )
    pts_photo = np.float32(
        [kp_photo[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    # ホモグラフィ行列（幾何補正行列）を算出 (RANSACで外れ値を除外)
    H, mask = cv2.findHomography(pts_photo, pts_pdf, cv2.RANSAC, 5.0)

    if H is None:
        return None, "幾何補正の計算に失敗しました。"

    # 撮影画像をPDFのサイズ・視点に合わせて射影変換
    h, w = img_pdf.shape[:2]
    aligned_photo = cv2.warpPerspective(img_photo, H, (w, h))

    return aligned_photo, None


# --- 4. メイン画面制御 ---
if check_password():
    st.title("🔍 刻印・図案 差分解析システム (ORB自動補正版)")

    col1, col2 = st.columns(2)
    with col1:
        pdf_file = st.file_uploader("正解データ (PDF)", type=["pdf"])
    with col2:
        photo_file = st.file_uploader(
            "撮影画像 (JPEG/PNG)", type=["jpg", "jpeg", "png"]
        )

    if pdf_file and photo_file:
        if st.button("解析実行", type="primary"):
            # メモリ上でバイナリ読み込み
            pdf_bytes = pdf_file.read()
            pdf_img = process_pdf(pdf_bytes)

            file_bytes = np.frombuffer(photo_file.read(), dtype=np.uint8)
            photo_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            st.info("ORB特徴点による自動位置合わせを実施中...")

            # ORBによる位置合わせ
            aligned_photo, error_msg = align_images_orb(pdf_img, photo_img)

            if aligned_photo is None:
                st.error(f"エラー: {error_msg}")
            else:
                # グレースケール変換とSSIM比較
                gray_pdf = cv2.cvtColor(pdf_img, cv2.COLOR_BGR2GRAY)
                gray_photo = cv2.cvtColor(aligned_photo, cv2.COLOR_BGR2GRAY)

                score, diff = ssim(gray_pdf, gray_photo, full=True)
                diff = (diff * 255).astype("uint8")

                # 差分領域の抽出
                thresh = cv2.threshold(
                    diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
                )[1]
                contours, _ = cv2.findContours(
                    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                result = aligned_photo.copy()
                diff_count = 0
                for c in contours:
                    if cv2.contourArea(c) > 50:  # ノイズ除外閾値
                        x, y, w_box, h_box = cv2.boundingRect(c)
                        cv2.rectangle(
                            result,
                            (x, y),
                            (x + w_box, y + h_box),
                            (0, 0, 255),
                            2,
                        )
                        diff_count += 1

                # 結果表示
                st.subheader("解析結果")
                if diff_count == 0:
                    st.success("判定: 一致（相違点は検出されませんでした）")
                else:
                    st.warning(
                        f"判定: 不一致（{diff_count} 箇所の相違点を検出）"
                    )

                st.image(
                    cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                    caption="解析結果画像 (自動補正済・差分箇所を赤枠で表示)",
                    use_container_width=True,
                )
