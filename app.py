import cv2
import fitz  # PyMuPDF
import numpy as np
import streamlit as st
from skimage.metrics import structural_similarity as ssim

st.set_page_config(page_title="刻印画像解析システム", layout="wide")


# --- 1. パスワード認証機能 ---
def check_password():
    def password_entered():
        # 【重要】運用時は任意のパスワードに変更してください
        if st.session_state["password"] == "G7k2M9xP":
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


# --- 2. 画像解析処理ロジック ---
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


def get_marker_corners(img):
    """ArUcoマーカー(4箇所)の検出"""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, _ = detector.detectMarkers(img)

    if ids is None or len(ids) < 4:
        return None

    marker_centers = {}
    for i, marker_id in enumerate(ids.flatten()):
        c = corners[i][0]
        marker_centers[marker_id] = c.mean(axis=0)

    # ID 0, 1, 2, 3 の順で取得
    try:
        pts = np.array(
            [
                marker_centers[0],
                marker_centers[1],
                marker_centers[2],
                marker_centers[3],
            ],
            dtype=np.float32,
        )
        return pts
    except KeyError:
        return None


# --- 3. メイン画面制御 ---
if check_password():
    st.title("🔍 刻印図案解析システム")

    col1, col2 = st.columns(2)
    with col1:
        pdf_file = st.file_uploader("正解データ (PDF)", type=["pdf"])
    with col2:
        photo_file = st.file_uploader(
            "撮影画像 (JPEG/PNG)", type=["jpg", "jpeg", "png"]
        )

    if pdf_file and photo_file:
        if st.button("解析実行", type="primary"):
            # メモリ上でバイナリ読み込み（ディスク非保存）
            pdf_bytes = pdf_file.read()
            pdf_img = process_pdf(pdf_bytes)

            file_bytes = np.frombuffer(photo_file.read(), dtype=np.uint8)
            photo_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # マーカー検出
            pdf_pts = get_marker_corners(pdf_img)
            photo_pts = get_marker_corners(photo_img)

            if pdf_pts is None or photo_pts is None:
                st.error(
                    "エラー: ArUcoマーカー（ID 0〜3の4箇所）が検出できませんでした。撮影画像またはPDFを確認してください。"
                )
            else:
                # 射影変換（位置・スケール・傾き補正）
                H, _ = cv2.findHomography(photo_pts, pdf_pts)
                h, w = pdf_img.shape[:2]
                aligned_photo = cv2.warpPerspective(photo_img, H, (w, h))

                # グレースケール変換とSSIM比較
                gray_pdf = cv2.cvtColor(pdf_img, cv2.COLOR_BGR2GRAY)
                gray_photo = cv2.cvtColor(aligned_photo, cv2.COLOR_BGR2GRAY)

                score, diff = ssim(gray_pdf, gray_photo, full=True)
                diff = (diff * 255).astype("uint8")

                thresh = cv2.threshold(
                    diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
                )[1]
                contours, _ = cv2.findContours(
                    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                result = aligned_photo.copy()
                diff_count = 0
                for c in contours:
                    if cv2.contourArea(c) > 50:  # 微小ノイズ除外
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
                    st.success("判定: 一致（相相違点は検出されませんでした）")
                else:
                    st.warning(
                        f"判定: 不一致（{diff_count} 箇所の相違点を検出）"
                    )

                st.image(
                    cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                    caption="解析結果画像 (差分箇所を赤枠で表示)",
                    use_container_width=True,
                )
