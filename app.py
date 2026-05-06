# app.py — Fire Detection Demo với Streamlit
# Cài đặt: pip install streamlit ultralytics opencv-python Pillow

import streamlit as st
import cv2
import numpy as np
import time
import tempfile
from ultralytics import YOLO
from PIL import Image

# ============================================================
# CẤU HÌNH TRANG — phải là lệnh đầu tiên
# ============================================================
st.set_page_config(
    page_title="🔥 Fire Detection",
    page_icon="🔥",
    layout="wide",                  # giao diện rộng 2 cột
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS TUỲ CHỈNH GIAO DIỆN
# ============================================================
st.markdown("""
<style>
    /* Nền tổng thể */
    .stApp { background-color: #0f0f0f; color: #f0f0f0; }

    /* Tiêu đề */
    h1 { color: #FF4B4B !important; text-align: center; }
    h3 { color: #FF8C00 !important; }

    /* Card kết quả */
    .fire-alert {
        background: linear-gradient(135deg, #FF4B4B22, #FF000044);
        border: 2px solid #FF4B4B;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        font-size: 1.4em;
        font-weight: bold;
        color: #FF4B4B;
        margin: 10px 0;
    }
    .safe-alert {
        background: linear-gradient(135deg, #00FF7722, #00CC4444);
        border: 2px solid #00FF77;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        font-size: 1.4em;
        font-weight: bold;
        color: #00FF77;
        margin: 10px 0;
    }
    .metric-card {
        background: #1e1e1e;
        border-radius: 10px;
        padding: 12px;
        text-align: center;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL — cache để không load lại mỗi lần
# ============================================================
@st.cache_resource
def load_model(path):
    """
    @st.cache_resource: load model 1 lần duy nhất
    giữ trong bộ nhớ suốt session → nhanh hơn
    """
    return YOLO(path)

MODEL_PATH = "best.pt"   # ← sửa đường dẫn
model = load_model(MODEL_PATH)

# ============================================================
# HÀM DETECT CHÍNH
# ============================================================
def detect_fire(img_bgr, conf_thresh):
    """
    Input : ảnh BGR (numpy), ngưỡng conf
    Output: ảnh đã vẽ box (RGB), dict metrics
    """
    H, W = img_bgr.shape[:2]

    # Chạy YOLO
    t0      = time.time()
    results = model(img_bgr, conf=conf_thresh, verbose=False)
    elapsed = time.time() - t0

    boxes     = results[0].boxes
    num_fires = len(boxes)
    output    = img_bgr.copy()

    # Vẽ từng bounding box
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf            = float(box.conf[0])

        # Màu gradient theo confidence: xanh→vàng→đỏ
        r = int(255 * conf)
        g = int(255 * (1 - conf))
        color = (0, g, r)   # BGR

        # Box
        cv2.rectangle(output, (x1,y1), (x2,y2), color, 2)

        # Label nền
        label       = f"Fire {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(output,
                      (x1, y1-th-10), (x1+tw+6, y1),
                      color, -1)
        cv2.putText(output, label, (x1+3, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255,255,255), 2)

    # Thông tin góc trên
    overlay = output.copy()
    cv2.rectangle(overlay, (0,0), (W,40), (20,20,20), -1)
    cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
    cv2.putText(output, f"FPS:{1/elapsed:.0f}",
                (8, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0,255,0), 2)
    cv2.putText(output, f"Fire:{num_fires}",
                (120, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0,200,255), 2)
    cv2.putText(output, f"Conf:{conf_thresh:.2f}",
                (250, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (255,200,0), 2)

    # Cảnh báo nếu có lửa
    if num_fires > 0:
        cv2.putText(output, "FIRE DETECTED!",
                    (10, H-15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0,0,255), 3)

    # Chi tiết từng box để hiển thị
    box_details = []
    for i, box in enumerate(boxes):
        x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
        box_details.append({
            "Box": f"#{i+1}",
            "Confidence": f"{float(box.conf[0]):.3f}",
            "Vị trí (x1,y1)": f"({x1}, {y1})",
            "Kích thước": f"{x2-x1}×{y2-y1}px",
        })

    return cv2.cvtColor(output, cv2.COLOR_BGR2RGB), {
        "num_fires": num_fires,
        "elapsed_ms": elapsed * 1000,
        "fps": 1 / elapsed,
        "box_details": box_details,
    }

# ============================================================
# SIDEBAR — cài đặt
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/emoji/96/fire.png", width=80)
    st.title("⚙️ Cài đặt")
    st.markdown("---")

    # Slider confidence
    conf_thresh = st.slider(
        "🎯 Confidence Threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Thấp → bắt nhiều (dễ nhầm) | Cao → chắc chắn (dễ bỏ sót)"
    )

    st.markdown("---")

    # Nguồn ảnh
    st.markdown("### 📥 Nguồn dữ liệu")
    source_mode = st.radio(
        "Chọn nguồn:",
        ["📁 Upload ảnh", "📸 Webcam", "🎥 Upload video"],
        index=0
    )

    st.markdown("---")
    st.markdown("### 📌 Thông tin model")
    st.info(f"""
    **Model:** YOLOv11n
    **Task:** Fire Detection
    **Classes:** Fire
    """)

# ============================================================
# TIÊU ĐỀ CHÍNH
# ============================================================
st.markdown("# 🔥 Fire Detection — YOLOv11n")
st.markdown("Phát hiện lửa theo thời gian thực bằng YOLOv11n")
st.markdown("---")

# ============================================================
# LAYOUT 2 CỘT CHÍNH
# ============================================================
col_input, col_output = st.columns(2, gap="large")

# ============================================================
# CỘT TRÁI — INPUT
# ============================================================
with col_input:
    st.markdown("### 📷 Ảnh đầu vào")

    img_input = None   # ảnh sẽ xử lý

    # --- Chế độ Upload ---
    if source_mode == "📁 Upload ảnh":
        uploaded = st.file_uploader(
            "Chọn ảnh (JPG, PNG)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        if uploaded:
            # Đọc file → numpy array BGR
            file_bytes = np.frombuffer(uploaded.read(), np.uint8)
            img_input  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.image(
                cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB),
                caption="Ảnh gốc",
                use_container_width=True
            )

    # --- Chế độ Webcam ---
    elif source_mode == "📸 Webcam":
        webcam_img = st.camera_input(
            "📸 Chụp ảnh từ webcam",
            label_visibility="collapsed"
        )
        if webcam_img:
            file_bytes = np.frombuffer(webcam_img.read(), np.uint8)
            img_input  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # --- Chế độ Video ---
    else:
        uploaded_video = st.file_uploader(
            "Chọn video (MP4, AVI, MOV)",
            type=["mp4", "avi", "mov"],
            label_visibility="collapsed"
        )
        if uploaded_video:
            st.info("Nhấn nút bên dưới để bắt đầu xử lý video")

    # --- Nút detect ---
    if source_mode == "🎥 Upload video":
        btn = st.button("▶️ Chạy Video", use_container_width=True, type="primary") if uploaded_video else False
    elif img_input is not None:
        btn = st.button("🔍 Phát hiện lửa", use_container_width=True, type="primary")
    else:
        st.info("👆 Vui lòng chọn nguồn dữ liệu")
        btn = False

# ============================================================
# CỘT PHẢI — OUTPUT
# ============================================================
with col_output:
    st.markdown("### 🔥 Kết quả phát hiện")

    if btn:
        if source_mode == "🎥 Upload video":
            # Xử lý video
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            tfile.flush()

            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Placeholders
            frame_placeholder = st.empty()
            metric_placeholder = st.empty()
            progress_bar = st.progress(0)

            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                frame_idx += 1
                progress_bar.progress(min(frame_idx / total_frames, 1.0))

                # Xử lý mỗi 2 khung hình để mượt hơn
                if frame_idx % 2 != 0: continue

                result_img, metrics = detect_fire(frame, conf_thresh)
                
                frame_placeholder.image(result_img, use_container_width=True)
                
                with metric_placeholder.container():
                    m1, m2, m3 = st.columns(3)
                    m1.metric("⏱ Thời gian", f"{metrics['elapsed_ms']:.1f} ms")
                    m2.metric("🚀 FPS", f"{metrics['fps']:.1f}")
                    m3.metric("📦 Số vùng lửa", metrics["num_fires"])

            cap.release()
            st.success("✅ Video xử lý xong!")

        elif img_input is not None:
            # Spinner khi đang xử lý ảnh
            with st.spinner("Đang phân tích..."):
                result_img, metrics = detect_fire(img_input, conf_thresh)

            # Hiển thị ảnh kết quả
            st.image(result_img,
                     caption="Kết quả YOLOv11n",
                     use_container_width=True)

            # --- Trạng thái cảnh báo ---
            if metrics["num_fires"] > 0:
                st.markdown(
                    f'<div class="fire-alert">'
                    f'🔥 PHÁT HIỆN {metrics["num_fires"]} VÙNG LỬA!'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="safe-alert">✅ KHÔNG PHÁT HIỆN LỬA</div>',
                    unsafe_allow_html=True
                )

            # --- Metrics 3 cột ---
            m1, m2, m3 = st.columns(3)
            m1.metric("⏱ Thời gian",
                      f"{metrics['elapsed_ms']:.1f} ms")
            m2.metric("🚀 FPS",
                      f"{metrics['fps']:.1f}")
            m3.metric("📦 Số box",
                      metrics["num_fires"])

            # --- Bảng chi tiết box ---
            if metrics["box_details"]:
                st.markdown("#### 📋 Chi tiết từng box")
                st.table(metrics["box_details"])

    else:
        # Placeholder khi chưa có ảnh
        st.markdown("""
        <div style='
            height:300px;
            display:flex;
            align-items:center;
            justify-content:center;
            background:#1a1a1a;
            border-radius:12px;
            border: 2px dashed #444;
            color:#666;
            font-size:1.1em;
        '>
            Kết quả sẽ hiển thị ở đây
        </div>
        """, unsafe_allow_html=True)