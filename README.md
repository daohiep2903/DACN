# 🔥 Fire Detection — YOLOv11n

Ứng dụng web phát hiện lửa theo thời gian thực sử dụng mô hình học sâu **YOLOv11n**, được xây dựng bằng **Streamlit** và **Ultralytics**.

---

## 📋 Tính năng

- **Upload ảnh** — Tải ảnh tĩnh (JPG/PNG) lên và phát hiện lửa ngay lập tức
- **Webcam** — Chụp ảnh trực tiếp từ camera máy tính để phân tích
- **Upload video** — Xử lý từng frame của video (MP4/AVI/MOV) theo thời gian thực
- **Điều chỉnh ngưỡng Confidence** — Slider từ 0.10 đến 0.95 để kiểm soát độ nhạy
- **Hiển thị kết quả chi tiết** — Bounding box, confidence score, tọa độ, kích thước từng vùng lửa
- **Giao diện tối chuyên nghiệp** — Dark theme tối ưu cho môi trường giám sát

---

## 🗂️ Cấu trúc thư mục

```
code_fire/
├── app.py          # Mã nguồn chính
├── best.pt         # Trọng số mô hình YOLOv11n đã huấn luyện
├── requirements.txt
└── README.md
```

---

## ⚙️ Yêu cầu hệ thống

| Thành phần | Yêu cầu tối thiểu |
|---|---|
| Python | 3.10 trở lên |
| RAM | 4 GB trở lên |
| GPU | Không bắt buộc (CPU vẫn chạy được, ~15–25 FPS) |
| OS | Windows / macOS / Linux |

---

## 🚀 Hướng dẫn cài đặt và chạy

### Bước 1 — Clone hoặc tải code về

```bash
git clone <your-repo-url>
cd code_fire
```

Hoặc giải nén file ZIP và di chuyển vào thư mục:

```bash
cd code_fire
```

### Bước 2 — Tạo môi trường ảo (khuyến nghị)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Bước 3 — Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### Bước 4 — Chạy ứng dụng

```bash
streamlit run app.py
```

Ứng dụng sẽ tự động mở trình duyệt tại địa chỉ:

```
http://localhost:8501
```

---

## 🎮 Hướng dẫn sử dụng

### Phát hiện từ ảnh
1. Chọn chế độ **"📁 Upload ảnh"** ở sidebar
2. Kéo thả hoặc nhấn **Browse files** để tải ảnh lên
3. Điều chỉnh ngưỡng **Confidence** nếu cần
4. Nhấn nút **"🔍 Phát hiện lửa"**
5. Xem kết quả: bounding box, badge cảnh báo và bảng chi tiết

### Phát hiện từ Webcam
1. Chọn chế độ **"📸 Webcam"**
2. Cho phép trình duyệt truy cập camera khi được hỏi
3. Nhấn nút chụp (nút tròn trắng) để chụp ảnh
4. Nhấn **"🔍 Phát hiện lửa"** để phân tích

### Phát hiện từ Video
1. Chọn chế độ **"🎥 Upload video"**
2. Tải file video lên (MP4, AVI, MOV)
3. Nhấn **"▶️ Chạy Demo Video"**
4. Theo dõi kết quả theo thời gian thực qua thanh tiến trình

---

## 🎯 Điều chỉnh ngưỡng Confidence

| Giá trị | Hành vi | Phù hợp khi |
|---|---|---|
| 0.10 – 0.30 | Nhạy cao, dễ báo nhầm | Muốn phát hiện lửa nhỏ, chấp nhận báo giả |
| 0.25 (mặc định) | Cân bằng | Sử dụng thông thường |
| 0.50 – 0.95 | Chắc chắn, dễ bỏ sót | Chỉ cảnh báo khi rất chắc chắn có lửa |

---

## 📊 Hiệu năng mô hình

| Chỉ số | Giá trị |
|---|---|
| mAP@50 | 0.9022 |
| mAP@50-95 | 0.5985 |
| Precision | 0.8974 |
| Recall | 0.8609 |
| F1-Score | 0.8788 |
| FPS (GPU Tesla T4) | 100.1 |

> Mô hình được huấn luyện trên tập dữ liệu **10.463 ảnh** thực tế từ Roboflow, bao gồm lửa trong nhà, ngoài trời, ban ngày và ban đêm.

---

## 🛠️ Công nghệ sử dụng

| Thư viện | Phiên bản | Vai trò |
|---|---|---|
| Streamlit | ≥ 1.32 | Giao diện Web |
| Ultralytics | ≥ 8.0 | Inference YOLOv11 |
| OpenCV | ≥ 4.8 | Xử lý ảnh, vẽ bounding box |
| Pillow | ≥ 10.0 | Đọc/ghi file ảnh |
| NumPy | ≥ 1.24 | Xử lý mảng dữ liệu |

---

## 🔧 Tuỳ chỉnh nâng cao

### Đổi model khác
Sửa dòng sau trong `app.py`:
```python
MODEL_PATH = "best.pt"   # ← đổi thành đường dẫn model của bạn
```

### Thay đổi ngưỡng diện tích phát hiện video
Trong hàm `detect_fire()`, mô hình xử lý cách 1 frame (`frame_idx % 2 != 0`) để tăng tốc. Có thể sửa thành `% 1` để xử lý mọi frame:
```python
if frame_idx % 1 != 0: continue   # xử lý mọi frame
```

---

## ❗ Lưu ý

- File `best.pt` phải nằm **cùng thư mục** với `app.py`
- Chế độ Webcam yêu cầu trình duyệt **cấp quyền camera** — nếu bị từ chối hãy vào Settings của trình duyệt để cho phép
- Khi chạy lần đầu, Streamlit sẽ tải thêm một số dependencies — cần kết nối Internet

---

## 👨‍💻 Tác giả

**Đào Văn Hiệp** — Mã sinh viên: 2022603324

Đồ án tốt nghiệp — Ngành Khoa học Máy tính

Đại học Công nghiệp Hà Nội — 2026

---

## 📄 Giấy phép

Dự án này được thực hiện cho mục đích học thuật.
