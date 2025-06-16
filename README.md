# Nhận diện chữ Kanji thời gian thực từ Webcam

## Tổng quan dự án

Dự án này triển khai một hệ thống nhận diện ký tự Kuzushiji (chữ viết tay cổ của Nhật Bản) theo thời gian thực. Hệ thống sử dụng Mạng nơ-ron tích chập (CNN) được xây dựng bằng TensorFlow/Keras và được triển khai với OpenCV để nhận diện qua webcam. Mục tiêu là phân loại 10 ký tự Kuzushiji khác nhau từ luồng video trực tiếp.

## Các tính năng chính

* **Mô hình Học sâu (Deep Learning):** Kiến trúc CNN tùy chỉnh được xây dựng bằng TensorFlow/Keras để phân loại ảnh mạnh mẽ.
* **Độ chính xác cao:** Đạt độ chính xác cao trên tập dữ liệu Kuzushiji-MNIST (ví dụ: ~98.03% trên tập kiểm thử).
* **Nhận diện thời gian thực qua Webcam:** Sử dụng OpenCV để thu nhận video từ webcam và thực hiện nhận diện ký tự trực tiếp.
* **Tiền xử lý ảnh nâng cao:** Tích hợp các kỹ thuật tiền xử lý ảnh (chuyển đổi ảnh xám, làm mờ Gaussian, phân ngưỡng Otsu, phát hiện đường viền, điều chỉnh kích thước và căn giữa) để chuẩn bị khung hình từ webcam cho đầu vào mô hình tối ưu.
* **Hỗ trợ hiển thị Unicode:** Sử dụng thư viện Pillow để hiển thị chính xác các ký tự tiếng Nhật (Kuzushiji) trong cửa sổ OpenCV.

## Công nghệ sử dụng

* **Python 3.10**
* **TensorFlow / Keras:** Để xây dựng và huấn luyện mô hình CNN.
* **OpenCV (`cv2`):** Để tích hợp webcam, xử lý và hiển thị ảnh.
* **NumPy:** Để thực hiện các phép toán số học và thao tác mảng.
* **Pillow (`PIL`):** Để hiển thị ký tự Unicode tiếng Nhật.
* **Matplotlib / Seaborn:** (Sử dụng trong giai đoạn huấn luyện để trực quan hóa dữ liệu/ma trận nhầm lẫn).

## Cấu trúc dự án

* `prepare_kanji_dataset.py`: Script để tải xuống, tiền xử lý và chia tách tập dữ liệu Kuzushiji-MNIST.
* `build_and_train_model.py`: Script để định nghĩa, huấn luyện, đánh giá và lưu mô hình CNN.
* `realtime_kanji_webcam.py`: Script để nhận diện ký tự thời gian thực bằng webcam và mô hình đã huấn luyện.
* `best_kanji_model.keras` (hoặc `.h5`): File mô hình học sâu đã được huấn luyện.
* `NotoSansJP-VariableFont_wght.ttf`: File font cần thiết để hiển thị các ký tự tiếng Nhật.
* `Confusion Matrix.png` (Tùy chọn): Biểu diễn trực quan hiệu suất của mô hình.


## Thiết lập và Cài đặt

1.  **Clone repository:**
    ```bash
    git clone [https://github.com/YourGitHubUsername/YourRepositoryName.git](https://github.com/YourGitHubUsername/YourRepositoryName.git)
    cd YourRepositoryName
    ```
    (Thay thế `YourGitHubUsername` và `YourRepositoryName` bằng thông tin GitHub của bạn.)

2.  **Tạo môi trường Conda (khuyến nghị):**
    ```bash
    conda create -n kanji_env python=3.10
    conda activate kanji_env
    ```

3.  **Cài đặt các thư viện cần thiết:**
    ```bash
    pip install  opencv-python numpy Pillow scikit-learn matplotlib seaborn
    ```

5.  **Cài đặt Tensorflow Cpu-Only:**
    ```bash
    pip install https://storage.googleapis.com/tensorflow/versions/2.19.0/tensorflow_cpu-2.19.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    ```

## Hướng dẫn sử dụng

1.  **Chuẩn bị tập dữ liệu:**
    ```bash
    python prepare_kanji_dataset.py
    ```
    Lệnh này sẽ tải xuống và lưu tập dữ liệu KMNIST.

2.  **Huấn luyện mô hình:**
    ```bash
    python build_and_train_model.py
    ```
    Lệnh này sẽ huấn luyện mô hình CNN và lưu file `best_kanji_model.keras`.

3.  **Chạy nhận diện thời gian thực từ Webcam:**
    ```bash
    python realtime_kanji_webcam.py
    ```
    Một cửa sổ webcam sẽ mở ra. Đặt một ký tự Kuzushiji (trong số 10 lớp đã được huấn luyện: お, き, す, つ, な, は, ま, や, り, を) vào trong khung màu xanh lá cây. Đảm bảo đủ ánh sáng và chữ viết rõ ràng để có kết quả tốt nhất. Nhấn 'q' để thoát.

---
