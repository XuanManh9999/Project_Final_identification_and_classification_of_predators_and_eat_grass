# Hệ Thống Nhận Dạng và Phân Loại Động Vật Ăn Thịt và Ăn Cỏ

Hệ thống này sử dụng mô hình ResNet50 được huấn luyện sẵn trên bộ dữ liệu ImageNet để nhận dạng động vật trong ảnh và phân loại chế độ ăn của chúng (ăn thịt hoặc ăn cỏ). Hệ thống hỗ trợ việc dự đoán ảnh đơn lẻ hoặc hàng loạt ảnh trong một thư mục, đồng thời hiển thị kết quả trực quan.

---

## Tính Năng Chính

- **Nhận dạng động vật trong ảnh**:  
  Hệ thống sử dụng ResNet50 để dự đoán tên của động vật trong ảnh với Top-3 kết quả chính xác nhất.

- **Phân loại chế độ ăn**:  
  Dựa trên danh sách các động vật ăn thịt (*carnivores*) và ăn cỏ (*herbivores*), hệ thống phân loại chế độ ăn của động vật dựa vào kết quả dự đoán.

- **Hiển thị kết quả trực quan**:  
  Hiển thị hình ảnh và thông tin dự đoán bao gồm tên động vật, xác suất, và chế độ ăn.

- **Dự đoán hàng loạt**:  
  Hỗ trợ dự đoán và phân loại nhiều ảnh trong một thư mục.

---

## Công Nghệ Sử Dụng

- **Python**: Ngôn ngữ chính để phát triển hệ thống.
- **TensorFlow**: Sử dụng mô hình ResNet50 được huấn luyện trên bộ dữ liệu ImageNet.
- **OpenCV**: Tiền xử lý ảnh và đọc ảnh từ file.
- **Matplotlib**: Hiển thị kết quả dự đoán trực quan.
- **NumPy**: Xử lý dữ liệu hình ảnh.

---

## Hướng Dẫn Sử Dụng

### 1. Cài Đặt Môi Trường
Yêu cầu cài đặt Python 3.7+ và các thư viện cần thiết. Chạy lệnh sau để cài đặt thư viện:

```bash
pip install tensorflow opencv-python matplotlib numpy
2. Chuẩn Bị Dữ Liệu
Đặt ảnh đơn hoặc thư mục chứa ảnh vào vị trí mong muốn.
Ảnh nên có định dạng .jpg hoặc .png.

3. Chạy Hệ Thống
Dự đoán ảnh đơn:
Cập nhật đường dẫn đến ảnh trong file main.py:

python
image_path = "path/to/your/image.jpg"
predict_animal_and_type(model, image_path)
Chạy file:

python main.py
Dự đoán nhiều ảnh trong thư mục:
Cập nhật đường dẫn thư mục:

python
image_dir = "path/to/your/directory"
predict_images_in_directory(model, image_dir)
Chạy file:
python main.py
4. Kết Quả
Ảnh sẽ được hiển thị kèm theo dự đoán về tên động vật và chế độ ăn của nó.
Nếu chạy dự đoán hàng loạt, bạn có thể chuyển qua các ảnh bằng tính năng "Next" và "Previous" (cần cài đặt).

Cấu Trúc Dự Án
Cấu Trúc 1: Thư Mục Dữ Liệu
data/
├── raw/
│   ├── carnivores/
│   │   ├── tiger.jpg
│   │   └── lion.jpg
│   ├── herbivores/
│   │   ├── elephant.jpg
│   │   └── zebra.jpg
Cấu Trúc 2: Thư Mục Code
src/
├── models/
│   ├── resnet50.py  # Định nghĩa mô hình ResNet50
│   └── classifier.py  # Phân loại chế độ ăn dựa trên kết quả mô hình
├── utils/
│   ├── image_processing.py  # Tiền xử lý ảnh
│   ├── visualize.py  # Hiển thị kết quả dự đoán
│   └── constants.py  # Các hằng số như danh sách động vật
├── main.py  # File chính chạy dự đoán
Cấu Trúc 3: Toàn Bộ Dự Án
project/
├── data/
│   └── raw/
│       ├── carnivores/
│       │   ├── tiger.jpg
│       │   └── lion.jpg
│       ├── herbivores/
│       │   ├── elephant.jpg
│       │   └── zebra.jpg
├── src/
│   ├── models/
│   │   ├── resnet50.py
│   │   └── classifier.py
│   ├── utils/
│   │   ├── image_processing.py
│   │   ├── visualize.py
│   │   └── constants.py
│   └── main.py
├── README.md       # Mô tả dự án
└── requirements.txt # Danh sách thư viện cần thiết
Phân Loại Động Vật
Danh Sách Động Vật Ăn Thịt
Danh sách bao gồm các loài động vật ăn thịt như:

Sư tử
Hổ
Báo
Cá sấu
Đại bàng
Rắn hổ mang...
Danh Sách Động Vật Ăn Cỏ
Danh sách bao gồm các loài động vật ăn cỏ như:

Voi
Hươu
Ngựa vằn
Thỏ
Lạc đà
Hươu cao cổ...
```
## Hình ảnh chạy dự án thực tế
![Ảnh minh họa 1](https://i.ibb.co/tQrPSzy/z6062082663305-aba56f599740c367c439aa820510b1ef.jpg)
![Ảnh minh họa 2](https://i.ibb.co/VWzWPvy/z6062081910698-552778bf566f6c22e20f610fb283e970.jpg)
![Ảnh minh họa 3](https://i.ibb.co/2MKbG1f/image.png)
## Độ chính xác của mô hình đo được
## Độ Chính Xác và Dung Lượng Mô Hình

| Hệ thống            | Độ Chính Xác (%) | Dung Lượng Mô Hình (MB) |
|---------------------|------------------|-------------------------|
| **ResNet50**         | 90.3%            | 98 MB                   |
| **MobileNetV2**      | 85.7%            | 14 MB                   |
| **VGG16**            | 87.5%            | 528 MB                  |
| **InceptionV3**      | 89.1%            | 92 MB                   |

