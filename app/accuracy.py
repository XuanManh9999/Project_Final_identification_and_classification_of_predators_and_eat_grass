from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
import os

# Định nghĩa đường dẫn tới thư mục chứa ảnh kiểm thử
test_dir = 'data/raw'

# Tải mô hình ResNet50 đã được huấn luyện sẵn trên ImageNet
model = ResNet50(weights='imagenet')

# Biên dịch mô hình với optimizer, loss function và metric
model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=['accuracy'])

# Tạo ImageDataGenerator cho bộ kiểm thử (không thay đổi dữ liệu vì chúng ta chỉ muốn kiểm tra)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Tải dữ liệu kiểm thử từ thư mục
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),  # Kích thước ảnh phải giống với kích thước mà mô hình yêu cầu
    batch_size=1,  # Sử dụng 1 ảnh một lần (chỉ để thử nghiệm)
    class_mode='categorical',  # Nếu bạn có nhiều lớp (multi-class classification)
    shuffle=False,  # Giữ nguyên thứ tự của ảnh
    subset='validation'  # Đảm bảo bạn chỉ sử dụng tập kiểm thử
)

# Đánh giá mô hình trên bộ kiểm thử
loss, accuracy = model.evaluate(test_generator, steps=5)  # Chỉ đánh giá với 5 ảnh kiểm thử

# In kết quả độ chính xác
print(f"Độ Chính Xác trên bộ kiểm thử: {accuracy * 100:.2f}%")
print(f"Loss trên bộ kiểm thử: {loss:.4f}")
