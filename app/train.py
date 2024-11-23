from app.data_processing import load_data
from app.model_training import build_resnet_model
from tensorflow.keras.callbacks import ModelCheckpoint

# Đường dẫn tới dữ liệu
DATA_DIR = "data/raw/"

# Tiền xử lý dữ liệu
train_data, val_data = load_data(DATA_DIR)

# Tạo callback lưu mô hình tốt nhất
checkpoint = ModelCheckpoint('models/best_model.h5', save_best_only=True, monitor='val_loss', mode='min')

# Xây dựng và huấn luyện mô hình
model = build_resnet_model(train_data, val_data, checkpoint)
