from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def build_resnet_model(train_data, val_data, checkpoint):
    """
    Build and fine-tune the ResNet50 model for binary classification.
    """
    # Load ResNet50 đã huấn luyện sẵn (pre-trained trên ImageNet)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Mở khóa các lớp của base model từ layer thứ 100 trở đi
    for layer in base_model.layers[:100]:
        layer.trainable = False
    for layer in base_model.layers[100:]:
        layer.trainable = True

    # Thêm các lớp mới cho bài toán phân loại động vật ăn thịt và ăn cỏ
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)  # Nhãn 0 hoặc 1 cho động vật ăn cỏ hoặc ăn thịt

    # Tạo mô hình
    model = Model(inputs=base_model.input, outputs=x)

    # Biên dịch mô hình
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    # Tiến hành huấn luyện
    model.fit(train_data, validation_data=val_data, epochs=50, callbacks=[checkpoint])

    return model