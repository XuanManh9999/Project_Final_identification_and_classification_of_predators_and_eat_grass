import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(data_dir, img_size=(224, 224)):
    """
    Load and split data into training and validation sets.
    """
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    train_data = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=32,
        class_mode='binary',
        subset='training'
    )
    
    val_data = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )
    
    return train_data, val_data
