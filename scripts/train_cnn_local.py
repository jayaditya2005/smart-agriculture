import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def train():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'raw', 'dataset')
    models_dir = os.path.join(base_dir, 'models')
    
    classes = [
        'apple_leaf_scab_disease',
        'corn_leaf_rust_disease',
        'crop_leaf_calcium_deficiency',
        'crop_leaf_magnesium_deficiency',
        'healthy_crop_leaf_isolated',
        'healthy_npk',
        'nitrogen_N_deficiency',
        'phosphorus_P_deficiency',
        'potassium_K_deficiency',
        'potato_leaf_late_blight_disease',
        'tomato_leaf_early_blight_disease'
    ]
    
    print("Preparing image generators (validation split 20%)...")
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    train_generator = datagen.flow_from_directory(
        data_dir, 
        target_size=(150, 150), 
        batch_size=32, 
        class_mode='categorical',
        subset='training',
        classes=classes
    )
    
    val_generator = datagen.flow_from_directory(
        data_dir, 
        target_size=(150, 150), 
        batch_size=32, 
        class_mode='categorical',
        subset='validation',
        classes=classes
    )
    
    print(f"Num classes detected: {train_generator.num_classes}")

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(train_generator.num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("Training Complete Authentic CNN Model...")
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10,
        verbose=1
    )
    
    save_path = os.path.join(models_dir, 'disease_cnn_model.h5')
    model.save(save_path)
    print(f"\\n✅ ML Structural & Authentic Training Complete! Base Model successfully saved to {save_path}")

if __name__ == "__main__":
    train()
