import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

# Load the Food101 dataset
dataset, info = tfds.load('food101', split=['train', 'validation'], with_info=True, as_supervised=True)

train_dataset, val_dataset = dataset
IMG_SIZE = 224  # Ukuran input gambar
BATCH_SIZE = 32

def preprocess_image(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))  # Ubah ukuran gambar
    image = image / 255.0  # Normalisasi ke range [0, 1]
    return image, label

# Apply preprocessing
train_dataset = train_dataset.map(preprocess_image).batch(BATCH_SIZE).shuffle(1000).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.map(preprocess_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # Freeze base model

# Build the model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(101, activation='softmax')  # 101 kelas
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=100
)

loss, accuracy = model.evaluate(val_dataset)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")