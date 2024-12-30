import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


train_dir = "path_to_train_data"  
val_dir = "path_to_validation_data"  
test_dir = "path_to_test_data" 


IMG_SIZE = 224  
BATCH_SIZE = 32  
EPOCHS = 10  
CONFIDENCE_THRESHOLD = 0.5  


train_datagen = ImageDataGenerator(rescale=1.0 / 255.0,
                                   rotation_range=20,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)


train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(IMG_SIZE, IMG_SIZE),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='binary')

val_generator = val_datagen.flow_from_directory(val_dir,
                                                target_size=(IMG_SIZE, IMG_SIZE),
                                                batch_size=BATCH_SIZE,
                                                class_mode='binary')

test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(IMG_SIZE, IMG_SIZE),
                                                  batch_size=1,
                                                  class_mode='binary',
                                                  shuffle=False)


base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                         include_top=False,
                         weights='imagenet')


base_model.trainable = False


model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  
])


model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])


history = model.fit(train_generator,
                    validation_data=val_generator,
                    epochs=EPOCHS,
                    steps_per_epoch=train_generator.samples // BATCH_SIZE,
                    validation_steps=val_generator.samples // BATCH_SIZE)


model.save("stop_sign_model.h5")

# Eğitim geçmişi grafiği
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc}")


predictions = model.predict(test_generator)
predicted_labels = (predictions > CONFIDENCE_THRESHOLD).astype(int)


true_labels = test_generator.classes
class_names = list(test_generator.class_indices.keys())
print(classification_report(true_labels, predicted_labels, target_names=class_names))


plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
