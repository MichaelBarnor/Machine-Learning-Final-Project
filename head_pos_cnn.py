# import tensorflow as tf
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras import layers, models

# dataset_path = "output_frames"
# image_size = (128, 128)
# batch_size = 32

# train_ds = tf.keras.utils.image_dataset_from_directory(
#     dataset_path,
#     image_size=image_size,
#     batch_size=batch_size,
#     validation_split=0.2,
#     subset="training",
#     seed=123,
#     shuffle = True
# )

# print("Class names:", train_ds.class_names)

# val_ds = tf.keras.utils.image_dataset_from_directory(
#     dataset_path,
#     image_size=image_size,
#     batch_size=batch_size,
#     validation_split=0.2,
#     subset="validation",
#     seed=123,
#     shuffle = True
# )

# AUTOTUNE = tf.data.AUTOTUNE
# train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
# base_model.trainable = False 

# model = models.Sequential([
#     base_model,
#     layers.GlobalAveragePooling2D(),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(2, activation='softmax')
# ])


# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])


# model.fit(train_ds, validation_data=val_ds, epochs=2)


# model.save("head_position_model.h5")


# ===================================================================# CNN #2 w/ more variance
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

dataset_path = "output_frames"  
image_size = (128, 128)
batch_size = 32


augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomBrightness(0.2),
])

def prepare(ds, shuffle=False):
    ds = ds.map(lambda x, y: (augmentation(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(500)
    return ds.prefetch(tf.data.AUTOTUNE)

raw_train = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    image_size=image_size,
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    seed=123,
)
raw_val = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    image_size=image_size,
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=123,
)

train_ds = prepare(raw_train, shuffle=True)
val_ds   = prepare(raw_val)

print("Classes:", raw_train.class_names)

base_model = MobileNetV2(input_shape=(*image_size,3),
                         include_top=False,
                         weights="imagenet")

for layer in base_model.layers[:-20]:
    layer.trainable = False
for layer in base_model.layers[-20:]:
    layer.trainable = True

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128,
                 activation="relu",
                 kernel_regularizer=regularizers.l2(1e-4)),
    layers.Dropout(0.3),
    layers.Dense(len(raw_train.class_names), activation="softmax")
])

opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(
    optimizer=opt,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(monitor="val_loss",
                  patience=3,
                  restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss",
                      factor=0.5,
                      patience=2,
                      min_lr=1e-6)
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=callbacks
)

model.save("head_position_model.h5")
