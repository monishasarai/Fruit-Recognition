
# Fruit Recognition Model

> **Repository**: Fruit image classification using TensorFlow / Keras.  
> This README describes the model, dataset layout, training & inference steps, dependencies, and tips to reproduce results.

---

## Project Overview

This project trains a convolutional neural network (Keras / TensorFlow) to recognize fruit types from images. The model file produced by the notebook is named `fruit_recognition_model.h5`. The notebook used to develop and evaluate the model is `Fruits.ipynb`.

The notebook includes code for:
- Preparing image data (train/validation/test)
- Defining and training a Keras model (CNN or transfer-learning backbone)
- Saving the trained model to `fruit_recognition_model.h5`
- Running inference on individual images

> **Note:** Paths and image sizes used in the notebook include `224x224` and `150x150`. The default inference size used in examples is `224x224`.

---

## Repository structure (suggested)

```
├── Fruits.ipynb
├── fruit_recognition_model.h5          # trained model produced by notebook
├── README.md                            # this file
├── requirements.txt                     # Python dependencies
├── data/
│   ├── train/
│   │   ├── apple/                       # images for class 'apple'
│   │   ├── banana/
│   │   └── ...
│   └── validation/
│       ├── apple/
│       └── ...
└── notebooks/
    └── Fruits.ipynb                     # original notebook (same as top-level)
```

> If your notebook uses a different layout, adapt the paths accordingly. In the uploaded notebook, test images are referenced at paths like `E:\Internship\test\test\0010.jpg` — replace those with local relative paths when running on other machines.

---

## Requirements

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate         # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Suggested `requirements.txt` (minimum):
```
tensorflow>=2.6
numpy
pillow
matplotlib
scikit-learn
jupyterlab           # if you want to open the notebook
opencv-python        # optional, if used in preprocessing
```

---

## How to train (high-level)

> The training code is contained in `Fruits.ipynb`. This is a high-level outline of the commands / steps used in the notebook:

1. Prepare dataset folders:
```
data/train/<class_name>/*.jpg
data/validation/<class_name>/*.jpg
data/test/<class_name>/*.jpg   # optional
```

2. Use `ImageDataGenerator` / `tf.keras.preprocessing.image` to create generators:
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 224
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, validation_split=0.2)
train_generator = train_datagen.flow_from_directory('data/train', target_size=(IMG_SIZE, IMG_SIZE), batch_size=32, class_mode='categorical', subset='training')
val_generator = train_datagen.flow_from_directory('data/train', target_size=(IMG_SIZE, IMG_SIZE), batch_size=32, class_mode='categorical', subset='validation')
```

3. Create or load a model (example using a simple transfer-learning backbone):
```python
import tensorflow as tf
base = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE,IMG_SIZE,3), include_top=False, weights='imagenet', pooling='avg')
x = tf.keras.layers.Dense(256, activation='relu')(base.output)
output = tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')(x)
model = tf.keras.Model(inputs=base.input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, validation_data=val_generator, epochs=10)
model.save('fruit_recognition_model.h5')
```

4. Monitor training using accuracy / loss plots (the notebook contains plotting code).

---

## Inference / How to use the saved model

Example Python script `predict_fruit.py`:

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

MODEL_PATH = 'fruit_recognition_model.h5'
IMG_SIZE = 224  # match the size used during training

model = load_model(MODEL_PATH)

def predict_image(img_path, model, target_size=(IMG_SIZE, IMG_SIZE)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    predicted_index = np.argmax(preds, axis=1)[0]
    # Replace the next line with the actual class names used in training:
    class_names = list(train_generator.class_indices.keys()) if 'train_generator' in globals() else None
    return predicted_index, preds[0]

if __name__ == '__main__':
    img_path = 'path_to_test_image.jpg'
    idx, probs = predict_image(img_path, model)
    print(f'Predicted class index: {idx}, probabilities: {probs}')
```

> Replace `class_names` resolution with the `train_generator.class_indices` mapping saved at training time, or save and load a `labels.json` file mapping indices→class names for deterministic inference.

---

## Reproducing results & tips

- Use a fixed random seed to make training deterministic where possible (`numpy.random.seed()`, `tf.random.set_seed()`).
- Save the `train_generator.class_indices` mapping after training to a JSON file (so that inference can map indices → labels reliably).
- If using transfer learning, tune learning rates and unfreeze layers gradually.
- Use `ModelCheckpoint` and `EarlyStopping` callbacks to avoid overfitting and to keep the best model (example in notebook).
- If GPU is available, ensure TensorFlow GPU is installed and visible (e.g., `nvidia-smi` shows your GPU).

---

## Expected artifacts

- `fruit_recognition_model.h5` — saved Keras model ready for inference.
- Training history plots (accuracy & loss) — available inside the notebook output.
- Optionally: `labels.json` mapping, `best_model.h5` (from ModelCheckpoint).

---

## Notes & Known paths from notebook

- The notebook references saved model `fruit_recognition_model.h5` and test images under `E:\Internship\test\test\` — if you move the project to a different environment, update the notebook to use relative paths.
- The notebook uses image sizes `224x224` and sometimes `150x150`. Use `224x224` by default for compatibility with common backbones.

---

## License

This repository is provided for educational purposes. You can add a license as needed, e.g. MIT:

```
MIT License
Copyright (c) 2025 <Your Name>
Permission is hereby granted...
```

---

## Contribution

If you want to improve the notebook or add better preprocessing/augmentation, open an issue or send a pull request. Suggestions:

- Export `labels.json` during training for deterministic inference.
- Add a Dockerfile for reproducible environments.
- Add unit tests for the preprocessing and inference pipeline.

---

If you'd like, I can:
- generate a `requirements.txt` for you (based on the notebook),
- create a `predict_fruit.py` script with automatic label-loading,
- or produce a `labels.json` by inspecting the notebook training generator (if available).
