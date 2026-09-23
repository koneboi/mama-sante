"""Full-integer int8 TFLite pipeline: Keras MobileNetV3 train -> TFLite convert -> verify.

Trains the same BUSI classification task with TensorFlow/Keras (the domain-standard
pathway for low-end Android: full-int8 TFLite is ~4x smaller than fp32 ORT and runs on
TFLite's mature int8 kernels). Evaluation reuses the identical 15% holdout split.
"""
from __future__ import annotations

import glob
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import tensorflow as tf

IMG_SIZE = 224
BATCH = 16
CLASSES = ["benign", "malignant", "normal"]
MEAN_RANGE = 1.0  # keras MobileNetV3 expects inputs scaled to [-1, 1]


def to_model_input(img: tf.Tensor) -> tf.Tensor:
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 127.5 - 1.0  # [0,255] -> [-1,1]
    return img


def split_paths(root: str, val_ratio: float = 0.15, seed: int = 0):
    rng = random.Random(seed)
    paths = sorted(glob.glob(os.path.join(root, "*", "*.png")))
    rng.shuffle(paths)
    n = int(len(paths) * (1 - val_ratio))
    return paths[:n], paths[n:]


def load(paths: list[str]) -> tuple[tf.Tensor, tf.Tensor]:
    imgs, labels = [], []
    for p in paths:
        img = tf.image.decode_png(tf.io.read_file(p), channels=3)
        imgs.append(to_model_input(img))
        labels.append(CLASSES.index(os.path.basename(os.path.dirname(p))))
    return tf.stack(imgs), tf.stack(labels)


def apply_aug(imgs: tf.Tensor, labels: tf.Tensor):
    imgs = tf.image.random_flip_left_right(imgs)
    return imgs, labels


def build_model() -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV3Small(
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        weights="imagenet",
        pooling="avg",
        include_preprocessing=False,  # inputs are already scaled to [-1, 1] by to_model_input
    )
    base.trainable = True
    x = tf.keras.layers.Dropout(0.2)(base.output)
    out = tf.keras.layers.Dense(len(CLASSES), activation="softmax")(x)
    model = tf.keras.Model(base.input, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def eval_model(model_or_interp, val_paths) -> float:
    ok = 0
    for p in val_paths:
        img = to_model_input(tf.image.decode_png(tf.io.read_file(p), channels=3)).numpy()[None]
        y = CLASSES.index(os.path.basename(os.path.dirname(p)))
        if hasattr(model_or_interp, "predict"):
            probs = model_or_interp.predict(img, verbose=0)[0]
        else:
            probs = model_or_interp(np.float32(img))
        ok += int(probs.argmax() == y)
    return ok / len(val_paths)


def main() -> None:
    t0 = time.time()
    train_paths, val_paths = split_paths("data/processed/busi_class")
    x_train, y_train = load(train_paths)
    print(f"train {len(train_paths)} / val {len(val_paths)}", flush=True)

    ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    counts = np.bincount(y_train.numpy(), minlength=3).astype(np.float32)
    weight = (1.0 / (counts + 1e-6))
    weight = weight / weight.sum() * 3.0
    ds = ds.map(apply_aug).shuffle(len(train_paths)).batch(BATCH)

    model = build_model()
    model.fit(ds, epochs=25, verbose=1)
    print(f"keras val acc: {eval_model(model, val_paths):.3f}", flush=True)
    model.save(os.path.join(ROOT, "models", "imaging_keras.h5"))
    print(f"keras save took {(time.time()-t0)/60:.1f} min")

    repr_ds = tf.data.Dataset.from_tensor_slices(x_train[:96]).batch(1)
    tf_model_dir = os.path.join(ROOT, "models", "tflite", "saved_model")
    os.makedirs(os.path.dirname(tf_model_dir), exist_ok=True)
    model.export(tf_model_dir)

    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)
    fp32 = converter.convert()
    with open(os.path.join(ROOT, "models", "imaging_fp32.tflite"), "wb") as f:
        f.write(fp32)

    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: (
        [img.numpy().astype(np.float32)] for img in iter(repr_ds)
    )
    converter.target_spec.supported_types = [tf.int8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    int8 = converter.convert()
    int8_path = os.path.join(ROOT, "models", "imaging_int8.tflite")
    with open(int8_path, "wb") as f:
        f.write(int8)

    os.makedirs(os.path.join(ROOT, "models", "tflite", "int8"), exist_ok=True)
    interp = tf.lite.Interpreter(
        model_path=int8_path,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
    )
    interp.allocate_tensors()
    in_d, out_d = interp.get_input_details()[0], interp.get_output_details()[0]
    scale, zp = in_d["quantization"]

    def int8_predict(img_np):
        q = np.clip(img_np / scale + zp, -128, 127).astype(np.int8)
        interp.set_tensor(in_d["index"], q)
        interp.invoke()
        o = interp.get_tensor(out_d["index"])
        return o.astype(np.float32).reshape(-1)

    ok = 0
    for p in val_paths:
        img = to_model_input(tf.image.decode_png(tf.io.read_file(p), channels=3)).numpy()[None]
        y = CLASSES.index(os.path.basename(os.path.dirname(p)))
        ok += int(int8_predict(img).argmax() == y)
    int8_acc = ok / len(val_paths)

    print(f"full-int8 TFLite val acc: {int8_acc:.3f}")
    print(f"sizes: fp32 {os.path.getsize(os.path.join(ROOT, 'models', 'imaging_fp32.tflite'))/1e6:.1f} MB, "
          f"int8 {os.path.getsize(int8_path)/1e6:.1f} MB")
    print(f"total took {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()