import subprocess
import sys

def _ensure_package(package, import_name=None):
    """Install package if not already available."""
    import_name = import_name or package
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

_ensure_package("lime")
_ensure_package("scikit-image", "skimage")

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from lime import lime_image
from skimage.segmentation import mark_boundaries


# ──────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def inspect_model(model):
    """Print wrapper model layers and last 10 layers of any sub-model."""
    print("=== Wrapper model layers ===")
    for i, layer in enumerate(model.layers):
        out_shape = getattr(layer, "output_shape", "?")
        is_submodel = isinstance(layer, tf.keras.Model)
        print(f"  [{i:2d}] {layer.name:40s}  output={out_shape}{'  <- SUB-MODEL' if is_submodel else ''}")
        if is_submodel:
            print(f"       Last 10 layers of '{layer.name}':")
            for sub in layer.layers[-10:]:
                print(f"         {sub.name:40s}  output={getattr(sub, 'output_shape', '?')}")


def _unwrap_dataset(dataset):
    """
    Unwrap prefetch/cache/repeat/shuffle wrappers to access .class_names.
    Safe to call on a plain dataset too — returns it unchanged.
    """
    ds = dataset
    while not hasattr(ds, 'class_names'):
        if not hasattr(ds, '_input_dataset'):
            raise AttributeError(
                "Could not find .class_names — dataset may not be a "
                "tf.data.Dataset created with image_dataset_from_directory()."
            )
        ds = ds._input_dataset
    return ds


def scale_for_display(img):
    """Scale any array to [0, 1] for imshow."""
    img = np.array(img, dtype=float)
    lo, hi = img.min(), img.max()
    if hi > lo:
        return (img - lo) / (hi - lo)
    return img


def _to_lime_range(img):
    img = np.array(img, dtype=np.float32)
    if img.max() > 1.0:
        # [0, 255] range
        return np.clip(img / 255.0, 0.0, 1.0)
    elif img.min() < 0:
        # [-1, 1] MobileNetV2 range
        return np.clip((img + 1.0) / 2.0, 0.0, 1.0)
    else:
        # already [0, 1]
        return np.clip(img, 0.0, 1.0)


def _get_backbone_and_head(model):
    """Returns (backbone, head_layers) for a wrapper model."""
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.Model):
            return layer, model.layers[i + 1:]
    raise ValueError("No nested sub-model (backbone) found inside `model`.")


def _run_model(model, img_array):
    """Full forward pass; returns softmax probs as 1-D numpy array (num_classes,)."""
    raw = model.predict(img_array, verbose=0)
    return tf.nn.softmax(raw[0]).numpy()


# ──────────────────────────────────────────────────────────────────────────────
# GRAD-CAM
# ──────────────────────────────────────────────────────────────────────────────

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None, grad_model=None):
    """
    Parameters
    ----------
    img_array            : np.ndarray (1, H, W, 3), pre-processed
    model                : full wrapper Keras model
    last_conv_layer_name : str  - layer name inside the backbone, e.g. "out_relu"
    pred_index           : int or None - class to explain; None = argmax

    Returns
    -------
    heatmap : np.ndarray (h, w), values in [0, 1]
    """

    backbone, head_layers = _get_backbone_and_head(model)

    # Check if we already have a built gradient model; if not, build it once
    if grad_model is None:
        try:
            conv_layer = backbone.get_layer(last_conv_layer_name)
        except ValueError:
            names = [l.name for l in backbone.layers]
            raise ValueError(
                f"Layer '{last_conv_layer_name}' not found in '{backbone.name}'.\n"
                f"Available: {names}"
            )

        grad_model = tf.keras.Model(
            inputs=backbone.inputs,
            outputs=[conv_layer.output, backbone.output],
        )

    with tf.GradientTape() as tape:
        img_tensor = tf.cast(img_array, tf.float32)
        # Use the reusable grad_model
        outputs = grad_model(img_tensor, training=False)

        conv_output = outputs[0]
        backbone_out = outputs[1]
        if isinstance(backbone_out, (list, tuple)):
            backbone_out = backbone_out[0]

        x = backbone_out
        for layer in head_layers:
            x = layer(x, training=False)
        final_preds = x

        if pred_index is None:
            pred_index = int(tf.argmax(final_preds[0], axis=-1).numpy())
        class_channel = final_preds[0, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    if grads is None:
        raise RuntimeError("Gradients are None — check the target layer is in the forward path.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_output[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    # Return both so the caller can keep hold of the model
    return heatmap.numpy(), grad_model 

def overlay_heatmap(img_single, heatmap, alpha=0.4):
    """
    Overlay a [0,1] heatmap onto an image (any value range).
    Returns float [0,1] RGB array.
    """
    H, W = img_single.shape[:2]
    display = scale_for_display(img_single)

    heatmap_resized = tf.image.resize(
        heatmap[..., tf.newaxis], (H, W)
    ).numpy()[..., 0]

    jet_rgb = cm.get_cmap("jet")(heatmap_resized)[..., :3]
    blended = jet_rgb * alpha + display * (1 - alpha)
    return np.clip(blended, 0, 1)


# ──────────────────────────────────────────────────────────────────────────────
# GRAD-CAM 4-UP GRID
# ──────────────────────────────────────────────────────────────────────────────

def plot_gradcam_from_dataset(
    dataset,
    model,
    last_conv_layer_name,
    n=4,
    alpha=0.4,
    figsize=(15, 12),
):
    """
    Randomly sample n images from the first batch, overlay Grad-CAM,
    show top-3 predictions vs. true label.
    Green title -> true label in top-3 | Red -> not in top-3
    """
    base_ds = _unwrap_dataset(dataset)
    for images, labels in dataset.take(1):
        images_np = images.numpy()
        labels_np = labels.numpy()
        class_names = base_ds.class_names

    n = min(n, len(images_np))
    indices = np.random.choice(len(images_np), n, replace=False)

    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).flatten()

    # --- STEP 1: Initialize the reusable model OUTSIDE the loop ---
    backbone, _ = _get_backbone_and_head(model)
    conv_layer = backbone.get_layer(last_conv_layer_name)
    reusable_grad_model = tf.keras.Model(
        inputs=backbone.inputs,
        outputs=[conv_layer.output, backbone.output],
    )

    for plot_i, idx in enumerate(indices):
        img = images_np[idx : idx + 1]
        true_name = class_names[labels_np[idx]]

        probs = _run_model(model, img)
        top3_idx = np.argsort(probs)[::-1][:3]
        top3_names = [class_names[j] for j in top3_idx]
        top3_confs = [f"{probs[j]:.2%}" for j in top3_idx]

        # --- STEP 2: Pass the reusable model in here ---
        heatmap, _ = make_gradcam_heatmap(
            img, model, last_conv_layer_name, grad_model=reusable_grad_model
        )
        
        overlay = overlay_heatmap(img[0], heatmap, alpha=alpha)

        ax = axes[plot_i]
        ax.imshow(overlay)
        ax.axis("off")

        in_top3 = true_name in top3_names
        title_lines = [f"True: {true_name}"]
        for rank, (name, conf) in enumerate(zip(top3_names, top3_confs), 1):
            suffix = " [CORRECT]" if name == true_name else ""
            title_lines.append(f"{rank}. {name} ({conf}){suffix}")

        ax.set_title(
            "\n".join(title_lines),
            color="green" if in_top3 else "red",
            fontsize=10,
            loc="left",
            linespacing=1.3,
        )

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# LIME + GRAD-CAM  2×2 PANEL
# ──────────────────────────────────────────────────────────────────────────────

def plot_ultimate_explanation(
    dataset,
    model,
    last_conv_layer_name,
    num_samples=500,
    figsize=(14, 12),
):
    """
    2×2 panel for a single randomly-sampled image:
      [0,0] Original image + true label
      [0,1] LIME: pros (green) and cons (red)
      [1,0] LIME: supportive features only
      [1,1] Grad-CAM overlay
    """

    # ── 1. Sample one image ──────────────────────────────────────────────────
    base_ds = _unwrap_dataset(dataset)
    for images, labels in dataset.take(1):
        images_np = images.numpy()
        labels_np = labels.numpy()
        class_names = base_ds.class_names

    idx = np.random.randint(0, len(images_np))
    img_tensor = images_np[idx : idx + 1]   # (1, H, W, 3) preprocessed
    true_label = class_names[labels_np[idx]]

    # ── 2. Model predictions ─────────────────────────────────────────────────
    probs = _run_model(model, img_tensor)
    pred_idx = int(np.argmax(probs))
    pred_label = class_names[pred_idx]
    confidence = probs[pred_idx]

    # ── 3. Convert to [0,1] for LIME and display ─────────────────────────────
    # _to_lime_range undoes [-1,1] MobileNetV2 preprocessing -> [0,1]
    lime_img = _to_lime_range(img_tensor[0])   # (H, W, 3) in [0, 1]

    # ── 4. LIME predict_fn ───────────────────────────────────────────────────
    def predict_fn(instances):
        # LIME gives [0,1] — scale back to [0,255] to match dataset/model expectation
        instances = (instances * 255.0).astype(np.float32)
        preds = model.predict(instances, verbose=0)
        return tf.nn.softmax(preds).numpy()

    # ── 5. Run LIME ──────────────────────────────────────────────────────────
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        lime_img.astype("double"),
        predict_fn,
        top_labels=5,
        hide_color=0,
        num_samples=num_samples,
    )

    lime_top_class_idx = explanation.top_labels[0]
    lime_top_class_name = class_names[lime_top_class_idx]

    # ── 6. Grad-CAM ──────────────────────────────────────────────────────────
    heatmap, _ = make_gradcam_heatmap(
    img_tensor, model, last_conv_layer_name, pred_index=pred_idx
    )
    gradcam_overlay = overlay_heatmap(img_tensor[0], heatmap, alpha=0.4)

    # ── 7. LIME images  ──────────────────────────────────────────────────────
    # get_image_and_mask returns an image in the same range as the input to
    # explain_instance, which is our lime_img ([0, 1]).
    # We display it directly — no further scaling needed except mark_boundaries.
    temp_pn, mask_pn = explanation.get_image_and_mask(
        lime_top_class_idx,
        positive_only=False,
        num_features=10,
        hide_rest=False,
    )
    temp_p, mask_p = explanation.get_image_and_mask(
        lime_top_class_idx,
        positive_only=True,
        num_features=5,
        hide_rest=True,
    )

    temp_pn, mask_pn = explanation.get_image_and_mask(
    lime_top_class_idx, positive_only=False, num_features=10, hide_rest=False
    )


    # ── 8. Plot ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # [0,0] Original
    axes[0, 0].imshow(lime_img)
    axes[0, 0].set_title(f"Original\nTrue: {true_label}", fontsize=12)
    axes[0, 0].axis("off")

    # [0,1] LIME pros & cons — temp_pn is already [0,1], clip for safety
    axes[0, 1].imshow(mark_boundaries(np.clip(temp_pn, 0, 1), mask_pn))
    axes[0, 1].set_title(
        f"LIME: Pros (Green) / Cons (Red)\n"
        f"LIME top: {lime_top_class_name}  |  Model: {pred_label} ({confidence:.1%})",
        fontsize=11,
    )
    axes[0, 1].axis("off")

    # [1,0] LIME supportive only
    axes[1, 0].imshow(mark_boundaries(np.clip(temp_p, 0, 1), mask_p))
    axes[1, 0].set_title("LIME: Top Supportive Features", fontsize=12)
    axes[1, 0].axis("off")

    # [1,1] Grad-CAM
    axes[1, 1].imshow(gradcam_overlay)
    axes[1, 1].set_title(f"Grad-CAM: {last_conv_layer_name}", fontsize=12)
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# USAGE
# ──────────────────────────────────────────────────────────────────────────────
# inspect_model(model)
#
# plot_gradcam_from_dataset(val_dataset, model, last_conv_layer_name="out_relu", n=4)
#
# plot_ultimate_explanation(val_dataset, model, last_conv_layer_name="out_relu", num_samples=1000)
