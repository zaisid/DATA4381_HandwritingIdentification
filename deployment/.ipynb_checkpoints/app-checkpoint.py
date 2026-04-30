"""
Handwriting Identification Demo
--------------------------------
Run with:  streamlit run app.py

Expects:
  - A weights .h5 file
  - A zip of test images (extracts to: test/w0001/img.png  or  w0001/img.png)
  - A CSV with columns: wid, gender, agegroup, handedness
"""

import os
import zipfile
import random
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
import plotly.express as px

# ─────────────────────────────────────────────
# CONFIG  — edit these to match your file paths
# ─────────────────────────────────────────────
MODEL_PATH      = "/Users/zainabsiddiqui/Downloads/Data_Capstone/Spring26/Progress3/models/HighRes3.keras"
TEST_ZIP_PATH   = "/Users/zainabsiddiqui/Downloads/Data_Capstone/Spring26/Progress3/test_sets/HighRes3_test.zip"
METADATA_CSV    = "/Users/zainabsiddiqui/Downloads/Data_Capstone/Spring26/Colab_Uploads/Handwriting_Metadata_clean.csv"
WEIGHTS_PATH    = "HighRes3_weights.weights.h5"
IMAGE_SIZE      = (442, 442)
EXTRACT_DIR     = tempfile.mkdtemp()

W_IDS_90 = [
    'w0001','w0002','w0003','w0004','w0005','w0006','w0009','w0010','w0011',
    'w0012','w0013','w0015','w0016','w0017','w0018','w0020','w0022','w0023',
    'w0024','w0025','w0026','w0027','w0028','w0029','w0030','w0031','w0032',
    'w0033','w0034','w0035','w0036','w0038','w0043','w0061','w0062','w0063',
    'w0064','w0066','w0069','w0070','w0071','w0073','w0074','w0075','w0076',
    'w0077','w0078','w0080','w0082','w0083','w0085','w0086','w0087','w0088',
    'w0089','w0091','w0092','w0093','w0094','w0095','w0121','w0122','w0123',
    'w0124','w0125','w0126','w0128','w0129','w0130','w0131','w0133','w0134',
    'w0135','w0136','w0137','w0138','w0139','w0142','w0143','w0144','w0145',
    'w0147','w0148','w0149','w0151','w0152','w0153','w0154','w0155','w0156',
]

# class_names MUST match training order (alphabetical = same as W_IDS_90)
CLASS_NAMES = W_IDS_90


# ─────────────────────────────────────────────
# CACHED LOADERS
# ─────────────────────────────────────────────

def make_transfer_model(base_model, input_shape, num_classes, name):
    import keras
    from keras import layers
    backbone = base_model
    backbone.trainable = False
    inputs = layers.Input(input_shape)
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = backbone(x)
    x = layers.Dropout(0.3)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation=None)(x)
    return keras.Model(inputs, outputs, name=name)


@st.cache_resource
def load_model():
    import keras
    base_model = keras.applications.MobileNetV2(
        input_shape=(442, 442, 3),
        include_top=False,
        weights=None,
    )
    model = make_transfer_model(base_model, (442, 442, 3), num_classes=90, name="hw_model")
    model.load_weights(WEIGHTS_PATH)
    return model


@st.cache_resource
def extract_and_index_images():
    """Unzip once, return dict: writer_id -> [list of absolute image paths]"""
    with zipfile.ZipFile(TEST_ZIP_PATH, 'r') as z:
        z.extractall(EXTRACT_DIR)
    index = {}
    for root, dirs, files in os.walk(EXTRACT_DIR):
        for fname in files:
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                writer_id = os.path.basename(root)
                if writer_id in W_IDS_90:
                    index.setdefault(writer_id, []).append(
                        os.path.join(root, fname)
                    )
    return index


@st.cache_data
def load_metadata():
    df = pd.read_csv(METADATA_CSV)
    df['wid'] = df['wid'].str.strip()
    return df.set_index('wid')


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────

def preprocess(img_path):
    """Load + resize only — model applies mobilenet preprocess_input internally."""
    img = Image.open(img_path).convert("RGB").resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def predict_top3(model, img_path):
    arr    = preprocess(img_path)
    logits = model.predict(arr, verbose=0)[0]
    probs  = tf.nn.softmax(logits).numpy()
    top3   = np.argsort(probs)[::-1][:3]
    return [(CLASS_NAMES[i], float(probs[i])) for i in top3]


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def fmt_meta(wid, meta_df):
    if wid not in meta_df.index:
        return "No demographic info available."
    row = meta_df.loc[wid]
    parts = []
    if str(row.get('gender',    'unknown')).lower() not in ('unknown', 'nan', ''):
        parts.append(row['gender'])
    if str(row.get('agegroup',  'unknown')).lower() not in ('unknown', 'nan', ''):
        parts.append(f"age group {row['agegroup']}")
    if str(row.get('handedness','unknown')).lower() not in ('unknown', 'nan', ''):
        parts.append(f"{row['handedness']}-handed")
    return ", ".join(parts) if parts else "Demographics: unknown"


def pick_example(image_index, writer_id, exclude_path=None):
    candidates = [p for p in image_index.get(writer_id, []) if p != exclude_path]
    return random.choice(candidates) if candidates else None


def build_lineup(top3, true_writer, available_writers, image_index, query_path):
    """
    Build the 4-option lineup ONCE per round.
    Returns (lineup, option_labels) where:
      lineup       = [(wid, conf, img_path), ...]  shuffled
      option_labels = {wid: "Option N"}            frozen to this shuffle
    """
    lineup = []
    for wid, conf in top3:
        ex = pick_example(image_index, wid, exclude_path=query_path)
        lineup.append((wid, conf, ex))

    top3_wids      = [w for w, _, _ in lineup]
    true_in_lineup = true_writer in top3_wids

    if true_in_lineup:
        distractors = [w for w in available_writers
                       if w not in top3_wids and w != true_writer]
        fourth_wid = random.choice(distractors) if distractors else None
    else:
        fourth_wid = true_writer

    if fourth_wid:
        ex = pick_example(image_index, fourth_wid, exclude_path=query_path)
        lineup.append((fourth_wid, 0.0, ex))

    random.shuffle(lineup)
    option_labels = {wid: f"Option {i+1}" for i, (wid, _, _) in enumerate(lineup)}
    return lineup, option_labels


def reset_round():
    for key in ['query_path', 'true_writer', 'top3',
                'lineup', 'option_labels', 'revealed', 'user_choice']:
        st.session_state.pop(key, None)


# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────

def init_state():
    defaults = {
        'score_user':      0,
        'score_model':     0,
        'rounds':          0,
        'selected_writer': W_IDS_90[0],
        'revealed':        False,
        'user_choice':     None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Who Wrote This?", layout="wide")
    init_state()

    with st.spinner("Loading model…"):
        model = load_model()
    with st.spinner("Indexing test images…"):
        image_index = extract_and_index_images()
    meta_df = load_metadata()

    available_writers = [w for w in W_IDS_90 if w in image_index and image_index[w]]

    # ── sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🖊️ Who Wrote This?")
        st.markdown("---")
        st.subheader("Select a writer to query")
        chosen_writer = st.selectbox(
            "Writer ID", available_writers,
            index=available_writers.index(st.session_state['selected_writer'])
                  if st.session_state['selected_writer'] in available_writers else 0,
            key="writer_select",
        )

        if st.button("🎲 Random writer", use_container_width=True):
            st.session_state['selected_writer'] = random.choice(available_writers)
            reset_round()
            st.rerun()

        if st.button("▶️ New round with this writer", use_container_width=True):
            st.session_state['selected_writer'] = chosen_writer
            reset_round()
            st.rerun()

        st.markdown("---")
        st.subheader("Score")
        c1, c2 = st.columns(2)
        c1.metric("You",   st.session_state['score_user'])
        c2.metric("Model", st.session_state['score_model'])
        st.caption(f"Rounds played: {st.session_state['rounds']}")

        if st.button("Reset scores", use_container_width=True):
            st.session_state['score_user']  = 0
            st.session_state['score_model'] = 0
            st.session_state['rounds']      = 0
            st.rerun()

        st.markdown("---")
        st.caption("The model sees 90 writers. It picks the 3 most likely — "
                   "can you spot which one matches?")

    # ── initialise round (runs ONCE per round, frozen into session_state) ────
    if 'query_path' not in st.session_state:
        writer = st.session_state['selected_writer']
        paths  = image_index.get(writer, [])
        if not paths:
            st.error(f"No images found for {writer}.")
            return

        qpath = random.choice(paths)
        with st.spinner("Running model…"):
            top3 = predict_top3(model, qpath)

        lineup, option_labels = build_lineup(
            top3, writer, available_writers, image_index, qpath
        )

        # ── freeze everything into session state ──
        st.session_state['query_path']    = qpath
        st.session_state['true_writer']   = writer
        st.session_state['top3']          = top3
        st.session_state['lineup']        = lineup
        st.session_state['option_labels'] = option_labels
        st.session_state['revealed']      = False
        st.session_state['user_choice']   = None

    # ── read frozen round state ───────────────────────────────────────────────
    query_path    = st.session_state['query_path']
    true_writer   = st.session_state['true_writer']
    top3          = st.session_state['top3']
    lineup        = st.session_state['lineup']
    option_labels = st.session_state['option_labels']
    revealed      = st.session_state['revealed']
    model_prediction = top3[0][0]

    # ── header ────────────────────────────────────────────────────────────────
    st.title("Who wrote this handwriting sample?")
    st.info(f"**Writer demographics:** {fmt_meta(true_writer, meta_df)}", icon="👤")

    # ── main layout ───────────────────────────────────────────────────────────
    left_col, right_col = st.columns([4, 3])

    with left_col:
        st.subheader("Query image")
        st.image(query_path, use_container_width=True)

    with right_col:
        st.subheader("Which writer is this?")
        row1 = st.columns(2)
        row2 = st.columns(2)
        grid = [row1[0], row1[1], row2[0], row2[1]]

        for cell, (wid, conf, ex_path) in zip(grid, lineup):
            with cell:
                st.markdown(f"**{option_labels[wid]}**")
                if ex_path:
                    st.image(ex_path, use_container_width=True)
                else:
                    st.write("_(no example image)_")

                if not revealed:
                    if st.button(f"Pick {option_labels[wid]}", key=f"pick_{wid}",
                                 use_container_width=True):
                        st.session_state['user_choice'] = wid
                        st.session_state['revealed']    = True
                        st.session_state['rounds']     += 1
                        if wid == true_writer:
                            st.session_state['score_user'] += 1
                        if model_prediction == true_writer:
                            st.session_state['score_model'] += 1
                        st.rerun()

    # ── reveal ────────────────────────────────────────────────────────────────
    if revealed:
        st.markdown("---")
        st.subheader("Results")

        user_choice   = st.session_state['user_choice']
        user_correct  = (user_choice == true_writer)
        model_correct = (model_prediction == true_writer)
        top3_wids     = [w for w, _ in top3]
        model_in_top3 = (not model_correct) and (true_writer in top3_wids)

        res_cols = st.columns(2)
        with res_cols[0]:
            if user_correct:
                st.success(f"✅ **You got it!** Correct — {true_writer}")
            else:
                st.error(
                    f"❌ **Your guess:** {option_labels.get(user_choice, '?')} ({user_choice})  \n"
                    f"**True writer was:** {option_labels.get(true_writer, 'not in lineup')} ({true_writer})"
                )

        with res_cols[1]:
            if model_correct:
                st.success(
                    f"🤖 **Model correct!** Top prediction: {model_prediction} "
                    f"({top3[0][1]:.1%})"
                )
            elif model_in_top3:
                top3_rank = top3_wids.index(true_writer) + 1
                st.warning(
                    f"🤖 **Model close!** True writer was #{top3_rank} in top-3  \n"
                    f"Top prediction: {model_prediction} ({top3[0][1]:.1%})"
                )
            else:
                st.error(
                    f"🤖 **Model wrong.** Predicted: {model_prediction} "
                    f"({top3[0][1]:.1%})  \nTrue: {true_writer}"
                )

        # confidence bar chart (model top-3 only, not the 4th distractor)
        st.markdown("**Model confidence — top 3:**")
        model_lineup = [(w, c) for w, c, _ in lineup if c > 0.0]
        bar_data = pd.DataFrame({
            "Writer":     [f"{option_labels[w]} ({w})" for w, c in model_lineup],
            "Confidence": [c for w, c in model_lineup],
        }).sort_values("Confidence", ascending=True)

        fig = px.bar(
            bar_data, x="Confidence", y="Writer", orientation="h",
            range_x=[0, 1.03],
            text=bar_data["Confidence"].map("{:.1%}".format),
        )
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=200)
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

        # full metadata reveal
        if true_writer in meta_df.index:
            row = meta_df.loc[true_writer]
            st.markdown(
                f"**Full writer profile — {true_writer}:** "
                f"Gender: {row.get('gender','?')} | "
                f"Age group: {row.get('agegroup','?')} | "
                f"Handedness: {row.get('handedness','?')}"
            )

        st.markdown("---")
        if st.button("▶️ Next round (same writer)", use_container_width=False):
            reset_round()
            st.rerun()
        if st.button("🔀 Next round (random writer)", use_container_width=False):
            st.session_state['selected_writer'] = random.choice(available_writers)
            reset_round()
            st.rerun()


if __name__ == "__main__":
    main()