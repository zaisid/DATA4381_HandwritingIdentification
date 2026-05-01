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
TEST_ZIP_PATH   = "../data/testsets/AllClass_bw_test.zip"
METADATA_CSV    = "../data/Handwriting_Metadata_clean.csv"
WEIGHTS_PATH    = "AllClass5_weights.weights.h5"
IMAGE_SIZE      = (384, 384)
INPUT_SIZE      = (384, 384, 3)
EXTRACT_DIR     = tempfile.mkdtemp()

W_IDS_475      = ['w0001', 'w0002', 'w0003', 'w0004', 'w0005', 'w0006', 'w0009', 'w0010', 'w0011', 'w0012', 'w0013', 'w0015', 'w0016', 'w0017', 'w0018', 'w0020', 'w0022', 'w0023', 'w0024', 'w0025', 'w0026', 'w0027', 'w0028', 'w0029', 'w0030', 'w0031', 'w0032', 'w0033', 'w0034', 'w0035', 'w0036', 'w0038', 'w0040', 'w0041', 'w0042', 'w0043', 'w0049', 'w0058', 'w0061', 'w0062', 'w0063', 'w0064', 'w0066', 'w0069', 'w0070', 'w0071', 'w0073', 'w0074', 'w0075', 'w0076', 'w0077', 'w0078', 'w0080', 'w0082', 'w0083', 'w0085', 'w0086', 'w0087', 'w0088', 'w0089', 'w0090', 'w0091', 'w0092', 'w0093', 'w0094', 'w0095', 'w0099', 'w0102', 'w0118', 'w0119', 'w0121', 'w0122', 'w0123', 'w0124', 'w0125', 'w0126', 'w0128', 'w0129', 'w0130', 'w0131', 'w0132', 'w0133', 'w0134', 'w0135', 'w0136', 'w0137', 'w0138', 'w0139', 'w0142', 'w0143', 'w0144', 'w0145', 'w0146', 'w0147', 'w0148', 'w0149', 'w0150', 'w0151', 'w0152', 'w0153', 'w0154', 'w0155', 'w0156', 'w0157', 'w0160', 'w0162', 'w0175', 'w0177', 'w0180', 'w0182', 'w0184', 'w0186', 'w0189', 'w0191', 'w0193', 'w0198', 'w0199', 'w0200', 'w0201', 'w0202', 'w0203', 'w0204', 'w0205', 'w0206', 'w0212', 'w0218', 'w0220', 'w0223', 'w0224', 'w0226', 'w0227', 'w0229', 'w0232', 'w0233', 'w0234', 'w0238', 'w0239', 'w0240', 'w0242', 'w0244', 'w0245', 'w0246', 'w0249', 'w0254', 'w0255', 'w0260', 'w0261', 'w0262', 'w0263', 'w0264', 'w0270', 'w0271', 'w0276', 'w0277', 'w0279', 'w0280', 'w0281', 'w0282', 'w0284', 'w0285', 'w0286', 'w0287', 'w0288', 'w0291', 'w0293', 'w0297', 'w0299', 'w0301', 'w0302', 'w0304', 'w0305', 'w0306', 'w0308', 'w0312', 'w0313', 'w0314', 'w0315', 'w0317', 'w0319', 'w0320', 'w0322', 'w0330', 'w0333', 'w0334', 'w0335', 'w0337', 'w0338', 'w0339', 'w0340', 'w0341', 'w0342', 'w0344', 'w0348', 'w0350', 'w0351', 'w0352', 'w0353', 'w0354', 'w0355', 'w0356', 'w0357', 'w0359', 'w0362', 'w0364', 'w0365', 'w0367', 'w0368', 'w0370', 'w0371', 'w0372', 'w0375', 'w0379', 'w0380', 'w0381', 'w0382', 'w0383', 'w0384', 'w0387', 'w0388', 'w0391', 'w0392', 'w0393', 'w0396', 'w0397', 'w0398', 'w0399', 'w0400', 'w0401', 'w0402', 'w0403', 'w0405', 'w0406', 'w0407', 'w0408', 'w0409', 'w0410', 'w0411', 'w0412', 'w0413', 'w0414', 'w0415', 'w0416', 'w0417', 'w0419', 'w0420', 'w0422', 'w0424', 'w0425', 'w0426', 'w0428', 'w0429', 'w0431', 'w0433', 'w0435', 'w0436', 'w0439', 'w0440', 'w0441', 'w0443', 'w0444', 'w0445', 'w0446', 'w0448', 'w0450', 'w0451', 'w0452', 'w0454', 'w0456', 'w0458', 'w0460', 'w0462', 'w0463', 'w0464', 'w0465', 'w0466', 'w0467', 'w0468', 'w0469', 'w0470', 'w0471', 'w0472', 'w0473', 'w0474', 'w0475', 'w0476', 'w0477', 'w0479', 'w0480', 'w0481', 'w0483', 'w0484', 'w0485', 'w0486', 'w0487', 'w0489', 'w0492', 'w0493', 'w0495', 'w0497', 'w0498', 'w0500', 'w0501', 'w0502', 'w0508', 'w0510', 'w0513', 'w0514', 'w0515', 'w0516', 'w0517', 'w0518', 'w0519', 'w0520', 'w0521', 'w0522', 'w0523', 'w0524', 'w0525', 'w0526', 'w0527', 'w0528', 'w0529', 'w0530', 'w0531', 'w0532', 'w0533', 'w0534', 'w0535', 'w0536', 'w0537', 'w0538', 'w0541', 'w0542', 'w0543', 'w0546', 'w0547', 'w0548', 'w0549', 'w0550', 'w0551', 'w0552', 'w0553', 'w0554', 'w0555', 'w0557', 'w0559', 'w0560', 'w0561', 'w0562', 'w0564', 'w0565', 'w0566', 'w0569', 'w0570', 'w0571', 'w0572', 'w0573', 'w0575', 'w0576', 'w0577', 'w0579', 'w0580', 'w0581', 'w0586', 'w0587', 'w0588', 'w0589', 'w0590', 'w0591', 'w0592', 'w0593', 'w0594', 'w0595', 'w0596', 'w0597', 'w0598', 'w0599', 'w0600', 'w0601', 'w0602', 'w0604', 'w0605', 'w0606', 'w0611', 'w0612', 'w0613', 'w0615', 'w0617', 'w0618', 'w0619', 'w0620', 'w0621', 'w0622', 'w0623', 'w0624', 'w0626', 'w0627', 'w0628', 'w0629', 'w0630', 'w0632', 'w0634', 'w0636', 'w0637', 'w0638', 'w0639', 'w0640', 'w0641', 'w0642', 'w0644', 'w0645', 'w0646', 'w0647', 'w0648', 'w0650', 'w0653', 'w0656', 'w0657', 'w0658', 'w0660', 'w0661', 'w0662', 'w0664', 'w0665', 'w0666', 'w0667', 'w0668', 'w0669', 'w0671', 'w0673', 'w0674', 'w0675', 'w0677', 'w0678', 'w0679', 'w0680', 'w0682', 'w0683', 'w0685', 'w0688', 'w0691', 'w0692', 'w0693', 'w0694', 'w0695', 'w0698', 'w0699', 'w0700', 'w0701', 'w0702', 'w0703', 'w0704', 'w0705', 'w0706', 'w0707', 'w0709', 'w0710', 'w0711', 'w0712', 'w0713', 'w0714', 'w0715', 'w0717', 'w0719', 'w0720', 'w0418', 'w0453', 'w0394', 'w0345', 'w0478', 'w0670', 'w0289', 'w0459', 'w0216', 'w0378']

# class_names MUST match training order (alphabetical = same as W_IDS_475)
CLASS_NAMES = sorted(W_IDS_475)

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
        input_shape=INPUT_SIZE,
        include_top=False,
        weights=None,
    )
    model = make_transfer_model(base_model, INPUT_SIZE, num_classes=475, name="hw_model")
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
                if writer_id in W_IDS_475:
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
        'selected_writer': W_IDS_475[0],
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

    available_writers = [w for w in W_IDS_475 if w in image_index and image_index[w]]

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
        st.caption("The model sees 475 writers. It picks the 3 most likely — "
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
