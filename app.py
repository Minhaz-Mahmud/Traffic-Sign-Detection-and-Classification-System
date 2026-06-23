"""
Traffic Sign Detection & Classification — Streamlit App
Wraps the original OpenCV pipeline (preprocessing → edge/color isolation →
boundary extraction → cropping → template matching → scoring) in an
interactive web UI, showing every intermediate output.

Run with:  streamlit run app.py
Expects a `train/` folder next to this file, containing the training images
referenced below (same layout as the original script).
"""

import os
import io
import numpy as np
import cv2
import streamlit as st
import matplotlib.pyplot as plt
from skimage import exposure
import pandas as pd

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Traffic Sign Detector", page_icon="🚦", layout="wide")

# ----------------------------------------------------------------------------
# CORE PIPELINE FUNCTIONS (ported 1:1 from the original script)
# ----------------------------------------------------------------------------

def apply_clahe(gray_image, clip_limit=2.0, tile_size=8):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(gray_image)


def gaussian_blur_canny(image, kernel_size=5, canny_low=50, canny_high=150, use_clahe=True):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    if use_clahe:
        gray = apply_clahe(gray)
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    return edges


def extract_shape_boundaries(image, min_area=100):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(image)
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            cv2.drawContours(result, [contour], -1, 255, 1)
    return result


def extract_inner_region(original_image, boundary_image):
    mask = boundary_image.copy()
    contours, _ = cv2.findContours(boundary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.fillPoly(mask, [contour], 255)
    mask_normalized = mask.astype(np.float32) / 255.0
    if len(original_image.shape) == 3:
        result = original_image.copy().astype(np.float32)
        for i in range(3):
            result[:, :, i] = result[:, :, i] * mask_normalized
        result = result.astype(np.uint8)
    else:
        result = (original_image.astype(np.float32) * mask_normalized).astype(np.uint8)
    return result, mask


def extract_inner_region_cropped(original_image, boundary_image):
    inner_image, mask = extract_inner_region(original_image, boundary_image)
    contours, _ = cv2.findContours(boundary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped = inner_image[y:y + h, x:x + w]
        return cropped, (x, y, w, h)
    return inner_image, None


def resize_image(image, target_size=512):
    h, w = image.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    if len(resized.shape) == 3:
        canvas = np.zeros((target_size, target_size, resized.shape[2]), dtype=resized.dtype)
    else:
        canvas = np.zeros((target_size, target_size), dtype=resized.dtype)
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    if len(resized.shape) == 3:
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w, :] = resized
    else:
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


def isolate_traffic_sign_colors_strict(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 100]);   upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 100]); upper_red2 = np.array([180, 255, 255])
    lower_blue = np.array([110, 150, 100]); upper_blue = np.array([120, 255, 220])
    lower_yellow = np.array([18, 120, 150]); upper_yellow = np.array([32, 255, 255])

    mask_red = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1),
                               cv2.inRange(hsv, lower_red2, upper_red2))
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    combined_mask = cv2.bitwise_or(cv2.bitwise_or(mask_red, mask_blue), mask_yellow)
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    return combined_mask


def structural_similarity_score(test_image, train_image):
    test_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY) if len(test_image.shape) == 3 else test_image.copy()
    train_gray = cv2.cvtColor(train_image, cv2.COLOR_BGR2GRAY) if len(train_image.shape) == 3 else train_image.copy()

    test_gray = apply_clahe(test_gray, clip_limit=3.0, tile_size=8)
    train_gray = apply_clahe(train_gray, clip_limit=3.0, tile_size=8)
    test_gray = cv2.bilateralFilter(test_gray, 11, 80, 80)
    train_gray = cv2.bilateralFilter(train_gray, 11, 80, 80)

    test_edges = cv2.Canny(test_gray, 50, 150)
    train_edges = cv2.Canny(train_gray, 50, 150)
    test_edges_norm = test_edges.astype(np.float32) / 255.0
    train_edges_norm = train_edges.astype(np.float32) / 255.0

    edge_intersection = np.sum(test_edges_norm * train_edges_norm)
    edge_union = np.sum(np.maximum(test_edges_norm, train_edges_norm))
    edge_similarity = edge_intersection / (edge_union + 1e-10)

    test_norm = (test_gray - np.mean(test_gray)) / (np.std(test_gray) + 1e-10)
    train_norm = (train_gray - np.mean(train_gray)) / (np.std(train_gray) + 1e-10)
    ncc = np.sum(test_norm * train_norm) / test_norm.size
    ncc_score = (ncc + 1) / 2

    return 0.7 * edge_similarity + 0.3 * ncc_score


def match_histograms(source, reference):
    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY) if len(source.shape) == 3 else source.copy()
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY) if len(reference.shape) == 3 else reference.copy()
    matched = exposure.match_histograms(source_gray, reference_gray, channel_axis=None)
    return np.uint8(matched)


def prepare_matching_image(image, method='clahe_hist'):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    gray = np.clip(gray.astype(np.float32), 0, 255).astype(np.uint8)

    if method == 'clahe_hist':
        gray = apply_clahe(gray, clip_limit=2.0, tile_size=8)
        gray = cv2.equalizeHist(gray)
        return np.clip(gray.astype(np.float32), 0, 255).astype(np.uint8)
    elif method == 'edge':
        gray = apply_clahe(gray, clip_limit=3.0, tile_size=8)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        kernel = np.ones((2, 2), np.uint8)
        return cv2.dilate(edges, kernel, iterations=1)
    return gray


def template_match(test_image, training_image, method=cv2.TM_CCOEFF_NORMED,
                    prep_method='clahe_hist', use_histogram_matching=False):
    if use_histogram_matching:
        test_image = match_histograms(test_image, training_image)
    test_gray = prepare_matching_image(test_image, method=prep_method)
    train_gray = prepare_matching_image(training_image, method=prep_method)
    if test_gray.shape[0] > train_gray.shape[0] or test_gray.shape[1] > train_gray.shape[1]:
        test_gray, train_gray = train_gray, test_gray
    result = cv2.matchTemplate(train_gray, test_gray, method)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return max_val, test_gray, train_gray


def process_image_pipeline(image_bgr, kernel_size=5, canny_low=50, canny_high=150,
                            min_area=500, target_size=512, title=""):
    """Returns (cropped_region, fig) — fig shows every intermediate stage."""
    img = resize_image(image_bgr, target_size=target_size)
    iso_img = isolate_traffic_sign_colors_strict(img)
    edges = gaussian_blur_canny(iso_img, kernel_size=kernel_size, canny_low=canny_low, canny_high=canny_high)
    boundaries = extract_shape_boundaries(edges, min_area=min_area)
    cropped_region, bbox = extract_inner_region_cropped(img, boundaries)

    inner_region, mask = extract_inner_region(img, boundaries)
    if cropped_region is None:
        cropped_region = inner_region
    cropped_region = resize_image(cropped_region, target_size=target_size)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Processing Pipeline: {title}', fontsize=14, fontweight='bold')

    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); axes[0, 0].set_title('Original (Resized)'); axes[0, 0].axis('off')
    axes[0, 1].imshow(gray, cmap='gray'); axes[0, 1].set_title('Grayscale'); axes[0, 1].axis('off')
    axes[0, 2].imshow(blurred, cmap='gray'); axes[0, 2].set_title('Gaussian Blur'); axes[0, 2].axis('off')
    axes[0, 3].imshow(edges, cmap='gray'); axes[0, 3].set_title('Canny Edges (on color-isolated mask)'); axes[0, 3].axis('off')
    axes[1, 0].imshow(boundaries, cmap='gray'); axes[1, 0].set_title('Shape Boundaries'); axes[1, 0].axis('off')
    axes[1, 1].imshow(cv2.cvtColor(inner_region, cv2.COLOR_BGR2RGB)); axes[1, 1].set_title('Inner Region'); axes[1, 1].axis('off')
    if len(cropped_region.shape) == 3:
        axes[1, 2].imshow(cv2.cvtColor(cropped_region, cv2.COLOR_BGR2RGB))
    else:
        axes[1, 2].imshow(cropped_region, cmap='gray')
    axes[1, 2].set_title('Cropped & Resized'); axes[1, 2].axis('off')
    axes[1, 3].axis('off')

    plt.tight_layout()
    return cropped_region, fig


def plot_matching_images(test_name, test_gray, train_names, train_grays, scores,
                          normalized_scores, best_idx, prep_method, use_hist_match):
    n_trains = len(train_names)
    fig, axes = plt.subplots(2, n_trains + 1, figsize=(4 * (n_trains + 1), 8))
    method_names = {'clahe_hist': 'CLAHE + Histogram Equalization', 'edge': 'Edge Detection (Canny)'}
    hist_match_str = " + Histogram Matching" if use_hist_match else ""
    fig.suptitle(f'Template Matching Images for Test: {test_name}\n'
                 f'Preprocessing: {method_names.get(prep_method, prep_method)}{hist_match_str}',
                 fontsize=14, fontweight='bold')

    axes[0, 0].imshow(test_gray, cmap='gray')
    axes[0, 0].set_title(f'Test: {test_name}\n(After Preprocessing)', fontweight='bold', fontsize=11)
    axes[0, 0].axis('off'); axes[1, 0].axis('off')

    for i, (name, train_gray, score, norm_score) in enumerate(zip(train_names, train_grays, scores, normalized_scores)):
        col = i + 1
        axes[0, col].imshow(train_gray, cmap='gray')
        title_color = 'green' if i == best_idx else 'black'
        axes[0, col].set_title(f'Train: {name}\n(After Preprocessing)',
                                fontweight='bold' if i == best_idx else 'normal',
                                color=title_color, fontsize=11)
        axes[0, col].axis('off')

        score_text = f'Score: {score:.4f}\n({norm_score:.1f}%)'
        axes[1, col].text(0.5, 0.5, score_text, ha='center', va='center', fontsize=12,
                           fontweight='bold', color=title_color, transform=axes[1, col].transAxes)
        axes[1, col].axis('off')

    plt.tight_layout()
    return fig


def plot_score_comparison(test_name, training_names, scores, normalized_scores, best_idx,
                           threshold, prep_method, use_hist_match):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    hist_match_str = " + Histogram Matching" if use_hist_match else ""
    fig.suptitle(f'Template Matching Scores for Test: {test_name}\n'
                 f'Preprocessing: {prep_method}{hist_match_str}', fontsize=14, fontweight='bold')

    colors = ['green' if i == best_idx else 'steelblue' for i in range(len(scores))]
    bars1 = ax1.bar(training_names, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    for bar, score in zip(bars1, scores):
        ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height(), f'{score:.4f}',
                  ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Average Score', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Training Classes', fontsize=12, fontweight='bold')
    ax1.set_title('Raw Scores', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=11); ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, max(scores) * 1.2 if max(scores) > 0 else 1])

    bars2 = ax2.bar(training_names, normalized_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    for bar, norm_score in zip(bars2, normalized_scores):
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height(), f'{norm_score:.1f}%',
                  ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Normalized Score (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Training Classes', fontsize=12, fontweight='bold')
    ax2.set_title('Normalized Scores (Sum = 100%)', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3); ax2.set_ylim([0, 100])

    plt.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# STATIC CONFIG (mirrors the original script — edit paths/messages as needed)
# ----------------------------------------------------------------------------
TRAIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train")

TRAINING_IMAGES = {
    'stop':       ['stop.jpg', 'stop_2.png', 'stop_3.jpg'],
    'limit2':     ['limit2.png', 'limit2_2.jpg', 'limit2_3.jpg'],
    'no_entry':   ['nop.png', 'nop_2.jpg', 'nop_3.jpg'],
    'pedestrain': ['pedes.png', 'pedes_2.jpg', 'pedes_3.jpg'],
    'caution':    ['caution.jpg', 'caution_2.jpg', 'caution_3.jpg'],
}

SIGN_MESSAGES = {
    'stop': '🛑 STOP! Come to a complete halt before proceeding.',
    'limit2': '⚠️ Speed Limit: 60 km/h — Slow down!',
    'no_entry': '🚫 No Entry — Authorized Vehicles Only.',
    'pedestrain': '🚶 Pedestrian Crossing Ahead — Watch for pedestrians!',
    'caution': '⚡ Caution! Proceed with care.',
}

# ----------------------------------------------------------------------------
# SIDEBAR — PARAMETERS
# ----------------------------------------------------------------------------
st.sidebar.title("⚙️ Pipeline Settings")
kernel_size = st.sidebar.slider("Gaussian kernel size (odd)", 3, 11, 5, step=2)
canny_low = st.sidebar.slider("Canny low threshold", 0, 200, 50)
canny_high = st.sidebar.slider("Canny high threshold", 50, 300, 150)
min_area = st.sidebar.slider("Minimum contour area", 50, 2000, 500, step=50)
target_size = st.sidebar.selectbox("Working resolution (px)", [256, 384, 512, 768], index=2)

st.sidebar.markdown("---")
st.sidebar.title("🧮 Matching Settings")
prep_method = st.sidebar.selectbox("Preprocessing for matching", ['edge', 'clahe_hist'], index=0)
use_structural_similarity = st.sidebar.checkbox("Use Structural Similarity (instead of Template Matching)", value=False)
use_histogram_matching = st.sidebar.checkbox("Use Histogram Matching", value=False)
threshold = st.sidebar.slider("Match threshold", 0.0, 1.0, 0.10, step=0.01)

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.title("🚦 Traffic Sign Detection & Classification")
st.caption(
    "Color isolation → Canny edges → boundary/contour extraction → crop → "
    "template matching against trained sign classes."
)

if not os.path.isdir(TRAIN_DIR):
    st.error(f"No `train/` folder found next to app.py. Expected at: `{TRAIN_DIR}`")
    st.stop()

# ----------------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------------
if "trained" not in st.session_state:
    st.session_state.trained = {}  # class -> list of cropped numpy arrays

# ----------------------------------------------------------------------------
# TRAINING PHASE (runs automatically during detection — no separate UI section)
# ----------------------------------------------------------------------------
show_training_plots = False

# Settings fingerprint — if the user changes sidebar params, we know to
# retrain automatically rather than reuse a stale cache.
train_settings_key = (kernel_size, canny_low, canny_high, min_area, target_size)


def run_training():
    """Processes all training images and stores cropped results + figures
    in session_state. Returns True on success (at least one class trained)."""
    for fig_list in st.session_state.get("train_figs", {}).values():
        for _, _, _, old_fig in fig_list:
            plt.close(old_fig)

    st.session_state.trained = {}
    st.session_state.train_figs = {}  # cls -> list of (fig, fname) when show_training_plots
    progress = st.progress(0)
    total = sum(len(v) for v in TRAINING_IMAGES.values())
    done = 0

    for cls, files in TRAINING_IMAGES.items():
        variants = []
        fig_list = []

        for i, fname in enumerate(files, 1):
            path = os.path.join(TRAIN_DIR, fname)
            if not os.path.isfile(path):
                done += 1
                progress.progress(done / total)
                continue
            img = cv2.imread(path, 1)
            if img is None:
                done += 1
                progress.progress(done / total)
                continue

            cropped, fig = process_image_pipeline(
                img, kernel_size=kernel_size, canny_low=canny_low, canny_high=canny_high,
                min_area=min_area, target_size=target_size, title=f"{cls} (variant {i})"
            )
            variants.append(cropped)
            fig_list.append((fname, i, cropped.copy(), fig))

            done += 1
            progress.progress(done / total)

        if variants:
            st.session_state.trained[cls] = variants
            st.session_state.train_figs[cls] = fig_list

    progress.empty()
    st.session_state.train_settings_key = train_settings_key
    return bool(st.session_state.trained)


def display_training_results():
    for cls, fig_list in st.session_state.get("train_figs", {}).items():
        st.subheader(f"Class: `{cls}`")
        if show_training_plots:
            for fname, i, cropped, fig in fig_list:
                st.pyplot(fig)
        else:
            cols = st.columns(len(fig_list))
            for col, (fname, i, cropped, fig) in zip(cols, fig_list):
                with col:
                    st.image(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB), caption=f"variant {i}: {fname}", use_container_width=True)
        st.success(f"Class `{cls}`: {len(fig_list)} variant(s) processed ✅")


needs_training = (
    not st.session_state.trained
    or st.session_state.get("train_settings_key") != train_settings_key
)

# ----------------------------------------------------------------------------
# TESTING PHASE
# ----------------------------------------------------------------------------
st.header("Detection")
st.caption("Upload an image to classify it against the trained sign set.")

uploaded = st.file_uploader("Upload a test image", type=["jpg", "jpeg", "png", "bmp"])

default_test_path = os.path.join(TRAIN_DIR, "test3.png")
use_default = False
if uploaded is None and os.path.isfile(default_test_path):
    use_default = st.checkbox("Use default test image (`train/test3.png`)", value=True)

run_btn = st.button("🚀 Run Detection", type="primary")

if run_btn:
    if needs_training:
        st.subheader("Training")
        if not run_training():
            st.error("No training images could be processed — check the `train/` folder.")
            st.stop()
        display_training_results()
        st.markdown("---")
    else:
        display_training_results()
        st.markdown("---")

    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        test_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        test_name = uploaded.name
    elif use_default:
        test_img = cv2.imread(default_test_path, 1)
        test_name = "test3.png"
    else:
        st.warning("Upload a test image, or enable the default test image checkbox.")
        st.stop()

    if test_img is None:
        st.error("Could not read the test image.")
        st.stop()

    st.subheader(f"Pipeline output — Test: `{test_name}`")
    test_cropped, fig = process_image_pipeline(
        test_img, kernel_size=kernel_size, canny_low=canny_low, canny_high=canny_high,
        min_area=min_area, target_size=target_size, title=f"Test: {test_name}"
    )
    st.pyplot(fig)
    plt.close(fig)

    # ---- Template matching against every class ----
    training_names = list(st.session_state.trained.keys())
    scores, train_grays, best_variant_indices = [], [], []
    test_gray = None

    st.subheader("Matching against each trained class")
    log_lines = []
    for cls in training_names:
        variants = st.session_state.trained[cls]
        best_score, best_variant_idx = 0, 0
        for v_idx, variant in enumerate(variants):
            if use_structural_similarity:
                score = structural_similarity_score(test_cropped, variant)
            else:
                score, _, _ = template_match(test_cropped, variant, prep_method=prep_method,
                                              use_histogram_matching=use_histogram_matching)
            if score > best_score:
                best_score, best_variant_idx = score, v_idx

        best_variant = variants[best_variant_idx]
        _, t_gray, tr_gray = template_match(test_cropped, best_variant, prep_method=prep_method,
                                             use_histogram_matching=use_histogram_matching)
        if test_gray is None:
            test_gray = t_gray

        scores.append(best_score)
        train_grays.append(tr_gray)
        best_variant_indices.append(best_variant_idx)
        log_lines.append(f"**{cls}**: {best_score:.4f}  (best variant #{best_variant_idx + 1})")

    st.markdown("  \n".join(log_lines))

    scores_array = np.array(scores)
    normalized_scores = (scores_array / scores_array.sum() * 100) if scores_array.sum() > 0 else np.zeros_like(scores_array)
    best_idx = int(np.argmax(normalized_scores))
    best_score = scores[best_idx]
    best_normalized_score = normalized_scores[best_idx]
    best_class = training_names[best_idx]
    is_match = best_score > threshold

    st.subheader("Visual comparison: test vs. each trained class")
    fig2 = plot_matching_images(test_name, test_gray, training_names, train_grays, scores,
                                 normalized_scores, best_idx, prep_method, use_histogram_matching)
    st.pyplot(fig2); plt.close(fig2)

    st.subheader("Score comparison charts")
    fig3 = plot_score_comparison(test_name, training_names, scores, normalized_scores, best_idx,
                                  threshold, prep_method, use_histogram_matching)
    st.pyplot(fig3); plt.close(fig3)

    st.subheader("📋 Results table")
    row = {"Test Image": test_name}
    for cls, score, norm in zip(training_names, scores, normalized_scores):
        row[cls] = f"{score:.4f} ({norm:.1f}%)"
    row["Result"] = f"{best_class} ({best_normalized_score:.1f}%)" if is_match else "NO MATCH"
    st.dataframe(pd.DataFrame([row]), use_container_width=True)

    st.subheader("✅ Final outcome")
    method_label = "Structural Similarity" if use_structural_similarity else "Template Matching"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Method", method_label)
    c2.metric("Preprocessing", prep_method)
    c3.metric("Best raw score", f"{best_score:.4f}")
    c4.metric("Threshold", f"{threshold:.2f}")

    if is_match:
        st.success(f"**MATCH** → `{best_class}`  ({best_normalized_score:.1f}% confidence)")
        message = SIGN_MESSAGES.get(best_class, "Sign detected — please observe traffic rules.")
        st.info(f"**Message:** {message}")
    else:
        st.error(f"**NO MATCH** — closest class was `{best_class}` "
                 f"({best_normalized_score:.1f}%, raw score {best_score:.4f}), below threshold {threshold:.2f}.")