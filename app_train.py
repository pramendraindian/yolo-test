import glob
import shutil
import subprocess
import streamlit as st
import wget
from PIL import Image
import torch
import cv2
import os
import time
import yaml
import zipfile

st.set_page_config(layout="wide")

cfg_model_path = 'models/yolov5s.pt'
model = None
confidence = .25

# Directories
os.makedirs("models", exist_ok=True)
os.makedirs("data/sample_images", exist_ok=True)
os.makedirs("data/sample_videos", exist_ok=True)
os.makedirs("data/uploaded_data", exist_ok=True)
os.makedirs("data/training_data", exist_ok=True)


# ─────────────────────────────────────────────
#  INFERENCE HELPERS
# ─────────────────────────────────────────────

def image_input(data_src):
    img_file = None
    if data_src == 'Sample data':
        img_path = glob.glob('data/sample_images/*')
        if not img_path:
            st.warning("No sample images found in data/sample_images/")
            return
        img_slider = st.slider("Select a test image.", min_value=1, max_value=len(img_path), step=1)
        img_file = img_path[img_slider - 1]
    else:
        img_bytes = st.sidebar.file_uploader("Upload an image", type=['png', 'jpeg', 'jpg'])
        if img_bytes:
            img_file = "data/uploaded_data/upload." + img_bytes.name.split('.')[-1]
            Image.open(img_bytes).save(img_file)

    if img_file:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_file, caption="Selected Image")
        with col2:
            img = infer_image(img_file)
            st.image(img, caption="Model Prediction")


def video_input(data_src):
    vid_file = None
    if data_src == 'Sample data':
        vid_file = "data/sample_videos/sample.mp4"
    else:
        vid_bytes = st.sidebar.file_uploader("Upload a video", type=['mp4', 'mpv', 'avi'])
        if vid_bytes:
            vid_file = "data/uploaded_data/upload." + vid_bytes.name.split('.')[-1]
            with open(vid_file, 'wb') as out:
                out.write(vid_bytes.read())

    if vid_file:
        cap = cv2.VideoCapture(vid_file)
        custom_size = st.sidebar.checkbox("Custom frame size")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if custom_size:
            width = st.sidebar.number_input("Width", min_value=120, step=20, value=width)
            height = st.sidebar.number_input("Height", min_value=120, step=20, value=height)

        fps = 0
        st1, st2, st3 = st.columns(3)
        with st1:
            st.markdown("## Height")
            st1_text = st.markdown(f"{height}")
        with st2:
            st.markdown("## Width")
            st2_text = st.markdown(f"{width}")
        with st3:
            st.markdown("## FPS")
            st3_text = st.markdown(f"{fps}")

        st.markdown("---")
        output = st.empty()
        prev_time = 0
        curr_time = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("Can't read frame, stream ended? Exiting ....")
                break
            frame = cv2.resize(frame, (width, height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_img = infer_image(frame)
            output.image(output_img)
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            st1_text.markdown(f"**{height}**")
            st2_text.markdown(f"**{width}**")
            st3_text.markdown(f"**{fps:.2f}**")

        cap.release()


def infer_image(img, size=None):
    model.conf = confidence
    result = model(img, size=size) if size else model(img)
    result.render()
    image = Image.fromarray(result.ims[0])
    return image


@st.cache_resource
def load_model(path, device):
    model_ = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=True)
    model_.to(device)
    print("model to ", device)
    return model_


@st.cache_resource
def download_model(url):
    model_file = wget.download(url, out="models")
    return model_file


def get_user_model():
    model_src = st.sidebar.radio("Model source", ["file upload", "url"])
    model_file = None
    if model_src == "file upload":
        model_bytes = st.sidebar.file_uploader("Upload a model file", type=['pt'])
        if model_bytes:
            model_file = "models/uploaded_" + model_bytes.name
            with open(model_file, 'wb') as out:
                out.write(model_bytes.read())
    else:
        url = st.sidebar.text_input("Model URL")
        if url:
            model_file_ = download_model(url)
            if model_file_.split(".")[-1] == "pt":
                model_file = model_file_

    return model_file


# ─────────────────────────────────────────────
#  TRAINING TAB
# ─────────────────────────────────────────────

def training_section():
    st.header("🏋️ Train a YOLOv5 Model")
    st.markdown(
        "Upload your labelled dataset as a **ZIP file** (YOLO format), configure "
        "training parameters, then click **Start Training**. The generated "
        "`best.pt` will be saved to `models/` and made available for inference."
    )

    # ── Dataset upload ──────────────────────────────────────────────────
    st.subheader("1. Upload Dataset")
    st.markdown(
        """
        Your ZIP must follow this structure:
        ```
        dataset.zip
        ├── images/
        │   ├── train/   ← training images
        │   └── val/     ← validation images
        ├── labels/
        │   ├── train/   ← YOLO .txt annotations
        │   └── val/
        └── data.yaml    ← class names + paths
        ```
        `data.yaml` example:
        ```yaml
        train: images/train
        val:   images/val
        nc:    3
        names: ['cat', 'dog', 'bird']
        ```
        """
    )

    zip_file = st.file_uploader("Upload dataset ZIP", type=["zip"])
    dataset_root = "data/training_data/dataset"
    yaml_path = None

    if zip_file:
        # Extract only once (or re-extract if user re-uploads)
        if os.path.exists(dataset_root):
            shutil.rmtree(dataset_root)
        os.makedirs(dataset_root, exist_ok=True)

        zip_save_path = "data/training_data/dataset.zip"
        with open(zip_save_path, "wb") as f:
            f.write(zip_file.read())

        with zipfile.ZipFile(zip_save_path, "r") as z:
            z.extractall(dataset_root)

        # Locate data.yaml (may be at root or one level deep)
        candidates = glob.glob(f"{dataset_root}/**/*.yaml", recursive=True) + \
                     glob.glob(f"{dataset_root}/*.yaml")
        if candidates:
            yaml_path = candidates[0]
            st.success(f"Dataset extracted. Found config: `{yaml_path}`")

            # Fix relative paths in data.yaml to absolute paths so YOLOv5 finds them
            with open(yaml_path, "r") as f:
                data_cfg = yaml.safe_load(f)

            base_dir = os.path.dirname(os.path.abspath(yaml_path))
            for key in ("train", "val", "test"):
                if key in data_cfg and not os.path.isabs(data_cfg[key]):
                    data_cfg[key] = os.path.join(base_dir, data_cfg[key])

            with open(yaml_path, "w") as f:
                yaml.dump(data_cfg, f)

            with st.expander("Preview data.yaml"):
                st.json(data_cfg)
        else:
            st.error("No `.yaml` config found inside the ZIP. Please add a `data.yaml`.")
            return

    # ── Hyperparameters ─────────────────────────────────────────────────
    st.subheader("2. Training Configuration")

    col1, col2, col3 = st.columns(3)
    with col1:
        base_model = st.selectbox(
            "Base YOLOv5 model (pretrained weights)",
            ["yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x"],
            index=1,
            help="Smaller = faster training; larger = better accuracy"
        )
        epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=50, step=10)

    with col2:
        img_size = st.selectbox("Image size (px)", [320, 416, 512, 640, 1280], index=3)
        batch_size = st.number_input("Batch size", min_value=1, max_value=128, value=16, step=4)

    with col3:
        project_name = st.text_input("Run / project name", value="yolov5_custom")
        freeze_layers = st.number_input(
            "Freeze backbone layers (0 = none)",
            min_value=0, max_value=24, value=0, step=1,
            help="Freeze the first N layers for transfer learning"
        )

    # Device
    device_str = "0" if torch.cuda.is_available() else "cpu"
    st.info(f"Training device: **{'GPU (cuda:0)' if device_str == '0' else 'CPU'}**")

    # ── Start training ───────────────────────────────────────────────────
    st.subheader("3. Train")

    if st.button("🚀 Start Training", disabled=(yaml_path is None)):
        output_dir = f"runs/train/{project_name}"
        best_pt_src = f"{output_dir}/weights/best.pt"
        best_pt_dst = f"models/{project_name}_best.pt"

        cmd = [
            "python", "-m", "yolov5.train",   # works if yolov5 installed as package
            "--img", str(img_size),
            "--batch", str(batch_size),
            "--epochs", str(epochs),
            "--data", os.path.abspath(yaml_path),
            "--weights", f"{base_model}.pt",
            "--project", "runs/train",
            "--name", project_name,
            "--device", device_str,
        ]
        if freeze_layers > 0:
            cmd += ["--freeze", str(freeze_layers)]

        st.markdown("**Training command:**")
        st.code(" ".join(cmd))

        log_area = st.empty()
        progress_bar = st.progress(0)
        log_lines = []

        with st.spinner("Training in progress — this may take a while..."):
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

                for line in process.stdout:
                    line = line.rstrip()
                    log_lines.append(line)

                    # Parse epoch progress from YOLOv5 output  e.g. "  1/50 "
                    if "/" in line and line.strip()[0].isdigit():
                        parts = line.strip().split("/")
                        try:
                            current_epoch = int(parts[0].strip().split()[-1])
                            progress_bar.progress(min(current_epoch / epochs, 1.0))
                        except (ValueError, IndexError):
                            pass

                    # Show last 20 lines in the log box
                    log_area.code("\n".join(log_lines[-20:]), language="bash")

                process.wait()

                if process.returncode == 0:
                    progress_bar.progress(1.0)
                    if os.path.isfile(best_pt_src):
                        shutil.copy(best_pt_src, best_pt_dst)
                        st.success(
                            f"✅ Training complete! Model saved to `{best_pt_dst}`\n\n"
                            "Switch to the **Inference** tab and select "
                            "**'Use your own model' → file upload** to use it."
                        )

                        # Offer download
                        with open(best_pt_dst, "rb") as f:
                            st.download_button(
                                label="⬇️ Download best.pt",
                                data=f,
                                file_name=f"{project_name}_best.pt",
                                mime="application/octet-stream",
                            )

                        # Show results image if available
                        results_img = f"runs/train/{project_name}/results.png"
                        if os.path.isfile(results_img):
                            st.image(results_img, caption="Training Results", use_column_width=True)
                    else:
                        st.warning(
                            f"Training finished but `best.pt` not found at `{best_pt_src}`. "
                            "Check logs above for errors."
                        )
                else:
                    st.error(f"Training failed (exit code {process.returncode}). See logs above.")

            except FileNotFoundError:
                st.error(
                    "Could not launch training. Make sure `yolov5` is installed:\n"
                    "```\npip install yolov5\n```\n"
                    "or clone the YOLOv5 repo and run from its root directory."
                )

    # ── Existing trained models ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("Previously Trained Models")
    trained_models = glob.glob("models/*_best.pt") + glob.glob("runs/train/**/weights/best.pt", recursive=True)
    if trained_models:
        for mp in trained_models:
            col_a, col_b = st.columns([4, 1])
            col_a.markdown(f"`{mp}`")
            with open(mp, "rb") as f:
                col_b.download_button(
                    "⬇️ Download",
                    data=f,
                    file_name=os.path.basename(mp),
                    mime="application/octet-stream",
                    key=mp,
                )
    else:
        st.info("No trained models found yet. Train one above!")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    global model, confidence, cfg_model_path

    st.title("🔍 Object Recognition Dashboard")

    # Top-level tabs
    tab_infer, tab_train = st.tabs(["📷 Inference", "🏋️ Train Model"])

    # ── INFERENCE TAB ────────────────────────────────────────────────────
    with tab_infer:
        st.sidebar.title("Settings")

        model_src = st.sidebar.radio(
            "Select YOLOv5 weight file",
            ["Use our demo model 5s", "Use your own model"]
        )

        if model_src == "Use your own model":
            user_model_path = get_user_model()
            if user_model_path:
                cfg_model_path = user_model_path
            st.sidebar.text(cfg_model_path.split("/")[-1])
            st.sidebar.markdown("---")

        if not os.path.isfile(cfg_model_path):
            st.warning("Model file not available! Add it to the `models/` folder or train one in the **Train Model** tab.", icon="⚠️")
        else:
            if torch.cuda.is_available():
                device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=False, index=0)
            else:
                device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=True, index=0)

            model = load_model(cfg_model_path, device_option)
            confidence = st.sidebar.slider('Confidence', min_value=0.1, max_value=1.0, value=.45)

            if st.sidebar.checkbox("Custom Classes"):
                model_names = list(model.names.values())
                assigned_class = st.sidebar.multiselect("Select Classes", model_names, default=[model_names[0]])
                classes = [model_names.index(name) for name in assigned_class]
                model.classes = classes
            else:
                model.classes = list(model.names.keys())

            st.sidebar.markdown("---")

            input_option = st.sidebar.radio("Select input type:", ['image', 'video'])
            data_src = st.sidebar.radio("Select input source:", ['Sample data', 'Upload your own data'])

            if input_option == 'image':
                image_input(data_src)
            else:
                video_input(data_src)

    # ── TRAINING TAB ─────────────────────────────────────────────────────
    with tab_train:
        training_section()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass