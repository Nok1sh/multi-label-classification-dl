import streamlit as st
import numpy as np
import torch
import cv2

from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from cnn.base_model import ResNetModel
from data.dataset_class import MultiLabelDataset


device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model():
    model = ResNetModel.load_checkpoint(0)
    model.unfreeze_last_layer
    model.to(device)
    model.eval()

    return model

model = load_model()

target_layer = [model.model.layer4[-1].conv2]

cam = GradCAM(
    model=model.model,
    target_layers=target_layer
)

classes = MultiLabelDataset.CLASSES


def predict_model(img):

    tags = []

    img = ResNetModel.TRANSFORM(img).unsqueeze(0).to(device)

    pred = model.predict_proba(img)

    cls_ind = 0
    for cls, p in zip(classes.values(), pred.numpy()[0]):
        if p >= 0.5:
            tags.append((cls, round(float(p), 2), cls_ind))
        cls_ind += 1
        
    return sorted(tags, key=lambda x: x[1], reverse=True)

def visualize_activity_map(img, cls_id):

    image = ResNetModel.TRANSFORM(img).unsqueeze(0).to(device)

    targets = [ClassifierOutputTarget(cls_id)]

    cam = GradCAM(model=model.model, target_layers=[model.model.layer4[-1].conv2])

    grayscale_cam = cam(input_tensor=image, targets=targets)[0]

    h, w, _ = img.shape
    grayscale_cam = cv2.resize(grayscale_cam, (w, h))

    rgb_img = np.float32(img) / 255.0

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return visualization


uploaded_file = st.file_uploader("Browse file", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)

if uploaded_file is not None:

    img = np.array(Image.open(uploaded_file).convert('RGB'))

    with col1:
        st.image(img)

    if st.button("Predict model"):
        with st.spinner("Prediction..."):
            results = predict_model(img)
            st.session_state.results = results

    if "results" in st.session_state:
        with col2:
            st.subheader("Classes")
            for cls, p, ind in st.session_state.results:
                st.write(cls)
                st.progress(p, text=f"{p}")

    if st.button("Class Activation Map"):
        with col1:
            overlay = visualize_activity_map(img, 5)
            st.subheader(f"Class: {list(classes.values())[5]}")
            st.image(overlay)