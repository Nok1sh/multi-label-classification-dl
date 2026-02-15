import streamlit as st
import numpy as np
import torch

from PIL import Image

from cnn.base_model import ResNetModel
from data.dataset_class import MultiLabelDataset
from cnn.utils import predict_model, visualize_activity_map
from ui_style import colored_progress, colored_button, text_styles, select_box_style


@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNetModel.load_checkpoint(0, validate=True)
    model.to(device)
    model.unfreeze_last_layer
    model.eval()

    return model

model = load_model()

classes = MultiLabelDataset.CLASSES


st.title("Multi-Label Classification")

st.set_page_config(layout="wide")

colored_button()
text_styles()
select_box_style()

uploaded_file = st.file_uploader("Browse file", type=["jpg", "jpeg", "png"])

col1, col2, col3, col4 = st.columns(4)

if uploaded_file is not None:
    current_file_name = uploaded_file.name

    if "last_file_name" not in st.session_state or st.session_state.last_file_name != current_file_name:
        st.session_state.clear()
        st.session_state.last_file_name = current_file_name

    img = np.array(Image.open(uploaded_file).convert('RGB'))

    with col1:
        st.image(img)

    if st.button("🔍 Predict"):
        with st.spinner("Prediction..."):
            results = predict_model(model, img)
            st.session_state.results = results

    if "results" in st.session_state:
        with col3:
            st.subheader("Classes")
            for cls, p, ind in st.session_state.results:
                st.markdown(f'<p class="text-subtitle">{cls}</p>', unsafe_allow_html=True)
                colored_progress(p)

            
            class_options = [(ind, cls) for cls, _, ind in st.session_state.results]
        
        with col4:
            st.markdown(f'<p class="text-body">Select class for Grad-CAM:</p>', unsafe_allow_html=True)
            selected = st.selectbox(
                "",
                options=class_options,
                format_func=lambda x: x[1],
            )
            if st.button("🖼️Class Activation Map"):
                with col2:
                    overlay = visualize_activity_map(model, img, selected[0])
                    st.image(overlay)
                    st.subheader(f"Class Activation Map: {list(classes.values())[selected[0]]}")

    