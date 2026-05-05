import os
import random
import torch
import torch.nn.functional as F
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from src.model import get_model
from src.dataset import get_transforms, load_split

DATA_DIR = "data/PokemonData"

MODELS = {
    "Exp1: ResNet50 (Pretrained, Full Finetune)": ("models/exp1_resnet50_pretrained_full_best.pth", "resnet50", True, "full"),
    "Exp2: ResNet50 (Pretrained, Frozen)": ("models/exp2_resnet50_pretrained_frozen_best.pth", "resnet50", True, "frozen"),
    "Exp3: ResNet50 (Scratch, Full Finetune)": ("models/exp3_resnet50_scratch_full_best.pth", "resnet50", False, "full"),
    "Exp4: ConvNeXt-Base (Pretrained, Full Finetune)": ("models/exp4_convnext_base_pretrained_full_best.pth", "convnext_base", True, "full"),
}


@st.cache_resource
def load_model(model_key):
    path, backbone, pretrained, finetune = MODELS[model_key]
    classes = get_classes()
    model = get_model(backbone, num_classes=len(classes), pretrained=False, finetune=finetune)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


@st.cache_data
def get_classes():
    return sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])


@st.cache_data(max_entries=2000)
def load_thumbnail(img_path):
    img = Image.open(img_path).convert("RGB")
    img.thumbnail((150, 150))
    return img


def predict(model, image):
    classes = get_classes()
    _, val_transform = get_transforms()
    tensor = val_transform(image).unsqueeze(0)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0]
    top5 = probs.topk(5)
    return [(classes[i], probs[i].item()) for i in top5.indices]


# 세션 상태 초기화
if "selected_image" not in st.session_state:
    st.session_state.selected_image = None
if "gallery_page" not in st.session_state:
    st.session_state.gallery_page = 0
if "gallery_images" not in st.session_state:
    _, val_samples, _ = load_split(DATA_DIR)
    images = [path for path, _ in val_samples]
    random.shuffle(images)
    st.session_state.gallery_images = images
if "thumbnails_ready" not in st.session_state:
    st.session_state.thumbnails_ready = False

st.set_page_config(page_title="Pokemon Classifier", layout="wide")

st.markdown("""
<style>
    .block-container { overflow: hidden; height: 100vh; padding-bottom: 0; padding-top: 1rem; margin-top: 0rem; }
    .gallery-scroll { height: 80vh; overflow-y: auto; }
    #MainMenu { visibility: hidden; }
    header { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='color: red;'>Pokemon Classifier</h1>", unsafe_allow_html=True)

if not st.session_state.thumbnails_ready:
    images = st.session_state.gallery_images
    bar = st.progress(0, text="갤러리 이미지 준비 중...")
    for i, img_path in enumerate(images):
        load_thumbnail(img_path)
        bar.progress((i + 1) / len(images), text=f"갤러리 이미지 준비 중... ({i+1}/{len(images)})")
    st.session_state.thumbnails_ready = True
    st.rerun()

col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("<p style='margin-bottom:0'>모델 선택</p>", unsafe_allow_html=True)
    model_key = st.selectbox("", list(MODELS.keys()), label_visibility="collapsed")
    st.markdown("**이미지 입력**")
    input_left, input_right = st.columns(2)
    with input_left:
        uploaded = st.file_uploader("드래그&드롭 또는 파일 선택", type=["jpg", "jpeg", "png"])
    with input_right:
        url = st.text_input("또는 이미지 URL 입력")

    image = None
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.session_state.selected_image = None
    elif url:
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            st.session_state.selected_image = None
        except:
            st.error("URL에서 이미지를 불러올 수 없습니다.")
    elif st.session_state.selected_image:
        image = Image.open(st.session_state.selected_image).convert("RGB")

    if image:
        model = load_model(model_key)
        results = predict(model, image)
        top_name, top_prob = results[0]

        with st.container(height=500):
            st.markdown(f"<h2>It's {top_name}!</h2>", unsafe_allow_html=True)
            st.progress(top_prob)
            st.write(f"**확률: {top_prob*100:.1f}%**")

            st.write("### Top 5 예측")
            for name, prob in results:
                st.write(f"**{name}**: {prob*100:.1f}%")
                st.progress(prob)

with col_right:
    st.subheader("Val 이미지 갤러리 (클릭해서 예측)")

    @st.fragment
    def gallery():
        gallery_images = st.session_state.gallery_images
        PAGE_SIZE = 50
        total_pages = (len(gallery_images) - 1) // PAGE_SIZE + 1
        page = st.session_state.gallery_page
        page_images = gallery_images[page * PAGE_SIZE:(page + 1) * PAGE_SIZE]

        with st.container(height=620):
            for row in range(0, len(page_images), 5):
                cols = st.columns(5)
                for j, img_path in enumerate(page_images[row:row+5]):
                    with cols[j]:
                        st.image(load_thumbnail(img_path), use_container_width=True)
                        cls_name = os.path.basename(os.path.dirname(img_path))
                        if st.button(cls_name, key=f"btn_{row+j}"):
                            st.session_state.selected_image = img_path
                            st.rerun(scope="app")

        pcol1, pcol2, pcol3 = st.columns([1, 3, 1])
        with pcol1:
            if st.button("◀ 이전") and page > 0:
                st.session_state.gallery_page -= 1
                st.rerun()
        with pcol2:
            st.markdown(f"<p style='text-align:center'>{page+1} / {total_pages}</p>", unsafe_allow_html=True)
        with pcol3:
            if st.button("다음 ▶") and page < total_pages - 1:
                st.session_state.gallery_page += 1
                st.rerun()

    gallery()
