import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageEnhance
import requests
import io
import os
import numpy as np

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Anime Sketch Colorizer",
    page_icon="🎨",
    layout="centered"
)

# ── Dark Theme CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #9ca3af;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .result-header {
        text-align: center;
        font-size: 1.1rem;
        color: #a78bfa;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #7c3aed, #2563eb);
        color: white;
        border: none;
        padding: 0.75rem;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }
    .upload-box {
        border: 2px dashed #4b5563;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        background: #1a1f2e;
    }
    div[data-testid="stSidebar"] { background-color: #161b27; }
    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #a78bfa;
        margin-bottom: 1rem;
    }
    .info-box {
        background: #1a1f2e;
        border-left: 3px solid #7c3aed;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.85rem;
        color: #9ca3af;
        margin-top: 0.5rem;
    }
    hr { border-color: #2d3748; }
</style>
""", unsafe_allow_html=True)

# ── Model Definition (must match training code exactly) ────────────────────
def encoder_block(in_channels, out_channels, use_batchnorm=True):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
    if use_batchnorm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2))
    return nn.Sequential(*layers)

def decoder_block(in_channels, out_channels, use_dropout=False):
    layers = [
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
    ]
    if use_dropout:
        layers.append(nn.Dropout(0.5))
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class UNetGenerator(nn.Module):
    def __init__(self):
        super(UNetGenerator, self).__init__()
        self.enc1       = encoder_block(3,   64,  use_batchnorm=False)
        self.enc2       = encoder_block(64,  128)
        self.enc3       = encoder_block(128, 256)
        self.enc4       = encoder_block(256, 512)
        self.enc5       = encoder_block(512, 512)
        self.enc6       = encoder_block(512, 512)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU()
        )
        self.dec1  = decoder_block(512,  512, use_dropout=True)
        self.dec2  = decoder_block(1024, 512, use_dropout=True)
        self.dec3  = decoder_block(1024, 512, use_dropout=True)
        self.dec4  = decoder_block(1024, 256)
        self.dec5  = decoder_block(512,  128)
        self.dec6  = decoder_block(256,  64)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        b  = self.bottleneck(e6)
        d1 = self.dec1(b);  d1 = torch.cat([d1, e6], dim=1)
        d2 = self.dec2(d1); d2 = torch.cat([d2, e5], dim=1)
        d3 = self.dec3(d2); d3 = torch.cat([d3, e4], dim=1)
        d4 = self.dec4(d3); d4 = torch.cat([d4, e3], dim=1)
        d5 = self.dec5(d4); d5 = torch.cat([d5, e2], dim=1)
        d6 = self.dec6(d5); d6 = torch.cat([d6, e1], dim=1)
        return self.final(d6)

# ── Load Model from HuggingFace ────────────────────────────────────────────
HF_REPO    = "adeelumar17/pix2pix"          # your HF repo
CHECKPOINT = "checkpoint_epoch_100.pth"      # your checkpoint filename
HF_URL     = f"https://huggingface.co/{HF_REPO}/resolve/main/{CHECKPOINT}"

@st.cache_resource(show_spinner=False)
def load_model():
    device = torch.device("cpu")  # Streamlit Cloud has no GPU
    model  = UNetGenerator().to(device)

    # ── Try HuggingFace token from Streamlit secrets ───────────────────────
    hf_token = None
    try:
        hf_token = st.secrets["HF_TOKEN"]
    except Exception:
        pass

    headers  = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    response = requests.get(HF_URL, headers=headers, stream=True)

    if response.status_code != 200:
        st.error(f"Failed to download model weights. Status: {response.status_code}\n"
                  "Make sure your HF repo is public or HF_TOKEN is set in Streamlit secrets.")
        st.stop()

    buf        = io.BytesIO(response.content)
    checkpoint = torch.load(buf, map_location=device)

    # strip module. prefix if saved with DataParallel
    def strip_prefix(state_dict):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(strip_prefix(checkpoint["generator_state_dict"]))
    model.eval()
    return model, device

# ── Inference ──────────────────────────────────────────────────────────────
def colorize(model, device, image: Image.Image, output_size: int) -> Image.Image:
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    inp = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(inp)
    out_img = (out.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)
    out_np  = (out_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    result  = Image.fromarray(out_np)
    if output_size != 128:
        result = result.resize((output_size, output_size), Image.LANCZOS)
    return result

def apply_post_processing(image: Image.Image, brightness: float, contrast: float) -> Image.Image:
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    return image

# ── UI ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🎨 Anime Sketch Colorizer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an anime sketch and generate a colorized version using pix2pix</div>', unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚙️ Generation Settings</div>', unsafe_allow_html=True)
    st.markdown("---")

    output_size = st.select_slider(
        "Output Image Size",
        options=[128, 256, 512],
        value=256,
        help="Larger sizes are upscaled from 128px output"
    )
    st.markdown('<div class="info-box">Model generates at 128px. Higher sizes apply LANCZOS upscaling.</div>',
                unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**🎨 Post-Processing**")

    brightness = st.slider(
        "Brightness",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.05,
        help="Adjust output brightness"
    )
    contrast = st.slider(
        "Contrast",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.05,
        help="Adjust output contrast"
    )

    st.markdown("---")
    st.markdown("**ℹ️ About**")
    st.markdown("""
    <div class="info-box">
    pix2pix conditional GAN trained on ~15K anime sketch-colorization pairs for 85 epochs.
    <br><br>
    Model: U-Net Generator + PatchGAN Discriminator
    </div>
    """, unsafe_allow_html=True)

# ── Main Area ──────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload your anime sketch",
    type=["png", "jpg", "jpeg"],
    help="Upload a black and white anime sketch"
)

if uploaded_file:
    input_image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="result-header">Input Sketch</div>', unsafe_allow_html=True)
        st.image(input_image, use_column_width=True)

    generate_btn = st.button("🎨 Generate Colorized Image")

    if generate_btn:
        with st.spinner("Loading model..."):
            model, device = load_model()

        with st.spinner("Colorizing your sketch..."):
            output_image = colorize(model, device, input_image, output_size)
            output_image = apply_post_processing(output_image, brightness, contrast)

        with col2:
            st.markdown('<div class="result-header">Generated Output</div>', unsafe_allow_html=True)
            st.image(output_image, use_column_width=True)

        # ── Download Button ────────────────────────────────────────────────
        buf = io.BytesIO()
        output_image.save(buf, format="PNG")
        buf.seek(0)
        st.download_button(
            label="⬇️ Download Colorized Image",
            data=buf,
            file_name="colorized_anime.png",
            mime="image/png"
        )

else:
    st.markdown("""
    <div class="upload-box">
        <div style="font-size: 3rem;">🖼️</div>
        <div style="color: #9ca3af; margin-top: 0.5rem;">Upload an anime sketch to get started</div>
        <div style="color: #6b7280; font-size: 0.8rem; margin-top: 0.3rem;">PNG, JPG, JPEG supported</div>
    </div>
    """, unsafe_allow_html=True)
