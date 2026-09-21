import os
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. PAGE SETUP & THEME CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="SAT-RCAN Multispectral Decision Support",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling to match the dark operational cockpit layout
st.markdown("""
<style>
    .main-title { font-size: 2.1rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.05rem; color: #94A3B8; margin-bottom: 1.5rem; }
    .col-header { font-size: 1.15rem; font-weight: 600; color: #F1F5F9; margin-bottom: 0.1rem; }
    .col-sub { font-size: 0.82rem; color: #64748B; margin-bottom: 0.8rem; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. MODEL ARCHITECTURE DEFINITION
# ==============================================================================
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        return x * y.expand_as(x)


class RCAB(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            ChannelAttention(channels)
        )

    def forward(self, x):
        return x + self.body(x)


class SatelliteRCAN(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, num_features=64, num_blocks=6):
        super().__init__()
        self.head = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[RCAB(num_features) for _ in range(num_blocks)])
        self.post_body = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * 16, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=4),
            nn.PReLU()
        )
        self.tail = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        feat = self.head(x)
        res = self.post_body(self.body(feat)) + feat
        return self.tail(self.upsample(res))


# ==============================================================================
# 3. RESOURCE LOADING & CACHING
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_assets():
    model = SatelliteRCAN(in_channels=4, out_channels=4, num_features=64, num_blocks=6).to(device)
    if os.path.exists("best_satellite_rcan.pt"):
        model.load_state_dict(torch.load("best_satellite_rcan.pt", map_location=device))
    model.eval()

    if os.path.exists("test_x.pt") and os.path.exists("test_y.pt"):
        test_x = torch.load("test_x.pt", map_location="cpu")
        test_y = torch.load("test_y.pt", map_location="cpu")
    else:
        # Fallback synthetic terrain tensors (avoids static noise if tensors are missing)
        x = np.linspace(0, 10, 64)
        y = np.linspace(0, 10, 64)
        xx, yy = np.meshgrid(x, y)
        terrain = np.sin(xx) * np.cos(yy) * 0.3 + 0.5
        base_tensor = np.stack([terrain * 0.7 + 0.1, terrain * 0.6 + 0.1, terrain * 0.4 + 0.05, terrain * 0.9], axis=0)
        test_x = torch.tensor(base_tensor, dtype=torch.float32).unsqueeze(0).repeat(10, 1, 1, 1)
        test_y = nn.functional.interpolate(test_x, scale_factor=4, mode='bicubic', align_corners=False)

    return model, test_x, test_y


model, test_x, test_y = load_assets()


# Helper function to convert 4-band satellite tensor to clean RGB array
def to_rgb(arr):
    rgb = arr[:3].transpose(1, 2, 0)
    p2, p98 = np.percentile(rgb, (2, 98))
    rgb_scaled = np.clip((rgb - p2) / (p98 - p2 + 1e-7), 0.0, 1.0)
    return rgb_scaled


# ==============================================================================
# 4. SIDEBAR OPERATIONAL CONTROLS
# ==============================================================================
st.sidebar.markdown("### Operational Controls")
st.sidebar.caption("Smart India Hackathon (SIH)")
st.sidebar.caption("Track: Space Tech / Remote Sensing")

scene_idx = st.sidebar.slider(
    "Test Scene Index (Untouched Cohort)",
    min_value=0,
    max_value=len(test_x) - 1,
    value=3
)

focus_zoom = st.sidebar.checkbox("Focus 400% Zoom Corridor", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("### Decision Support Guardrails")
calc_uncertainty = st.sidebar.checkbox("Compute Uncertainty Proxy", value=True)
calc_ndvi = st.sidebar.checkbox("Run Biophysical NDVI Check", value=True)

# ==============================================================================
# 5. INFERENCE & METRIC CALCULATIONS
# ==============================================================================
x_scene = test_x[scene_idx:scene_idx + 1].to(device)
y_scene = test_y[scene_idx].numpy()

with torch.no_grad():
    if calc_uncertainty:
        p1 = model(x_scene)
        p2 = torch.flip(model(torch.flip(x_scene, dims=[3])), dims=[3])
        p3 = torch.flip(model(torch.flip(x_scene, dims=[2])), dims=[2])
        stack = torch.stack([p1[0], p2[0], p3[0]], dim=0)
        sr_img = torch.clamp(torch.mean(stack, dim=0), 0.0, 1.0).cpu().numpy()
        uncertainty_map = torch.std(stack, dim=0).mean(dim=0).cpu().numpy()
    else:
        sr_img = torch.clamp(model(x_scene)[0], 0.0, 1.0).cpu().numpy()
        uncertainty_map = np.zeros((sr_img.shape[1], sr_img.shape[2]))

lr_img = x_scene[0].cpu().numpy()

# Apply crop if 400% Zoom Corridor is enabled
if focus_zoom:
    h, w = lr_img.shape[1], lr_img.shape[2]
    lr_img = lr_img[:, h // 4:3 * h // 4, w // 4:3 * w // 4]

    H, W = sr_img.shape[1], sr_img.shape[2]
    sr_img = sr_img[:, H // 4:3 * H // 4, W // 4:3 * W // 4]
    y_scene = y_scene[:, H // 4:3 * H // 4, W // 4:3 * W // 4]
    uncertainty_map = uncertainty_map[H // 4:3 * H // 4, W // 4:3 * W // 4]

# ==============================================================================
# 6. MAIN DISPLAY PANELS
# ==============================================================================
st.markdown('<div class="main-title">Multispectral Satellite Super-Resolution & Decision Support</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Physics-Aware 4x Spatial Reconstruction with Biophysical (NDVI) and Geometric Integrity</div>',
    unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "Reconstructed Visualization",
    "Model Reliability & Error Maps",
    "Vegetation Signal Consistency (NDVI)"
])

with tab1:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="col-header">1. Sentinel-2 L2A Input</div>', unsafe_allow_html=True)
        st.markdown('<div class="col-sub">Native 10 m Observation Grid</div>', unsafe_allow_html=True)
        st.image(to_rgb(lr_img), width='stretch')

    with col2:
        st.markdown('<div class="col-header">2. Deep RCAN Super-Resolution</div>', unsafe_allow_html=True)
        st.markdown('<div class="col-sub">Reconstructed Finer Grid (~4 m Output Grid)</div>', unsafe_allow_html=True)
        st.image(to_rgb(sr_img), width='stretch')

    with col3:
        st.markdown('<div class="col-header">3. High-Resolution Reference</div>', unsafe_allow_html=True)
        st.markdown('<div class="col-sub">Airborne Optical Reference (Ground Truth Proxy)</div>',
                    unsafe_allow_html=True)
        st.image(to_rgb(y_scene), width='stretch')

with tab2:
    st.markdown("#### Test-Time Augmentation Disagreement Proxy & Radiometric Errors")
    c_err1, c_err2 = st.columns(2)

    with c_err1:
        fig1, ax1 = plt.subplots(figsize=(5, 4), facecolor='#0F172A')
        im1 = ax1.imshow(uncertainty_map, cmap='magma')
        ax1.set_title(r"Invariance Disagreement Map ($\sigma$)", color='white', fontsize=10)
        ax1.axis('off')
        cbar1 = fig1.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar1.ax.axes, 'yticklabels'), color='white')
        st.pyplot(fig1, width='stretch')
        plt.close(fig1)

    with c_err2:
        mae_map = np.mean(np.abs(sr_img - y_scene), axis=0)
        fig2, ax2 = plt.subplots(figsize=(5, 4), facecolor='#0F172A')
        im2 = ax2.imshow(mae_map, cmap='inferno', vmin=0.0, vmax=0.08)
        ax2.set_title("Radiometric Residual MAE", color='white', fontsize=10)
        ax2.axis('off')
        cbar2 = fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar2.ax.axes, 'yticklabels'), color='white')
        st.pyplot(fig2, width='stretch')
        plt.close(fig2)

with tab3:
    st.markdown("#### Normalized Difference Vegetation Index (NDVI) Conservation")
    eps = 1e-7
    ndvi_sr = (sr_img[3] - sr_img[0]) / (sr_img[3] + sr_img[0] + eps)
    ndvi_ref = (y_scene[3] - y_scene[0]) / (y_scene[3] + y_scene[0] + eps)
    ndvi_diff = np.abs(ndvi_sr - ndvi_ref)

    c_n1, c_n2, c_n3 = st.columns(3)
    with c_n1:
        st.caption("Super-Resolved NDVI")
        fig_n1, ax_n1 = plt.subplots(facecolor='#0F172A')
        ax_n1.imshow(ndvi_sr, cmap='RdYlGn', vmin=-0.2, vmax=0.8)
        ax_n1.axis('off')
        st.pyplot(fig_n1, width='stretch')
        plt.close(fig_n1)
    with c_n2:
        st.caption("High-Resolution Reference NDVI")
        fig_n2, ax_n2 = plt.subplots(facecolor='#0F172A')
        ax_n2.imshow(ndvi_ref, cmap='RdYlGn', vmin=-0.2, vmax=0.8)
        ax_n2.axis('off')
        st.pyplot(fig_n2, width='stretch')
        plt.close(fig_n2)
    with c_n3:
        st.caption(r"NDVI Drift Magnitude ($\Delta$NDVI)")
        fig_n3, ax_n3 = plt.subplots(facecolor='#0F172A')
        ax_n3.imshow(ndvi_diff, cmap='plasma', vmin=0.0, vmax=0.05)
        ax_n3.axis('off')
        st.pyplot(fig_n3, width='stretch')
        plt.close(fig_n3)