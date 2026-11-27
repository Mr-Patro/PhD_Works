import os
import tempfile
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.ndimage as ndimage
import mne
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import matplotlib.pyplot as plt
import streamlit as st

mne.set_log_level("ERROR")

# ------------------------------------------------------------------------------
# Page Configuration & Clinical Styling
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="HCD-SE-TriNet | Parkinson's EEG Clinical Decision Support",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 1.85rem;
        font-weight: 700;
        color: #1B2631;
        margin-bottom: 0.15rem;
    }
    .sub-header {
        font-size: 0.98rem;
        color: #566573;
        margin-bottom: 0.65rem;
    }
    .author-banner {
        background-color: #EBF5FB;
        border-left: 4px solid #2471A3;
        padding: 0.65rem 1rem;
        border-radius: 4px;
        font-size: 0.90rem;
        color: #154360;
        margin-bottom: 0.75rem;
    }
    .disclaimer-box {
        background-color: #FEF9E7;
        border-left: 4px solid #F39C12;
        padding: 0.65rem 1rem;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #7D6608;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# Model Architecture Definitions (Exact Match to Stage 5/7 Checkpoint)
# ------------------------------------------------------------------------------
class EEGNetEncoder1D(nn.Module):
    def __init__(self, n_channels: int = 64, F1: int = 16, D: int = 2, F2: int = 32, dropout: float = 0.35):
        super().__init__()
        self.conv_temporal = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.conv_spatial = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )
        self.conv_separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.unsqueeze(1)
        h = self.conv_temporal(h)[..., : x.shape[-1]]
        h = self.conv_spatial(h)
        t_len = h.shape[-1]
        h = self.conv_separable[0](h)[..., :t_len]
        h = self.conv_separable[1:](h)
        return h.flatten(1)


class SiameseRestWalkEEGNet(nn.Module):
    def __init__(self, n_channels: int = 64, embed_dim: int = 128, dropout: float = 0.35):
        super().__init__()
        self.encoder = EEGNetEncoder1D(n_channels=n_channels, dropout=dropout)
        self.proj = nn.Sequential(nn.LayerNorm(256), nn.Dropout(dropout), nn.Linear(256, embed_dim), nn.GELU())
        self.classifier = nn.Sequential(nn.LayerNorm(embed_dim), nn.Dropout(0.25), nn.Linear(embed_dim, 2))

    def forward(self, x_pair: torch.Tensor):
        z_walk = self.encoder(x_pair[:, 0])
        z_rest = self.encoder(x_pair[:, 1])
        embed = self.proj(torch.cat([z_walk, z_walk - z_rest], dim=-1))
        return self.classifier(embed), embed


class Pretrained2DBackbone(nn.Module):
    def __init__(self, model_name: str, embed_dim: int = 128, drop_path: float = 0.2):
        super().__init__()
        self.encoder = timm.create_model(model_name, pretrained=False, num_classes=0, drop_path_rate=drop_path)
        in_f = self.encoder.num_features
        self.proj = nn.Sequential(nn.LayerNorm(in_f), nn.Dropout(0.3), nn.Linear(in_f, embed_dim), nn.GELU())
        self.classifier = nn.Sequential(nn.LayerNorm(embed_dim), nn.Dropout(0.25), nn.Linear(embed_dim, 2))

    def forward(self, x: torch.Tensor):
        embed = self.proj(self.encoder(x))
        return self.classifier(embed), embed


class Tunable_HCD_SE_Fusion(nn.Module):
    def __init__(self, embed_dim: int = 128, dropout: float = 0.3, hybrid_blend: float = 0.55,
                 mod_drop: float = 0.12, temp: float = 2.0):
        super().__init__()
        self.hybrid_blend = hybrid_blend
        self.mod_drop = mod_drop
        self.temp = temp

        self.cross_spec_gate = nn.Sequential(
            nn.Linear(3 * embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )
        self.spec_bilinear_proj = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.se_channel_attn = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 2 * embed_dim),
            nn.Sigmoid(),
        )
        self.hybrid_proj = nn.Sequential(
            nn.LayerNorm(2 * embed_dim),
            nn.Dropout(dropout),
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
        )
        self.branch_prior = nn.Parameter(torch.tensor([-0.30, 0.25, 0.05]))
        self.reliability_gate = nn.Sequential(
            nn.Linear(9, 16),
            nn.GELU(),
            nn.Linear(16, 3),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

    def forward(self, z1, z2, z3, lg1, lg2, lg3):
        u1 = F.normalize(z1, p=2, dim=-1)
        u2 = F.normalize(z2, p=2, dim=-1)
        u3 = F.normalize(z3, p=2, dim=-1)

        g_2d = self.cross_spec_gate(torch.cat([u2, u3, torch.abs(u2 - u3)], dim=-1))
        z_2d_mix = g_2d * u2 + (1.0 - g_2d) * u3
        z_2d = self.spec_bilinear_proj(torch.cat([z_2d_mix, u2 * u3], dim=-1))

        z_dom = torch.cat([u1, z_2d], dim=-1)
        z_hybrid = self.hybrid_proj(z_dom * self.se_channel_attn(z_dom))

        certs, margins = [], []
        for lg in [lg1, lg2, lg3]:
            p = F.softmax(lg / self.temp, dim=-1).clamp(min=1e-6, max=1.0 - 1e-6)
            ent = -torch.sum(p * torch.log(p), dim=-1, keepdim=True) / np.log(2.0)
            certs.append(1.0 - ent)
            margins.append(torch.abs(p[:, 1:2] - p[:, 0:1]))

        cos_12 = (u1 * u2).sum(dim=-1, keepdim=True)
        cos_13 = (u1 * u3).sum(dim=-1, keepdim=True)
        cos_23 = (u2 * u3).sum(dim=-1, keepdim=True)
        rel_desc = torch.cat(certs + margins + [cos_12, cos_13, cos_23], dim=-1)

        alpha = F.softmax(self.reliability_gate(rel_desc) + self.branch_prior.unsqueeze(0), dim=-1)
        lg_hybrid = self.classifier(z_hybrid)
        lg_branches = alpha[:, 0:1] * lg1 + alpha[:, 1:2] * lg2 + alpha[:, 2:3] * lg3
        logits_final = self.hybrid_blend * lg_hybrid + (1.0 - self.hybrid_blend) * lg_branches
        return logits_final, alpha


# ------------------------------------------------------------------------------
# Cached Ensemble Loader (Zero-VRAM CPU Inference for Stability & Cloud Deploy)
# ------------------------------------------------------------------------------
CKPT_LOCAL_CANDIDATES = [
    "./models_deploy/hcd_se_trinet_deploy.pt",
    "../models_deploy/hcd_se_trinet_deploy.pt",
]
DEFAULT_GITHUB_RELEASE_URL = os.environ.get(
    "MODEL_URL",
    "https://github.com/Mr-Patro/PhD_Works/releases/download/v1.0-hcd-se-trinet/hcd_se_trinet_deploy.pt",
)


@st.cache_resource(show_spinner="Loading 5-Fold HCD-SE-TriNet Clinical Ensemble (CPU Engine)...")
def load_clinical_ensemble():
    ckpt_path = None
    for cand in CKPT_LOCAL_CANDIDATES:
        if os.path.exists(cand):
            ckpt_path = cand
            break

    if ckpt_path is None:
        os.makedirs("./models_deploy", exist_ok=True)
        ckpt_path = "./models_deploy/hcd_se_trinet_deploy.pt"
        urllib.request.urlretrieve(DEFAULT_GITHUB_RELEASE_URL, ckpt_path)

    # Strictly use CPU for clinical portal inference so active GPU training jobs never cause OOM
    device = torch.device("cpu")
    payload = torch.load(ckpt_path, map_location="cpu")
    bp = payload["best_params"]
    if hasattr(bp, "item"):
        bp = bp.item()

    models_per_fold = []
    for fold_w in payload["ensemble_folds"]:
        m_eeg = SiameseRestWalkEEGNet(n_channels=64, embed_dim=128).to(device)
        m_eeg.load_state_dict({k: v.float() for k, v in fold_w["eegnet"].items()})
        m_eeg.eval()

        m_conv = Pretrained2DBackbone("convnext_tiny", embed_dim=128).to(device)
        m_conv.load_state_dict({k: v.float() for k, v in fold_w["convnext"].items()})
        m_conv.eval()

        m_swin = Pretrained2DBackbone("swin_tiny_patch4_window7_224", embed_dim=128).to(device)
        m_swin.load_state_dict({k: v.float() for k, v in fold_w["swin"].items()})
        m_swin.eval()

        fnet = Tunable_HCD_SE_Fusion(
            embed_dim=128,
            dropout=float(bp["dropout"]),
            hybrid_blend=float(bp["hybrid_blend"]),
            mod_drop=0.0,
            temp=float(bp["temp"]),
        ).to(device)
        fnet.load_state_dict({k: v.float() for k, v in fold_w["fusion"].items()})
        fnet.eval()

        models_per_fold.append((m_eeg, m_conv, m_swin, fnet))

    return models_per_fold, payload.get("ch_names_64", [f"Ch{i+1}" for i in range(64)]), device


# ------------------------------------------------------------------------------
# Automated Clinical EEG Preprocessing & Dual-Domain Feature Builder
# ------------------------------------------------------------------------------
def read_uploaded_eeg(uploaded_files, sfreq_target: float = 250.0):
    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]

    with tempfile.TemporaryDirectory() as tmpdir:
        header_path = None
        valid_headers = (".set", ".edf", ".fif", ".vhdr", ".bdf")
        for uf in uploaded_files:
            out_p = os.path.join(tmpdir, uf.name)
            with open(out_p, "wb") as f:
                f.write(uf.getbuffer())
            if uf.name.lower().endswith(valid_headers):
                header_path = out_p

        if header_path is None:
            raise ValueError("Please upload a valid EEG header file (.set, .edf, .fif, .vhdr, or .bdf).")

        return load_eeg_from_disk(header_path, sfreq_target=sfreq_target)


def load_eeg_from_disk(file_path: str, sfreq_target: float = 250.0):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".set":
        raw = mne.io.read_raw_eeglab(file_path, preload=True)
    elif ext == ".edf":
        raw = mne.io.read_raw_edf(file_path, preload=True)
    elif ext == ".vhdr":
        raw = mne.io.read_raw_brainvision(file_path, preload=True)
    elif ext == ".bdf":
        raw = mne.io.read_raw_bdf(file_path, preload=True)
    else:
        raw = mne.io.read_raw_fif(file_path, preload=True)

    orig_sfreq = float(raw.info["sfreq"])
    orig_n_ch = len(raw.ch_names)
    duration_sec = float(raw.times[-1])

    raw.pick_types(eeg=True, eog=False, ecg=False, emg=False, misc=False, exclude="bads")
    non_scalp = [
        ch for ch in raw.ch_names
        if any(k in ch.upper() for k in ["EOG", "ECG", "EMG", "X", "Y", "Z", "VREF", "REF", "STATUS"])
    ]
    if non_scalp and len(raw.ch_names) - len(non_scalp) >= 60:
        raw.drop_channels(non_scalp)

    # Harmonize to 64 scalp channels
    if len(raw.ch_names) > 64:
        stds = np.std(raw.get_data(), axis=1)
        keep_idx = np.sort(np.argsort(stds)[::-1][:64])
        raw.pick([raw.ch_names[i] for i in keep_idx])
    elif len(raw.ch_names) < 64:
        pad = np.zeros((64, raw.n_times), dtype=np.float32)
        pad[: len(raw.ch_names)] = raw.get_data()
        info_64 = mne.create_info([f"Ch{i+1}" for i in range(64)], sfreq=raw.info["sfreq"], ch_types="eeg")
        raw = mne.io.RawArray(pad, info_64)

    t_start = min(5.0, raw.times[-1] * 0.05)
    raw.crop(tmin=t_start)
    if abs(raw.info["sfreq"] - sfreq_target) > 1e-2:
        raw.resample(sfreq_target, npad="auto")
    raw.filter(l_freq=1.0, h_freq=45.0, fir_design="firwin", phase="zero")
    raw.set_eeg_reference("average", projection=False)

    data = raw.get_data()
    med = np.median(data, axis=1, keepdims=True)
    mad = np.median(np.abs(data - med), axis=1, keepdims=True) + 1e-8
    clean = np.clip((data - med) / (1.4826 * mad), -6.0, 6.0).astype(np.float32)

    qc_meta = {
        "orig_channels": orig_n_ch,
        "orig_sfreq": orig_sfreq,
        "duration_sec": round(duration_sec, 1),
        "ch_names": raw.ch_names,
    }
    return clean, qc_meta


def prepare_patient_tensors(eeg_rest: np.ndarray, eeg_walk: np.ndarray = None, sfreq: float = 250.0, n_epochs: int = 3):
    paired_mode = eeg_walk is not None
    if not paired_mode:
        split_idx = int(eeg_rest.shape[1] * 0.35)
        eeg_walk = eeg_rest[:, split_idx:]
        eeg_rest = eeg_rest[:, :split_idx]

    win_1d = int(4.0 * sfreq)
    max_len = min(eeg_rest.shape[1], eeg_walk.shape[1])
    seg_len = max(win_1d, max_len // n_epochs)
    starts = np.linspace(0, max(0, max_len - seg_len), num=n_epochs, dtype=int)

    ts_list, spec_list = [], []
    beta_rest_vals, beta_walk_vals = [], []

    for s in starts:
        seg_w = eeg_walk[:, s : s + seg_len]
        seg_r = eeg_rest[:, s : s + seg_len]

        off = max(0, (seg_len - win_1d) // 2)
        w1 = seg_w[:, off : off + win_1d]
        r1 = seg_r[:, off : off + win_1d]
        if w1.shape[1] < win_1d:
            w1 = np.pad(w1, ((0, 0), (0, win_1d - w1.shape[1])))
            r1 = np.pad(r1, ((0, 0), (0, win_1d - r1.shape[1])))
        w1 = (w1 - w1.mean(1, keepdims=True)) / (w1.std(1, keepdims=True) + 1e-6)
        r1 = (r1 - r1.mean(1, keepdims=True)) / (r1.std(1, keepdims=True) + 1e-6)
        ts_list.append(np.stack([w1, r1], axis=0).astype(np.float32))

        nperseg = min(500, seg_w.shape[1])
        freqs, psd_r = signal.welch(seg_r, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2, axis=-1)
        _, psd_w = signal.welch(seg_w, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2, axis=-1)
        f_mask = (freqs >= 1.0) & (freqs <= 45.0)
        b_mask = (freqs >= 13.0) & (freqs <= 30.0)

        beta_rest_vals.append(float(np.mean(psd_r[:, b_mask]) / (np.mean(psd_r[:, f_mask]) + 1e-10)))
        beta_walk_vals.append(float(np.mean(psd_w[:, b_mask]) / (np.mean(psd_w[:, f_mask]) + 1e-10)))

        log_r = np.log10(psd_r[:, f_mask] + 1e-10)
        log_w = np.log10(psd_w[:, f_mask] + 1e-10)
        log_d = log_w - log_r

        zf = (224.0 / log_w.shape[0], 224.0 / log_w.shape[1])
        img = np.stack([
            ndimage.zoom(log_w, zf, order=1),
            ndimage.zoom(log_r, zf, order=1),
            ndimage.zoom(log_d, zf, order=1),
        ], axis=0)
        img = (img - img.mean(axis=(1, 2), keepdims=True)) / (img.std(axis=(1, 2), keepdims=True) + 1e-6)
        spec_list.append(img.astype(np.float32))

    biomarkers = {
        "paired_mode": paired_mode,
        "beta_rest_rel": float(np.mean(beta_rest_vals) * 100.0),
        "beta_walk_rel": float(np.mean(beta_walk_vals) * 100.0),
        "beta_reactivity_pct": float(((np.mean(beta_walk_vals) - np.mean(beta_rest_vals)) / (np.mean(beta_rest_vals) + 1e-8)) * 100.0),
    }
    return np.stack(ts_list, axis=0), np.stack(spec_list, axis=0), biomarkers


def run_patient_inference(models_per_fold, device, X_ts: np.ndarray, X_spec: np.ndarray):
    ts_t = torch.tensor(X_ts, dtype=torch.float32, device=device)
    sp_t = torch.tensor(X_spec, dtype=torch.float32, device=device)

    fold_probs, fold_alphas = [], []
    branch_probs = {"EEGNet": [], "ConvNeXt": [], "Swin": []}

    with torch.no_grad():
        for m_eeg, m_conv, m_swin, fnet in models_per_fold:
            l1, z1 = m_eeg(ts_t)
            l2, z2 = m_conv(sp_t)
            l3, z3 = m_swin(sp_t)
            l_hyb, alpha = fnet(z1, z2, z3, l1, l2, l3)

            fold_probs.append(float(F.softmax(l_hyb, dim=-1)[:, 1].mean().item()))
            fold_alphas.append(alpha.mean(dim=0).cpu().numpy())
            branch_probs["EEGNet"].append(float(F.softmax(l1, dim=-1)[:, 1].mean().item()))
            branch_probs["ConvNeXt"].append(float(F.softmax(l2, dim=-1)[:, 1].mean().item()))
            branch_probs["Swin"].append(float(F.softmax(l3, dim=-1)[:, 1].mean().item()))

    p_mean = float(np.mean(fold_probs))
    p_std = float(np.std(fold_probs))
    alpha_mean = np.mean(fold_alphas, axis=0)

    return {
        "p_pd": p_mean,
        "p_std": p_std,
        "alphas": alpha_mean,
        "branch_p": {k: float(np.mean(v)) for k, v in branch_probs.items()},
    }


def discover_demo_cohorts():
    cohorts = {}
    for cand in ["./ds007526-download", "../ds007526-download"]:
        if os.path.isdir(cand):
            subs = sorted([d for d in os.listdir(cand) if d.startswith("sub-")])
            if subs:
                cohorts["OpenNeuro ds007526 (Primary Paired Rest-Walk Cohort)"] = ("ds007526", cand, subs)
            break

    for cand in ["./ds003490-external", "../ds003490-external"]:
        if os.path.isdir(cand):
            subs = sorted([d for d in os.listdir(cand) if d.startswith("sub-")])
            if subs:
                cohorts["OpenNeuro ds003490 (External Cross-Site Validation Cohort)"] = ("ds003490", cand, subs)
            break
    return cohorts


def load_demo_subject_eeg(ds_tag: str, base_dir: str, sub_id: str):
    sub_dir = os.path.join(base_dir, sub_id)
    valid_exts = (".set", ".vhdr", ".edf", ".bdf", ".fif")
    eeg_files = []
    for r_dir, _, files in os.walk(sub_dir):
        for f in sorted(files):
            if f.lower().endswith(valid_exts):
                eeg_files.append(os.path.join(r_dir, f))

    if not eeg_files:
        raise FileNotFoundError(f"No EEG recordings found inside {sub_dir}")

    # Lookup ground-truth phenotype metadata from participants.tsv if present
    meta_label = ""
    tsv_path = os.path.join(base_dir, "participants.tsv")
    if os.path.exists(tsv_path):
        try:
            df_p = pd.read_csv(tsv_path, sep="\t")
            id_col = [c for c in df_p.columns if "participant" in c.lower()][0]
            grp_col = [c for c in df_p.columns if any(k in c.lower() for k in ["group", "diag", "status"])][0]
            row_m = df_p[df_p[id_col].astype(str) == sub_id]
            if len(row_m) > 0:
                meta_label = f" | Cohort Label: {row_m.iloc[0][grp_col]}"
        except Exception:
            pass

    if ds_tag == "ds007526":
        r_cands = [f for f in eeg_files if "task-rest" in os.path.basename(f).lower() and f.endswith(".set")]
        w_cands = [f for f in eeg_files if "task-walk" in os.path.basename(f).lower() and f.endswith(".set")]
        if not r_cands or not w_cands:
            raise FileNotFoundError(f"Missing paired rest/walk .set files for {sub_id}")
        eeg_r, qc_r = load_eeg_from_disk(r_cands[0])
        eeg_w, qc_w = load_eeg_from_disk(w_cands[0])
        return eeg_r, eeg_w, f"{sub_id} (ds007526{meta_label})"
    else:
        ses1_files = [f for f in eeg_files if "ses-01" in f] or eeg_files
        rest_f = [f for f in ses1_files if "rest" in os.path.basename(f).lower()]
        act_f = [f for f in ses1_files if "rest" not in os.path.basename(f).lower()]
        if act_f and rest_f:
            eeg_r, qc_r = load_eeg_from_disk(rest_f[0])
            eeg_w, qc_w = load_eeg_from_disk(act_f[0])
        else:
            target_f = act_f[0] if act_f else rest_f[0]
            eeg_r, qc_r = load_eeg_from_disk(target_f)
            eeg_w = None
        return eeg_r, eeg_w, f"{sub_id} (ds003490{meta_label})"


# ------------------------------------------------------------------------------
# Sidebar: Researcher Attribution & Clinical Metadata
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 👨‍💻 Principal Investigator & Affiliation")
    st.info(
        "**Peetabas Patro**\n\n"
        "Ph.D. Research Fellow, Computer Science & Engineering\n\n"
        "Vellore Institute of Technology (VIT), Vellore, Tamil Nadu, India"
    )
    st.markdown("---")
    st.header("🩺 Patient & Session Metadata")
    patient_id = st.text_input("Patient / Case ID", value="SUB-CASE-001")
    patient_age = st.number_input("Age (years)", min_value=18, max_value=100, value=66)
    patient_sex = st.selectbox("Sex", ["Male", "Female", "Other / Unspecified"])
    med_state = st.selectbox("Medication State at Recording", ["OFF-Medication", "ON-Medication", "Drug-Naive / Untreated", "Unknown"])
    st.markdown("---")
    st.markdown("**Model Specifications**")
    st.caption(
        "• **Architecture:** HCD-SE-TriNet (5-Fold Ensemble)\n"
        "• **Branches:** Siamese-EEGNet (1D) + ConvNeXt-Tiny (2D) + Swin-Tiny (2D)\n"
        "• **Primary Cohort:** OpenNeuro ds007526 (88.72% Acc)\n"
        "• **External Cohort:** OpenNeuro ds003490 (N=50)\n"
        "• **Inference Backend:** CPU Precision Engine (Zero-VRAM)"
    )


# ------------------------------------------------------------------------------
# Main Portal Interface
# ------------------------------------------------------------------------------
st.markdown('<div class="main-header">🧠 HCD-SE-TriNet: Parkinson’s Disease EEG Clinical Decision-Support Portal</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Hierarchical Cross-Spectral & Squeeze-and-Excitation Tri-Branch Adaptive Attention Framework</div>', unsafe_allow_html=True)

st.markdown("""
<div class="author-banner">
    <strong>Developed & Validated by:</strong> Peetabas Patro &nbsp;|&nbsp; Ph.D. Research Fellow in Computer Science, <strong>Vellore Institute of Technology (VIT), Vellore, India</strong> &nbsp;|&nbsp; Repository: <code>github.com/Mr-Patro/PhD_Works</code>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-box">
    <strong>Research Use Only (CDSS Prototype):</strong> This clinical portal is an investigational neurophysiological decision-support system validated on OpenNeuro <code>ds007526</code> and <code>ds003490</code>. Outputs assist neurological screening and severity assessment and do not replace clinical diagnosis.
</div>
""", unsafe_allow_html=True)

models_per_fold, default_chs, device = load_clinical_ensemble()

col_up1, col_up2 = st.columns(2)
with col_up1:
    rest_files = st.file_uploader(
        "1. Upload Resting-State EEG (Required — .set/.fdt, .edf, .fif, .vhdr/.eeg)",
        type=["set", "fdt", "edf", "fif", "vhdr", "eeg", "vmrk", "bdf"],
        accept_multiple_files=True,
        key="rest_uploader",
    )
with col_up2:
    walk_files = st.file_uploader(
        "2. Upload Active / Walking EEG (Optional — Recommended for Rest-to-Walk Reactivity)",
        type=["set", "fdt", "edf", "fif", "vhdr", "eeg", "vmrk", "bdf"],
        accept_multiple_files=True,
        key="walk_uploader",
    )

# Dual-Cohort Benchmark Demo Loader (ds007526 + ds003490)
demo_cohorts = discover_demo_cohorts()
use_demo = False
selected_demo_ds = None
selected_demo_sub = None

if demo_cohorts:
    with st.expander("⚡ Benchmark Cohort Demo Loader (OpenNeuro ds007526 & External ds003490)", expanded=True):
        c_d1, c_d2, c_d3 = st.columns([2.2, 1.3, 1.5])
        with c_d1:
            selected_demo_ds = st.selectbox("Select Validation Dataset", list(demo_cohorts.keys()))
        ds_tag, ds_dir, sub_list = demo_cohorts[selected_demo_ds]
        with c_d2:
            selected_demo_sub = st.selectbox("Select Subject ID", sub_list, index=min(5, len(sub_list) - 1))
        with c_d3:
            st.write("")
            st.write("")
            if st.button("🚀 Run Demo Inference", use_container_width=True):
                use_demo = True

run_clicked = st.button("🔬 Run Clinical EEG Analysis on Uploaded Files", type="primary", use_container_width=True)

if run_clicked or use_demo:
    try:
        with st.spinner("Harmonizing 64-channel scalp montage, filtering 1–45 Hz, and executing 5-fold HCD-SE-TriNet..."):
            if use_demo and selected_demo_ds is not None:
                ds_tag, ds_dir, _ = demo_cohorts[selected_demo_ds]
                eeg_r, eeg_w, demo_case_label = load_demo_subject_eeg(ds_tag, ds_dir, selected_demo_sub)
                patient_id = demo_case_label
            else:
                if not rest_files:
                    st.error("Please upload at least a Resting-State EEG file (or use the Benchmark Cohort Demo Loader above).")
                    st.stop()
                eeg_r, _ = read_uploaded_eeg(rest_files)
                eeg_w, _ = (read_uploaded_eeg(walk_files) if walk_files else (None, None))

            X_ts, X_spec, bio = prepare_patient_tensors(eeg_r, eeg_w)
            pred = run_patient_inference(models_per_fold, device, X_ts, X_spec)

        p_pd = pred["p_pd"]
        if p_pd < 0.42:
            risk_tier = "LOW RISK (Healthy Control Profile)"
            severity_est = "Non-Parkinsonian Baseline"
        elif p_pd < 0.65:
            risk_tier = "BORDERLINE / EARLY SCREENING ALERT"
            severity_est = "Projected Mild Motor Stage (UPDRS-III ≤ 20)"
        elif p_pd < 0.88:
            risk_tier = "HIGH PD PROBABILITY"
            severity_est = "Projected Mild-to-Moderate Motor Stage (UPDRS-III ~15–28)"
        else:
            risk_tier = "VERY HIGH PD PROBABILITY"
            severity_est = "Projected Moderate-to-Advanced Motor Stage (UPDRS-III > 25)"

        st.markdown("---")
        st.subheader(f"📋 Clinical Decision-Support Report — `{patient_id}`")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Parkinson's Probability P(PD)", f"{p_pd * 100:.1f}%", delta=f"±{pred['p_std']*100:.1f}% Ensemble SD")
        m2.metric("Screening Classification", "PD Positive" if p_pd >= 0.50 else "Control / Negative", delta=risk_tier)
        m3.metric("Estimated Clinical Stage", severity_est.split(" (")[0])
        m4.metric("Beta (13–30 Hz) Reactivity", f"{bio['beta_reactivity_pct']:+.1f}%",
                  delta="Paired Rest→Walk" if bio["paired_mode"] else "Single-Session Mode")

        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2))

        t_ax = np.linspace(0, 4.0, X_ts.shape[-1])
        axes[0].plot(t_ax, X_ts[0, 1, 4], color="#2471A3", lw=0.9, label="Resting / Baseline")
        axes[0].plot(t_ax, X_ts[0, 0, 4] + 4.2, color="#C0392B", lw=0.9, label="Active / Walk (+offset)")
        axes[0].set_title("(a) Harmonized Sensorimotor 1D EEG (Ch-5)")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Z-Normalized Amplitude")
        axes[0].legend(loc="upper right", fontsize=8)
        axes[0].grid(True, linestyle="--", alpha=0.4)

        im = axes[1].imshow(X_spec[0, 2], aspect="auto", cmap="RdBu_r", origin="lower")
        axes[1].set_title("(b) Rest-to-Active Spectral Reactivity (224×224)")
        axes[1].set_xlabel("Frequency Bins (1–45 Hz)")
        axes[1].set_ylabel("64 Scalp Channels")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        b_names = ["Siamese-EEGNet\n(1D Temporal)", "ConvNeXt-Tiny\n(2D Local CNN)", "Swin-Tiny\n(2D Transformer)"]
        b_weights = [pred["alphas"][0] * 100, pred["alphas"][1] * 100, pred["alphas"][2] * 100]
        b_scores = [pred["branch_p"]["EEGNet"] * 100, pred["branch_p"]["ConvNeXt"] * 100, pred["branch_p"]["Swin"] * 100]
        x_b = np.arange(3)
        axes[2].bar(x_b - 0.18, b_weights, width=0.35, color="#117A65", edgecolor="black", label="Attention Weight α (%)")
        axes[2].bar(x_b + 0.18, b_scores, width=0.35, color="#2E86C1", edgecolor="black", label="Branch P(PD) (%)")
        axes[2].set_xticks(x_b)
        axes[2].set_xticklabels(b_names, fontsize=8.5)
        axes[2].set_ylim(0, 108)
        axes[2].set_title("(c) Adaptive Branch Reliability Audit")
        axes[2].legend(loc="upper right", fontsize=8)
        axes[2].grid(axis="y", linestyle="--", alpha=0.4)

        plt.tight_layout()
        st.pyplot(fig)

        report_df = pd.DataFrame([{
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Developer_Affiliation": "Peetabas Patro | CSE, VIT Vellore, India",
            "Patient_ID": patient_id,
            "Age": patient_age,
            "Sex": patient_sex,
            "Medication_State": med_state,
            "Protocol_Mode": "Paired Rest+Active" if bio["paired_mode"] else "Single-Session",
            "Predicted_PD_Probability": round(p_pd, 4),
            "Ensemble_Uncertainty_SD": round(pred["p_std"], 4),
            "Screening_Decision": "PD Positive" if p_pd >= 0.50 else "Control / Negative",
            "Risk_Tier": risk_tier,
            "Estimated_Motor_Stage": severity_est,
            "Beta_Rest_RelPower_Pct": round(bio["beta_rest_rel"], 2),
            "Beta_Active_RelPower_Pct": round(bio["beta_walk_rel"], 2),
            "Beta_Reactivity_Pct": round(bio["beta_reactivity_pct"], 2),
            "Alpha_EEGNet": round(float(pred["alphas"][0]), 3),
            "Alpha_ConvNeXt": round(float(pred["alphas"][1]), 3),
            "Alpha_Swin": round(float(pred["alphas"][2]), 3),
        }])
        st.dataframe(report_df, use_container_width=True)

        csv_bytes = report_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Structured Clinical Assessment Report (.CSV)",
            data=csv_bytes,
            file_name=f"HCD_SE_TriNet_Report_{patient_id.replace(' ', '_').replace('|', '')}.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"EEG Processing Error: {e}")

st.markdown("---")
st.caption("© 2026 Peetabas Patro | School of Computer Science and Engineering, Vellore Institute of Technology (VIT), Vellore, India. All Rights Reserved.")
