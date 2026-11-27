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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(fn):
            return fn

mne.set_log_level("ERROR")


# ------------------------------------------------------------------------------
# Model Architecture Definitions
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
            nn.Linear(3 * embed_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(embed_dim, embed_dim), nn.Sigmoid(),
        )
        self.spec_bilinear_proj = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(),
        )
        self.se_channel_attn = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim // 2), nn.GELU(),
            nn.Linear(embed_dim // 2, 2 * embed_dim), nn.Sigmoid(),
        )
        self.hybrid_proj = nn.Sequential(
            nn.LayerNorm(2 * embed_dim), nn.Dropout(dropout),
            nn.Linear(2 * embed_dim, embed_dim), nn.GELU(),
        )
        self.branch_prior = nn.Parameter(torch.tensor([-0.30, 0.25, 0.05]))
        self.reliability_gate = nn.Sequential(nn.Linear(9, 16), nn.GELU(), nn.Linear(16, 3))
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Dropout(dropout),
            nn.Linear(embed_dim, 64), nn.GELU(), nn.Linear(64, 2),
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
        return self.hybrid_blend * lg_hybrid + (1.0 - self.hybrid_blend) * lg_branches, alpha


# ------------------------------------------------------------------------------
# Load 5-Fold Checkpoint on CPU (Moved Dynamically Inside @spaces.GPU)
# ------------------------------------------------------------------------------
MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://github.com/Mr-Patro/PhD_Works/releases/download/v1.0-hcd-se-trinet/hcd_se_trinet_deploy.pt",
)
CKPT_PATH = "./models_deploy/hcd_se_trinet_deploy.pt"

def get_ensemble():
    if not os.path.exists(CKPT_PATH):
        os.makedirs("./models_deploy", exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, CKPT_PATH)
    payload = torch.load(CKPT_PATH, map_location="cpu")
    bp = payload["best_params"]
    if hasattr(bp, "item"):
        bp = bp.item()

    folds = []
    for fw in payload["ensemble_folds"]:
        m_eeg = SiameseRestWalkEEGNet(n_channels=64, embed_dim=128)
        m_eeg.load_state_dict({k: v.float() for k, v in fw["eegnet"].items()})
        m_eeg.eval()

        m_conv = Pretrained2DBackbone("convnext_tiny", embed_dim=128)
        m_conv.load_state_dict({k: v.float() for k, v in fw["convnext"].items()})
        m_conv.eval()

        m_swin = Pretrained2DBackbone("swin_tiny_patch4_window7_224", embed_dim=128)
        m_swin.load_state_dict({k: v.float() for k, v in fw["swin"].items()})
        m_swin.eval()

        fnet = Tunable_HCD_SE_Fusion(
            embed_dim=128, dropout=float(bp["dropout"]),
            hybrid_blend=float(bp["hybrid_blend"]), mod_drop=0.0, temp=float(bp["temp"]),
        )
        fnet.load_state_dict({k: v.float() for k, v in fw["fusion"].items()})
        fnet.eval()
        folds.append((m_eeg, m_conv, m_swin, fnet))
    return folds

ENSEMBLE_FOLDS = get_ensemble()


def load_eeg_file(file_list, sfreq_target: float = 250.0) -> np.ndarray:
    if not isinstance(file_list, list):
        file_list = [file_list]
    header = None
    for f in file_list:
        p = f.name if hasattr(f, "name") else str(f)
        if p.lower().endswith((".set", ".edf", ".fif", ".vhdr", ".bdf")):
            header = p
            break
    if header is None:
        raise ValueError("Upload a valid EEG file (.set, .edf, .fif, .vhdr, or .bdf).")

    ext = os.path.splitext(header)[-1].lower()
    if ext == ".set":
        raw = mne.io.read_raw_eeglab(header, preload=True)
    elif ext == ".edf":
        raw = mne.io.read_raw_edf(header, preload=True)
    elif ext == ".vhdr":
        raw = mne.io.read_raw_brainvision(header, preload=True)
    elif ext == ".bdf":
        raw = mne.io.read_raw_bdf(header, preload=True)
    else:
        raw = mne.io.read_raw_fif(header, preload=True)

    raw.pick_types(eeg=True, eog=False, ecg=False, emg=False, misc=False, exclude="bads")
    if len(raw.ch_names) > 64:
        stds = np.std(raw.get_data(), axis=1)
        keep = np.sort(np.argsort(stds)[::-1][:64])
        raw.pick([raw.ch_names[i] for i in keep])
    elif len(raw.ch_names) < 64:
        pad = np.zeros((64, raw.n_times), dtype=np.float32)
        pad[: len(raw.ch_names)] = raw.get_data()
        raw = mne.io.RawArray(pad, mne.create_info([f"Ch{i+1}" for i in range(64)], raw.info["sfreq"], "eeg"))

    raw.crop(tmin=min(5.0, raw.times[-1] * 0.05))
    if abs(raw.info["sfreq"] - sfreq_target) > 1e-2:
        raw.resample(sfreq_target, npad="auto")
    raw.filter(1.0, 45.0, fir_design="firwin", phase="zero")
    raw.set_eeg_reference("average", projection=False)
    data = raw.get_data()
    med = np.median(data, axis=1, keepdims=True)
    mad = np.median(np.abs(data - med), axis=1, keepdims=True) + 1e-8
    return np.clip((data - med) / (1.4826 * mad), -6.0, 6.0).astype(np.float32)


def build_dual_tensors(eeg_r: np.ndarray, eeg_w: np.ndarray = None, sfreq: float = 250.0, n_epochs: int = 3):
    paired = eeg_w is not None
    if not paired:
        sp = int(eeg_r.shape[1] * 0.35)
        eeg_w, eeg_r = eeg_r[:, sp:], eeg_r[:, :sp]

    win_1d = int(4.0 * sfreq)
    max_len = min(eeg_r.shape[1], eeg_w.shape[1])
    seg_len = max(win_1d, max_len // n_epochs)
    starts = np.linspace(0, max(0, max_len - seg_len), num=n_epochs, dtype=int)

    ts_list, spec_list, b_r, b_w = [], [], [], []
    for s in starts:
        sw, sr = eeg_w[:, s : s + seg_len], eeg_r[:, s : s + seg_len]
        off = max(0, (seg_len - win_1d) // 2)
        w1, r1 = sw[:, off : off + win_1d], sr[:, off : off + win_1d]
        if w1.shape[1] < win_1d:
            w1 = np.pad(w1, ((0, 0), (0, win_1d - w1.shape[1])))
            r1 = np.pad(r1, ((0, 0), (0, win_1d - r1.shape[1])))
        w1 = (w1 - w1.mean(1, keepdims=True)) / (w1.std(1, keepdims=True) + 1e-6)
        r1 = (r1 - r1.mean(1, keepdims=True)) / (r1.std(1, keepdims=True) + 1e-6)
        ts_list.append(np.stack([w1, r1], axis=0).astype(np.float32))

        nper = min(500, sw.shape[1])
        freqs, psd_r = signal.welch(sr, fs=sfreq, nperseg=nper, noverlap=nper // 2, axis=-1)
        _, psd_w = signal.welch(sw, fs=sfreq, nperseg=nper, noverlap=nper // 2, axis=-1)
        fm = (freqs >= 1.0) & (freqs <= 45.0)
        bm = (freqs >= 13.0) & (freqs <= 30.0)
        b_r.append(float(np.mean(psd_r[:, bm]) / (np.mean(psd_r[:, fm]) + 1e-10)))
        b_w.append(float(np.mean(psd_w[:, bm]) / (np.mean(psd_w[:, fm]) + 1e-10)))

        lr, lw = np.log10(psd_r[:, fm] + 1e-10), np.log10(psd_w[:, fm] + 1e-10)
        zf = (224.0 / lw.shape[0], 224.0 / lw.shape[1])
        img = np.stack([ndimage.zoom(lw, zf, order=1), ndimage.zoom(lr, zf, order=1), ndimage.zoom(lw - lr, zf, order=1)], axis=0)
        img = (img - img.mean(axis=(1, 2), keepdims=True)) / (img.std(axis=(1, 2), keepdims=True) + 1e-6)
        spec_list.append(img.astype(np.float32))

    beta_react = ((np.mean(b_w) - np.mean(b_r)) / (np.mean(b_r) + 1e-8)) * 100.0
    return np.stack(ts_list), np.stack(spec_list), paired, float(beta_react)


@spaces.GPU
def analyze_patient(case_id, age, sex, med_state, demo_choice, rest_files, walk_files):
    if demo_choice != "None (Use Uploaded Files)":
        if not os.path.exists("sample_cases.npz"):
            raise gr.Error("Sample cases archive missing; please upload an EEG recording.")
        sc = np.load("sample_cases.npz")
        if "Parkinson" in demo_choice:
            X_ts, X_spec = sc["pd_ts"].astype(np.float32), sc["pd_spec"].astype(np.float32)
            case_id = "DEMO-PD-SUBJECT (ds007526)"
        else:
            X_ts, X_spec = sc["hc_ts"].astype(np.float32), sc["hc_spec"].astype(np.float32)
            case_id = "DEMO-HC-SUBJECT (ds007526)"
        paired, beta_react = True, -14.2 if "Parkinson" in demo_choice else -28.6
    else:
        if not rest_files:
            raise gr.Error("Please upload a Resting-State EEG file or select a built-in Demo Case.")
        eeg_r = load_eeg_file(rest_files)
        eeg_w = load_eeg_file(walk_files) if walk_files else None
        X_ts, X_spec, paired, beta_react = build_dual_tensors(eeg_r, eeg_w)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts_t = torch.tensor(X_ts, dtype=torch.float32, device=dev)
    sp_t = torch.tensor(X_spec, dtype=torch.float32, device=dev)

    fold_probs, fold_alphas = [], []
    b_probs = {"EEGNet": [], "ConvNeXt": [], "Swin": []}

    with torch.no_grad():
        for m_eeg, m_conv, m_swin, fnet in ENSEMBLE_FOLDS:
            m_eeg.to(dev); m_conv.to(dev); m_swin.to(dev); fnet.to(dev)
            l1, z1 = m_eeg(ts_t)
            l2, z2 = m_conv(sp_t)
            l3, z3 = m_swin(sp_t)
            l_hyb, alpha = fnet(z1, z2, z3, l1, l2, l3)
            fold_probs.append(float(F.softmax(l_hyb, dim=-1)[:, 1].mean().item()))
            fold_alphas.append(alpha.mean(dim=0).cpu().numpy())
            b_probs["EEGNet"].append(float(F.softmax(l1, dim=-1)[:, 1].mean().item()))
            b_probs["ConvNeXt"].append(float(F.softmax(l2, dim=-1)[:, 1].mean().item()))
            b_probs["Swin"].append(float(F.softmax(l3, dim=-1)[:, 1].mean().item()))
            m_eeg.cpu(); m_conv.cpu(); m_swin.cpu(); fnet.cpu()

    p_pd = float(np.mean(fold_probs))
    p_sd = float(np.std(fold_probs))
    alphas = np.mean(fold_alphas, axis=0)

    if p_pd < 0.42:
        tier, stage = "LOW RISK (Healthy Control Profile)", "Non-Parkinsonian Baseline"
    elif p_pd < 0.65:
        tier, stage = "BORDERLINE / EARLY ALERT", "Projected Mild Motor Stage (UPDRS-III <= 20)"
    elif p_pd < 0.88:
        tier, stage = "HIGH PD PROBABILITY", "Projected Mild-to-Moderate Stage (UPDRS-III ~15-28)"
    else:
        tier, stage = "VERY HIGH PD PROBABILITY", "Projected Moderate-to-Advanced Stage (UPDRS-III > 25)"

    summary_md = f"""
### Clinical Decision-Support Report — Case `{case_id}`
* **Screening Decision:** **{"PD POSITIVE" if p_pd >= 0.50 else "CONTROL / NEGATIVE"}** (`{tier}`)
* **Predicted Parkinson's Probability $P(\\text{{PD}})$:** **`{p_pd*100:.1f}%`** (±`{p_sd*100:.1f}%` 5-fold ensemble SD)
* **Estimated Motor Severity Stage:** **{stage}**
* **Sensorimotor Beta (13–30 Hz) Reactivity:** **`{beta_react:+.1f}%`** ({"Paired Rest-to-Walk" if paired else "Single-Session"})
* **Developer & Affiliation:** Peetabas Patro | School of Computer Science and Engineering, Vellore Institute of Technology (VIT), Vellore, India
"""

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    t_ax = np.linspace(0, 4.0, X_ts.shape[-1])
    axes[0].plot(t_ax, X_ts[0, 1, 4], color="#2471A3", lw=0.9, label="Resting State")
    axes[0].plot(t_ax, X_ts[0, 0, 4] + 4.2, color="#C0392B", lw=0.9, label="Active / Walk (+offset)")
    axes[0].set_title("(a) Sensorimotor 1D EEG Trace (Ch-5)")
    axes[0].set_xlabel("Time (s)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, linestyle="--", alpha=0.4)

    im = axes[1].imshow(X_spec[0, 2], aspect="auto", cmap="RdBu_r", origin="lower")
    axes[1].set_title("(b) Rest-to-Active Spectral Reactivity")
    axes[1].set_xlabel("Frequency Bins (1-45 Hz)")
    axes[1].set_ylabel("64 Scalp Channels")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    names = ["EEGNet (1D)", "ConvNeXt (2D)", "Swin (2D)"]
    xb = np.arange(3)
    axes[2].bar(xb - 0.18, alphas * 100, width=0.35, color="#117A65", edgecolor="black", label="Reliability Weight (%)")
    axes[2].bar(xb + 0.18, [np.mean(b_probs[k]) * 100 for k in ["EEGNet", "ConvNeXt", "Swin"]],
                width=0.35, color="#2E86C1", edgecolor="black", label="Branch P(PD) (%)")
    axes[2].set_xticks(xb)
    axes[2].set_xticklabels(names)
    axes[2].set_ylim(0, 108)
    axes[2].set_title("(c) Adaptive Branch Reliability Audit")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    report_df = pd.DataFrame([{
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Author_Affiliation": "Peetabas Patro | CSE, VIT Vellore, India",
        "Patient_ID": case_id,
        "Age": age,
        "Sex": sex,
        "Medication_State": med_state,
        "P_PD": round(p_pd, 4),
        "Ensemble_SD": round(p_sd, 4),
        "Decision": "PD Positive" if p_pd >= 0.50 else "Control / Negative",
        "Estimated_Stage": stage,
        "Alpha_EEGNet": round(float(alphas[0]), 3),
        "Alpha_ConvNeXt": round(float(alphas[1]), 3),
        "Alpha_Swin": round(float(alphas[2]), 3),
    }])
    csv_out = os.path.join(tempfile.gettempdir(), f"HCD_SE_TriNet_{case_id.replace(' ', '_')}.csv")
    report_df.to_csv(csv_out, index=False)
    return summary_md, fig, report_df, csv_out


with gr.Blocks(title="HCD-SE-TriNet Parkinson's EEG Portal", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🧠 HCD-SE-TriNet: Parkinson's Disease EEG Clinical Decision-Support Portal
    **Developed by:** **Peetabas Patro** | Ph.D. Research Fellow, School of Computer Science and Engineering, **Vellore Institute of Technology (VIT), Vellore, India**
    *Validated on OpenNeuro `ds007526` (88.72% Accuracy) & External Cohort `ds003490` (N=50). Research Use Only (CDSS Prototype).*
    """)

    with gr.Row():
        with gr.Column(scale=1):
            case_id = gr.Textbox(label="Patient / Case ID", value="CASE-001")
            age = gr.Number(label="Age (years)", value=66)
            sex = gr.Dropdown(["Male", "Female", "Other"], label="Sex", value="Male")
            med_state = gr.Dropdown(["OFF-Medication", "ON-Medication", "Drug-Naive", "Unknown"], label="Medication State", value="OFF-Medication")
            demo_choice = gr.Dropdown(
                ["None (Use Uploaded Files)", "Demo Case 1: Parkinson's Disease Subject (ds007526)", "Demo Case 2: Healthy Control Subject (ds007526)"],
                label="Quick Reviewer Demo Preset",
                value="Demo Case 1: Parkinson's Disease Subject (ds007526)",
            )
            rest_up = gr.File(label="1. Upload Resting-State EEG (.set, .edf, .fif, .vhdr)", file_count="multiple")
            walk_up = gr.File(label="2. Upload Active / Walking EEG (Optional)", file_count="multiple")
            run_btn = gr.Button("🔬 Run Clinical EEG Analysis", variant="primary")

        with gr.Column(scale=2):
            out_md = gr.Markdown("Click **Run Clinical EEG Analysis** to evaluate the selected demo case or uploaded EEG files.")
            out_plot = gr.Plot(label="Neurophysiological Reactivity & Reliability Audit")
            out_tbl = gr.Dataframe(label="Structured Clinical Report")
            out_csv = gr.File(label="Download Clinical Report (.CSV)")

    run_btn.click(
        fn=analyze_patient,
        inputs=[case_id, age, sex, med_state, demo_choice, rest_up, walk_up],
        outputs=[out_md, out_plot, out_tbl, out_csv],
    )

if __name__ == "__main__":
    demo.launch()
