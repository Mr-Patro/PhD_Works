---
title: HCD-SE-TriNet Parkinson's EEG Clinical Portal
emoji: 🧠
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

# HCD-SE-TriNet: Clinical Decision-Support Portal for Parkinson's Disease EEG

Prototype web portal implementing the **Hierarchical Cross-Spectral & Squeeze-and-Excitation Tri-Branch Network (`HCD-SE-TriNet`)** trained on paired resting-state and walking 64-channel EEG (OpenNeuro `ds007526`, 88.72% accuracy) and externally validated on OpenNeuro `ds003490`.

## Local Execution (Ubuntu / Linux)
    pip install -r requirements.txt
    streamlit run app.py
