# AI-Workforce-Analysis-and-Deep-Learning-Classification

Prototype of a Data Processing Tool for Analyzing Various Sections of Workforce Using Artificial Intelligence.

## 📌 Project Overview
Higher education curricula descriptions in Lithuania are published in Lithuanian as unstructured free text via national registries (e.g. Open Data Portal `data.gov.lt`), whereas standardized international occupational taxonomies (**ISCO-08** by the ILO) are structured in English. 

This project bridges the cross-lingual and many-to-many semantic gap by fine-tuning multilingual Transformers (**XLM-RoBERTa-base**) to automatically map Lithuanian study programs to ISCO-08 occupational codes, paired with a Human-in-the-Loop (HITL) review dashboard adhering to the EU AI Act (Article 14).

---

## 🏗 End-to-End Architecture
1. **Data Ingestion**: Ingests Lithuanian study programs via `data.gov.lt` REST API and ISCO-08 taxonomies from the ILO.
2. **Preprocessing**: Normalizes text (Unicode NFC, preserves Lithuanian diacritics: `ą, č, ę, ė, į, š, ų, ū, ž`), handles SentencePiece BPE subword tokenization, and stratifies dataset splits (70/15/15%).
3. **Deep Learning Model**: Fine-tuned `xlm-roberta-base` (278M parameters, 12 layers) with custom 2-layer classification head ($768 \to 256 \to 43$) and inverse-frequency class-weighted `BCEWithLogitsLoss`.
4. **Human-in-the-Loop (HITL)**: Streamlit interactive interface (`app.py`) for top-$k$ ranked predictions, attention heatmaps, and audit logging.

---

## 📂 Repository Structure
```
├── app.py                     # Streamlit HITL web dashboard prototype
├── bpmn/                      # Process models (BPMN diagrams)
├── data/                      # Raw and processed datasets (see data/README.md)
│   ├── raw/
│   └── processed/
├── diagrams/                  # C4, UML Use Case, and Activity diagram sources
├── figures/                   # Rendered diagrams, loss curves, and distribution plots
├── src/                       # Pipeline source code
│   ├── build_dataset.py
│   ├── fetch_programs.py
│   ├── load_isco.py
│   ├── plot_distribution.py
│   ├── plot_loss_curves.py
│   ├── preprocessing.py
│   └── verify_tokenization.py
├── test_xlm.py                # Model smoke test
└── thesis.md                  # Comprehensive thesis research document
```

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install requirements:
```bash
git clone https://github.com/NamanJaiswal7/AI-Workforce-Analysis-and-Deep-Learning-Classification.git
cd AI-Workforce-Analysis-and-Deep-Learning-Classification
python3 -m venv venv
source venv/bin/activate
pip install torch transformers streamlit scikit-learn pandas
```

### 2. Run the Dashboard
```bash
streamlit run app.py
```
