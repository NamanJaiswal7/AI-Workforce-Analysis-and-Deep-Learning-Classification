# Dataset Directory Documentation

This directory contains the raw and preprocessed datasets used in the thesis: **"Prototype of a Data Processing Tool for Analyzing Various Sections of Workforce Using Artificial Intelligence"**.

---

## 📁 Directory Structure

```
data/
├── raw/
│   ├── study_programs_raw.json    # 6,304 records fetched from data.gov.lt API
│   └── isco08_en.csv              # 43 ISCO-08 2-digit groups (code, title, description)
└── processed/
    ├── programs_cleaned.json      # 2,240 active programs with cleaned Lithuanian text
    ├── train.csv                  # 1,568 rows (70% split)
    ├── val.csv                    # 336 rows (15% split)
    └── test.csv                   # 336 rows (15% split)
```

---

## 🎯 Formulation of $(X, y)$

In this supervised multi-label learning setting:
- **$X$ (Feature / Input):** Unstructured Lithuanian study program text concatenated from program title (`prg_pavad_lt`), study field (`stud_sritis_lt`), and subfield (`stud_posritis_lt`).
  - Preprocessing includes HTML tag stripping, Unicode NFC normalization, preservation of Lithuanian diacritics (`ą, č, ę, ė, į, š, ų, ū, ž`), lowercase transformation, and SentencePiece BPE tokenization.
- **$y$ (Target / Output):** One or more ISCO-08 2-digit Sub-Major Group codes (represented as multi-hot binary vectors across 43 classes).

---

## 🔍 Examples of $(X, y)$ in the Dataset

Below are concrete samples extracted directly from [`train.csv`](./processed/train.csv):

| Example | Feature Input $X$ (Lithuanian Study Program Text) | Label Output $y$ (ISCO-08 Codes) | ISCO-08 Classification Title(s) | Category |
| :---: | :--- | :---: | :--- | :---: |
| **1** | `php programuotojo modulinė profesinio mokymo programa informacijos ir ryšio technologijos programinės įrangos taikomųjų programų kūrimas ir analizė` | `25` | **25:** Information and Communications Technology Professionals | Single-label |
| **2** | `žaidimų kūrimas informacijos ir ryšio technologijos programinės įrangos taikomųjų programų kūrimas ir analizė` | `14` | **14:** Hospitality, Retail and Other Services Managers | Single-label |
| **3** | `raidos ir edukacinė psichologija socialiniai mokslai psichologija` | `24, 93` | **24:** Business and Administration Professionals<br>**93:** Labourers in Mining, Construction, Manufacturing and Transport | Multi-label |
| **4** | `komunikacija ir skaitmeninė rinkodara socialinė gerovė none` | `24, 12` | **24:** Business and Administration Professionals<br>**12:** Administrative and Commercial Managers | Multi-label |
| **5** | `šviesos technologijos fiziniai mokslai fizika` | `24` | **24:** Business and Administration Professionals | Single-label |
| **6** | `multimedijos techniko modulinė profesinio mokymo programa informacijos ir ryšio technologijos kompiuterio taikymas ir kompiuterinis raštingumas` | `25, 14` | **25:** Information and Communications Technology Professionals<br>**14:** Hospitality, Retail and Other Services Managers | Multi-label |
| **7** | `elektroninės įrangos derintojo modulinė profesinio mokymo programa inžinerija ir inžinerinės profesijos elektronika ir automatika` | `54` | **54:** Food Preparation Assistants | Single-label |
| **8** | `sporto vadyba verslas ir administravimas vadyba ir administravimas` | `24` | **24:** Business and Administration Professionals | Single-label |

---

## 📊 Label Space Reference
The target space comprises 43 2-digit sub-major groups from the ILO ISCO-08 taxonomy (e.g., `21` = Science and Engineering Professionals, `25` = ICT Professionals, `24` = Business & Administration Professionals, `12` = Administrative & Commercial Managers). Detailed definitions are located in [`raw/isco08_en.csv`](./raw/isco08_en.csv).
