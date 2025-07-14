
---

# XRD-Soil-Mineral-ID: A tool for semi-quantitative identification of soil minerals using XRD data

**XRD-Soil-Mineral-ID** is a Python-based tool for semi-quantitative identification and visualization of soil minerals from X-ray diffraction (XRD) data. The script detects peaks, matches them with known minerals, and estimates relative abundances using peak area and reference intensity ratios (RIRs).

Developed by **Eduardo Gutierrez Brito**  
Contributor: **Gustavo A. Roa** ([@Gustavo-Roa](https://github.com/Gustavo-Roa))

> [![DOI](https://zenodo.org/badge/1019710134.svg)](https://doi.org/10.5281/zenodo.15882593)

---

## 📁 Folder Structure

```
XRD-Soil-Mineral-ID/
├── data/
│   └── example_xrd_soil_data.xlsx
├── xrd_soil_mineral_id.py
├── xrd_soil_mineral_id.ipynb
├── requirements.txt
├── LICENSE
└── README.md
```


---

## 🚀 Getting Started

### 1. Repository

You have two options to get the code:

#### 📦 Option A: Clone with Git

```bash
git clone https://github.com/Gustavo-Roa/XRD-Soil-Mineral-ID.git
cd XRD-Soil-Mineral-ID
```

#### 📁 Option B: Download ZIP

1. Click the green **Code** button on the GitHub page.
2. Select **Download ZIP**.
3. Extract the contents and navigate into the extracted folder.

---

### 2. Install Dependencies

Install all required Python packages using:

```bash
pip install -r requirements.txt
```

> **Note:** Python 3.8 or higher is recommended.

---

### 3. Run the Tool

Choose one of the following options:

#### ✅ Option A: Run as a script

```bash
python xrd_soil_mineral_id.py
```

#### ✅ Option B: Open the notebook

```bash
jupyter notebook xrd_soil_mineral_id.ipynb
```


---

## 📊 Features

* Load XRD data from Excel file
* Automatically detect significant peaks
* Match peaks with known soil mineral 2θ references
* Estimate mineral abundances using peak area and RIRs
* Generate annotated XRD plots and tabular summaries

---

## 📥 Input Format

The input Excel file should contain the following columns:

* `two_theta_1`: 2θ angles
* `intensity_1`: Corresponding intensities

Ensure the file is located in the `data/` folder with the name `soil_xrd_data_example.xlsx`.

---

## 📤 Output

* Annotated XRD plot with labeled peaks
* Table of mineral percentage by sample

---

## 📚 Reference

If you use this repository, please cite:

> Gutierrez, E., & Roa, G. A. (2025). *XRD-Soil-Mineral-ID: A tool for semi-quantitative identification of soil minerals using XRD data.*  
> [![DOI](https://zenodo.org/badge/1019710134.svg)](https://doi.org/10.5281/zenodo.15882593)

---

## 📝 License

MIT License. See the LICENSE file for details.

---

## 🤝 Contribute

For questions, contributions, or suggestions, feel free to open an issue, submit a pull request, or contact either of us directly:

- 📧 Eduardo Gutierrez: eegutier@ksu.edu  
- 📧 Gustavo Roa: groa@ksu.edu

