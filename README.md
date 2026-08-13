
# EZPro-Fit 🧬🤖

> **EZPro-Fit: A Thermophile-Informed Cross-Modal Generative Framework Integrating DNA Language and Protein Structure for Enzyme Mutant Design**

---

## 📖 Overview

**EZPro-Fit** is an end-to-end generative framework designed to bridge the gap between structural protein design and genetic encoding. While traditional models operate strictly at the amino acid level, EZPro-Fit jointly models **protein conformations, DNA coding sequences (CDS), and taxonomic context** by leveraging multimodal data from thermophilic microorganisms.

By capturing the coupled distribution of structure and coding information, EZPro-Fit enables **coherent structure-sequence co-design** and **bidirectional inference** between amino acid structures and nucleotide encodings, all while aligning outputs with specific host expression preferences (e.g., *Escherichia coli*).

---

## 🛠️ Environment Setup & Installation

This project depends on **OpenFold** for structural processing and features. Follow the steps below to set up your environment using the provided `environment.yml`.

### 1. Clone the Repository & Submodules

```bash
git clone https://github.com/Ran-Xu1213/EZPro-Fit.git
cd EZPro-Fit

```

### 2. Create and Activate the Conda Environment

Using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate ezprofit

```

### 3. Configure OpenFold

Ensure that OpenFold is properly installed and linked within your environment, as it is required for handling structural features during data loading and training.

---

## 🚀 Training

To train the model using the provided training script, run:

```bash
python3.8 train.py

```

## 💡 Citation

If you use EZPro-Fit in your research, please cite our work:

```bibtex
@article{ezprofit2026,
  title={EZPro-Fit: A Thermophile-Informed Cross-Modal Generative Framework Integrating DNA Language and Protein Structure for Enzyme Mutant Design},  author={Xu, Ran et al.},
  year={2026}
}

```

---

## 📜 License

This project is licensed under the terms of the [MIT License](https://www.google.com/search?q=LICENSE).
