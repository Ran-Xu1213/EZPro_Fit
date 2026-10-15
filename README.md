
# EZPro-Fit 🧬🤖

> **EZPro-Fit: A Thermophile-Informed Cross-Modal Generative Framework Integrating DNA Language and Protein Structure for Enzyme Mutant Design**

---

## 📖 Overview

**EZPro-Fit** is an end-to-end generative framework designed to bridge the gap between structural protein design and genetic encoding. While traditional models operate strictly at the amino acid level, EZPro-Fit jointly models **protein conformations, DNA coding sequences (CDS), and taxonomic context** by leveraging multimodal data from thermophilic microorganisms.

By capturing the coupled distribution of structure and coding information, EZPro-Fit enables **coherent structure-sequence co-design** and **bidirectional inference** between amino acid structures and nucleotide encodings, all while aligning outputs with specific host expression preferences (e.g., *Escherichia coli*).

---

## Dataset Description

The dataset comprises protein amino acid sequences, DNA sequences, and associated grouping annotations. The `train_list.csv` file contains **286,146 records** and serves as the source dataset for the training and validation subsets, with the validation subset obtained by partitioning this file. The test dataset, provided in [`test_dataset.csv`](EZPro-Fit/data/test_dataset.csv), contains **9,863 records** and is used for model evaluation. All record counts exclude the CSV header.

Each record contains seven fields: `Entry`, the entry identifier; `Sequence`, the protein amino acid sequence; `dna_sequence`, the DNA sequence; and `50_grouping`, `100_grouping`, `500_grouping`, and `1000_grouping`, which specify group assignments under four grouping configurations.


Due to its large file size, `train_list.csv` will be hosted on Zenodo. The repository link will be provided here once it becomes available. In the meantime, requests for access to the training data may be directed to [p2314443@mpu.edu.mo](mailto:p2314443@mpu.edu.mo).

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
