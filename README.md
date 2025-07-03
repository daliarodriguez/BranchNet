# BranchNet

BranchNet is a neuro-symbolic learning framework that converts decision tree ensembles into sparse, partially connected feedforward neural networks. Each branch in the tree ensemble corresponds to a hidden neuron, preserving symbolic structure while enabling gradient-based learning.

This repository contains the code to reproduce the results from the paper:

**BranchNet: A Neuro-Symbolic Learning Framework for Structured Multi-Class Classification**\
Dalia Rodriguez-Salas, Christian Riess\
[https://arxiv.org/abs/2507.01781](https://arxiv.org/abs/2507.01781)

---

## ✨ Key Features

BranchNet offers a unique approach to structured data classification by:

* **Neuro-Symbolic Integration:** Combining the inherent interpretability of decision tree ensembles with the powerful, end-to-end gradient-based optimization capabilities of neural networks.
* **Structured Sparsity:** Enforcing sparsity by directly mapping each decision path (branch) from the tree ensemble to a distinct hidden neuron, preserving tree-derived connectivity patterns.
* **Interpretable by Design:** Maintaining strong symbolic interpretability throughout the learning process due to the direct correspondence between hidden neurons and decision paths, and the use of a frozen symbolic output layer ($W_2$).
* **Automatic Architecture Tuning:** The model's architecture is entirely data-driven, being derived directly from the decision tree ensemble, thereby eliminating the need for manual architectural tuning.
* **Strong Performance:** Consistently outperforming XGBoost in accuracy on structured multi-class tabular data, with statistically significant gains.
* **Adaptive Weight Refinement:** Allowing the input-to-hidden layer weights ($W_1$) to be updated through gradient-based training, adaptively fine-tuning the symbolic prior.

---

## 🔍 Overview of Evaluation

BranchNet was rigorously evaluated on datasets from two OpenML benchmark suites, leveraging the methodology used in the [BEExAI project](https://github.com/SquareResearchCenter-AI/BEExAI):

* **`multi_class`**: A curated set of 8 multi-class datasets from the OpenML-CC18 benchmark, primarily focused on 10-class problems (with one 3-class dataset).
* **`clf_num`**: A binary-class set of 15 numeric-only datasets from the `inria-soda/tabular-benchmark` suite.

---

## 📂 Contents

The repository is structured as follows:

* `BranchNet.py`: Contains the core BranchNet neural network architecture definition.
* `BranchNetFramework.py`: Implements the BranchNet learning framework, including the `fit` function, `train_step`, `validation_step`, `predict`, and `predict_proba` methods.
* `train.py`: The main script responsible for building and training both BranchNet and XGBoost models, and evaluating their performance. This script also handles the initial tree ensemble generation used to construct the BranchNet.
* `openml_download.py`: A utility script for programmatic downloading and preprocessing of the evaluated datasets from OpenML.
* `benchmetrics.py`: Defines the training configurations for each dataset and orchestrates the running of the benchmark experiments.
* `output/benchmarks/models/get_stats.py`: A post-processing script used to aggregate raw benchmark outputs and generate the formatted result tables presented in the paper.
* `requirements.txt`: Lists all Python dependencies required to run the code.
* `LICENSE`: Details the licensing terms for the project.

---

## 🚀 Getting Started

### Prerequisites

* Python 3.10 or higher.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/daliarodriguez/Branchnet
    cd BranchNet
    ```
2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

---

## 🏃 Usage

### 1. Download Datasets

First, download the necessary datasets using the provided script:

```bash
python openml_download.py
```
### 2. Run Experiments

To reproduce the full set of experiments as presented in the paper, execute the benchmetrics.py script:

```bash
python benchmetrics.py
```
This script will iterate through all specified datasets and models (BranchNet and XGBoost) and run experiments across multiple random seeds (10 seeds, as used in the paper) to ensure robust results. This process may take a significant amount of time depending on your hardware.

### 3. Generate Result Tables

After the benchmetrics.py run is complete, you can generate the summary tables (similar to those in the paper) using the get_stats.py script:

```bash
python output/benchmarks/models/get_stats.py
```
This will process the raw output files from the experiments and print the aggregated results, allowing for easy comparison and analysis.

---

## 📊 Results Summary

BranchNet outperforms XGBoost on all 8 multi-class benchmarks (Wilcoxon p < 0.01). Performance on binary tasks is more mixed, showing strong results in a few cases, but underperforming in others.

For reproducibility, see:

- `output/benchmarks/models/performance_summary.csv`
- `output/benchmarks/models/branchnet_summary.csv`

---

## 📘 Citation

If you use this work, please cite:

```bibtex
@misc{rodriguezsalas2025branchnet,
  title={BranchNet: A Neuro-Symbolic Learning Framework for Structured Multi-Class Classification},
  author={Dalia Rodriguez-Salas and Christian Riess},
  year={2025},
  eprint={arXiv:2507.01781},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

---

## 🔧 License

MIT License. See `LICENSE` file for details.

---

## 📫 Contact

For questions or feedback:

- Dalia Rodriguez-Salas — [dalia.rodriguez@fau.de](mailto\:dalia.rodriguez@fau.de)
