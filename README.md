# road-pci-intelligence

An end-to-end deep learning and decision-support system designed to automate road distress detection, calculate pavement health metrics, and optimize infrastructure maintenance. Manual road pavement inspection is dangerous, subjective, and difficult to scale across municipal networks. By leveraging deep semantic segmentation to identify and quantify critical distress categories (longitudinal cracks, transverse cracks, alligator cracking, and potholes) from roadway imagery, this system computes an automated, simplified Pavement Condition Index (PCI) score and runs budget-constrained repair prioritization simulations to help municipal planners allocate road maintenance funds effectively.

---

## Planned Pipeline

```mermaid
flowchart LR
    A[Roadway Imagery] --> B[Deep Learning Segmentation]
    B --> C[Distress Classification & Severity Extraction]
    C --> D[Simplified PCI Calculation]
    D --> E[Budget-Constrained Repair Prioritization]
    E --> F[Interactive Dashboard Application]
```

1. **Data Ingestion & Preprocessing**: Sourcing, exploring, and unifying pavement distress benchmarks with pixel-level segmentation masks and multi-class annotations.
2. **Deep Semantic Segmentation**: Training deep segmentation networks (e.g., U-Net / Feature Pyramid Networks via `segmentation-models-pytorch`) to classify distress types (longitudinal, transverse, alligator cracks, potholes).
3. **Pavement Condition Index (PCI) Engine**: Measuring defect density and severity from predicted masks to formulate an objective, simplified PCI score (0–100 scale).
4. **Maintenance Optimization Simulation**: Formulating budget-constrained resource allocation algorithms (greedy heuristic & linear programming) to prioritize street repairs for maximum network condition improvement.
5. **Interactive Dashboard**: A Streamlit web application enabling users to upload inspection imagery, view segmentation overlays, inspect distress metrics, and simulate maintenance budgets.

---

## Status Checklist

- [x] **Step 0: Environment & Tooling Verification** (Python 3.14, Git, Kaggle CLI, GitHub CLI authenticated)
- [x] **Step 1: Project Scaffolding & Version Control** (Directory structure, requirements, git initialization, GitHub repository)
- [ ] **Step 2: Dataset Discovery & Selection** (Evaluate Kaggle benchmarks for crack types & potholes)
- [ ] **Step 3: Exploratory Data Analysis Notebook** (Image resolution checks, mask formats, class distributions via Kaggle Kernels)
- [ ] **Step 4: Dataset Preprocessing & Standardization** (Convert annotations to uniform binary/multi-class masks, train/val/test splits)
- [ ] **Step 5: Model Training & Evaluation** (Semantic segmentation benchmarks, mIoU / Dice loss tracking)
- [ ] **Step 6: Automated PCI Computation Module** (Defect severity calculation from mask geometry)
- [ ] **Step 7: Budget Optimization Engine** (Cost estimation, knapsack/priority allocation)
- [ ] **Step 8: Interactive Streamlit Web Application** (User uploads, mask visualizations, budget scenario runner)

---

## Project Structure

```
road-pci-intelligence/
├── app/                  # Streamlit application UI and components
├── data/                 # Raw and processed datasets (git-ignored)
├── notebooks/            # Jupyter/Kaggle exploration & training notebooks
├── outputs/              # Model checkpoints, evaluation plots, and reports (git-ignored)
├── src/                  # Core modular source code (data loaders, model, PCI, optimization)
├── .gitignore            # Git ignore configuration
├── README.md             # Project overview and roadmap
└── requirements.txt      # Project Python dependencies
```
