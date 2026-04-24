# AI-Assisted Human Annotation Platform for Medical Chest X-Ray Images

**CSE 881 — Course Project**

Velamuru Sai Yashwitha Reddy | Jampala Harshitha | Duke Ethan | Gagan Anandan

---

## Project Overview

This project builds a human-in-the-loop annotation platform for chest X-ray images and radiology reports. A ResNet50 model trained on the Kaggle Chest X-Ray Pneumonia dataset generates Normal/Pneumonia predictions with confidence scores for 7,467 unlabelled NLM OpenI NLMCXR images. High-confidence predictions (>= 0.70) are auto-accepted. Low-confidence predictions (< 0.70) are routed to human annotators for review. All records, predictions, and annotation decisions are stored in MongoDB Atlas. A separate text pipeline uses a RAG-based LLM (Llama 3.3 70B via Groq) to extract CheXpert-14 multi-label conditions from paired radiology reports.

---

## Datasets

### Dataset 1 — Kaggle Chest X-Ray Pneumonia (Labelled — used for model training)
- Download from: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- After downloading, extract to: C:\Users\HP\Desktop\archive\chest_xray
- 5,856 images with real doctor-verified labels: NORMAL and PNEUMONIA
- Pre-split into train / val / test folders

Expected folder structure after extraction:

    chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/

### Dataset 2 — NLM OpenI NLMCXR (Unlabelled — annotation target)
- Download images from: https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz
- Download reports from: https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz
- Extract images to: C:\Users\HP\Desktop\CSE881_ProjectData\NLMCXR_png
- 7,470 PNG chest X-ray images from 3,851 unique patients
- Zero diagnostic labels — this is what the platform annotates
- Paired XML radiology reports available for the text pipeline

---

## Requirements

Python version: 3.9.10

Install all dependencies by running this in your terminal:

    pip install torch torchvision pymongo scikit-learn scikit-image tqdm matplotlib seaborn pandas numpy pillow scipy

If torch fails to install, use:

    python -m pip install torch torchvision

---

## MongoDB Atlas Setup

1. Create a free account at https://www.mongodb.com/cloud/atlas/register
2. Create a free M0 cluster (choose any region)
3. Under Database Access — create a user with username xray_user and a password of your choice
4. Under Network Access — click Add IP Address and Allow Access From Anywhere (0.0.0.0/0)
5. Under Database, click Connect, then Drivers, then Python — copy your connection string
6. It looks like: mongodb+srv://xray_user:YOUR_PASSWORD@cluster0.xxxxx.mongodb.net/
7. Paste this URI into the MONGO_URI variable at the top of notebooks/xray_final_project.ipynb
8. Make sure to remove the angle brackets around your password

---

## How to Run

### Phase 1 and 2 — IDA, EDA, and Preprocessing

Open and run all cells in order in:

    notebooks/xray_full_pipeline.ipynb

Before running, update this line at the top of the notebook:

    DATASET_PATH = r"C:\Users\HP\Desktop\CSE881_ProjectData\NLMCXR_png"

This notebook will:
- Inventory all 7,470 NLMCXR images and check for corrupted files and duplicates
- Perform full EDA including class distribution, resolution analysis, and pixel intensity analysis
- Preprocess all images: grayscale, CLAHE, resize 224x224, 3-channel stack, normalize to 0-1, save as .npy
- Create a patient-level 70/15/15 train/val/test split
- Save preprocessed_metadata.csv to the working directory

Output files produced:
- preprocessed_metadata.csv
- preprocessing_stats.json
- image_inventory.csv
- All EDA plots saved as PNG files in the working directory

---

### Phase 3, 4, and 5 — Model Training, MongoDB, and Evaluation

Open and run all cells in order in:

    notebooks/xray_final_project.ipynb

Before running, update these variables at the top of the notebook:

    KAGGLE_PATH  = r"C:\Users\HP\Desktop\archive\chest_xray"
    NLMCXR_PATH  = r"C:\Users\HP\Desktop\CSE881_ProjectData\NLMCXR_png"
    METADATA_CSV = r"C:\Users\HP\Desktop\CSE881_ProjectData\preprocessed_metadata.csv"
    MONGO_URI    = "your_mongodb_atlas_connection_string_here"

Make sure preprocessed_metadata.csv exists from Phase 1 and 2 before running this notebook.

This notebook will:
- Run EDA on the Kaggle labelled dataset
- Train ResNet50 (ImageNet pretrained, transfer learning) on Kaggle Normal/Pneumonia images
- Evaluate the model on the Kaggle test set against real ground truth labels
- Generate predictions and confidence scores on all 7,467 NLMCXR images
- Insert all records into MongoDB Atlas
- Simulate 500 human annotations ordered by lowest confidence first
- Evaluate model performance and human-AI annotation agreement
- Save all output files and plots

Output files produced:
- best_resnet50_pneumonia.pth (trained model weights)
- nlmcxr_predictions_metadata.csv (all 7,467 predictions)
- evaluation_summary.json (all evaluation metrics)
- baseline_comparison.csv
- annotations_export.csv
- All result plots saved as PNG files

Note: Training on CPU takes approximately 1.5 to 2 hours for 10 epochs. A GPU is recommended for faster training.

---

## Key Results

Metric                               | Value
-------------------------------------|----------
ResNet50 Test Accuracy               | 0.8478
ResNet50 ROC AUC                     | 0.9268
F1 Score (weighted)                  | 0.8497
Majority Class Baseline Accuracy     | 0.6250
Improvement over Majority Baseline   | +22.3%
NORMAL Recall                        | 88.5% (207/234)
PNEUMONIA Recall                     | 82.6% (322/390)
NLMCXR Images Predicted              | 7,467
Mean AI Confidence                   | 0.8292
Auto-Accepted (confidence >= 0.70)   | 5,834 images (78.1%)
Routed to Human Review               | 1,633 images (21.9%)
Annotations Completed                | 500
Human-AI Agreement Rate              | 80%
Human Correction Rate                | 20%
Cohen's Kappa                        | 0.6002

---

## MongoDB Database Structure

Database name: xray_annotation_db

Collection     | Contents
---------------|--------------------------------------------------------------------------
images         | 7,467 records — filename, patient ID, AI prediction, confidence, status
annotations    | One record per human decision — AI label, human label, corrected, timestamp
users          | Annotator profiles and task completion counts
evaluation     | Final evaluation metrics
reports        | Parsed radiology reports with CheXpert-14 labels (text pipeline)

---

## Repository Structure

    cse881-xray-annotation-platform/
    │
    ├── README.md
    │
    ├── notebooks/
    │   ├── xray_full_pipeline.ipynb        # Phase 1+2: IDA, EDA, Preprocessing
    │   └── xray_final_project.ipynb        # Phase 3-5: Model Training, MongoDB, Evaluation
    │
    ├── outputs/
    │   ├── preprocessing_stats.json
    │   ├── evaluation_summary.json
    │   ├── baseline_comparison.csv
    │   ├── annotations_export.csv
    │   ├── preprocessed_metadata.csv
    │   └── nlmcxr_predictions_metadata.csv
    │
    ├── plots/
    │   ├── eda_class_distribution.png
    │   ├── eda_sample_images.png
    │   ├── eda_intensity_size.png
    │   ├── eda_resolution.png
    │   ├── training_history.png
    │   ├── evaluation_kaggle.png
    │   ├── baseline_comparison.png
    │   ├── nlmcxr_predictions.png
    │   ├── nlmcxr_sample_predictions.png
    │   ├── annotation_stats.png
    │   └── correction_by_confidence.png
    │
    └── docs/
        └── project_proposal.pdf

---

## Important Notes

- Raw image datasets (Kaggle and NLMCXR) are not included in this repository due to file size
- Download both datasets from the links provided above before running the notebooks
- The processed .npy image arrays are not included due to file size and are generated automatically by running xray_full_pipeline.ipynb
- Update all file paths at the top of each notebook to match your local machine before running
- MongoDB Atlas free tier M0 Sandbox is sufficient for this project

---

## Contact

For questions about this repository please contact the project team via MSU email.
