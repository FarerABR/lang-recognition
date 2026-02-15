# Language Recognition from Audio

A comprehensive machine learning project for automatic language recognition from audio samples using both supervised and unsupervised learning techniques.

## Overview

This project implements a complete ML pipeline to classify spoken language from audio files. It uses audio feature extraction (librosa) combined with multiple classification and clustering algorithms to achieve high accuracy in language recognition.

**Languages Supported:** German, Italian, Korean, Spanish

**Dataset:** 720 audio samples (MP3 format)
- 4 languages × 2 genders (male/female) × 90 samples per category
- Audio samples from multiple speakers per language/gender combination

## Project Structure

```
lang-recognition/
├── Data_Cleaning_and_Feature_Extraction.ipynb  # Stage 1: Data preprocessing & feature extraction
├── Clustering.ipynb                            # Stage 2: Unsupervised learning analysis
├── Classification.ipynb                        # Stage 3: Supervised classification models
├── Evaluation.ipynb                            # Stage 4: Results analysis & visualization
├── requirements.txt                            # Python dependencies
├── LICENSE                                     # MIT License
├── README.md                                   # This file
├── raw/                                        # Original audio files
│   ├── German/
│   │   ├── Female/
│   │   └── Male/
│   ├── Italian/
│   │   ├── Female/
│   │   └── Male/
│   ├── Korean/
│   │   ├── Female/
│   │   └── Male/
│   └── Spanish/
│       ├── Female/
│       └── Male/
├── data/                                       # Processed datasets
│   ├── augmented/                              # Augmented training data
│   ├── no_augmentation/                        # Original training data
│   ├── metadata.csv                            # Audio file metadata
│   └── feature_names.npy                       # Feature column names
├── outputs/                                    # Clustering analysis outputs
│   └── outputs_YYYYMMDD_HHMMSS/
│       ├── step1_raw_and_optimized_data/
│       ├── step2_kmeans_compare/
│       ├── step3_dbscan_compare/
│       └── step4_optics_compare/
├── classifier_result/                          # Classification results
│   ├── metrics.csv                             # Model performance metrics
│   ├── confusion_matrices.npy                  # Confusion matrices for all models
│   └── confusion_matrices.png                  # Visualized confusion matrices
└── Report/                                     # Project documentation & visualizations
```

## Workflow Pipeline

The project follows a **4-stage pipeline**:

### Stage 1: Data Cleaning & Feature Extraction
**Notebook:** `Data_Cleaning_and_Feature_Extraction.ipynb`

- Loads 720 MP3 audio files
- Validates audio quality and duration
- Extracts **86 acoustic features** using librosa:
  - MFCCs (Mel-frequency cepstral coefficients)
  - Spectral features (centroid, bandwidth, contrast, rolloff)
  - Chroma features (STFT and CQT)
  - Zero-crossing rate
  - Tempogram and tempo
  - RMS energy
- Splits data: 80% training (576 samples) / 20% testing (144 samples)
- Applies data augmentation (time-stretching, pitch-shifting) to training data
- Normalizes features using StandardScaler
- Saves processed datasets to `data/` directory

### Stage 2: Clustering Analysis
**Notebook:** `Clustering.ipynb`

Performs comprehensive unsupervised learning analysis with multiple clustering algorithms:

- **Step 1:** PCA optimization (dimensionality reduction to 90% variance)
- **Step 2:** K-Means clustering (tests k=2-9, optimal selection via silhouette score)
- **Step 3:** DBSCAN (density-based clustering with epsilon tuning)
- **Step 4:** OPTICS (ordering points clustering with reachability plots)

Compares performance on:
- Raw feature space vs. PCA-optimized space
- Augmented vs. non-augmented datasets

Outputs silhouette scores, purity metrics, cluster visualizations, and PCA projections.

### Stage 3: Classification
**Notebook:** `Classification.ipynb`

Trains and evaluates **7 supervised learning models**:

1. **Logistic Regression** - 100% accuracy ✓
2. **MLP Neural Network** - 100% accuracy ✓
3. **Random Forest** - 100% accuracy ✓
4. **SVM (RBF kernel)** - 99.3% accuracy
5. **K-Nearest Neighbors** - 98.6% accuracy
6. **Decision Tree** - 95.1% accuracy
7. **Naïve Bayes** - 82.6% accuracy

All models trained on augmented dataset with standardized features.

### Stage 4: Evaluation
**Notebook:** `Evaluation.ipynb`

- Analyzes classification metrics (accuracy, precision, recall, F1-score)
- Compares clustering algorithm performance
- Generates confusion matrices
- Displays purity scores and cluster quality metrics
- Provides comprehensive performance comparison across all methods

## Results Summary

### Classification Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 100% | 100% | 100% | 100% |
| MLP Neural Network | 100% | 100% | 100% | 100% |
| Random Forest | 100% | 100% | 100% | 100% |
| SVM (RBF) | 99.3% | 99.3% | 99.3% | 99.3% |
| KNN | 98.6% | 98.7% | 98.6% | 98.6% |
| Decision Tree | 95.1% | 95.2% | 95.1% | 95.2% |
| Naïve Bayes | 82.6% | 86.1% | 82.6% | 82.1% |

**Key Findings:**
- Three models (Logistic Regression, MLP, Random Forest) achieve **perfect 100% accuracy**
- All top models demonstrate excellent generalization on test data
- Audio features extracted by librosa provide highly discriminative information for language classification

## Requirements

```bash
# Install dependencies
pip install -r requirements.txt
```

**Key Dependencies:**
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `librosa` - Audio analysis and feature extraction
- `soundfile` - Audio file I/O
- `scikit-learn` - Machine learning algorithms
- `matplotlib`, `seaborn` - Visualization
- `jupyter` - Notebook environment

## Usage

1. **Clone the repository:**
```bash
git clone https://github.com/FarerABR/lang-recognition.git
cd lang-recognition
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the notebooks in order:**
   - Start with `Data_Cleaning_and_Feature_Extraction.ipynb`
   - Continue with `Clustering.ipynb` (optional, for unsupervised analysis)
   - Run `Classification.ipynb` for supervised model training
   - View results in `Evaluation.ipynb`

4. **Explore results:**
   - Classification metrics: `classifier_result/metrics.csv`
   - Confusion matrices: `classifier_result/confusion_matrices.png`
   - Clustering analysis: `outputs/` directory

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Audio features extracted using [librosa](https://librosa.org/) library.
Machine learning models implemented with [scikit-learn](https://scikit-learn.org/).
