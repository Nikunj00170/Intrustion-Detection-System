# Network Intrusion Detection System (IDS)

A machine learning-based intrusion detection system that identifies malicious network activities and distinguishes them from normal traffic using the UNSW-NB15 dataset.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-FF6F00.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Project Overview

This project implements and compares multiple machine learning algorithms to detect network intrusions with high accuracy. The system can classify network packets as either normal traffic or potential attacks, making it valuable for cybersecurity applications.

## 🚀 Key Features

- **Multi-Algorithm Comparison**: Implements 7 ML algorithms including ensemble methods and dimensionality reduction
- **Binary & Multi-class Classification**: Detects both attack vs. normal and specific attack types
- **Feature Engineering**: Robust preprocessing with PCA dimensionality reduction
- **High Accuracy**: Achieves 87%+ test accuracy with ensemble methods
- **Visualization**: Comprehensive confusion matrices and performance metrics

## 📊 Dataset

- **Source**: UNSW-NB15 Dataset
- **Training Samples**: 175,341
- **Testing Samples**: 82,332
- **Features**: 44 network traffic characteristics
- **Classes**: Normal traffic + 9 attack categories (Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms)

## 🛠️ Technologies Used

```
Python 3.10+
├── Data Processing
│   ├── NumPy
│   ├── Pandas
│   └── Scikit-learn
├── Visualization
│   ├── Matplotlib
│   └── Seaborn
├── Machine Learning
│   ├── Scikit-learn (Classical ML)
│   ├── XGBoost (Gradient Boosting)
│   └── TensorFlow/Keras (Deep Learning)
└── Development
    └── Jupyter Notebook
```

## 📈 Model Performance

| Model                    | Training Accuracy | Test Accuracy | Precision | Recall |
| ------------------------ | ----------------- | ------------- | --------- | ------ |
| Random Forest            | 99.82%            | 87.15%        | 81.85%    | 98.49% |
| Decision Tree            | 99.82%            | 86.42%        | 82.50%    | 95.62% |
| PCA Random Forest        | 99.82%            | 85.59%        | 80.52%    | 97.39% |
| K-Nearest Neighbors      | 94.91%            | 85.70%        | 81.23%    | 96.28% |
| SVM (Linear)             | 90.43%            | 81.66%        | 77.20%    | 94.65% |
| Logistic Regression      | 88.09%            | 74.12%        | 69.25%    | 95.34% |
| Naive Bayes              | 75.30%            | 73.97%        | 80.12%    | 70.12% |

_Random Forest demonstrated the best overall performance with 87.15% test accuracy and excellent recall (98.49%)._

## 🔧 Installation

### Prerequisites

```bash
Python 3.10 or higher
pip package manager
```

### Clone Repository

```bash
git clone https://github.com/Nikunj00170/Intrustion-Detection-System.git
cd Intrustion-Detection-System
```

### Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow xgboost jupyter
```

Or use requirements.txt:

```bash
pip install -r requirements.txt
```

## 💻 Usage

### Quick Start

1. **Open Jupyter Notebook**:

```bash
jupyter notebook ML-project-UPDATED.ipynb
```

2. **Add Dataset Files**:
   Place your UNSW-NB15 dataset files in the `data/unsw-nb15/` directory:

```bash
cp UNSW_NB15_training-set.csv data/unsw-nb15/
cp UNSW_NB15_testing-set.csv data/unsw-nb15/
```

Then update the file paths in the notebook:

```python
data_train = pd.read_csv('data/unsw-nb15/UNSW_NB15_training-set.csv')
data_test = pd.read_csv('data/unsw-nb15/UNSW_NB15_testing-set.csv')
```

3. **Run All Cells**:
   Execute the notebook cells sequentially to:
   - Load and explore data
   - Preprocess features
   - Train multiple models
   - Compare performance metrics

### Example: Using a Trained Model

```python
import joblib
import numpy as np

# Load trained model (after training)
model = joblib.load('models/random_forest_ids.pkl')

# Predict on new network packet data
network_packet = np.array([[...]])  # Your feature vector
prediction = model.predict(network_packet)

print("Attack" if prediction[0] == 1 else "Normal")
```

## 🔬 Methodology

### 1. Data Preprocessing

- Handle missing values
- Encode categorical features (protocol, service, state)
- Apply RobustScaler for numerical feature normalization
- Binary label encoding (normal=0, attack=1)

### 2. Feature Engineering

- Extract 44 network traffic features
- Apply Principal Component Analysis (PCA) for dimensionality reduction
- Reduce from 194 features to 20 principal components

### 3. Model Training

Train and evaluate 7 different algorithms:

- **Classical ML**: Logistic Regression, KNN, Naive Bayes, SVM
- **Tree-based**: Decision Tree, Random Forest (including PCA-reduced variant)

### 4. Evaluation Metrics

- Accuracy, Precision, Recall
- Confusion Matrix
- Feature Importance Analysis

## 📁 Project Structure

```
Intrustion-Detection-System/
│
├── ML-project-UPDATED.ipynb           # Main Jupyter notebook
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── data/                              # Dataset directory
│   ├── unsw-nb15/                    # UNSW-NB15 dataset
│   │   ├── Training and Testing Sets/
│   │   │   ├── UNSW_NB15_training-set.csv
│   │   │   └── UNSW_NB15_testing-set.csv
│   │   └── (other UNSW-NB15 files)
│   │
│   └── raw-datasets/                 # Alternative datasets
│       └── nsl-kdd/                  # NSL-KDD dataset (optional)
│
├── models/                            # Saved models (generated)
│   ├── random_forest_ids.pkl
│   └── xgboost_ids.pkl
│
└── visualizations/                    # Generated plots
    ├── confusion_matrices/
    └── feature_importance/
```

## 🧪 Key Findings

1. **Ensemble Methods Win**: Random Forest and XGBoost significantly outperform other algorithms
2. **PCA Impact**: Dimensionality reduction maintains 85%+ accuracy while reducing computation time
3. **Real-time Viability**: Linear SVM and Neural Networks offer good speed-accuracy tradeoffs for production

## 🔮 Future Enhancements

- [ ] Implement real-time network traffic monitoring dashboard
- [ ] Add LSTM/CNN architectures for temporal pattern recognition
- [ ] Integrate with SIEM systems for automated threat response
- [ ] Implement online learning for adaptive threat detection
- [ ] Add explainability features (SHAP/LIME) for model interpretability
- [ ] Develop API for model deployment
- [ ] Add support for zero-day attack detection

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Author

- Nikunj Sharma ([@nikunj](https://github.com/Nikunj00170))

## 🙏 Acknowledgments

- **Dataset**: UNSW-NB15 by Moustafa & Slay (2015)
- **Libraries**: Scikit-learn, TensorFlow, XGBoost communities

## 📚 References

1. Moustafa, N., & Slay, J. (2015). "UNSW-NB15: A Comprehensive Dataset for Network Intrusion Detection Systems"
2. Revathi, S., & Malathi, A. (2013). "A Detailed Analysis on NSL-KDD Dataset Using Various Machine Learning Techniques"
3. [Scikit-learn Documentation](https://scikit-learn.org/)
4. [XGBoost Documentation](https://xgboost.readthedocs.io/)
5. [TensorFlow Documentation](https://www.tensorflow.org/)

## 📧 Contact

For questions or feedback, please open an issue or contact the team members directly.

---

⭐ **Star this repository if you find it helpful!**
