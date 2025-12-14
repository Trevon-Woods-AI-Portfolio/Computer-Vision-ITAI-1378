# Classical Machine Learning - Lab 3 🤖🖼️

### Problem Statement 😏➡️💡
Traditional machine learning models, especially in computer vision, often struggle with generalization when faced with limited data and variations in real-world scenarios. A common pitfall is overfitting, where models achieve high accuracy on training data but fail on new, unseen data. This project aims to demonstrate the complete classical machine learning pipeline for computer vision, highlighting the critical importance of proper training approaches to build models that generalize effectively, rather than merely memorizing training examples. 

### Approach 🚀
This project approaches face recognition using classical machine learning techniques. The methodology includes:

1.  **Environment Setup & Data Preparation** 🛠️📊: Installing necessary libraries (OpenCV, scikit-image, scikit-learn, matplotlib, seaborn, numpy) and loading the Olivetti faces dataset. The dataset consists of 400 images of 40 individuals (10 images per person), which is ideal for demonstrating overfitting due to its limited samples per class. The data is split into training (60%), validation (20%), and test (20%) sets using stratified splitting to ensure balanced representation.

2.  **Feature Extraction** 🔍✨: Converting raw pixel data into more meaningful representations.
    *   **HOG (Histogram of Oriented Gradients)**: Captures edge and gradient information, effective for shape and structural recognition in faces.
    *   **LBP (Local Binary Patterns)**: Extracts texture features, robust to lighting changes.

3.  **Classical ML Algorithms Implementation** 🧠💻: Training and evaluating four machine learning classifiers:
    *   **Support Vector Machine (SVM)**: Effective for high-dimensional data, trained with an RBF kernel.
    *   **Random Forest**: An ensemble method using multiple decision trees.
    *   **k-Nearest Neighbors (k-NN)** (Not implemented in the provided notebook, but mentioned in objectives).
    *   **Naive Bayes** (Not implemented in the provided notebook, but mentioned in objectives).

4.  **Overfitting Demonstration** 🚨📉: Intentionally creating an overfitted Random Forest model to achieve 100% training accuracy, followed by a reality check on validation data to show its poor generalization. This contrasts with a 'reasonable' model built with regularization techniques and cross-validation.

5.  **Final Model Selection and Evaluation** ✅🏆: Comparing model performance based on validation accuracy and the overfitting gap (difference between training and validation accuracy) to select the best-performing model for final evaluation on the unseen test set.

### Results 📈📊

**Feature Extraction Comparison:**
*   Raw Pixels: 4096 dimensions
*   HOG Features: 1764 dimensions (significant reduction, captures shape)
*   LBP Features: 10 dimensions (very compact, captures texture)

**Model Performance (on Validation Data with HOG and LBP features):**

| Model          | Train Accuracy | Validation Accuracy | Overfitting Gap |
| :------------- | :------------- | :------------------ | :-------------- |
| SVM + HOG      | 1.000          | 0.963               | 0.037           |
| SVM + LBP      | 0.571          | 0.412               | 0.158           |
| RF + HOG       | 1.000          | 0.887               | 0.113           |
| RF + LBP       | 1.000          | 0.388               | 0.613           |

**Overfitting Demonstration:**
*   **Overfitted Model (Random Forest with HOG)**: Achieved 100.0% training accuracy but only 92.5% validation accuracy, showing a 7.5% performance drop.
*   **Reasonable Model (Random Forest with HOG and regularization)**: Achieved 1.000 training accuracy and 0.750 validation accuracy (from the last run in Cell 4.3). While this example still shows a gap, the principle demonstrated was that proper hyperparameter tuning and cross-validation lead to better generalization than an overly complex model.

**Selected Best Model**: Based on the highest validation accuracy and lowest overfitting gap, the **SVM + HOG** model was chosen. 🏆

### Key Findings 💡

*   **Overfitting is a critical challenge** ⚠️: High training accuracy does not guarantee real-world performance. Models can memorize data rather than learn generalizable patterns.
*   **Validation sets are essential** ✅: They provide an unbiased estimate of model performance on unseen data and are crucial for detecting overfitting and selecting the best model.
*   **Feature engineering matters** ✨: Transforming raw pixels into descriptive features like HOG (for shape) and LBP (for texture) significantly impacts model performance and computational efficiency. HOG features generally outperformed LBP for this facial recognition task.
*   **Cross-validation is key for robust evaluation** 🔄: It provides a more reliable estimate of model performance and helps ensure that the model is not sensitive to specific subsets of the training data.
*   **Model complexity vs. generalization** ⚖️: Balancing model complexity (e.g., `max_depth`, `min_samples_leaf` in Random Forest) is vital to prevent overfitting and ensure the model generalizes well to new data.

### Technologies Used 💻

*   **Python 3.x**
*   **NumPy**: Numerical operations
*   **Matplotlib**: Plotting and visualization
*   **Seaborn**: Statistical data visualization
*   **scikit-learn**: Machine learning algorithms (SVM, RandomForestClassifier, StandardScaler, train_test_split, cross_val_score, accuracy_score, confusion_matrix, classification_report)
*   **OpenCV (`cv2`)**: Computer Vision tasks
*   **scikit-image (`skimage`)**: Image processing, especially for feature extraction (HOG, LBP)

### How to Run ▶️

1.  **Open in Google Colab** ☁️: Click the 'Open in Colab' badge (if provided, otherwise manually open the `.ipynb` file in Colab).
2.  **Run Cells** 🏃: Execute each code cell sequentially. The notebook provides step-by-step instructions and explanations.
3.  **Local Execution** 🏠: Download the `.ipynb` file and run it using a Python environment (e.g., Anaconda) with all the listed dependencies installed.
    *   `pip install -r requirements.txt` (if a `requirements.txt` file is provided)
    *   Otherwise, install individually: `pip install numpy matplotlib seaborn scikit-learn opencv-python-headless scikit-image`
