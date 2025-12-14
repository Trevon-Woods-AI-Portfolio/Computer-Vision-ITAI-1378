# Chihuahua or Muffin Classifier using CNN 🐶🧁

## 📝 Problem Statement

This project aims to accurately classify images as either a "Chihuahua" or a "Muffin" using a Convolutional Neural Network (CNN). It builds upon previous work that utilized traditional Neural Networks, demonstrating the enhanced capabilities of CNNs for image classification tasks.

## ✨ Approach

The solution involves building, training, and evaluating a CNN model in PyTorch. The key steps are:

1.  **Data Preparation**: 📚
    *   The dataset, pre-divided into `train` and `validation` sets, consists of Chihuahua and Muffin images.
    *   **Data Transformations**: Images are resized to 128x128 pixels. The training set undergoes data augmentation (random horizontal flips and rotations) to improve generalization. Both sets are converted to PyTorch tensors and normalized using ImageNet's mean and standard deviation.
    *   **Dataloaders**: `DataLoader` objects are created to efficiently batch and shuffle the data for training and validation.

2.  **Model Definition**: 🧠
    *   A custom `ChihuahuaMuffinCNN` model is defined using PyTorch's `nn.Module`. 
    *   The model architecture includes several sequential blocks of `Conv2d` (Convolutional Layer), `ReLU` (Activation Function), and `MaxPool2d` (Pooling Layer) to extract features.
    *   These convolutional features are then flattened and passed through `Linear` (Fully Connected) layers for classification, with a `Dropout` layer for regularization.

3.  **Training Setup**: ⚙️
    *   **Loss Function**: `nn.CrossEntropyLoss` is used, suitable for multi-class classification.
    *   **Optimizer**: The `Adam` optimizer with a learning rate of 0.001 is chosen to update model parameters.

4.  **Model Training**: 🚀
    *   The model is trained for 30 epochs. Each epoch involves training on the training dataset and evaluating on the validation dataset.
    *   Loss and accuracy metrics are tracked and printed for both phases.

## 📈 Results

After training for 30 epochs, the model achieved impressive performance:

*   **Final Training Loss**: ~0.0059
*   **Final Training Accuracy**: 100.00%
*   **Final Validation Loss**: ~0.0215
*   **Final Validation Accuracy**: 100.00%

The CNN model demonstrated superior performance compared to a traditional Neural Network (from a previous workshop, not detailed here) for this image classification task, successfully distinguishing between Chihuahuas and muffins with high accuracy on the validation set.

## 💡 Key Findings

*   **CNN Effectiveness**: Convolutional Neural Networks are highly effective for image classification, capable of learning complex spatial features directly from raw pixel data.
*   **Data Augmentation**: Techniques like random flips and rotations significantly enhance the model's ability to generalize to unseen data by introducing variability in the training set.
*   **Overfitting**: While a 100% validation accuracy is excellent, it also highlights the potential for overfitting on small datasets. More diverse and larger datasets would be beneficial for real-world robustness.

## 🛠️ Technologies Used

*   **Python** 🐍
*   **PyTorch**: Deep learning framework
*   **torchvision**: Image dataset and transformation utilities
*   **NumPy**: Numerical operations
*   **Matplotlib**: Plotting and visualization
*   **tqdm**: Progress bars
*   **torchsummary**: Model summary visualization

## 🚀 How to Run

1.  **Open in Google Colab**: Click the "Open in Colab" badge (if available) or upload the `.ipynb` file to Google Colab.
2.  **Execute Cells**: Run all code cells sequentially from top to bottom. Ensure you have access to the `/content/drive/MyDrive/muffin_chihuahua_data` directory, which should contain the `train` and `validation` subdirectories with image data.

## 📦 Requirements/Dependencies

The main libraries required are:

*   `torch`
*   `torchvision`
*   `numpy`
*   `matplotlib`
*   `tqdm`
*   `torchsummary` (can be installed via `!pip install torchsummary`)

These can be installed using pip if running locally:

```bash
pip install torch torchvision numpy matplotlib tqdm torchsummary
```

```
