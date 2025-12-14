# Chihuahua Or Muffin CNN - Lab 04 🤖

### Problem Statement 🧠
This project aims to build a neural network classifier to distinguish between images of muffins and chihuahuas. 🧁🐶

### Approach 🚀
The solution involves building a simple feed-forward neural network using PyTorch. The data is loaded using `torchvision.datasets.ImageFolder` and `torch.utils.data.DataLoader`, with image transformations applied to resize and normalize the images. The model is trained using `nn.CrossEntropyLoss` as the loss function and `optim.SGD` as the optimizer, iterating over multiple epochs to minimize the loss and improve accuracy.

### Results 📊
After 50 epochs of training, the model achieved a training accuracy of 97.50% and a validation accuracy of 96.67%. The validation loss converged to approximately 0.3628.

### Key Findings ✨
The simple neural network architecture is effective in classifying muffins and chihuahuas with high accuracy. The training process demonstrates how to set up a basic image classification pipeline in PyTorch.

### Technologies Used 🛠️
-   **PyTorch**: Deep learning framework
-   **torchvision**: Dataset management and image transformations
-   **matplotlib**: Plotting and visualization
-   **PIL (Pillow)**: Image handling
-   **tqdm**: Progress bars for training loops

### How to Run ▶️
1.  **Open in Google Colab**: Upload this notebook to Google Colab. ☁️
2.  **Run Cells**: Execute all code cells sequentially (Shift + Enter) to train the model and visualize the results. 🏃‍♀️

### Requirements/Dependencies 📦
This project primarily uses the following libraries:
-   `torch`
-   `torchvision`
-   `matplotlib`
-   `Pillow` (imported as `PIL`)
-   `tqdm`

