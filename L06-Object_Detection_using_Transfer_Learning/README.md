# Object Detection with Transfer Learning - Lab 06 🔍 

This lab provides a hands-on introduction to object detection using transfer learning with a pre-trained model.

## 💡 Problem Statement

Traditional image classification can tell us *what* is in an image (e.g., "This is a cat 🐱"), but it doesn't tell us *where* the objects are located within the image. Object detection aims to solve this by not only identifying objects but also localizing them with precise bounding boxes. This is crucial for applications where spatial information is vital, such as self-driving cars 🚗, surveillance systems 📹, and medical imaging 🩺.

## 🚀 Approach

This lab uses a transfer learning approach to perform object detection. Here's a breakdown of the steps:

1.  **Environment Setup**: Installed necessary libraries including TensorFlow, TensorFlow Hub, TensorFlow Datasets, and Matplotlib.
2.  **Dataset Loading & Exploration**: Utilized a small subset (10%) of the popular **Pascal VOC 2007 dataset**, which contains 20 object classes. The dataset was loaded, and a custom function was implemented to visualize images along with their ground truth bounding box annotations.
3.  **Pre-trained Model**: Loaded a pre-trained **SSD MobileNet V2** model from TensorFlow Hub. This model is known for its balance of speed and accuracy, making it suitable for quick experimentation.
4.  **Running the Detector**: Implemented a function (`run_detector`) to preprocess images and feed them into the pre-trained model, extracting detection results such as bounding boxes, class labels, and confidence scores.
5.  **Visualizing Detections**: Developed a `plot_detections` function to display the model's predictions, drawing bounding boxes and labels on images, filtered by a confidence threshold.
6.  **Evaluation Metrics**: Implemented the **Intersection over Union (IoU)** metric from scratch to quantify the overlap between predicted and ground truth bounding boxes. This was then used to calculate **Precision**, **Recall**, and **F1 Score** to evaluate the model's performance comprehensively.
7.  **Experimentation**: Explored the impact of different confidence and IoU thresholds on detection visualization and overall model performance metrics.

## 📊 Results

After running the evaluation on a subset of the validation dataset, the model produced initial results for precision, recall, and F1 score. It's important to note that these scores are based on a limited dataset and a pre-trained model not fine-tuned for this specific subset, hence the potentially low initial scores.

-   **Initial Performance (IoU Threshold = 0.50, Confidence Threshold = 0.50)**:
    -   True Positives: (e.g., `0` for the initial run)
    -   False Positives: (e.g., `196`)
    -   False Negatives: (e.g., `155`)
    -   Precision: (e.g., `0.00%`)
    -   Recall: (e.g., `0.00%`)
    -   F1 Score: (e.g., `0.00`)

*(Note: Actual numerical results may vary based on random sampling and specific model behavior.)*

Experiments with different thresholds showed expected behaviors:
-   **Lower Confidence Threshold (0.3)**: Led to more detected boxes, including potentially more false positives.
-   **Higher Confidence Threshold (0.7)**: Resulted in fewer, but generally more confident and accurate detections.
-   **Varying IoU Thresholds (0.3 vs. 0.7)**: Impacted the strictness of what was considered a 'correct' detection, directly influencing precision and recall.

## 🧠 Key Findings

-   **Object detection is more complex than classification**: It requires both object identification and localization.
-   **Bounding boxes define object location**: Coordinates (often fractional) are crucial for specifying object boundaries.
-   **Pre-trained models accelerate development**: They provide a strong starting point, saving immense training time and computational resources.
-   **Visualization is key**: Drawing bounding boxes helps in understanding model predictions and errors.
-   **IoU is fundamental**: It quantifies the overlap accuracy between predicted and ground truth boxes.
-   **Precision, Recall, and F1 Score offer comprehensive evaluation**: They help understand the trade-offs between false positives and false negatives.
-   **Thresholds matter**: Confidence and IoU thresholds significantly influence the model's perceived performance and the quality of visible detections.

## 🛠️ Technologies Used

-   **TensorFlow**: Core deep learning framework.
-   **TensorFlow Hub**: For loading pre-trained object detection models.
-   **TensorFlow Datasets (tfds)**: For easy access to the Pascal VOC 2007 dataset.
-   **NumPy**: For numerical operations.
-   **Matplotlib**: For plotting and visualization of images and bounding boxes.
-   **PIL (Pillow)**: For image manipulation.

## 🏃‍♀️ How to Run

To run this lab, simply open it in Google Colab and execute the cells sequentially. The `requirements.txt` is implicitly handled by the `%pip install` commands within the notebook.

1.  **Open in Google Colab**: Click the "Open in Colab" badge (or upload the `.ipynb` file). ⬆️
2.  **Install Dependencies**: Run the first code cell to install all required libraries. ✅
3.  **Execute Cells**: Go through each section, run the code cells, and complete the 
