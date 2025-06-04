# assignment2
2D Convolution with Custom Kernel using TensorFlow/Keras

This project demonstrates the application of a custom 3×3 kernel (Laplacian filter) on a 5×5 input matrix using TensorFlow's `Conv2D` layer. It shows the impact of different stride and padding combinations on the output feature maps.

---

 Input Matrix

```plaintext
[[ 1,  2,  3,  4,  5],
 [ 6,  7,  8,  9, 10],
 [11, 12, 13, 14, 15],
 [16, 17, 18, 19, 20],
 [21, 22, 23, 24, 25]]
-----------------------------------------------------------------------------------------------

# Convolution on Custom Matrix using TensorFlow/Keras


This project demonstrates how to apply a 2D convolution operation using a custom kernel on a manually created input matrix in TensorFlow/Keras. It explores different stride and padding settings and visualizes the output feature maps.

 Objective

- Apply a custom 3x3 kernel (Laplacian filter) to a 5x5 input matrix.
- Use TensorFlow/Keras `Conv2D` layer without bias.
- Observe the effect of different stride and padding values:
  - Stride = 1, Padding = 'VALID'
  - Stride = 1, Padding = 'SAME'
  - Stride = 2, Padding = 'VALID'
  - Stride = 2, Padding = 'SAME'

Concepts Covered

- 2D Convolution operation
- Stride and Padding in CNNs
- Manual input and kernel definition
- Keras Sequential model for simple convolution tasks

 Dependencies

- Python 3.x
- NumPy
- TensorFlow 2.x

Install requirements using pip:

```bash
pip install tensorflow numpy
------------------------------------------------------------------------
Sobel Edge Detection with OpenCV and NumPy

This project demonstrates how to apply **Sobel filters** to detect horizontal and vertical edges in a grayscale image using OpenCV's `filter2D` function. The results are visualized using Matplotlib.

---
 Objective

- Load and process a grayscale image.
- Apply **Sobel X** and **Sobel Y** filters to extract edge information.
- Display the original and edge-detected images side by side.

---
 Filters Used

 Sobel X Filter (detects vertical edges)

```plaintext
[[-1,  0,  1],
 [-2,  0,  2],
 [-1,  0,  1]]
----------------------------------------------------------------------------------------------------------------------
 Sobel Edge Detection using OpenCV and NumPy

This project demonstrates how to apply **Sobel X** and **Sobel Y** filters for edge detection on a grayscale image. The input image is automatically downloaded from a URL (Lena image), processed with custom convolution filters, and displayed using Matplotlib.

---

 Sample Image

The script downloads the popular **Lena** test image from OpenCV's GitHub repository:
-------------------------------------------------------------------------------------------------------------------
 Max Pooling vs. Average Pooling in TensorFlow

This project demonstrates the difference between **MaxPooling** and **AveragePooling** operations using TensorFlow and a randomly generated 4×4 input matrix.

---

 Objective

- Generate a random 4×4 matrix as input.
- Apply **2×2 MaxPooling** and **2×2 AveragePooling**.
- Observe how pooling layers reduce spatial dimensions and extract key features.

---

 Pooling Explained

- **Max Pooling** selects the maximum value from each 2×2 region.
- **Average Pooling** computes the average of each 2×2 region.
- Both operations reduce dimensionality while retaining important information.

---

 Requirements

- Python 3.x
- TensorFlow ≥ 2.x
- NumPy

Install using pip:
```bash
pip install tensorflow numpy
-------------------------------------------------------------------------------------------
# 🌸 Iris Dataset: Normalization vs. Standardization

This project demonstrates how **Min-Max Normalization** and **Z-score Standardization** affect feature distribution and model performance on the Iris dataset using a Logistic Regression classifier.

---

 Objectives

1. Load and explore the Iris dataset.
2. Apply **Min-Max Normalization** and **Z-score Standardization**.
3. Visualize distributions of transformed features.
4. Train and evaluate a **Logistic Regression** model using:
   - Original data
   - Min-Max Normalized data
   - Z-score Standardized data
5. Analyze when to use normalization vs. standardization in machine learning.

---

 Requirements
- Python 3.x
- pandas
- numpy
- matplotlib
- scikit-learn

Install dependencies with:

```bash
pip install pandas numpy matplotlib scikit-learn





