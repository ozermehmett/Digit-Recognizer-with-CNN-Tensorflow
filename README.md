# Digit-Recognizer-with-CNN-Tensorflow

This project implements a convolutional neural network (CNN) for recognizing handwritten digits using TensorFlow. The model is trained and tested on the MNIST dataset and achieved a rank of 70 on the Kaggle leaderboard for the digit recognition challenge.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Conclusion](#conclusion)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project demonstrates the use of a CNN to classify handwritten digits from the MNIST dataset. The MNIST dataset contains 60,000 training images and 10,000 test images of handwritten digits (0-9). Each image is 28x28 pixels.

## Dataset

The dataset used in this project is the MNIST dataset, which is a standard benchmark dataset in the field of machine learning and image processing.

## Model Architecture

The CNN model is built using the TensorFlow library and consists of the following layers:

- **Input Layer**: Accepts input images of shape (28, 28, 1).
- **Convolutional Layers**: 
  - 64 filters of size 3x3 with ReLU activation, followed by Batch Normalization and Dropout.
  - 64 filters of size 3x3 with ReLU activation, followed by Batch Normalization and Max Pooling.
  - 128 filters of size 3x3 with ReLU activation, followed by Batch Normalization and Dropout.
  - 128 filters of size 3x3 with ReLU activation, followed by Batch Normalization and Max Pooling.
  - 256 filters of size 3x3 with ReLU activation, followed by Batch Normalization and Dropout, and Max Pooling.

- **Flatten Layer**: Flattens the 2D matrix into a vector.
- **Dense Layers**: 
  - Fully connected layers with ReLU activation, followed by Batch Normalization and Dropout.

- **Output Layer**: Fully connected layer with 10 neurons (one for each digit) and softmax activation.

## Training

The model is trained using the following configurations:

- **Optimizer**: Adam with a learning rate of 0.0001
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 200
- **Batch Size**: 64

Early stopping and learning rate reduction on plateau callbacks are applied during training.

## Evaluation

The model is evaluated on the test set using accuracy as the primary metric. Additionally, a confusion matrix is generated to analyze the performance of the model across different classes.

## Results

The model achieved a high accuracy on the test set, demonstrating its effectiveness in recognizing handwritten digits. Below are some sample predictions from the test set:

![__results__numbers](https://github.com/ozermehmett/Digit-Recognizer-with-CNN-Tensorflow/assets/115498182/ae06013c-354b-4c7d-ba02-5b6c8d6454f6)

The training and validation loss/accuracy over epochs is visualized below:

![__results__acc](https://github.com/ozermehmett/Digit-Recognizer-with-CNN-Tensorflow/assets/115498182/99ee5fac-4171-4b22-b553-18e2a71d1445)

## Conclusion

This project showcases the power of CNNs in image classification tasks. The model performs exceptionally well on the MNIST dataset, making it a strong candidate for similar image recognition tasks.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/ozermehmett/Digit-Recognizer-with-CNN-Tensorflow.git
    cd Digit-Recognizer-with-CNN-Tensorflow
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Download the MNIST dataset (automatically handled by the code):
    ```python
    from tensorflow.keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    ```

## Usage

To train and evaluate the model, simply run the Jupyter notebook:

```bash
jupyter notebook digit-recognizer-with-cnn-tensorflow.ipynb
```

Follow the steps in the notebook to train the model and make predictions on the test set.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
