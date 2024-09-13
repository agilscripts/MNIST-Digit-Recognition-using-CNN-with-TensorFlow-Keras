# MNIST-Digit-Recognition-using-CNN-with-TensorFlow-Keras
This project demonstrates the implementation of a Convolutional Neural Network (CNN) for handwritten digit recognition using the MNIST dataset. The model is built using TensorFlow and Keras, achieving high accuracy through data preparation, augmentation, and a well-structured neural network architecture.


### **README.md:**

```markdown
# MNIST Digit Recognition using CNN with TensorFlow & Keras

## Project Overview
This project involves building a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. The dataset consists of 28x28 grayscale images of digits (0-9), and the goal is to correctly predict the digit represented by each image. This project was developed in Jupyter Notebook using Python, TensorFlow, and Keras.

## Dataset
- **MNIST dataset**: The dataset contains 60,000 training images and 10,000 test images of handwritten digits (0-9). Each image is 28x28 pixels and is labeled with the corresponding digit.
- Train data is pre-processed by normalizing and reshaping it to a 3D format suitable for a CNN.
- Labels are one-hot encoded.

## Project Structure
1. **Data Preparation**:
   - Load and preprocess the MNIST dataset.
   - Normalize pixel values to the [0,1] range.
   - Reshape images into 28x28x1 format.
   - Split the training data into training and validation sets.
   
2. **Model Architecture**:
   - 5-layer CNN using the Keras Sequential API.
   - Layers include convolutional, max pooling, dropout, and dense layers.
   - Regularization using dropout to prevent overfitting.

3. **Model Training**:
   - The model is compiled using RMSprop optimizer and categorical cross-entropy loss.
   - The training runs for 10 epochs with a batch size of 86.
   - Validation accuracy is monitored to adjust learning rate dynamically.

4. **Evaluation**:
   - Accuracy and loss curves are plotted for both training and validation data.
   - Confusion matrix and error analysis help identify areas of improvement.

## Key Results
- Achieved over 98% accuracy on the validation set after training for 10 epochs.
- The CNN model performs robustly on the MNIST dataset, demonstrating strong generalization.

## Requirements
- Python 3.x
- TensorFlow 2.x
- Keras (integrated in TensorFlow)
- Jupyter Notebook
- Required Python libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn

To install the required libraries, run:

```bash
pip install -r requirements.txt
```

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/mnist-digit-recognition.git
   ```
2. Navigate to the project directory:
   ```bash
   cd mnist-digit-recognition
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Jupyter notebook:
   ```bash
   jupyter notebook
   ```
5. Open the notebook file and run all cells to train the CNN model.

## Future Improvements
- Implement data augmentation to further improve accuracy.
- Experiment with different CNN architectures or optimizers.
- Fine-tune hyperparameters such as learning rate, batch size, and number of epochs.

## License
This project is licensed under the MIT License.

## Acknowledgements
- **Keras & TensorFlow**: For providing a powerful and user-friendly deep learning framework.
- **MNIST**: For the open dataset used in this project.
```
