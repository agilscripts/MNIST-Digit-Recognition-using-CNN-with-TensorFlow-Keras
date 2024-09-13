"# MNIST-Digit-Recognition-using-CNN-with-TensorFlow-Keras
This project demonstrates the implementation of a Convolutional Neural Network (CNN) for handwritten digit recognition using the MNIST dataset. The model is built using TensorFlow and Keras, achieving high accuracy through data preparation, augmentation, and a well-structured neural network architecture.


### **README.md:**

```markdown
# MNIST Digit Recognition using CNN with TensorFlow & Keras

## Project Overview
This project involves building a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. The dataset consists of 28x28 grayscale images of digits (0-9), and the goal is to correctly predict the digit represented by each image. This project was developed in Jupyter Notebook using Python, TensorFlow, and Keras.

## Dataset
- **MNIST dataset**: The dataset contains 60,000 training images and 10,000 test images of handwritten digits (0-9). Each image is 28x28 pixels and is labeled with the corresponding digit.
- The train data is pre-processed by normalizing and reshaping it to a 3D format suitable for a CNN.
- Labels are one-hot encoded for multi-class classification.

## Project Structure
1. **Data Preparation**:
   - Load and preprocess the MNIST dataset.
   - Normalize pixel values to the [0,1] range to ensure faster convergence.
   - Reshape images into 28x28x1 format to fit the input shape for CNNs.
   - Split the training data into training and validation sets.
   
2. **Model Architecture**:
   - The CNN consists of multiple layers:
     - Two 32-filter Conv2D layers with ReLU activation.
     - Max Pooling and Dropout for down-sampling and regularization.
     - Two additional Conv2D layers with 64 filters.
     - Dense layers at the end for final classification.
   - Dropout layers are included to prevent overfitting and improve generalization.
   
3. **Model Training**:
   - The model is compiled using the **RMSprop optimizer** and **categorical cross-entropy loss** function.
   - Trained for 10 epochs with a batch size of 86.
   - Validation accuracy is tracked to dynamically adjust the learning rate using ReduceLROnPlateau.

4. **Evaluation**:
   - **Training & Validation Performance**: The model achieved over 98% validation accuracy.
   - **Accuracy and Loss Curves**: Plotted to observe the convergence of the model over epochs.
   - **Confusion Matrix**: Analyzed the misclassifications to better understand where the model struggled.
   - **Misclassification Analysis**: Identified where the model confused similar digits, such as 4 and 9, 1 and 8.

### Results
- **Final Validation Accuracy**: ~99.4%
- **Model Summary**: A 5-layer CNN that efficiently recognizes digits with high accuracy.
- **Training Performance**:
  - Achieved accuracy of 99% after 10 epochs.
  - Loss and accuracy curves show steady improvement without significant overfitting.

### Misclassifications
- Example misclassifications:
  - Predicted: 8, Actual: 1
  - Predicted: 4, Actual: 9
  - Predicted: 8, Actual: 3
- These misclassifications indicate areas where digits may overlap in appearance due to image quality or stroke style.

### Key Graphs

- **Training & Validation Accuracy and Loss**:
  ![Accuracy and Loss](path/to/your/accuracy-loss-graph.png)

- **Confusion Matrix and Misclassifications**:
  Visual analysis of model misclassifications.

## Requirements
- Python 3.x
- TensorFlow 2.x
- Keras (integrated with TensorFlow)
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
   git clone https://github.com/agilscripts/MNIST-Digit-Recognition-using-CNN-with-TensorFlow-Keras.git
   ```
2. Navigate to the project directory:
   ```bash
   cd MNIST-Digit-Recognition-using-CNN-with-TensorFlow-Keras
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Jupyter notebook:
   ```bash
   jupyter notebook
   ```
5. Open the notebook file `NumRecognizer.ipynb` and run all cells to train the CNN model.

## Future Improvements
- **Data Augmentation**: Applying data augmentation techniques like rotation, zoom, or shifting to increase model robustness.
- **Hyperparameter Tuning**: Further optimizing the learning rate, batch size, and number of epochs.
- **Exploring Different Architectures**: Trying different CNN architectures or more advanced networks like ResNet or EfficientNet.

## License
This project is licensed under the MIT License.

## Acknowledgements
- **Keras & TensorFlow**: For providing a powerful and user-friendly deep learning framework.
- **MNIST**: For the open dataset used in this project.
```
