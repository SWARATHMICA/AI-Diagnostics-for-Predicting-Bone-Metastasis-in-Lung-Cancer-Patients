# AI Diagnostics for Predicting Bone Metastasis in Lung Cancer Patients

## Project Overview
This project leverages deep learning to predict bone metastasis in lung cancer patients using a fine-tuned VGG16 model. The model classifies bone images into three categories: normal, benign, and malignant, achieving an accuracy of 92% after 1000 epochs. The goal is to assist healthcare providers in diagnosing bone metastasis, improving prognosis and treatment planning.

## Dataset
The dataset used in this project is the [IQ-OTHNCCD Lung Cancer Dataset](https://www.kaggle.com/datasets/adityamahimkar/iqothnccd-lung-cancer-dataset) from Kaggle. It includes labeled images of lung cancer patients’ bone scans, categorized into 'normal,' 'malignant,' and 'benign' classes.

## Model Architecture
We used the VGG16 model pre-trained on the ImageNet dataset as the base model and modified the top layers for a multi-class classification task:
- **Flatten layer** for feature extraction.
- Fully connected layers with **ReLU activation** for learning complex patterns.
- **Dropout layers** to prevent overfitting.
- Output layer with **softmax activation** for multi-class classification.

## Code Outline
1. **Data Preparation**:
   - The dataset is split into training, validation, and testing sets with a class balance.
   - Image augmentation is applied to the training data to improve generalization.
2. **Model Training**:
   - The model is compiled with categorical cross-entropy loss and trained using the Adam optimizer.
   - Early stopping was used to prevent overfitting, monitoring accuracy with patience of 10 epochs.
3. **Evaluation and Visualization**:
   - Accuracy and loss curves are plotted to monitor training progress.
   - Confusion matrix and classification report to assess model performance.

## Usage
To train and evaluate this model:
1. Clone the repository and download the dataset.
2. Install the required packages:
   ```bash
   pip install tensorflow sklearn matplotlib seaborn
3.Run the script to preprocess data, train the model, and generate predictions.

## Results
After 1000 epochs, the model achieved:

Accuracy: 92%
Loss: Stable with good convergence.

## Visualizations
Accuracy and loss curves for both training and validation sets.Confusion matrix and classification report to provide a comprehensive view of model performance across classes.

## Conclusion
This deep learning model is a step forward in the application of AI for diagnostic support in oncology, aiming to assist in early detection of bone metastasis in lung cancer patients. Future improvements could include model fine-tuning and exploring advanced architectures to further enhance accuracy.
