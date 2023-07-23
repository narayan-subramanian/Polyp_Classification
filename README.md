# Polyp_Classification

This GitHub repository contains the code and resources for a polyp classification project using different models. 
The goal of this project is to classify polyps accurately using various deep learning models. The following models have been implemented and evaluated:

### Classification using DNN
Implementation of a Deep Neural Network (DNN) for polyp classification.

### Small CNNs: 1 Conv + 1 FC
Implementation of a small Convolutional Neural Network (CNN) architecture with 1 Convolutional layer and 1 Fully Connected (FC) layer.

### Small CNNs: 2 Conv + 1 FC
Implementation of a small Convolutional Neural Network (CNN) architecture with 2 Convolutional layers and 1 Fully Connected (FC) layer.

### Small CNNs: 3 Conv - 1 FC
Implementation of a small Convolutional Neural Network (CNN) architecture with 3 Convolutional layers and 1 Fully Connected (FC) layer.

### Hyperparameter Optimization for 3 Conv - FC
Hyperparameter optimization for the Small CNNs model with 3 Convolutional layers and 1 Fully Connected (FC) layer to fine-tune its performance.
VGG16 Transfer Learning

### Using the pre-trained VGG16 model for feature extraction and classification on the polyp dataset.
Calculate the outputs of bottom VGG model for training and validation

### Extracting and saving the intermediate outputs of the pre-trained bottom VGG16 model for training and validation datasets.
Train the top FC model using the outputs from the pre-trained bottom VGG16

### Implementing a Fully Connected (FC) model on top of the pre-trained bottom VGG16 model and training it using the saved outputs.
VGG16 Fine Tuning

### Fine-tuning the pre-trained VGG16 model on the polyp dataset to improve its performance.


## Dataset
https://drive.google.com/drive/folders/13Qnx0SM-s9CXTcTpjNMdcvM-tXi5vzFR?usp=sharing

The dataset can be downloaded from here


## Usage

1. Clone the repository to your local machine:

2. Make sure you have Jupyter Notebook installed. If you don't have it, you can install it using pip:
3. pip install notebook
4. Launch Jupyter Notebook:
5. Once the Jupyter Notebook server is running, navigate to the project folder and open the .ipynb file.
