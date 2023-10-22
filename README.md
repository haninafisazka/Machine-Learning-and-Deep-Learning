# Machine-Learning-and-Deep-Learning

## Assignment Ch. 4 - Machine Learning and Deep Learning
Welcome to Chapter 4 of the Machine Learning and Deep Learning course in the Artificial Intelligence Track at Startup Campus. In this assignment, we will explore various aspects of machine learning, including clustering, supervised learning, and deep learning. The assignment is designed to help us build and evaluate models and understand neural networks.

## Assignment Outcomes (Target Portfolio)
By completing this assignment, we should be able to:
1. Perform data segmentation using clustering methods.
2. Design and compare multiple supervised learning models.
3. Design and build a simple neural network model for the MNIST Handwritten Digit Dataset.


## TASK 1 - Bank Customer Churn Prediction ##

This project focuses on predicting customer churn in the banking industry. Customer churn, also known as customer attrition, refers to the decision of customers to discontinue their use of the bank's services. Predicting churn can help banks take proactive measures to retain their customers and improve customer satisfaction.

### Project Overview
1. Data Collection
The dataset is obtained from this source. It contains various features related to bank customers, such as age, balance, geographical location, and whether they are active members.

2. Data Preprocessing
Initial data preprocessing includes removing irrelevant columns and one-hot encoding categorical features.

3. Data Split
The dataset is split into training and testing sets with a 75-25 split ratio.

4. Model Selection
This project explores the performance of different machine learning models to predict customer churn. The models considered are:
    - CatBoost Classifier
    - XGBoost Classifier
    - LightGBM Classifier
  
5. Model Training and Evaluation
Each model is trained using the training data and evaluated using various performance metrics such as accuracy. The goal is to determine the best-performing model.

### Usage
You can use this code as a template for predicting customer churn in your own dataset. Here's how to use it:
1. Replace the dataset URL with your own dataset or provide the appropriate path to your data.
2. Modify the data preprocessing steps as needed to fit your data.
3. Select and initialize the machine learning models you want to evaluate.
4. Customize the hyperparameter tuning process for your chosen models if necessary.
5. Run the code and evaluate the model performance on your dataset.

## TASK 2 - Data Clustering Analysis
This project involves the analysis and clustering of data from an external source to uncover meaningful patterns and group data points into clusters. The data is visualized and analyzed to identify the optimal number of clusters.

### Project Overview
1. Data Source
The data is obtained from this source.
2. Data Preprocessing
The initial step includes reading the data and dropping unnecessary columns.
3. Data Visualization
The data is visualized to gain insights into its structure and distribution.
4. Cluster Analysis
The project aims to identify the optimal number of clusters in the data using the silhouette score. The AgglomerativeClustering algorithm is employed for this analysis.

### Usage
You can use this code as a template for clustering analysis on your own dataset. Here's how to use it:
1. Replace the dataset URL with your own dataset or provide the appropriate path to your data.
2. Modify the data preprocessing steps as needed to fit your data.
3. Customize the range of cluster numbers to test for your dataset.
4. Run the code and analyze the silhouette score to determine the optimal number of clusters.
5. Visualize the clustering results to gain insights into your data's structure.

## TASK 3 - MNIST Handwritten Digit Classification using PyTorch
This project is focused on performing handwritten digit classification using a neural network implemented with PyTorch. The MNIST dataset is used, and the process includes data loading, visualization, model design, hyperparameter setup, training, and model evaluation.

### Project Overview
1. Data Loading
The MNIST dataset is loaded using PyTorch's data loader, and the data is transformed for further processing.
2. Data Visualization
The project visualizes several MNIST digits to understand the dataset's structure.
3. Model Design
A neural network model is designed for handwritten digit classification. The model consists of fully connected layers.
4. Hyperparameter Setup
Hyperparameters like the loss function, optimizer, and learning rate are configured for model training.
5. Model Training
The training loop is implemented, and the model is trained on the MNIST training dataset.
6. Model Evaluation
The trained model is evaluated using various performance metrics such as accuracy, confusion matrix, and classification report.

### Usage
You can use this code as a template for handwritten digit classification tasks using your own dataset or variations of the MNIST dataset. Here's how to use it:
1. Replace the dataset loading code with your own dataset or provide the appropriate path to your data.
2. Modify the model architecture and hyperparameters according to your task.
3. Run the code to train the model and evaluate its performance.
4. Analyze the model's accuracy and other metrics to assess its performance.

