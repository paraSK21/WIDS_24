# Classification of MRI images 🧬

The project focuses on developing a classification model using deep learning techniques to analyze brain MRI images. The dataset that will be used contains four classes. We would make a binary classification model (tumor vs no-tumor predictor), then would go on to build a model capable of multiclass classification.

Data preprocessing techniques will be applied to increase the training dataset's diversity and prevent overfitting. The model architecture would be based on a Convolutional Neural Network (CNN) with convolutional layers, max-pooling layers, and fully connected layers for binary/multiclass classification. After training the model, its performance will be evaluated on a separate test set, and various evaluation metrics will be calculated.

# Prerequisites
Basic Python Skills and a lot of Enthusiasm to learn about Deep Learning and Neural Networks.

# Tentative Timeline

| Week | Work | 
| :---   | :--- |
| Week 1 | Brush up Python Programming, Numpy, Pandas, Matplotlib |
| Week 2 | Learn ML basics, Neural Networks, get familiar with PyTorch |
| Week 3 | Learn about CNNs, EDA (Exploratory Data Analysis) and Data pre-processing |
| Week 4 | Building and Training the model, Fine Tuning model hyper-parameters, Model Evaluation and Inference |

# Resources
## Week 1
### Aim
During Week 1, we will review Python programming along with basics of NumPy, Pandas and Matplotlib. This knowledge will prove valuable when developing models. Note that you are not required to remember all the functions of these libraries, just go through them once.
### Important Links
* [Python in One Video](https://www.youtube.com/watch?v=L5sZ6WgOnj0) <br/>
* [NumPy Basics](https://medium.com/nerd-for-tech/a-complete-guide-on-numpy-for-data-science-c54f47dfef8d) <br/>
* [Pandas Basics](https://medium.com/edureka/python-pandas-tutorial-c5055c61d12e) <br/>
* [Matplotlib Basics](https://youtu.be/7-eg-wqOIcA?si=AkI9syiB6VQNwTCp) <br/>

## Week 2
### Aim
Get Acquainted with neural networks and the math behind it. You need not understand every nitty-griity of it, but this shall be your building blocks of deep learning to develop the intuition. You will get to know how to utilize PyTorch.
### Important links
* [Linear and Logistic Regression](https://www.youtube.com/watch?v=0pJlY_IDB8w) <br/>
* [Basic Machine Learning concepts and an introduciton to Neural Networks](https://medium.com/towards-data-science/simple-introduction-to-neural-networks-ac1d7c3d7a2c)
* [Introduction to Deep FeedForward Neural Networks](https://towardsdatascience.com/an-introduction-to-deep-feedforward-neural-networks-1af281e306cd) 
* [If you want video lectures](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&pp=iAQB) </br>
* [Building neural networks from scratch in Python](https://medium.com/hackernoon/building-a-feedforward-neural-network-from-scratch-in-python-d3526457156b) Go through how neural networks were implemented long before libraries existed. </br>
* [PyTorch For Beginners](https://www.youtube.com/playlist?list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4) (You do not need to watch the entire playlist) <br/>
* [Andrew Ng's Lectures for Machine Learning](https://youtube.com/playlist?list=PLkDaE6sCZn6FNC6YRfRQc_FbeQrF8BwGI&si=AuRQywAxT-_bdSO3)<br/>
### Assignment 1 and 2
Find the Assignment 1 [here](https://colab.research.google.com/drive/1mdV2FyO0Ket1TX0sxLNAx_zQuqwZCmVF?usp=sharing) <br/>
Dataset for Assignment 2 can be downloaded from [here](https://www.kaggle.com/datasets/abhishek14398/salary-dataset-simple-linear-regression) <br/>

Some instructions and important points:
* For Assignment 1, Go to the link and copy the file to your drive <br/>
![instruction](https://github.com/siddhant3c/WiDS-2023/assets/119072231/079d25c0-3f4a-4e2b-b78d-1052bc6e9d57)<br/>
* For Assignment 1, Follow the instructions mentioned in the file.
* For Assignment 2, use `from sklearn.linear_model import LogisticRegression, LinearRegression` and `from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error`. Use them to make a linear regression model and also for computing evaluation metrics. Train the model on the given dataset, and perform some EDA before model training.
* For both assignments, you need to submit your code (one .ipynb file for each assignment)
* The submission for both assignments would be via a single google form. (Link will be shared soon)
* Deadline is 1st Jan


## Week 3
### Aim
This week is for you to develop an understanding of Convolutional Neural Networks (CNNs). Also, we will perform EDA on the dataset, and will pre-process the data to handle class imbalance. Here is what to keep in mind- we want a binary classification model, and thus need only 2 classes. You will need to merge the 3 tumor classes into a single class. Think of various data augmentation techniques you can use to overcome the imbalance.
### Important Links
* [Binary Classification using CNNs](https://medium.com/@mayankverma05032001/binary-classification-using-convolution-neural-network-cnn-model-6e35cdf5bdbb) </br>
* [Convolutional Neural Network(CNN) with Practical Implementation](https://medium.com/machine-learning-researcher/convlutional-neural-network-cnn-2fc4faa7bb63)
* [Brain Tumor MRI dataset](https://drive.google.com/file/d/1yh7nSiDYAmNUsDcvEOtGBSdVHipvQ0-d/view?usp=sharing)</br>
* [nn.Module Documentation](https://pytorch.org/docs/stable/nn.html) </br>
* [Implementation of a CNN based Image Classifier using PyTorch](https://www.geeksforgeeks.org/implementation-of-a-cnn-based-image-classifier-using-pytorch) </br>

### Assignment 3
* Download [assignment_3.ipynb](https://github.com/siddhant3c/WiDS-2023/blob/main/Assignment_3.ipynb), make a copy of it and complete it. Make sure to build two models (linear and CNN based)
* Both models will have same training function
* Deadline is 9th Jan, form link to be shared soon.

## Week 4
### Aim
In this week your task is to build a classification model for the final project.

* The dataset link has been provided within the week 3 resources. The initial step entails visualizing the data, wherein it becomes apparent that there exist four classes with unevenly distributed examples.

Participants have the flexibility to choose between two options:

1. Binary Classification
2. Multiclass Classification

| **Binary Classification** | Multiclass Classificatin | 
| :---   | :--- |
| Consolidate the three tumor classes into a singular class. Subsequently, execute data augmentation to equalize the number of training and test data samples for both classes | Implement data augmentation to ensure uniformity in the number of samples across all classes |

Subsequently, the following procedures are applicable to both scenarios:

1. Attribute target labels to the input training data.
2. Divide the data into training, validation, and test sets.
3. Construct the network architecture, with the freedom to set hyperparameters according to individual preferences.
4. Conduct training and validation on the model, and assess its performance using various evaluation metrics such as F1 score, ROC, etc.

**15th Jan** is the final project submission deadline (as per the Analytics Club)
