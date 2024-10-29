# Human Emotions for image classification

Emotion recognition is a groundbreaking field in artificial intelligence that seeks to decode human emotions through facial expressions, creating more responsive and human-centric technologies. This project, Emotion Recognition Using AI, utilizes advanced deep learning algorithms to classify images of human faces into six fundamental emotional categories: happiness, sadness, anger, pain, fear, and disgust. By training the model on facial features unique to each emotion, this project aims to bridge the gap between technology and empathetic human interaction.  

With the growing presence of AI in everyday life, understanding emotions has become essential for developing applications that respond intuitively to users. Emotions are a core component of human communication, conveying our thoughts, feelings, and intentions. Recognizing this, our project harnesses AI to identify subtle emotional cues in facial expressions. The application of emotion recognition spans multiple fields—from enhancing customer service experiences and providing mental health support to adapting virtual education based on student engagement. This project thus stands at the forefront of human-centered AI, contributing to advancements in health, education, social welfare, and user experience.

# Prerequisites
Basic Python Skills and a lot of Enthusiasm to learn about Deep Learning and Neural Networks.

# Tentative Timeline

| Week | Work | 
| :---   | :--- |
| Week 1 | Brush up Python Programming, Numpy, Pandas, Matplotlib |
| Week 2 - Week 3 | Learn ML basics, Neural Networks, get familiar with PyTorch |
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

## Week 2 - Week 3
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
* For Assignment 1, Go to the link and copy the file to your drive 
* For Assignment 1, Follow the instructions mentioned in the file.
* For Assignment 2, use `from sklearn.linear_model import LogisticRegression, LinearRegression` and `from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error`. Use them to make a linear regression model and also for computing evaluation metrics. Train the model on the given dataset, and perform some EDA before model training.
* For both assignments, you need to submit your code (one .ipynb file for each assignment)

## Week 4
### Aim
In this week your task is to build a classification model for the final project.

* Final Project Dataset :- https://drive.google.com/drive/folders/1dvXIrVzgOWgOZxpvknyzT-wXyjfnm6F3?usp=sharing

Participants have the flexibility to choose between two options:

1. Binary Classification
2. Multiclass Classification

| **Binary Classification** | Multiclass Classificatin | 
| :---   | :--- |
| Consolidate the six classes into a singular class. Subsequently, execute data augmentation to equalize the number of training and test data samples for both classes | Implement data augmentation to ensure uniformity in the number of samples across all classes |

Subsequently, the following procedures are applicable to both scenarios:

1. Attribute target labels to the input training data.
2. Divide the data into training, validation, and test sets.
3. Construct the network architecture, with the freedom to set hyperparameters according to individual preferences.
4. Conduct training and validation on the model, and assess its performance using various evaluation metrics such as F1 score, ROC, etc.
