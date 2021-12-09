# Challenge13_Repo
To predict whether Alphabet Soup funding applicants will be successful using a deep neural network

## User Story
Role: Risk management associate at Alphabet Soup, a venture capital firm.

Goal: Create a model that predicts whether applicants will be successful if funded by Alphabet Soup. 

Reason: Alphabet Soup’s business team receives many funding applications from startups every day. With my knowledge of machine learning and neural networks, the task is to decide to use the features in the provided dataset to create a binary classifier model that will predict whether an applicant will become a successful business.

## General information
The business team has given you a CSV file containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. The CSV file contains a variety of information about each business, including whether or not it ultimately became successful. 


## Technology
Jupyter notebook that contains data preparation, analysis, and visualizations %matplotlib inline Python

## Libraries used in the analysis
The following libraries were imported and dependencies were used:
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder

## Analysis and Model optimization
To optimize your model for a predictive accuracy as close to 1 as possible, the following techniques were used:
• Adjust the input data by dropping different features columns to ensure that no variables or outliers confuse the model. Done
• Add more neurons (nodes) to a hidden layer. Done
• Add more hidden layers. Done
• Use different activation functions for the hidden layers. Done
• Add to or reduce the number of epochs in the training regimen. Done

## Deliverables
This project consists of three technical deliverables as follows:
• Preprocess data for use on a neural network model.
• Use the model-fit-predict pattern to compile and evaluate a binary classification model using a neural network.
• Optimize the neural network model.
. After finishing your models, display the accuracy scores achieved by each model, and compare the results. 
. Save each models as an HDF5 file, labeled as:
   Original model as -  AlphabetSoup.h5
   Alernative model 1 as - A1_AlphabetSoup.h5
   Alternative model 2 as - A2_AlphabetSoup.h5
   
## Results
The accuracy and loss of the three neural network models did not change significantly despite trying different activation functions for the hidden layers. The original model has the best result with accuracy of 0.537 and a loss of 0.903.

#### Original Model nn with training data
Model: "sequential_13"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_37 (Dense)            (None, 6)                 18        
                                                                 
 dense_38 (Dense)            (None, 6)                 42        
                                                                 
 dense_39 (Dense)            (None, 1)                 7         
                                                                 
=================================================================
Total params: 67
Trainable params: 67
Non-trainable params: 0
_________________________________________________________________
Original Model Results
268/268 - 0s - loss: 0.6911 - accuracy: 0.5335 - 307ms/epoch - 1ms/step
Loss: 0.6910977959632874, Accuracy: 0.533527672290802

####  Alternative Model nn_A1 with test data
Model: "sequential_24"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_85 (Dense)            (None, 6)                 18        
                                                                 
 dense_86 (Dense)            (None, 12)                84        
                                                                 
 dense_87 (Dense)            (None, 1)                 13        
                                                                 
 dense_88 (Dense)            (None, 1)                 2         
                                                                 
=================================================================
Total params: 117
Trainable params: 117
Non-trainable params: 0
_________________________________________________________________
Alternative Model 1 Results
268/268 - 0s - loss: 0.6906 - accuracy: 0.5338 - 438ms/epoch - 2ms/step
Loss: 0.6906254887580872, Accuracy: 0.5337609052658081


#### Alternative Model nn_A2 with test data
Model: "sequential_35"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_102 (Dense)           (None, 6)                 18        
                                                                 
 dense_103 (Dense)           (None, 18)                126       
                                                                 
 dense_104 (Dense)           (None, 1)                 19        
                                                                 
=================================================================
Total params: 163
Trainable params: 163
Non-trainable params: 0
_________________________________________________________________
Alternative Model 2 Results
268/268 - 1s - loss: 0.6909 - accuracy: 0.5335 - 639ms/epoch - 2ms/step
Loss: 0.6908572912216187, Accuracy: 0.533527672290802

Conclusion:
By comparing the three models, it is observed that Alternative model 1 has aslightly better accuracy than the original training model and the Alternative 2 model.