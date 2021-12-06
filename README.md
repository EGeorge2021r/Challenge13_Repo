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
The accuracy and loss of the three neural network models did not change significantly despite trying different values for output neurons and different activation functions for the hidden layers. The original model has the best result with accuracy of 0.537 and a loss of 0.903.

#### Model: "sequential_11"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_23 (Dense)            (None, 3)                 9         
                                                                 
 dense_24 (Dense)            (None, 2)                 8         
                                                                 
 dense_25 (Dense)            (None, 1)                 3         
                                                                 
=================================================================
Total params: 20
Trainable params: 20
Non-trainable params: 0
_________________________________________________________________
Original Model Results
804/804 - 1s - loss: 0.6903 - accuracy: 0.5376 - 801ms/epoch - 996us/step
Loss: 0.6903157830238342, Accuracy: 0.5375913381576538


#### Model: "sequential_12"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_26 (Dense)            (None, 3)                 9         
                                                                 
 dense_27 (Dense)            (None, 1)                 4         
                                                                 
=================================================================
Total params: 13
Trainable params: 13
Non-trainable params: 0
_________________________________________________________________
Alternative Model 1 Results
804/804 - 1s - loss: 0.6906 - accuracy: 0.5355 - 778ms/epoch - 968us/step
Loss: 0.6905868053436279, Accuracy: 0.5355310440063477


#### Model: "sequential_19"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_40 (Dense)            (None, 3)                 9         
                                                                 
 dense_41 (Dense)            (None, 1)                 4         
                                                                 
=================================================================
Total params: 13
Trainable params: 13
Non-trainable params: 0
_________________________________________________________________
Alternative Model 2 Results
804/804 - 1s - loss: 0.6908 - accuracy: 0.5346 - 737ms/epoch - 917us/step
Loss: 0.6907570958137512, Accuracy: 0.5345591902732849