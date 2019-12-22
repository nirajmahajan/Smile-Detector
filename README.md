# Smile Recogniser

A simple emotion classifier trained in **Keras**. It has an accuracy of 98.00% with the training set and 94.67% with the validation set

## Code Structure:

The code structure for all the models is quite similar. Each model has the following files:

- **helpers.py** : Contains all essential imports and functions
- **train.py** : Used for training and testing the accuracy of a model
- **predict.py** : Used for predicting the digit in a given image
- **model/** : Has a pre-trained model
- **data/** : Has the dataset in the .h5 format
- **samples/** : Contain a few sample testcases

## Model:

## ![](/home/legolas/Github/Smile-Detector/model/trained_model.png)

## Usage of Code:

1. To train the model:

   ```bash
   python3 train.py # Note that this will replace the pre existing model
   ```

2. To check the accuracy of the model (present in the 'model' directory):

   ```bash
   python3 train.py -use_trained_model
   ```

3. To predict the emotion from an image:

   ```bash
   python3 predict.py --path <Path To Image>
   ```

