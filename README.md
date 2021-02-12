# CNN_genetic_algorithm

The only problem i previously used Genetic Algorithm(GA) was in knapsack problem. So I got a really cool idea of classifying the IRIS dataset usin CNN, where GA is used to train the weights.

Training a CNN with genetic algorithm using the classic Iris dataset

## Requirements:
```bash
$ pip install tensorflow
$ pip install keras
$ pip install numpy
$ pip install pandas
$ pip install scikit-learn
$ pip install pygad
```

[This](https://blog.paperspace.com/train-keras-models-using-genetic-algorithm-with-pygad/) blog is used as the main inspiration in training the IRIS dataset using a Genetic Algorithm based CNN architecture.

[This](https://pygad.readthedocs.io/en/latest/) is the documentation to read more about PyGAD, PyGAD is a python library for implementing genetic algorithm, and it also supports keras integration.

[This](https://www.kaggle.com/kstaud85/iris-data-visualization) kaggle post provides a detailed visual representation of the IRIS dataset.

## Network Architecture of the CNN used:
```
Layer (type)                 Output Shape              Param #   
=================================================================
input_10 (InputLayer)        [(None, 4)]               0         
_________________________________________________________________
dense_25 (Dense)             (None, 16)                80        
_________________________________________________________________
dense_26 (Dense)             (None, 8)                 136       
_________________________________________________________________
dense_27 (Dense)             (None, 3)                 27        
=================================================================
Total params: 243
Trainable params: 243
Non-trainable params: 0
_________________________________________________________________
```

## Hyperparameters used in training:
There are two basic hyperparameters used in the GA model, those are: num_generations, and num_parents_mating.
Where the parameter values used are:
```python
num_generations = 250
num_parents_mating = 5
```

## Absolute error
The model reaches an Absolute Error :  0.026681786

## Fitness vs Generation graph:
![Fitness Graph](/genetic_algorithm/Iris_GA.png "Fitness Graph")
