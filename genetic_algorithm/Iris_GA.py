'''
Training a CNN with genetic algorithm
Using the classic Iris dataset
'''
import tensorflow.keras
import pygad.kerasga
import numpy
import pygad
import pandas as pd


def fitness_func(solution, sol_idx):
    global X_train, y_train, keras_ga, model
    model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model, weights_vector=solution)
    model.set_weights(weights=model_weights_matrix)
    predictions = model.predict(X_train)
    mae = tensorflow.keras.losses.MeanAbsoluteError()
    abs_error = mae(y_train, predictions).numpy() + 0.00000001
    solution_fitness = 1.0 / abs_error
    return solution_fitness

def callback_generation(ga_instance):
    if(ga_instance.generations_completed%20 == 0):
        print("Generation = {generation}".format(generation=ga_instance.generations_completed))
        print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

input_layer  = tensorflow.keras.layers.Input(4)
dense_layer1 = tensorflow.keras.layers.Dense(16, activation="swish")(input_layer)
dense_layer2 = tensorflow.keras.layers.Dense(8, activation="swish")(dense_layer1)
output_layer = tensorflow.keras.layers.Dense(3, activation="softmax")(dense_layer2)
model = tensorflow.keras.Model(inputs=input_layer, outputs=output_layer)
weights_vector = pygad.kerasga.model_weights_as_vector(model=model)
keras_ga = pygad.kerasga.KerasGA(model=model, num_solutions=10)


# Loading and pre-processing data
data  = pd.read_csv('Iris.csv')
# converting species data to categorical
# Import label encoder 
from sklearn import preprocessing 
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
# Encode labels in column 'species'. 
data['Species']= label_encoder.fit_transform(data['Species']) 
# Shuffling the data
data = data.sample(frac = 1)
# converting to numpy array
data = data.to_numpy()
# the inputs are the 2nd, 3rd, 4th and 5th column
data_inputs = data[:,1:5]
# the outputs is the last colimn
data_outputs = data[:,5]
# Using one Hot Encoding to encode the data
from sklearn.preprocessing import OneHotEncoder
# Convert data to a single column
data_outputs = data_outputs.reshape(-1, 1)
# One Hot encode the class labels
encoder = OneHotEncoder(sparse=False)
data_outputs = encoder.fit_transform(data_outputs)

# splitting into test and train set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_inputs, data_outputs, test_size=0.2, random_state=42)


# Setting the hyperparameters
num_generations = 250
num_parents_mating = 5
initial_population = keras_ga.population_weights

# creating an instance of genetic algorithm
ga_instance = pygad.GA(num_generations=num_generations, 
                       num_parents_mating=num_parents_mating, 
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       on_generation=callback_generation)

# running the instance
ga_instance.run()

# After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
ga_instance.plot_result(title="PyGAD & Keras - Iteration vs. Fitness", linewidth=4)

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

# Fetch the parameters of the best solution.
best_solution_weights = pygad.kerasga.model_weights_as_matrix(model=model, weights_vector=solution)
model.set_weights(best_solution_weights)
predictions = model.predict(X_test)
print("Predictions : \n", predictions)

mae = tensorflow.keras.losses.MeanAbsoluteError()
abs_error = mae(y_test, predictions).numpy()
print("Absolute Error : ", abs_error)
