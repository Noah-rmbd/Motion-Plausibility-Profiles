import pickle
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM

distance_matrix = np.loadtxt("Models-Weights-And-Results/trajectory_distance_matrix.txt", delimiter=",")

print("Now the training")
model = OneClassSVM(kernel='precomputed')
y = model.fit_predict(distance_matrix)
np.savetxt('Models-Weights-And-Results/trajectory_predictions.txt', y)
print("Finally ended")
print(np.shape(y))