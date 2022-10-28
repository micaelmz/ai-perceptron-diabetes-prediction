"""
Solução do desafio do processo seletivo da Liga de IA 2022.2
Criado por: Micael Muniz
Hora de finalização: 18:45 19/09/2022
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random as python_random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

SEED = 0

# Set seed for reproducibility
np.random.seed(SEED)

# Data set URL
url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

original_data = pd.read_csv(url, sep=',')

# talvez a primeira posição seja o ID do paciente, então não é relevante
data = original_data.iloc[:, 1:]

# Todos dados menos se tem ou não diabetes
data_patient = data.iloc[:, :-1]

# Somente se tem ou não diabetes
result_patient = data.iloc[:, -1:]

print(f"No total temos {len(data_patient)} pacientes")

# Teste e treino
# 15% para teste e 85% para treino, isso cria um pequeno fator de aleatoriedade, o que pode gerar resultados diferentes
train_data, test_data, train_result, test_result = train_test_split(data_patient, result_patient, test_size=0.15)

# Normalização dos dados
# Isso faz com que os dados fiquem entre 0 e 1, facilitando o treino da rede neural, é como se fosse "uma regra e tres"
# ou uma distriuição gaussiana
scaler = StandardScaler()
scaler.fit(train_data)
train_data = scaler.transform(train_data.values)
test_data = scaler.transform(test_data.values)

# Criação da rede neural

np.random.seed(SEED)
python_random.seed(SEED)
tf.random.set_seed(SEED)
ann = keras.Sequential()
ann.add(layers.Dense(6, activation='relu'))
ann.add(layers.Dense(1, activation='sigmoid'))

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann.fit(train_data, train_result, batch_size=16, epochs=10000)

result_prediction = ann.predict(train_data)
result_prediction = (result_prediction > 0.5)
matrix = confusion_matrix(train_result, result_prediction)

print(f"A acurancia da IA é de {round(accuracy_score(train_result, result_prediction) * 100, 2)}%")
print(f"{matrix[0][0]} pacientes não tem diabetes e foram classificados corretamente")
print(f"{matrix[0][1]} pacientes não tem diabetes e foram classificados incorretamente")
print(f"{matrix[1][0]} pacientes tem diabetes e foram classificados incorretamente")
print(f"{matrix[1][1]} pacientes tem diabetes e foram classificados corretamente")