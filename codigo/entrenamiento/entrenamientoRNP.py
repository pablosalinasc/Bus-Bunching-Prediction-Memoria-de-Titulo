########################################################
# Algoritmo de entrenamiento para version final de RNP #
########################################################

import os
import gc
import numpy
import h5py
import math
from pandas import read_csv
from keras.callbacks import CSVLogger
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

numeroEntrenamiento = '002'
epocas = 50
tasaApendizaje = 1e-06
neuronas = 100
porcentajeEntrenamiento = 0.8
features = 4
paraderos = 43

def cls():
    os.system('cls' if os.name=='nt' else 'clear')

def normalizarX(dataset):
	dataset[:, 0] = [ (k / 7)*2-1 for k in dataset[:, 0]]
	dataset[:, 1] = [ (k / 43)*2-1 for k in dataset[:, 1]]
	dataset[:, 2] = [ (k / 4800)*2-1 for k in dataset[:, 2]]
	dataset[:, 3] = [ (k / 43)*2-1 for k in dataset[:, 3]]
	return dataset

def denormalizarX(dataset):
	dataset[:, 0] = [ 7*(k+1)/2 for k in dataset[:, 0]]
	dataset[:, 1] = [ 43*(k+1)/2 for k in dataset[:, 1]]
	dataset[:, 2] = [ 4800*(k+1)/2 for k in dataset[:, 2]]
	dataset[:, 3] = [ 43*(k+1)/2 for k in dataset[:, 3]]
	return dataset

def normalizarY(dataset):
	dataset = [ math.tanh((k/1800)*2-1) for k in dataset]
	return dataset

def denormalizarY(dataset):
	dataset = [ 1800*(math.atanh(k)+1)/2 for k in dataset]
	return dataset

# Permite hacer print con colores
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Objeto viaje
class Viaje():
	def __init__(self,id_viaje):
		self.id = id_viaje;
		self.x = numpy.array([]);
		self.y = numpy.array([]);

# Fija la semilla del random
numpy.random.seed(7)

# Carga el dataset
dataframe = read_csv('../../datos/datasetDefinitivoParaderosI09.tsv', engine = 'python', sep = "\t", header = None, names = ['ID','x1','x2','x3','x4','y'])
# Solo se dejan las consultas que dan valores hasta media hora en el futuro (1800 segundos)
filtrado = dataframe[ (dataframe['x4'] - dataframe['x2'] >= 30) & (dataframe['x4'] - dataframe['x2'] <= 43)]
dataset = filtrado.values
dataset = dataset.astype(float)

# paraderoProm = 0
# contador = 0
# for i in range(len(dataset)):
# 	if dataset[i,5] > 1800 and dataset[i,5] < 1900:
# 		contador = contador + 1
# 		paraderoProm = paraderoProm + dataset[i,4] - dataset[i,2]

# paraderoProm = paraderoProm / contador
# print(paraderoProm)
# # En saltos de 26.42 paraderos hay tiempos de viaje de media hora en promedio
# input()

# Calculo de valores maximos y minimos para cada columna
# maxColumnas = dataset.max(axis = 0)
# minColumnas = dataset.min(axis = 0)
# print(maxColumnas)
# print(minColumnas)
print(dataset.std(axis = 0))
input()

# Toma los viajes por cada id distinto (columna 0) e ingresarlo a otro arreglo 
xTemp = dataset[:,1:5]
yTemp = dataset[:,5]

# Solo deja la seccion utilizada para entrenamiento 
train_size = int(len(xTemp) * porcentajeEntrenamiento)
x = xTemp[0:train_size]

train_size = int(len(yTemp) * porcentajeEntrenamiento)
y = yTemp[0:train_size]

del(xTemp)
del(yTemp)
gc.collect()

# Corrige dimensionalidad de entrada a  [samples, time steps, features]
# Normaliza x e y para estar entre 0 y 1 
x = normalizarX(x)
x = numpy.reshape(x, (len(x), len(x[0])))
y = normalizarY(y)
y = numpy.reshape(y, (len(y),1))

# Creacion del modelo
model = Sequential()
model.add(Dense(neuronas,input_shape=(features,),activation='tanh'))
model.add(Dense(neuronas,activation='tanh'))
model.add(Dense(1))
optimizer = Adam(lr=tasaApendizaje)
model.compile(loss = 'mean_squared_error', optimizer = optimizer)

# Entrenamiento
csv_logger = CSVLogger('../../resultados/errores/entrenamientoParaderos{0}.tsv'.format(numeroEntrenamiento), append=True, separator='\t')
for i in range(epocas):
		print(bcolors.ENDC+'Entrenamiento red paraderos (modelo 1) {}'.format(numeroEntrenamiento))
		print(bcolors.HEADER + 'Epoca {0}/{1}'.format(i+1,epocas)+bcolors.ENDC)
		model.fit(x, y, epochs = 1, batch_size = 1, verbose = 1, shuffle = True, callbacks=[csv_logger])
		model.reset_states()
		gc.collect()
		cls()
		model.save('../../resultados/modelos/modeloParaderos{0}e{1}.h5'.format(numeroEntrenamiento,i+1))

model.save('../../resultados/modelos/modeloParaderos{0}e{1}.h5'.format(numeroEntrenamiento,i+1))
