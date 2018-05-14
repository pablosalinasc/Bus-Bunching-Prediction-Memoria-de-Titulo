#############################################################################
#	Algoritmo de prueba de predictibilidad de serie de tiempo de RND 		#
#	implementa modelo de persistencia para serie de tiempo de distancias	#
#############################################################################

import os
import gc
import numpy
import h5py
import math
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def cls():
	os.system('cls' if os.name=='nt' else 'clear')

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
		self.x=numpy.array([]);
		self.y=numpy.array([]);

def square(list):
    return [i ** 2 for i in list]

def sqrt(list):
	return [math.sqrt(x) for x in list]

porcentajeEntrenamiento = 0.8

# Fija la semilla del random
numpy.random.seed(7)

# Carga el dataset
dataframe = read_csv('../../datos/datasetDefinitivoGPSv4.tsv', engine = 'python', sep = "\t")
dataset = dataframe.values
dataset = dataset.astype(float)

# Calculo de valores maximos y minimos por columna
# maxColumnas = dataset.max(axis = 0)
# minColumnas = dataset.min(axis = 0)
# print(maxColumnas)
# print(minColumnas)
# input()

# Toma los viajes por cada id distinto (columna 0) e ingresarlo a otro arreglo 
dataset_viajes = []
for i in range(len(dataset)):
	id_lectura = dataset[i][0]
	#En caso que el dataset_viaje tenga datos
	if len(dataset_viajes) > 0:
		#Agrega puntos al mismo viaje
		if dataset_viajes[len(dataset_viajes)-1].id == id_lectura:
			dataset_viajes[len(dataset_viajes)-1].x = numpy.append(dataset_viajes[len(dataset_viajes)-1].x,[dataset[i][1:5]],axis=0)
			dataset_viajes[len(dataset_viajes)-1].y = numpy.append(dataset_viajes[len(dataset_viajes)-1].y,[dataset[i][5]],axis=0)
		#Agrega un nuevo viaje
		else:
			dataset_viajes.append(Viaje(id_lectura))
			dataset_viajes[len(dataset_viajes)-1].x = numpy.array([dataset[i][1:5]])
			dataset_viajes[len(dataset_viajes)-1].y = numpy.array([dataset[i][5]])
	#En caso que sea el primer viaje
	else:
		dataset_viajes.append(Viaje(id_lectura))
		dataset_viajes[len(dataset_viajes)-1].x = numpy.array([dataset[i][1:5]])
		dataset_viajes[len(dataset_viajes)-1].y = numpy.array([dataset[i][5]])

dataset = dataset_viajes
del(dataset_viajes)
gc.collect()


# separa entre dataset de entrenamiento y prueba
train_size = int(len(dataset) * porcentajeEntrenamiento)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size], dataset[train_size:len(dataset)]
del(dataset)
gc.collect()

deltas = []
for i in range(len(train)):
	for j in range(len(train[i].x)):
		if i==0:
			deltas = [abs(train[i].x[j][3] - train[i].y[j])]
		else:
			deltas.append(abs(train[i].x[j][3] - train[i].y[j]))

MIN_train = numpy.min(deltas)
MAX_train = numpy.max(deltas)
MAE_train = numpy.mean(deltas)
MSE_train = numpy.mean(square(deltas))
RMSE_train = math.sqrt(MSE_train)

deltas = []
for i in range(len(test)):
	for j in range(len(test[i].x)):
		if i==0:
			deltas = [abs(test[i].x[j][3] - test[i].y[j])]
		else:
			deltas.append(abs(test[i].x[j][3] - test[i].y[j]))

MIN_test = numpy.min(deltas)
MAX_test = numpy.max(deltas)
MAE_test = numpy.mean(deltas)
MSE_test = numpy.mean(square(deltas))
RMSE_test = math.sqrt(MSE_test)

print("Modelo de persistencia (RND)")
print("Dataset\tMIN\tMAX\tMAE\tMSE\tRMSE")
print("Entrenamiento\t{}\t{}\t{}\t{}\t{}".format(MIN_train,MAX_train,MAE_train,MSE_train,RMSE_train))
print("Prueba\t{}\t{}\t{}\t{}\t{}\n".format(MIN_test,MAX_test,MAE_test,MSE_test,RMSE_test))
