#####################################################
#	Algoritmo de prueba de rendimiento para RNP		#
#	usando dataset de entrenamiento y prueba		#
#####################################################

import os
import gc
import numpy
import h5py
import math
from pandas import read_csv
from keras.callbacks import CSVLogger
from keras.models import *
from keras.layers import *
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_array

porcentajeEntrenamiento = 0.8
features = 4
sizeSeq = 4
paraderos = 43
nombreModelo = 'modeloParaderos002e1'
carpeta = 'modeloParaderos9'
saltos = 11

def cls():
    os.system('cls' if os.name=='nt' else 'clear')

def normalizarX(dataset):
	dataset[:, 0] = [ (k / 7)*2-1 for k in dataset[:, 0]]
	dataset[:, 1] = [ (k / 43)*2-1 for k in dataset[:, 1]]
	dataset[:, 2] = [ (k / 4800)*2-1 for k in dataset[:, 2]] # en caso de velocidad promedio 12 km/h
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

def mean_absolute_percentage_error(y_true, y_pred):
	mape = 0.0
	for i in range(len(y_true)):
		if y_true[i] == 0:
			y_true[i] = 0.001
		mape = mape + numpy.abs((y_true[i] - y_pred[i]) / y_true[i])[0]
	return mape * 100/len(y_true) 

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

# Carga el dataset
dataframe = read_csv('../../datos/datasetDefinitivoParaderosI09.tsv', engine = 'python', sep = "\t", header = None, names = ['ID','x1','x2','x3','x4','y'])
#filtrado = dataframe[(dataframe['x4'] - dataframe['x2'] >= 11) & (dataframe['x4'] - dataframe['x2'] <= 19)]
#filtrado = dataframe[ (dataframe['y'] < 3600) & (dataframe['y'] > 3000)]
dataset = dataframe.values
dataset = dataset.astype(float)

# frecuenciasPorSalto = numpy.zeros(43)
# for i in range(len(dataset)):
# 	salto = dataset[i][4] - dataset[i][2]
# 	frecuenciasPorSalto[int(salto)] = frecuenciasPorSalto[int(salto)] + 1

# print(frecuenciasPorSalto)
# input()

# Calculo de valores maximos y minimos para cada columna
# meanColumnas = dataset.mean(axis = 0)
# maxColumnas = dataset.min(axis = 0)
# minColumnas = dataset.max(axis = 0)
# print(minColumnas)
# print(meanColumnas)
# print(maxColumnas)
# input()

# Toma los viajes por cada id distinto (columna 0) e ingresarlo a otro arreglo 

xTemp = dataset[:,1:5]
yTemp = dataset[:,5]

train_size_x = int(len(xTemp) * porcentajeEntrenamiento)

trainX = xTemp[0:train_size_x]
testX = xTemp[train_size_x:len(dataset)]

train_size_y = int(len(yTemp) * porcentajeEntrenamiento)

trainY = yTemp[0:train_size_y]
testY = yTemp[train_size_y:len(dataset)]

del(xTemp)
del(yTemp)
gc.collect()

# Corrige dimensionalidad de entrada a  [samples, time steps, features]
# Normaliza x e y para estar entre 0 y 1 
trainX = normalizarX(trainX)
trainX = numpy.reshape(trainX, (len(trainX), len(trainX[0])))
trainY = numpy.reshape(trainY, (len(trainY),1))

testX = normalizarX(testX)
testX = numpy.reshape(testX, (len(testX), len(testX[0])))
testY = numpy.reshape(testY, (len(testY),1))

#Carga modelo
model = load_model('../../resultados/modelos/Paraderos/{}/{}.h5'.format(carpeta,nombreModelo))

# Calcula el MSE y RMSE para entrenamiento y test

f = open("../../resultados/pruebas/prueba_{}.txt".format(nombreModelo),"w")
f.write("DATASET\tMSE\tRMSE\tMAPE\tMAE\tMIN_MAE\tMAX_MAE\tDESV_MAE\n")

# Prueba con datos de entrenamiento
print('Prueba red paraderos (modelo 1)\n- Dataset: Entrenamiento\n- Salto: {}\n- Archivo: {}'.format(saltos,nombreModelo))

predicciones = model.predict(trainX	, batch_size = 1, verbose = 1)
# print('min {} max {}'.format(min(predicciones),max(predicciones)))
predicciones = denormalizarY(predicciones)
referencia = trainY
del(trainY)

#Calcula el MSE y RMSE para la ultima prediccion (las anteriores eran para ayudar a la prediccion)
MSE_train = 0.0
RMSE_train = 0.0
MAPE_train = 0.0
errores=numpy.array([])
errores = [abs(referencia[i] - predicciones[i]) for i in range(len(referencia))]
MSE_train = mean_squared_error(referencia, predicciones)
MINIMO_train = numpy.amin(errores)
MAXIMO_train = numpy.amax(errores)
MAE_train = numpy.mean(errores)
DESV_train = numpy.std(errores)
RMSE_train = math.sqrt(MSE_train)
MAPE_train = mean_absolute_percentage_error(referencia, predicciones)

model.reset_states()
cls()

f.write("Entrenamiento \t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(MSE_train,RMSE_train,MAPE_train,MAE_train,MINIMO_train,MAXIMO_train,DESV_train))
del(trainX)
gc.collect()

# Prueba con datos de PRUEBA
rmsePrueba = 0.0
msePrueba = 0.0

print('Prueba red paraderos (modelo 1)\n- Dataset: Prueba\n- Salto: {}\n- Archivo: {}'.format(saltos,nombreModelo))

predicciones = model.predict(testX, batch_size = 1, verbose = 1)
predicciones = denormalizarY(predicciones)
referencia = testY
del(testY)

#Calcula el MSE y RMSE para la ultima prediccion (las anteriores eran para ayudar a la prediccion)
MSE_test = 0.0
RMSE_test = 0.0
MAPE_test = 0.0
errores=numpy.array([])
errores = [abs(referencia[i] - predicciones[i]) for i in range(len(referencia))]
MSE_test = mean_squared_error(referencia, predicciones)
MINIMO_test = numpy.amin(errores)
MAXIMO_test = numpy.amax(errores)
MAE_test = numpy.mean(errores)
DESV_test = numpy.std(errores)
RMSE_test = math.sqrt(MSE_test)
MAPE_test = mean_absolute_percentage_error(referencia, predicciones)

model.reset_states()
cls()

f.write("Prueba \t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(MSE_test,RMSE_test,MAPE_test,MAE_train,MINIMO_test,MAXIMO_test,DESV_test))
f.close()
del(testX)
gc.collect()

#Muestra RMSE Y MSE promedios para cada dataset
print(bcolors.FAIL +'RESULTADOS:')
print("DATASET\tMSE\tRMSE\tMAPE\tMAE\tMIN_MAE\tMAX_MAE\tDESV_MAE")
print("Entrenamiento \t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(MSE_train,RMSE_train,MAPE_train,MAE_train,MINIMO_train,MAXIMO_train,DESV_train))
print("Prueba \t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(MSE_test,RMSE_test,MAPE_test,MAE_test,MINIMO_test,MAXIMO_test,DESV_test)+bcolors.ENDC)
