#####################################################
#	Algoritmo de prueba de rendimiento para RND 	#
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

porcentajeEntrenamiento = 0.8
features = 4
sizeSeq = 10
nombreModelo = 'modeloGPS001e34'

def cls():
	os.system('cls' if os.name=='nt' else 'clear')

def normalizarX(dataset):
#	dataset[:, 0] = [k / (5) for k in dataset[:, 0]]
#	dataset[:, 1] = [k / (7) for k in dataset[:, 1]]
#	dataset[:, 2] = [k / (43) for k in dataset[:, 2]]
	dataset[:, 3] = [k / (1.2) for k in dataset[:, 3]]
	return dataset

def denormalizarX(dataset):
#	dataset[:, 0] = [k * (5) for k in dataset[:, 0]]
#	dataset[:, 1] = [k * (7) for k in dataset[:, 1]]
#	dataset[:, 2] = [k * (43) for k in dataset[:, 2]]
	dataset[:, 3] = [k * (1.2) for k in dataset[:, 3]]
	return dataset

def normalizarY(dataset):
	dataset = [k/(1.2)  for k in dataset]
	return dataset

def denormalizarY(dataset):
	dataset = [k*(1.2) for k in dataset]
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
		self.x=numpy.array([]);
		self.y=numpy.array([]);

def mean_absolute_percentage_error(y_true, y_pred):
	mape = 0.0
	for i in range(len(y_true)):
		if y_true[i] == 0:
			y_true[i] = 0.001
		mape = mape + numpy.abs((y_true[i] - y_pred[i]) / y_true[i])[0]
	return mape * 100/len(y_true)

# Carga el dataset
dataframe = read_csv('../../datos/datasetDefinitivoGPSv4.tsv', engine = 'python', sep = "\t")
dataset = dataframe.values
dataset = dataset.astype(float)

# Calculo de valores maximos y minimos por columna
maxColumnas = dataset.max(axis = 0)
minColumnas = dataset.min(axis = 0)

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

# Corrige dimensionalidad de entrada a  [samples, time steps, features]
# Normaliza los datos para estar entre 0 y 1 
for i in range(len(train)):
	train[i].x = normalizarX(train[i].x)
	train[i].x = numpy.reshape(train[i].x, (1,len(train[i].x), len(train[i].x[0])))
	train[i].y = normalizarY(train[i].y)
	train[i].y = numpy.reshape(train[i].y, (len(train[i].y),1))

for i in range(len(test)):
	test[i].x = normalizarX(test[i].x)
	test[i].x = numpy.reshape(test[i].x, (1,len(test[i].x), len(test[i].x[0])))
	test[i].y = normalizarY(test[i].y)
	test[i].y = numpy.reshape(test[i].y, (len(test[i].y),1))

#Carga modelo
model = load_model('../../resultados/modelos/{}.h5'.format(nombreModelo))

# Calcula el las metricas a partir de las predicciones de la red
MSE_train = 0.0
RMSE_train = 0.0
MAE_train = 0.0
MAPE_train = 0.0
DESV_train = 0.0

f = open("../../resultados/pruebas/prueba_{}_entrenamiento.txt".format(nombreModelo),"w")
f.write("Viaje\tMSE\tRMSE\tMAE\tMAPE\tDESV\n")

# Prueba con datos de entrenamiento
for i in range(len(train)):

	entrada_metadatos = numpy.array([])
	entrada_recurrente = numpy.array([])
	salida_secuencia = numpy.array([])
	
	for j in range(len(train[i].x[0])-sizeSeq):


		for k in range(len(train[j].x[0])-sizeSeq): 
			# Genera las secuencias para un viaje
			if k==0:
				entrada_metadatos = numpy.array( [train[j].x[0][k+sizeSeq][0:3]] )
				entrada_recurrente = numpy.array( train[j].x[:,k:k+sizeSeq,3:4] )
				salida_secuencia = numpy.array( [ train[j].y[sizeSeq] ] )
			else:
				entrada_metadatos = numpy.append(entrada_metadatos , [ train[j].x[0][k+sizeSeq][0:3] ] , axis = 0)
				secuenciaTemp = train[j].x[:,k:k+sizeSeq,3:4]
				entrada_recurrente = numpy.append(entrada_recurrente , secuenciaTemp , axis = 0)
				indice = int(k + sizeSeq)
				salida_secuencia = numpy.append(salida_secuencia , [ train[j].y[indice] ] , axis = 0)
				
	print(bcolors.ENDC + 'Prueba red GPS (modelo 1)\n-Dataset: entrenamiento\n')
	print(bcolors.WARNING + 'Viaje {0}/{1}'.format(i+1,len(train)) + bcolors.ENDC)
	
	predicciones = model.predict([entrada_metadatos,entrada_recurrente], batch_size = 1, verbose = 1)
	predicciones = denormalizarY(predicciones)
	referencia = denormalizarY(salida_secuencia)
		
	# Calcula Metricas
	errores=numpy.array([])
	errores = [abs(referencia[k] - predicciones[k]) for k in range(len(referencia))]

	MSE = mean_squared_error(referencia, predicciones)
	RMSE = math.sqrt(MSE)
	MAE = numpy.mean(errores)
	MAPE = mean_absolute_percentage_error(referencia, predicciones)
	DESV = numpy.std(errores)
		
	f.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(i+1,MSE,RMSE,MAE,MAPE,DESV))
	MSE_train = MSE_train + MSE
	RMSE_train = RMSE_train + RMSE
	MAE_train = MAE_train + MAE
	MAPE_train = MAPE_train + MAPE
	DESV_train = DESV_train + DESV
#	model.reset_states()
	cls()

RMSE_train = RMSE_train/len(train)
MSE_train = MSE_train/len(train)
MAE_train = MAE_train/len(train)
MAPE_train = MAPE_train/len(train)
DESV_train = DESV_train/len(train)
f.write("PROMEDIO\t{}\t{}\t{}\t{}\t{}\n".format(MSE_train,RMSE_train,MAE_train,MAPE_train,DESV_train))
f.close()
del(train)
gc.collect()

f = open("../../resultados/pruebas/prueba_{0}_prueba.txt".format(nombreModelo),"w")
f.write("Viaje\tMSE\tRMSE\tMAE\tMAPE\tDESV\n")

# Prueba con datos de prueba
MSE_test = 0.0
RMSE_test = 0.0
MAE_test = 0.0
MAPE_test = 0.0
DESV_test = 0.0

for i in range(len(test)):
	
	entrada_metadatos = numpy.array([])
	entrada_recurrente = numpy.array([])
	salida_secuencia = numpy.array([])
	
	for j in range(len(test[i].x[0])-sizeSeq):

		if j==0:
			entrada_metadatos = numpy.array( [ test[i].x[0][j+sizeSeq][0:3] ] )
			entrada_recurrente = numpy.array( test[i].x[:,j:j+sizeSeq,3:4] )
			salida_secuencia = numpy.array( [ test[i].y[sizeSeq] ] )
		else:
			entrada_metadatos = numpy.append(entrada_metadatos , [ test[i].x[0][j+sizeSeq][0:3] ] , axis = 0)
			secuenciaTemp = test[i].x[:,j:j+sizeSeq,3:4]
			entrada_recurrente = numpy.append(entrada_recurrente , secuenciaTemp , axis = 0)
			indice = int(j + sizeSeq)
			salida_secuencia = numpy.append(salida_secuencia , [ test[i].y[indice] ] , axis = 0)
		
	print(bcolors.ENDC + 'Prueba red GPS (modelo 1)\n-Dataset: prueba\n')
	print(bcolors.WARNING + 'Viaje {0}/{1}'.format(i+1,len(test)) + bcolors.ENDC)
	
	predicciones = model.predict([entrada_metadatos,entrada_recurrente], batch_size = 1, verbose = 1)
	predicciones = denormalizarY(predicciones)
	referencia = denormalizarY(salida_secuencia)
	
	# Calcula Metricas
	errores=numpy.array([])
	errores = [abs(referencia[k] - predicciones[k]) for k in range(len(referencia))]

	#Calcula el MSE y RMSE
	MSE = mean_squared_error(referencia, predicciones)
	RMSE = math.sqrt(MSE)
	MAE = numpy.mean(errores)
	MAPE = mean_absolute_percentage_error(referencia, predicciones)
	DESV = numpy.std(errores)

	
	f.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(i+1,MSE,RMSE,MAE,MAPE,DESV))
	MSE_test = MSE_test + MSE
	RMSE_test = RMSE_test + RMSE
	MAE_test = MAE_test + MAE
	MAPE_test = MAPE_test + MAPE
	DESV_test = DESV_test + DESV
	cls()

RMSE_test = RMSE_test/len(test)
MSE_test = MSE_test/len(test)
MAE_test = MAE_test/len(test)
MAPE_test = MAPE_test/len(test)
DESV_test = DESV_test/len(test)
f.write("PROMEDIO\t{}\t{}\t{}\t{}\t{}\n".format(MSE_test,RMSE_test,MAE_test,MAPE_test,DESV_test))
f.close()
del(test)
gc.collect()

# Muestra RMSE Y MSE promedios para cada dataset
# Muestra RMSE Y MSE promedios para cada dataset
print(bcolors.FAIL +'RESULTADOS:')
print("DATASET\tMSE\tRMSE\tMAE\tMAPE\tDESV_MAE")
print("Entrenamiento \t{}\t{}\t{}\t{}\t{}".format(MSE_train,RMSE_train,MAE_train,MAPE_train,DESV_train))
print("Prueba \t{}\t{}\t{}\t{}\t{}".format(MSE_test,RMSE_test,MAE_test,MAPE_test,DESV_test)+bcolors.ENDC)
