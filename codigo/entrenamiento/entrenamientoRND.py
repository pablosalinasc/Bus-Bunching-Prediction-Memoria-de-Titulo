########################################################
# Algoritmo de entrenamiento para version final de RND #
########################################################

import os
import gc
import numpy
import h5py
import math
import theano
from pandas import read_csv
from keras.callbacks import CSVLogger
from keras.models import *
from keras.layers import *
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

numeroEntrenamiento = '001'
epocas = 50
tasaApendizaje = 1e-04
bloquesLSTM = 50
porcentajeEntrenamiento = 0.8
features = 4
sizeSeq = 10

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

# print(theano.config)
# input()

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
del(test)
gc.collect()

# Corrige dimensionalidad de entrada a  [samples, time steps, features]
# Normaliza los datos para estar entre 0 y 1 
for i in range(len(train)):
	train[i].x = normalizarX(train[i].x)
	train[i].x = numpy.reshape(train[i].x, (1,len(train[i].x), len(train[i].x[0])))
	train[i].y = normalizarY(train[i].y)
	train[i].y = numpy.reshape(train[i].y, (len(train[i].y),1))

# Creacion del modelo
entradaMetadatos = Input(shape=(3,))
embedding = Embedding(64,6)(entradaMetadatos)
capa2d =Flatten()(embedding)
densa = Dense(bloquesLSTM,activation='sigmoid')(capa2d)

entradaRecurrente = Input(shape=(sizeSeq,1))
lstm1 = LSTM(bloquesLSTM,unroll=True, return_sequences = True)(entradaRecurrente)
lstm2 = LSTM(bloquesLSTM,unroll=True)(lstm1)

concatenacion = concatenate([densa, lstm2])
sigmoidal = Dense(2*bloquesLSTM, activation="sigmoid")(concatenacion)
salida = Dense(1)(sigmoidal)

optimizer = Adam(lr = tasaApendizaje)
modelo = Model(inputs=[entradaMetadatos, entradaRecurrente], outputs=salida)
modelo.compile(loss = 'mean_squared_error', optimizer = optimizer)

# Entrenamiento 
# 	- Realiza 1 backpropagation por secuencia
nModelo = 1
csv_logger = CSVLogger('../../resultados/errores/entrenamientoGPS{0}.tsv'.format(numeroEntrenamiento), append=True, separator='\t')
for i in range(epocas):
	for j in range(len(train)):
		entrada_metadatos = numpy.array([])
		entrada_recurrente = numpy.array([])
		salida_secuencia = numpy.array([])

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

		print(bcolors.ENDC+'Entrenamiento red GPS (modelo 1) {}'.format(numeroEntrenamiento))
		print(bcolors.HEADER + 'Epoca {0}/{1}'.format(i+1,epocas)+bcolors.ENDC)
		print(bcolors.WARNING + 'Par de viajes {0}/{1}'.format(j+1,len(train))+bcolors.ENDC+bcolors.OKBLUE)
		modelo.fit([entrada_metadatos, entrada_recurrente], salida_secuencia, epochs = 1, batch_size = 1, verbose = 1, shuffle = False, callbacks=[csv_logger])
		modelo.reset_states()
		gc.collect()
		cls()

	modelo.save('../../resultados/modelos/modeloGPS{0}e{1}.h5'.format(numeroEntrenamiento,nModelo))
	nModelo = nModelo + 1
