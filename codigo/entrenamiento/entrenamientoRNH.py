########################################################
# Algoritmo de entrenamiento para version final de RNH #
########################################################

import os
import gc
import numpy
import math
import theano
from pandas import read_csv
from keras.callbacks import CSVLogger
from keras.models import *
from keras.layers import *
from keras.optimizers import Adam

#print('Importa correctamente los paquetes')
# Se elimina la variable creciente en lo datos de secuencia, dejando solo el dato del headway

# print(theano.config)
# input()

numeroEntrenamiento = '001'
epocas = 1
tasaApendizaje = 1e-04
bloquesLSTM = 50
porcentajeEntrenamiento = 0.8
features = 3
sizeSeq = 4

def cls():
	os.system('cls' if os.name=='nt' else 'clear')

def normalizarX(dataset):
#	dataset[:, 0] = [k / (4800) for k in dataset[:, 0]]
#	dataset[:, 1] = [k / 8 for k in dataset[:, 1]]
	#dataset[:, 2] = [k / (4800) for k in dataset[:, 2]]
	dataset[:, 3] = [k / (3600) for k in dataset[:, 3]]
	return dataset

def denormalizarX(dataset):
#	dataset[:, 0] = [k * (4800) for k in dataset[:, 0]]
#	dataset[:, 1] = [k * (8) for k in dataset[:, 1]]
	#dataset[:, 2] = [k * (4800) for k in dataset[:, 2]]
	dataset[:, 3] = [k * (3600) for k in dataset[:, 3]]
	return dataset

def normalizarY(dataset):
	dataset = [k/(600)  for k in dataset]
	return dataset

def denormalizarY(dataset):
	dataset = [k*(600) for k in dataset]
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

# Fija la semilla del random
numpy.random.seed(7)

# Carga el dataset
dataframe = read_csv('../../datos/datasetDefinitivoModelo2v4.tsv', engine = 'python', sep = "\t", header = None, names = ['ID','xm1','xm2','xm3','xs4','y'])
dataset = dataframe.values
dataset = dataset.astype(float)

# Calculo de valores maximos y minimos por columna
# maxColumnas = dataset.max(axis = 0)
# minColumnas = dataset.min(axis = 0)
# print(minColumnas)
# print(maxColumnas)
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

#Cuenta viajes que superan cierto valor minimo
# valorMax = 3600
# contador = 0
# minimo = 3600.0
# for i in range(len(dataset)):
# 	if max(dataset[i].x[:,3]) > valorMax:
# 		contador = contador + 1
# 		if min(dataset[i].x[:,3]) < minimo:
# 			minimo = min(dataset[i].x[:,3])
# print('contador: {}\t valorMax: {}\t minimo: {}'.format(contador,valorMax,minimo))
# input()
#########################

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

# Creacion del modelo
entradaMetadatos = Input(shape=(3,))
embedding = Embedding(43,8)(entradaMetadatos)
capa2d =Flatten()(embedding)
densa = Dense(bloquesLSTM,activation='tanh')(capa2d)

entradaRecurrente = Input(shape=(sizeSeq,1))
lstm1 = LSTM(bloquesLSTM,unroll=True, return_sequences = True)(entradaRecurrente)
lstm2 = LSTM(bloquesLSTM,unroll=True)(lstm1)

concatenacion = concatenate([densa, lstm2])
tanh = Dense(2*bloquesLSTM,activation="tanh")(concatenacion)
salida = Dense(1)(tanh)

optimizer = Adam(lr = tasaApendizaje)
modelo = Model(inputs=[entradaMetadatos, entradaRecurrente], outputs=salida)
modelo.compile(loss = 'mean_squared_error', optimizer = optimizer)


# Entrenamiento 
# 	- Realiza 1 backpropagation por secuencia
nModelo = 1
csv_logger = CSVLogger('../../resultados/errores/entrenamientoHeadway{0}.tsv'.format(numeroEntrenamiento), append=True, separator='\t')
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
				salida_secuencia = numpy.array( [ train[j].y[k+sizeSeq] ] )
			else:
				entrada_metadatos = numpy.append(entrada_metadatos , [ train[j].x[0][k+sizeSeq][0:3] ] , axis = 0)
				secuenciaTemp = train[j].x[:,k:k+sizeSeq,3:4]
				entrada_recurrente = numpy.append(entrada_recurrente , secuenciaTemp , axis = 0)
				salida_secuencia = numpy.append(salida_secuencia , [ train[j].y[ k + sizeSeq ] ] , axis = 0)
		
		print(bcolors.ENDC+'Entrenamiento red Headway (modelo 2) {}'.format(numeroEntrenamiento))
		print(bcolors.HEADER + 'Epoca {0}/{1}'.format(i+1,epocas)+bcolors.ENDC)
		print(bcolors.WARNING + 'Par de viajes {0}/{1}'.format(j+1,len(train))+bcolors.ENDC+bcolors.OKBLUE)
		modelo.fit([entrada_metadatos, entrada_recurrente], salida_secuencia, epochs = 1, batch_size = 1, verbose = 1, shuffle = False, callbacks=[csv_logger])
		modelo.reset_states()
		gc.collect()
		cls()
	
	# Guarda el modelo al final de cada epoca
	modelo.save('../../resultados/modelos/modeloHeadway{0}e{1}.h5'.format(numeroEntrenamiento,nModelo))
	nModelo = nModelo + 1
