#####################################################
#	Algoritmo de prediccion de bunching usando RNP	#
#	para una ventana de tiempo especifica			#
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
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error

featuresParaderos = 4
featuresGPS = 5
secuenciaGPS = 10
paraderos = 43

fechaVentana = '20161205'
inicioVentana = 8*3600
finVentana = 8.5*3600

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
	def __init__(self,id_viaje,ultimoParadero,inicioViaje,finViaje):
		self.id = id_viaje;
		self.ultimoParadero = ultimoParadero
		self.inicioViaje = inicioViaje;
		self.finViaje = finViaje;
		self.x = numpy.array([]);
		self.y = numpy.array([]);

# Fija la semilla del random
# numpy.random.seed(7)

# Carga el dataset de headway
dataframe = read_csv('../../datos/datasetBrutos/datasetBrutoI09Modelo2.tsv', engine = 'python', sep = "\t")
dataset = dataframe.values
dataset = dataset.astype(float)
del(dataframe)
gc.collect()

# Almacena los id de viajes que esten dentro de la ventana
lista_ids = []
id_viaje = 0
print('ID viaje 	INICIO 		FIN 	UltimoParadero')
for i in range(len(dataset)):
	if dataset[i][2] == 1:
		if id_viaje != 0:
			finViaje = dataset[i-1][3]
			if finViaje > finVentana and inicioViaje < finVentana and '{0:12.0f}'.format(id_viaje).find(fechaVentana) > -1:
				# Se revisa el paradero en que termina la ventana
				j = 2
				while dataset[i-j][3] > finVentana:
					j = j + 1

				ultimoParaderoVentana = dataset[i-j][2]

				print('{0:11.0f}	{1:11.0f}	{2:11.0f}	{3:2.0f}'.format(id_viaje, inicioViaje, finViaje,ultimoParaderoVentana))
				lista_ids.append(['{0:11.0f}'.format(id_viaje),ultimoParaderoVentana,inicioViaje,finViaje])

		id_viaje = dataset[i][0]
		inicioViaje = dataset[i][3]

# Carga el dataset de paraderos
dataframe = read_csv('../../datos/datasetDefinitivoParaderosI09.tsv', engine = 'python', sep = "\t")
dataset = dataframe.values
dataset = dataset.astype(float)

# Toma los viajes por cada id distinto (columna 0) e ingresarlo a otro arreglo 
dataset_viajes = []
for i in range(len(dataset)):
	id_lectura = dataset[i][0]
	#En caso que el dataset_viaje tenga datos
	encontrado = False
	for j in range(len(lista_ids)):
		if '{0:11.0f}'.format(id_lectura) == lista_ids[j][0]:
			encontrado = True
			ultimoParadero = lista_ids[j][1]
			inicioViaje = lista_ids[j][2]
			finViaje = lista_ids[j][3]
			break

	if encontrado:
		if len(dataset_viajes) > 0:
			#Agrega puntos al mismo viaje
			if dataset_viajes[len(dataset_viajes)-1].id == id_lectura:
				dataset_viajes[len(dataset_viajes)-1].x = numpy.append(dataset_viajes[len(dataset_viajes)-1].x,[dataset[i][1:5]],axis=0)
				dataset_viajes[len(dataset_viajes)-1].y = numpy.append(dataset_viajes[len(dataset_viajes)-1].y,[dataset[i][5]],axis=0)
			#Agrega un nuevo viaje
			else:
				dataset_viajes.append(Viaje(id_lectura,ultimoParadero,inicioViaje,finViaje))
				dataset_viajes[len(dataset_viajes)-1].x = numpy.array([dataset[i][1:5]])
				dataset_viajes[len(dataset_viajes)-1].y = numpy.array([dataset[i][5]])
		#En caso que sea el primer viaje
		else:
			dataset_viajes.append(Viaje(id_lectura,ultimoParadero,inicioViaje,finViaje))
			dataset_viajes[len(dataset_viajes)-1].x = numpy.array([dataset[i][1:5]])
			dataset_viajes[len(dataset_viajes)-1].y = numpy.array([dataset[i][5]])

dataset = dataset_viajes
del(dataset_viajes)
gc.collect()

for i in range(len(dataset)):
	dataset[i].x = normalizarX(dataset[i].x)
	dataset[i].x = numpy.reshape(dataset[i].x, (1,len(dataset[i].x), len(dataset[i].x[0])))
#	dataset[i].y = normalizarY(dataset[i].y)
	dataset[i].y = numpy.reshape(dataset[i].y, (len(dataset[i].y),1))

print('\nCantidad de viajes en dataset paraderos: {}'.format(len(dataset)))

modelo = load_model('../../resultados/modelos/Paraderos/modeloParaderos15/modeloParaderos002e50.h5')

# Se hacen las predicciones para cada viaje

numpy.set_printoptions(precision=3, suppress=False)

print('ID par\tparadero\theadway prediccion\theadway real\tbunching prediccion\tbunching real')

for i in range(len(dataset)):
#	print("")
#	print("ID: {}\nUltimoParaderoVentana: {}".format(dataset[i].id,dataset[i].ultimoParadero))
	
	dataset[i].TTreales = numpy.array([dataset[i].inicioViaje])
	dataset[i].TTprediccion = numpy.array([dataset[i].inicioViaje])
	
	ultimoTiempoVentanaPred = dataset[i].inicioViaje
	ultimoTiempoVentanaReal = dataset[i].inicioViaje
	# j es el paradero a consultar
	for j in range(paraderos-1):
		# Si aun no se llega al paradero limite de la ventana, se predice avanzando en paraderos consecutivos
		if j+1 < dataset[i].ultimoParadero:
			# busca la consulta correcta segun los datos que se poseen
#			print('busca para el paradero {}'.format(j+2))
			for k in range(len(dataset[i].x[0])):
				if round(43*(dataset[i].x[0][k][1]+1)/2) == j+1 and round(43*(dataset[i].x[0][k][3]+1)/2) == j+2:
					dataset[i].TTreales = numpy.append(dataset[i].TTreales, dataset[i].y[k] + dataset[i].TTreales[len(dataset[i].TTreales)-1],axis=0)
					prediccionTemp = modelo.predict(numpy.array([dataset[i].x[0][k]]), batch_size = 1, verbose = 0)
					valorDenormalizado = denormalizarY(prediccionTemp) + dataset[i].TTprediccion[len(dataset[i].TTprediccion)-1] 
					# dataset[i].TTprediccion = numpy.append(dataset[i].TTprediccion,valorDenormalizado,axis=0)
					dataset[i].TTprediccion = numpy.append(dataset[i].TTprediccion, [dataset[i].TTreales[len(dataset[i].TTreales)-1]],axis=0)
					# print('if: Agrega una prediccion para el paradero {}'.format(j+2))
					# ultimoTiempoVentanaPred = valorDenormalizado[0]
					ultimoTiempoVentanaPred = dataset[i].TTreales[len(dataset[i].TTreales)-1]
					ultimoTiempoVentanaReal = dataset[i].TTreales[len(dataset[i].TTreales)-1]
					break
		else:
			# En este caso el paradero desde donde se hacen las consultas es siempre el ultimo de la ventana
#			print('busca para el paradero {}'.format(j+2))
			for k in range(len(dataset[i].x[0])):
				if round(43*(dataset[i].x[0][k][1]+1)/2) == dataset[i].ultimoParadero and round(43*(dataset[i].x[0][k][3]+1)/2) == j+2:
					dataset[i].TTreales = numpy.append(dataset[i].TTreales, dataset[i].y[k] + ultimoTiempoVentanaReal,axis=0)
					entrada = numpy.array([[dataset[i].x[0][k][0],dataset[i].x[0][k][1],float(((ultimoTiempoVentanaPred - dataset[i].inicioViaje) / 4800)*2-1),dataset[i].x[0][k][3]]])
					prediccionTemp = modelo.predict(entrada, batch_size = 1, verbose = 0)
					valorDenormalizado = denormalizarY(prediccionTemp) + ultimoTiempoVentanaPred
					dataset[i].TTprediccion = numpy.append(dataset[i].TTprediccion,valorDenormalizado,axis=0)
					# print('else: Agrega una prediccion para el paradero {}'.format(j+2))
					break

	print(dataset[i].TTreales)
	print(dataset[i].TTprediccion)

# Una vez realizada las predicciones se realiza la combinatoria de los viajes para calcular las secuencias de headway
input()


for i in range(len(dataset)):
	for j in [1+i,2+i]:
		if j < len(dataset):
			# Solo se toman secuencias de headway con valores iniciales menores a 3600 
			if (dataset[j].inicioViaje - dataset[i].inicioViaje) < 3600:
				# Restricciones el ultimo paradero de cada para es el minimo entre ambos
				print("")
				ultimoParadero = min(dataset[j].ultimoParadero,dataset[i].ultimoParadero)
				idPar = '{0:13.0f}{1:3.0f}'.format(dataset[i].id,round(dataset[j].id/1000)*1000-dataset[j].id)
				for k in range(len(dataset[i].TTreales)):
					if k <= ultimoParadero:
						headwayInicial = dataset[j].inicioViaje - dataset[i].inicioViaje
						headwayPrediccion = dataset[j].TTprediccion[k]-dataset[i].TTprediccion[k]
						headwayReal = dataset[j].TTreales[k]-dataset[i].TTreales[k]
									
						#Revisa si hay bunching o no
						bunchingPrediccion = False
						bunchingReal = False
						if abs(headwayPrediccion) < 0.25*headwayInicial:
							bunchingPrediccion = True
						
						if abs(headwayReal) < 0.25*headwayInicial:
							bunchingReal = True

						print(bcolors.OKGREEN+'{0}\t{1}\t{2:4.0f}\t{3:4.0f}\t{4}\t{5}'.format(idPar,k+1,   round(headwayPrediccion),headwayReal,bunchingPrediccion,bunchingReal)+bcolors.ENDC)
					else:
						headwayInicial = dataset[j].inicioViaje - dataset[i].inicioViaje
						headwayPrediccion = dataset[j].TTprediccion[k]-dataset[i].TTprediccion[k]
						headwayReal = dataset[j].TTreales[k]-dataset[i].TTreales[k]
									
						#Revisa si hay bunching o no
						bunchingPrediccion = False
						bunchingReal = False
						if abs(headwayPrediccion) < 0.25*headwayInicial:
							bunchingPrediccion = True
						
						if abs(headwayReal) < 0.25*headwayInicial:
							bunchingReal = True
						
						print(bcolors.FAIL+'{0}\t{1}\t{2:4.0f}\t{3:4.0f}\t{4}\t{5}'.format(idPar,k+1,round(headwayPrediccion),headwayReal,bunchingPrediccion,bunchingReal)+bcolors.ENDC)
