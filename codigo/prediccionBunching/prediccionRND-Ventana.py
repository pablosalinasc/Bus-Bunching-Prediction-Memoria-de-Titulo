#####################################################
#	Algoritmo de prediccion de bunching usando RND	#
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

features = 4
secuencia = 10
paraderos = 43

fechaVentana = '20161205'
inicioVentana = 8*3600
finVentana = 8.5*3600

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
	def __init__(self,id_viaje,ultimoParadero,inicioViaje,finViaje):
		self.id = id_viaje;
		self.ultimoParadero = ultimoParadero
		self.inicioViaje = inicioViaje;
		self.finViaje = finViaje;
		self.x = numpy.array([]);
		self.y = numpy.array([]);

# Fija la semilla del random
# numpy.random.seed(7)
numpy.set_printoptions(precision=3, suppress=False)

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
dataframe = read_csv('../../datos/datasetDefinitivoGPSv4.tsv', engine = 'python', sep = "\t")
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

# modelo = load_model('../../resultados/modelos/Distancias/modeloGPS11/modeloGPS001e51.h5')
modelo = load_model('../../resultados/modelos/Distancias/modeloGPS10/modeloGPS001e6.h5')

# Se hacen las predicciones para cada viaje
for i in range(len(dataset)):
	
	distanciasAcumulada = [0.0]
	for j in range(secuencia):
		distanciasAcumulada = distanciasAcumulada + dataset[i].y[j] 
		ultimoParaderoVisitado = int(dataset[i].x[0][j][2])
		
		if j ==0:
			dataset[i].distanciasReales = numpy.array([[distanciasAcumulada[0],ultimoParaderoVisitado]])
			dataset[i].distanciasPrediccion = numpy.array([[distanciasAcumulada[0],ultimoParaderoVisitado]])
		else:
			dataset[i].distanciasReales = numpy.append(dataset[i].distanciasReales,[[distanciasAcumulada[0],ultimoParaderoVisitado]],axis=0)
			dataset[i].distanciasPrediccion = numpy.append(dataset[i].distanciasPrediccion,[[distanciasAcumulada[0],ultimoParaderoVisitado]],axis=0)

	distanciaParaderosPrediccion = numpy.array([])

	for k in range(len(dataset[i].x[0])-secuencia):
		# Si aun no se llega al paradero limite de la ventana, se predice avanzando en paraderos consecutivos
		if dataset[i].x[0][k+secuencia][2] < dataset[i].ultimoParadero:
			entrada_metadatos = numpy.array( [ dataset[i].x[0][k+secuencia][0:3] ] )
			entrada_recurrente = numpy.array( dataset[i].x[:,k:k+secuencia,3:4] )
			salida_secuencia = numpy.array( [ dataset[i].y[k+secuencia] ] )

			ultimoParaderoVisitado = dataset[i].x[0][k+secuencia][2]

			dataset[i].distanciasReales = numpy.append(dataset[i].distanciasReales, [[distanciasAcumulada[0] + salida_secuencia[0][0], ultimoParaderoVisitado]], axis=0)
			
			prediccionTemp = modelo.predict([entrada_metadatos, entrada_recurrente], batch_size = 1, verbose = 0)

			predicionDenormalizada = prediccionTemp[0][0]*1.2 + distanciasAcumulada[0]
			
			dataset[i].distanciasPrediccion = numpy.append(dataset[i].distanciasPrediccion, [[predicionDenormalizada,ultimoParaderoVisitado]], axis=0)
			
			distanciasAcumulada = distanciasAcumulada + salida_secuencia[0][0]
			
		else:
			# Solo se almacenan las distancias reales
			distanciasAcumulada = distanciasAcumulada + dataset[i].y[k+secuencia]
			dataset[i].distanciasReales = numpy.append(dataset[i].distanciasReales, [[distanciasAcumulada[0],dataset[i].x[0][k+secuencia][2]]], axis=0)
			
			#En caso que se avance al siguiente paradero del recorrido
			if ultimoParaderoVisitado < dataset[i].x[0][k+secuencia][2]:
				distanciaParaderosPrediccion = numpy.append(distanciaParaderosPrediccion, [(dataset[i].distanciasReales[len(dataset[i].distanciasReales)-2][0] + dataset[i].distanciasReales[len(dataset[i].distanciasReales)-1][0])/2.0], axis = 0)
				
			ultimoParaderoVisitado = dataset[i].x[0][k+secuencia][2]

	while dataset[i].distanciasPrediccion[len(dataset[i].distanciasPrediccion)-1][0] < dataset[i].distanciasReales[len(dataset[i].distanciasReales)-1][0]:

		# Se elige el id del ultimo paradero, segun las distncias capturadas de los datos de reales
		paraderoActual = int(dataset[i].ultimoParadero)
		revisado = False
		for j in range(len(distanciaParaderosPrediccion)):
			if distanciaParaderosPrediccion[j] > dataset[i].distanciasPrediccion[len(dataset[i].distanciasPrediccion)-1][0]:
				paraderoActual = int(dataset[i].ultimoParadero + j)
				revisado = True
				break

		if revisado == False:
			paraderoActual = int(dataset[i].distanciasPrediccion[len(dataset[i].distanciasPrediccion)-1][1])

		# El paradero correspondiete a la ultima posicion
		entrada_metadatos = numpy.array( [[ dataset[i].x[0][0][0], dataset[i].x[0][0][1], paraderoActual ]] )
		entrada_recurrente = numpy.append(entrada_recurrente[0][1:secuencia], [(dataset[i].distanciasPrediccion[len(dataset[i].distanciasPrediccion)-1][0]-dataset[i].distanciasPrediccion[len(dataset[i].distanciasPrediccion)-2][0])/1.2] )
		entrada_recurrente = numpy.reshape(entrada_recurrente, (1,secuencia, 1))

		prediccionTemp = modelo.predict([entrada_metadatos, entrada_recurrente], batch_size = 1, verbose = 0)

		predicionDenormalizada = prediccionTemp[0]*1.2 + dataset[i].distanciasPrediccion[len(dataset[i].distanciasPrediccion)-1][0] 
		dataset[i].distanciasPrediccion = numpy.append(dataset[i].distanciasPrediccion,[[predicionDenormalizada[0],paraderoActual]],axis=0)

	# Se extrae el tiempo de llegada a cada paradero
	dataset[i].TTreales = numpy.array([dataset[i].inicioViaje])
	dataset[i].TTprediccion = numpy.array([dataset[i].inicioViaje])

	paraderoActual = dataset[i].distanciasReales[0][1]-1
	for j in range(len(dataset[i].distanciasReales)):
		if dataset[i].distanciasReales[j][1] > paraderoActual:
			dataset[i].TTreales = numpy.append(dataset[i].TTreales,[dataset[i].inicioViaje+300+30*j], axis = 0)
		paraderoActual = dataset[i].distanciasReales[j][1]
	
	dataset[i].TTreales = numpy.append(dataset[i].TTreales,[dataset[i].inicioViaje+300+30*(len(dataset[i].distanciasReales)-1)],axis=0)

	paraderoActual = dataset[i].distanciasPrediccion[0][1]-1
	for j in range(len(dataset[i].distanciasPrediccion)):
		if dataset[i].distanciasPrediccion[j][1] > paraderoActual:
			dataset[i].TTprediccion = numpy.append(dataset[i].TTprediccion,[dataset[i].inicioViaje+300+30*j], axis = 0)
		paraderoActual = dataset[i].distanciasPrediccion[j][1]

	dataset[i].TTprediccion = numpy.append(dataset[i].TTprediccion,[dataset[i].inicioViaje+300+30*(len(dataset[i].distanciasPrediccion)-1)],axis=0)

	#print('Viaje {}:'.format(dataset[i].id))
	#print('{}'.format(dataset[i].distanciasReales))
	#print('{}'.format(dataset[i].distanciasPrediccion))
	#print('{}'.format(dataset[i].TTreales))
	#print('{}'.format(dataset[i].TTprediccion))

print('ID par\tparadero\theadway prediccion\theadway real\tbunching prediccion\tbunching real')

for i in range(len(dataset)):
	for j in [1+i,2+i]:
		if j < len(dataset):
			# Solo se toman secuencias de headway con valores iniciales menores a 3600 
			if (dataset[j].inicioViaje - dataset[i].inicioViaje) < 3600:
				# Restricciones el ultimo paradero de cada para es el minimo entre ambos
				print("")
				ultimoParadero = min(dataset[j].ultimoParadero,dataset[i].ultimoParadero)
				idPar = '{0:13.0f}{1:3.0f}'.format(dataset[i].id,round(dataset[j].id/1000)*1000-dataset[j].id)
				for k in range(min(len(dataset[i].TTreales),len(dataset[j].TTreales),len(dataset[i].TTprediccion),len(dataset[j].TTprediccion))):
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

						print(bcolors.OKGREEN+'{0}\t{1}\t{2:4.0f}\t{3:4.0f}\t{4}\t{5}'.format(idPar,k+1,round(headwayPrediccion),headwayReal,bunchingPrediccion,bunchingReal)+bcolors.ENDC)
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
