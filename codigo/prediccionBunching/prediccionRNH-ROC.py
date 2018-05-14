#####################################################
#	Algoritmo de prediccion de bunching usando RNH	#
#	exportando valores de clasificacion para ROC	#
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
secuencia = 4

fechasVentana = ['20161107','20161108','20161109','20161110','20161111','20161114','20161115','20161116','20161117','20161118','20161121','20161122','20161123','20161124','20161125','20161128','20161129','20161130','20161201','20161202','20161205','20161206','20161207','20161208','20161209']
inicioPruebas = 7
finPruebas = 21

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
	def __init__(self,id_viaje,limiteVentana):
		self.id = id_viaje;
		self.limiteVentana = limiteVentana;
		self.predicciones = numpy.array([]);
		self.x=numpy.array([]);
		self.y=numpy.array([]);

# Carga el dataset de headway
dataframe = read_csv('../../datos/datasetBrutos/datasetBrutoI09Modelo2.tsv', engine = 'python', sep = "\t")
datasetBruto = dataframe.values
datasetBruto = datasetBruto.astype(float)

# Carga el dataset de headway
dataframe = read_csv('../../datos/datasetDefinitivoModelo2v4.tsv', engine = 'python', sep = "\t")
datasetDefinitivo = dataframe.values
datasetDefinitivo = datasetDefinitivo.astype(float)
del(dataframe)
gc.collect()

# precisionesDetalle = numpy.zeros(len(range(inicioPruebas*3600, finPruebas*3600, 1800))*len(fechasVentana))
# precisionesBinarias = numpy.zeros(len(range(inicioPruebas*3600, finPruebas*3600, 1800))*len(fechasVentana))
# sensibilidades = numpy.zeros(len(range(inicioPruebas*3600, finPruebas*3600, 1800))*len(fechasVentana))
# especificidades = numpy.zeros(len(range(inicioPruebas*3600, finPruebas*3600, 1800))*len(fechasVentana))

# f = open("../../resultados/pruebas/prediccionBunchingHeadway.tsv","w")
# f.write("Fecha\tHora fin\tCantidad de pares de viajes\tPrecision binaria\tPrecision caso a caso\n")

f2 = open("../../resultados/ROC/datosROCheadway2.tsv","w")

for dia in fechasVentana:
	for finVentana in range(inicioPruebas*3600, finPruebas*3600, 1800):

		print("Ventana:\n- Dia: {}\n- finVentana: {}\n".format(dia,finVentana))

		# Almacena los id de viajes que esten dentro de la ventana
		lista_ids = []
		id_viaje = 0
		print('ID viaje 	INICIO 		FIN 	UltimoParadero')
		for i in range(len(datasetBruto)):
			if datasetBruto[i][2] == 1:
				if id_viaje != 0:
					finViaje = datasetBruto[i-1][3]
					if finViaje > finVentana and inicioViaje < finVentana and '{0:12.0f}'.format(id_viaje).find(dia) > -1:
						# Se revisa el paradero en que termina la ventana
						j = 2
						while datasetBruto[i-j][3] > finVentana:
							j = j + 1

						ultimoParaderoVentana = datasetBruto[i-j][2]

						print('{0:11.0f}	{1:11.0f}	{2:11.0f}	{3:2.0f}'.format(id_viaje, inicioViaje, finViaje,ultimoParaderoVentana))
						lista_ids.append(['{0:11.0f}'.format(id_viaje),ultimoParaderoVentana])

				id_viaje = datasetBruto[i][0]
				inicioViaje = datasetBruto[i][3]

		if len(lista_ids)>1:

			# Toma los viajes por cada id distinto (columna 0) e ingresarlo a otro arreglo 
			dataset_viajes = []
			for i in range(len(datasetDefinitivo)):
				id_lectura = datasetDefinitivo[i][0]
				#En caso que el dataset_viaje tenga datos
				idviaje1 = '{}{}'.format('{0:8.0f}'.format(id_lectura)[:9],'{0:16.0f}'.format(id_lectura)[11:13])
				idviaje2 = '{}{}'.format('{0:8.0f}'.format(id_lectura)[:9],'{0:16.0f}'.format(id_lectura)[14:16])
				
				check1 = False
				check2 = False
				idLimiteVentana1 = -1
				idLimiteVentana2 = -1

				for j in range(len(lista_ids)):
					#print("\'{}\' \'{}\' \'{}\'".format(lista_ids[j][:11],idviaje1[:11],idviaje2[:11]))
					if idviaje1[:11] == lista_ids[j][0][:11]:
						idLimiteVentana1 = lista_ids[j][1]-1
						check1 = True
					if idviaje2[:11] == lista_ids[j][0][:11]:
						idLimiteVentana2 = lista_ids[j][1]-1
						check2 = True
				
				# Si ambos viajes estan en la ventana
				if check1 and check2:

					indiceVentana = min([idLimiteVentana2,idLimiteVentana1])
					#print('{} {}'.format(idviaje1,idviaje2))
					if len(dataset_viajes) > 0:
						#Agrega puntos al mismo viaje
						if dataset_viajes[len(dataset_viajes)-1].id == id_lectura:
							dataset_viajes[len(dataset_viajes)-1].x = numpy.append(dataset_viajes[len(dataset_viajes)-1].x,[datasetDefinitivo[i][1:5]],axis=0)
							dataset_viajes[len(dataset_viajes)-1].y = numpy.append(dataset_viajes[len(dataset_viajes)-1].y,[datasetDefinitivo[i][5]],axis=0)
						#Agrega un nuevo viaje
						else:
							dataset_viajes.append(Viaje(id_lectura,indiceVentana))
							dataset_viajes[len(dataset_viajes)-1].x = numpy.array([datasetDefinitivo[i][1:5]])
							dataset_viajes[len(dataset_viajes)-1].y = numpy.array([datasetDefinitivo[i][5]])
					#En caso que sea el primer viaje
					else:
						dataset_viajes.append(Viaje(id_lectura,indiceVentana))
						dataset_viajes[len(dataset_viajes)-1].x = numpy.array([datasetDefinitivo[i][1:5]])
						dataset_viajes[len(dataset_viajes)-1].y = numpy.array([datasetDefinitivo[i][5]])

			dataset = dataset_viajes
			del(dataset_viajes)
			gc.collect()

			for i in range(len(dataset)):
				dataset[i].x = normalizarX(dataset[i].x)
				dataset[i].x = numpy.reshape(dataset[i].x, (1,len(dataset[i].x), len(dataset[i].x[0])))
			#	dataset[i].y = normalizarY(dataset[i].y)
				dataset[i].y = numpy.reshape(dataset[i].y, (len(dataset[i].y),1))

			#print('\nCantidad de pares de viajes: {}'.format(len(dataset)))

			# Importa el modelo de la red predictora
			modelo = load_model('../../resultados/modelos/Headway/modeloHeadway10/modeloHeadway002e51.h5')

			# Debe preparar secuencias hasta que la prediccion ya no este dentro de la ventana
			# Las nuevas secuencias se preparan en base a las predicciones anteriores

			# precisionBinaria = 0.0
			# precisionDetalle = float(0.0)
			# sensibilidad = 0.0
			# especificidad = 0.0

			cantidadParesViajes = 0.0

			#print('ID par\tparadero\tsalida prediccion\tsalida real\theadway prediccion\theadway real\tbunching prediccion\tbunching real')
			for i in range(len(dataset)):
				# SE captura el headway inicial para determinar el bunching
				headwayInicial = dataset[i].x[0][0][3]*3600				
				#print(headwayInicial)

				# tempPrecisionPrediccion = 0
				# tempPrecisionReal = 0
				# tempPrecisionDetalle = 0 
				# tempSensibilidad = 0.0
				# tempEspecificidad = 0.0
				# cantBunching = 0.0
				# cantNoBunching = 0.0

				if dataset[i].limiteVentana < len(dataset[i].x[0])-1 and dataset[i].limiteVentana > secuencia:
					cantidadParesViajes = cantidadParesViajes + 1.0
					ultimaSecuencia = numpy.array([])
					for j in range(len(dataset[i].x[0])-secuencia):
						# En caso que se encuentre dentro de la ventana
						if (j+secuencia) <= dataset[i].limiteVentana:
							# Se alimenta a la red para realizar predicciones
							entrada_metadatos = numpy.array( [ dataset[i].x[0][j+secuencia][0:3] ] )
							entrada_recurrente = numpy.array( dataset[i].x[:,j:j+secuencia,3:4] )
							salida_secuencia = numpy.array( [ dataset[i].y[j+secuencia] ] )
							
							prediccionesTemp = modelo.predict([entrada_metadatos,entrada_recurrente], batch_size = 1, verbose = 0)

							prediccion = prediccionesTemp[0][0]*600
							headwayPrediccion = prediccionesTemp[0][0]*600+dataset[i].x[0][j+secuencia][3]*3600
							headwayReal = salida_secuencia[0][0]+dataset[i].x[0][j+secuencia][3]*3600

							#Revisa si hay bunching o no
							bunchingPrediccion = False
							bunchingReal = False
							if abs(headwayPrediccion) < 0.25*headwayInicial:
								bunchingPrediccion = True
							
							if abs(headwayReal) < 0.25*headwayInicial:
								bunchingReal = True

							#print(bcolors.OKGREEN+'{0:13.0f}\t{1}\t{2:4.0f}\t{3:4.0f}\t{4}\t{5}'.format(dataset[i].id,j+1+secuencia,headwayReal,headwayReal,bunchingReal,bunchingReal)+bcolors.ENDC)
							
							if j == 0:
								dataset[i].predicciones = numpy.array([headwayReal])
								# dataset[i].predicciones = prediccionesTemp[0]*600+dataset[i].x[0][j+secuencia][3]*3600
							else:
								dataset[i].predicciones = numpy.append(dataset[i].predicciones,[headwayReal],axis=0)
								# dataset[i].predicciones = numpy.append(dataset[i].predicciones,prediccionesTemp[0]*600+dataset[i].x[0][j+secuencia][3]*3600,axis=0)

							if (j+secuencia) == dataset[i].limiteVentana:
								# Debe generar secuencia siguiente
								ultimaSecuencia = numpy.array( dataset[i].x[:,j+1:j+secuencia+1,3:4] )
								# ultimaSecuencia = numpy.append(ultimaSecuencia,[prediccionesTemp],axis=1)
								#print(ultimaSecuencia)

						# En caso se tener que utilizar predicciones anteriores para generar las restantes
						else:
							entrada_metadatos = numpy.array( [[ j+1+secuencia, dataset[i].x[0][0][1],dataset[i].x[0][0][2] ]] )
							
							#print(entrada_metadatos)

							entrada_recurrente = ultimaSecuencia

							salida_secuencia = numpy.array( [ dataset[i].y[j+secuencia] ] )

							siguientePrediccion = modelo.predict([entrada_metadatos,entrada_recurrente], batch_size = 1, verbose = 0)

							prediccion = siguientePrediccion[0][0]*600
							headwayPrediccion = siguientePrediccion[0][0]*600+dataset[i].predicciones[len(dataset[i].predicciones)-1]
							headwayReal = salida_secuencia[0][0]+dataset[i].x[0][j+secuencia][3]*3600
							
							#Revisa si hay bunching o no
							if abs(headwayPrediccion) < 0.25*headwayInicial:
								bunchingPrediccion = True
								# tempPrecisionPrediccion = 1

							
							if abs(headwayReal) < 0.25*headwayInicial:
								bunchingReal = True
								# tempPrecisionReal = 1
									
							# if bunchingReal == bunchingPrediccion:
							# 	tempPrecisionDetalle = tempPrecisionDetalle + 1

							# if bunchingReal== True and bunchingReal == bunchingPrediccion:
							# 	tempSensibilidad = tempSensibilidad + 1
							# 	cantBunching = cantBunching + 1


							# if bunchingReal== False and bunchingReal == bunchingPrediccion:
							# 	tempEspecificidad = tempEspecificidad + 1
							# 	cantNoBunching = cantNoBunching + 1

							f2.write('{0:4.8f}\t{1:4.8f}\t{2}\t{3}\n'.format(round(abs(headwayPrediccion))/headwayInicial,abs(headwayReal)/headwayInicial,bunchingPrediccion*1,bunchingReal*1))
							
							dataset[i].predicciones = numpy.append(dataset[i].predicciones,[headwayPrediccion],axis=0)
							
							if j < (len(dataset[i].x[0]) - 1):
								ultimaSecuencia = numpy.array([ultimaSecuencia[0][1:secuencia]])
								ultimaSecuencia = numpy.append(ultimaSecuencia,[siguientePrediccion],axis=1)
								#print(ultimaSecuencia)
					
					# # Despues de recorrer el viaje completo calcula las precisiones
					# precisionDetalle = precisionDetalle + float(tempPrecisionDetalle)/float((len(dataset[i].x[0]))-dataset[i].limiteVentana-1)
					# if cantBunching > 0:
					# 	sensibilidad = sensibilidad + float(tempSensibilidad)/float(cantBunching)
					# if cantNoBunching > 0:
					# 	especificidad = especificidad + float(tempEspecificidad)/float(cantNoBunching)

					# aciertoBinario = 0
					# if tempPrecisionReal == tempPrecisionPrediccion:
					# 	precisionBinaria = precisionBinaria + 1
					# 	aciertoBinario = 1 

					# print("\nPar {}\nAcierto existencia de bunching: {}\nAciertos binarios acumulados: {}\nParaderos acertados: {}\nPrecision paraderos: {}\nPrecisionAcumulada: {}\n".format(dataset[i].id,aciertoBinario,precisionBinaria,tempPrecisionDetalle,float(tempPrecisionDetalle)/float((len(dataset[i].x[0]))-dataset[i].limiteVentana-1),precisionDetalle))

			# if cantidadParesViajes > 0:
			# 	sensibilidad = float(sensibilidad)/float(cantidadParesViajes)
			# 	especificidad = float(especificidad)/float(cantidadParesViajes)
			# 	precisionDetalle = float(precisionDetalle)/float(cantidadParesViajes)
			# 	precisionBinaria = precisionBinaria/cantidadParesViajes

			# 	print(bcolors.OKBLUE+"\nPrecisionBinaria: {0:3.4f}\tPrecisionDetalle: {1:3.4f}\tSensibilidad: {2:3.4f}\tEspecificidad: {3:3.4f}\n".format(precisionBinaria*100, precisionDetalle*100,sensibilidad*100,especificidad*100)+bcolors.ENDC)
			# 	f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(dia,finVentana,cantidadParesViajes,precisionBinaria,precisionDetalle,sensibilidad,especificidad))

			# 	for p in range(len(precisionesBinarias)):
			# 		if precisionesBinarias[p] == 0.0:
			# 			precisionesBinarias[p] = precisionBinaria
			# 			break

			# 	for p in range(len(precisionesDetalle)):
			# 		if precisionesDetalle[p] == 0.0:
			# 			precisionesDetalle[p] = precisionDetalle
			# 			break

			# 	for p in range(len(sensibilidades)):
			# 		if sensibilidades[p] == 0.0:
			# 			sensibilidades[p] = sensibilidad
			# 			break

			# 	for p in range(len(especificidades)):
			# 		if especificidades[p] == 0.0:
			# 			especificidades[p] = especificidad
			# 			break


# promedioPrecisionBinaria = 0.0
# contador = 0.0
# for i in range(len(precisionesBinarias)):
# 	if precisionesBinarias[i] != 0:
# 		contador = contador + 1
# 		promedioPrecisionBinaria = promedioPrecisionBinaria + precisionesBinarias[i]

# promedioPrecisionBinaria = promedioPrecisionBinaria / contador

# promedioPrecisionDetalle = 0.0
# contador = 0.0
# for i in range(len(precisionesDetalle)):
# 	if precisionesDetalle[i] != 0:
# 		contador = contador + 1
# 		promedioPrecisionDetalle = promedioPrecisionDetalle + precisionesDetalle[i]

# promedioPrecisionDetalle = promedioPrecisionDetalle / contador


# promedioSensibilidad = 0.0
# contador = 0.0
# for i in range(len(sensibilidades)):
# 	if sensibilidades[i] != 0:
# 		contador = contador + 1
# 		promedioSensibilidad = promedioSensibilidad + sensibilidades[i]

# promedioSensibilidad = promedioSensibilidad / contador


# promedioEspecificidad = 0.0
# contador = 0.0
# for i in range(len(especificidades)):
# 	if especificidades[i] != 0:
# 		contador = contador + 1
# 		promedioEspecificidad = promedioEspecificidad + especificidades[i]

# promedioEspecificidad = promedioEspecificidad / contador

# print(bcolors.FAIL+"\nPrecision Binaria: {}\tPrecision Detalle: {}\n".format(promedioPrecisionBinaria,promedioPrecisionDetalle,promedioSensibilidad,promedioEspecificidad)+bcolors.ENDC)
# f.write("Promedio\t\t\t{}\t{}\t{}\t{}\n".format(promedioPrecisionBinaria,promedioPrecisionDetalle,promedioSensibilidad,promedioEspecificidad))
# f.close()
f2.close()