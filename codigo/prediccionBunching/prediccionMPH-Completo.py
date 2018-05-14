#####################################################
#	Algoritmo de prediccion de bunching				#
#	para el conjunto completo de ventanas de tiempo	#
#	usando MPH										#
#####################################################

import os
import gc
import numpy
import math
import datetime
from pandas import read_csv
from sklearn.preprocessing import *
from sklearn.metrics import *

features = 4
paraderos = 43
porcentajeEntrenamiento = 0.8
cantidadTramos = 7
cantidadDias = 5

fechasVentana = ['20161107','20161108','20161109','20161110','20161111','20161114','20161115','20161116','20161117','20161118','20161121','20161122','20161123','20161124','20161125','20161128','20161129','20161130','20161201','20161202','20161205','20161206','20161207','20161208','20161209']
inicioPruebas = 7
finPruebas = 21

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
	def __init__(self,id_viaje,ultimoParadero,inicioViaje,finViaje):
		self.id = id_viaje;
		self.ultimoParadero = ultimoParadero
		self.inicioViaje = inicioViaje;
		self.finViaje = finViaje;
		self.x = numpy.array([]);
		self.y = numpy.array([]);

#####################################
#### Calcula los viajes promedio ####
#####################################

dataframe = read_csv('../../datos/datasetModeloPromedio.tsv', engine = 'python', sep = "\t", names = ['ID','Dia','Tramo','Paradero','Timestamp'])
dataset = dataframe.values
dataset = dataset.astype(float)

IDs, frecuencias = numpy.unique(dataset[:,0], return_counts=True)
cantidadViajes= len(frecuencias)
print('Cantidad de viajes: {}'.format(cantidadViajes))

viajesPromedio = numpy.zeros((cantidadDias,cantidadTramos,paraderos), dtype=numpy.int)
frecuenciasCasos = numpy.zeros((cantidadDias,cantidadTramos,paraderos), dtype=numpy.int)

for j in range(len(dataset)):
	if dataset[j,3] != 1:
		#Calcula promedios parciales
		viajesPromedio[int(dataset[j,1]-1),int(dataset[j,2]-1),int(dataset[j,3]-1)] = (viajesPromedio[int(dataset[j,1]-1),int(dataset[j,2]-1),int(dataset[j,3]-1)] + (dataset[j,4]-dataset[j-1,4]))
		frecuenciasCasos[int(dataset[j,1]-1),int(dataset[j,2]-1),int(dataset[j,3]-1)] = frecuenciasCasos[int(dataset[j,1]-1),int(dataset[j,2]-1),int(dataset[j,3]-1)] + 1

for i in range(cantidadDias):
	for j in range(cantidadTramos):
		#print('Dia {} Tramo {}\n\tParadero\tTT\n\t1\t{}'.format(i+1,j+1,viajesPromedio[i,j,0]))
		TTacumulado = 0.0
		for k in range(paraderos-1):
			tiempoTemp =  int(viajesPromedio[i,j,k+1] / frecuenciasCasos[i,j,k+1])
			TTacumulado = TTacumulado + tiempoTemp
			viajesPromedio[i,j,k+1] = TTacumulado
			#print('\t{}\t{}'.format(k+2,viajesPromedio[i,j,k+1]))
		#print('')

del(dataset)
gc.collect()

#############################################
#### Carga los dataset que va a utilizar ####
#############################################

# Carga el dataset de headway
dataframe = read_csv('../../datos/datasetBrutos/datasetBrutoI09Modelo2.tsv', engine = 'python', sep = "\t")
datasetBruto = dataframe.values
datasetBruto = datasetBruto.astype(float)

# Carga el dataset de paraderos
dataframe = read_csv('../../datos/datasetDefinitivoParaderosI09.tsv', engine = 'python', sep = "\t")
datasetDefinitivo = dataframe.values
datasetDefinitivo = datasetDefinitivo.astype(float)
del(dataframe)
gc.collect()


precisionesDetalle = numpy.zeros(len(range(inicioPruebas*3600, finPruebas*3600, 1800))*len(fechasVentana))
precisionesBinarias = numpy.zeros(len(range(inicioPruebas*3600, finPruebas*3600, 1800))*len(fechasVentana))
sensibilidades = numpy.zeros(len(range(inicioPruebas*3600, finPruebas*3600, 1800))*len(fechasVentana))
especificidades = numpy.zeros(len(range(inicioPruebas*3600, finPruebas*3600, 1800))*len(fechasVentana))

f = open("../../resultados/pruebas/prediccionBunchingPromedio.tsv","w")
f.write("Fecha\tHora fin\tCantidad de pares de viajes\tPrecision binaria\tPrecision caso a caso\n")

f2 = open("../../resultados/ROC/datosROCpromedio2.tsv","w")

for dia in fechasVentana:
	for finVentana in range(inicioPruebas*3600, finPruebas*3600, 1800):

		######################################################
		#### Extrae los datos de los viajes de la ventana ####
		######################################################

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
						lista_ids.append(['{0:11.0f}'.format(id_viaje),ultimoParaderoVentana,inicioViaje,finViaje])

				id_viaje = datasetBruto[i][0]
				inicioViaje = datasetBruto[i][3]

		##########################################
		#### Extrae los tiempos de los viajes ####
		##########################################

		if len(lista_ids)>1:

			# Toma los viajes por cada id distinto (columna 0) e ingresarlo a otro arreglo 
			dataset_viajes = []
			for i in range(len(datasetDefinitivo)):
				id_lectura = datasetDefinitivo[i][0]
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
							dataset_viajes[len(dataset_viajes)-1].x = numpy.append(dataset_viajes[len(dataset_viajes)-1].x,[datasetDefinitivo[i][1:5]],axis=0)
							dataset_viajes[len(dataset_viajes)-1].y = numpy.append(dataset_viajes[len(dataset_viajes)-1].y,[datasetDefinitivo[i][5]],axis=0)
						#Agrega un nuevo viaje
						else:
							dataset_viajes.append(Viaje(id_lectura,ultimoParadero,inicioViaje,finViaje))
							dataset_viajes[len(dataset_viajes)-1].x = numpy.array([datasetDefinitivo[i][1:5]])
							dataset_viajes[len(dataset_viajes)-1].y = numpy.array([datasetDefinitivo[i][5]])
					#En caso que sea el primer viaje
					else:
						dataset_viajes.append(Viaje(id_lectura,ultimoParadero,inicioViaje,finViaje))
						dataset_viajes[len(dataset_viajes)-1].x = numpy.array([datasetDefinitivo[i][1:5]])
						dataset_viajes[len(dataset_viajes)-1].y = numpy.array([datasetDefinitivo[i][5]])

			dataset = dataset_viajes
			del(dataset_viajes)
			gc.collect()

			print('\nCantidad de viajes en dataset paraderos: {}'.format(len(dataset)))

			# La idea es que las predicciones de los viajes sean proporcionales a la concordancia de los datos que existen con los promedios
			# Si los tiempos en son el doble que el promedio entonces las nuevas predicciones van a ser del doble del promedio tambien

			####################################################
			#### Predice proporcional a los viajes promedio ####
			####################################################
			
			diaSemana = datetime.date(int(dia[0:4]), int(dia[4:6]), int(dia[6:8])).weekday() + 1

			for i in range(len(dataset)):
				#print("")
				#print('ID\tparadero\tTTprediccion\tTTreal')

				dataset[i].TTprediccion = numpy.array([dataset[i].inicioViaje])
				dataset[i].TTreal = numpy.array([dataset[i].inicioViaje])

				proporcion = 1.0
				for j in range(paraderos-1):
					# Se actualiza la proporcion del viaje promedio contra el viaje real
					if j+1 < dataset[i].ultimoParadero:
						proporcion = proporcion + dataset[i].y[j]/viajesPromedio[diaSemana-1,int(dataset[i].x[j][0]-1),int(dataset[i].x[j][3]-1)]
						dataset[i].TTprediccion = numpy.append(dataset[i].TTreal, [dataset[i].y[j] + dataset[i].inicioViaje],axis=0)
						dataset[i].TTreal = numpy.append(dataset[i].TTreal, [dataset[i].y[j] + dataset[i].inicioViaje],axis=0)
						#print("{0:11.0f}\t{1}\t{2:4.0f}\t{3:4.0f}".format(dataset[i].id,j+1,dataset[i].TTreal[j+1],dataset[i].TTreal[j+1]))
					#Se utiliza el promedio de proporcion para el resto del viaje
					else:
						dataset[i].TTprediccion = numpy.append(dataset[i].TTprediccion, [viajesPromedio[diaSemana-1,int(dataset[i].x[j][0]-1),int(dataset[i].x[j][3]-1)]*proporcion/(dataset[i].ultimoParadero+1) + dataset[i].inicioViaje],axis=0)
						dataset[i].TTreal = numpy.append(dataset[i].TTreal, [dataset[i].y[j] + dataset[i].inicioViaje],axis=0)
						#print("{0:11.0f}\t{1}\t{2:4.0f}\t{3:4.0f}".format(dataset[i].id,j+1,dataset[i].TTprediccion[j+1],dataset[i].TTreal[j+1]))

			###########################################
			#### Produce las secuencias de headway ####
			###########################################

			precisionBinaria = 0.0
			precisionDetalle = float(0.0)
			sensibilidad = 0.0
			especificidad = 0.0
			cantidadParesViajes = 0.0

			for i in range(len(dataset)):
				for j in [1+i,2+i]:
					if j < len(dataset):
						# Solo se toman secuencias de headway con valores iniciales menores a 3600 
						if (dataset[j].inicioViaje - dataset[i].inicioViaje) < 3600:
							# Restricciones el ultimo paradero de cada para es el minimo entre ambos
							ultimoParadero = min(dataset[j].ultimoParadero,dataset[i].ultimoParadero)
							# Si no hay paraderos que predecir, entonces no se cuenta como par
							if ultimoParadero < len(dataset[i].TTreal):
								cantidadParesViajes = cantidadParesViajes + 1.0
								idPar = '{0:13.0f}{1:3.0f}'.format(dataset[i].id,round(dataset[j].id/1000)*1000-dataset[j].id)
								
								tempPrecisionPrediccion = 0
								tempPrecisionReal = 0
								tempPrecisionDetalle = 0
								tempSensibilidad = 0.0
								tempEspecificidad = 0.0
								cantBunching = 0.0
								cantNoBunching = 0.0


								for k in range(len(dataset[i].TTreal)):
									if k < ultimoParadero:
										headwayInicial = dataset[j].inicioViaje - dataset[i].inicioViaje
										headwayPrediccion = dataset[j].TTprediccion[k]-dataset[i].TTprediccion[k]
										headwayReal = dataset[j].TTreal[k]-dataset[i].TTreal[k]
													
										#Revisa si hay bunching o no
										bunchingPrediccion = False
										bunchingReal = False
										if abs(headwayPrediccion) < 0.25*headwayInicial:
											bunchingPrediccion = True
										
										if abs(headwayReal) < 0.25*headwayInicial:
											bunchingReal = True

										#f2.write('{0}\t{1}\t{2:4.0f}\t{3:4.0f}\t{4}\t{5}'.format(idPar,k+1,round(headwayReal),headwayReal,bunchingPrediccion,bunchingReal))
									else:
										headwayInicial = dataset[j].inicioViaje - dataset[i].inicioViaje
										headwayPrediccion = dataset[j].TTprediccion[k]-dataset[i].TTprediccion[k]
										headwayReal = dataset[j].TTreal[k]-dataset[i].TTreal[k]
													
										#Revisa si hay bunching o no
										bunchingPrediccion = False
										bunchingReal = False
										
										if abs(headwayPrediccion) < 0.25*headwayInicial:
											bunchingPrediccion = True
											tempPrecisionPrediccion = 1

										if abs(headwayReal) < 0.25*headwayInicial:
											bunchingReal = True
											tempPrecisionReal = 1
										
										if bunchingReal == bunchingPrediccion:
											tempPrecisionDetalle = tempPrecisionDetalle + 1
										
										if bunchingReal== True and bunchingReal == bunchingPrediccion:
											tempSensibilidad = tempSensibilidad + 1
											cantBunching = cantBunching + 1

										if bunchingReal== False and bunchingReal == bunchingPrediccion:
											tempEspecificidad = tempEspecificidad + 1
											cantNoBunching = cantNoBunching + 1


										f2.write('{0:4.8f}\t{1:4.8f}\t{2}\t{3}\n'.format(round(abs(headwayPrediccion))/headwayInicial,abs(headwayReal)/headwayInicial,bunchingPrediccion*1,bunchingReal*1))
								
								# Despues de recorrer el viaje completo calcula las precisiones
								precisionDetalle = precisionDetalle + float(tempPrecisionDetalle)/float((len(dataset[i].TTreal)-ultimoParadero))
								if cantBunching > 0:
									sensibilidad = sensibilidad + float(tempSensibilidad)/float(cantBunching)
								if cantNoBunching > 0:
									especificidad = especificidad + float(tempEspecificidad)/float(cantNoBunching)
								
								aciertoBinario = 0
								if tempPrecisionReal == tempPrecisionPrediccion:
									precisionBinaria = precisionBinaria + 1
									aciertoBinario = 1 

								#print("Par {}\nAcierto existencia de bunching: {}\nAciertos binarios acumulados: {}\nParaderos acertados: {}\nPrecision paraderos: {}\nPrecisionAcumulada: {}\n".format(idPar,aciertoBinario,precisionBinaria,tempPrecisionDetalle,float(tempPrecisionDetalle)/float((len(dataset[i].TTreales)-ultimoParadero)),precisionDetalle))

			# Luego de recorrer la ventana completa calcula las metricas para esa ventana
			if cantidadParesViajes > 0:

				sensibilidad = float(sensibilidad)/float(cantidadParesViajes)
				especificidad = float(especificidad)/float(cantidadParesViajes)
				precisionDetalle = float(precisionDetalle)/float(cantidadParesViajes)
				precisionBinaria = precisionBinaria/cantidadParesViajes

				print(bcolors.OKBLUE+"\nPrecisionBinaria: {0:3.4f}\tPrecisionDetalle: {1:3.4f}\tSensibilidad: {2:3.4f}\tEspecificidad: {3:3.4f}\n".format(precisionBinaria*100, precisionDetalle*100,sensibilidad*100,especificidad*100)+bcolors.ENDC)
				f.write("{}\t{}\t{}\t{}\t{}\n".format(dia,finVentana,cantidadParesViajes,precisionBinaria,precisionDetalle,sensibilidad,especificidad))

				for p in range(len(precisionesBinarias)):
					if precisionesBinarias[p] == 0.0:
						precisionesBinarias[p] = precisionBinaria
						break

				for p in range(len(precisionesDetalle)):
					if precisionesDetalle[p] == 0.0:
						precisionesDetalle[p] = precisionDetalle
						break

				for p in range(len(sensibilidades)):
					if sensibilidades[p] == 0.0:
						sensibilidades[p] = sensibilidad
						break

				for p in range(len(especificidades)):
					if especificidades[p] == 0.0:
						especificidades[p] = especificidad
						break

promedioPrecisionBinaria = 0.0
contador = 0.0
for i in range(len(precisionesBinarias)):
	if precisionesBinarias[i] != 0:
		contador = contador + 1
		promedioPrecisionBinaria = promedioPrecisionBinaria + precisionesBinarias[i]

promedioPrecisionBinaria = promedioPrecisionBinaria / contador

promedioPrecisionDetalle = 0.0
contador = 0.0
for i in range(len(precisionesDetalle)):
	if precisionesDetalle[i] != 0:
		contador = contador + 1
		promedioPrecisionDetalle = promedioPrecisionDetalle + precisionesDetalle[i]

promedioPrecisionDetalle = promedioPrecisionDetalle / contador


promedioSensibilidad = 0.0
contador = 0.0
for i in range(len(sensibilidades)):
	if sensibilidades[i] != 0:
		contador = contador + 1
		promedioSensibilidad = promedioSensibilidad + sensibilidades[i]

promedioSensibilidad = promedioSensibilidad / contador


promedioEspecificidad = 0.0
contador = 0.0
for i in range(len(especificidades)):
	if especificidades[i] != 0:
		contador = contador + 1
		promedioEspecificidad = promedioEspecificidad + especificidades[i]

promedioEspecificidad = promedioEspecificidad / contador

print(bcolors.FAIL+"\nPrecision Binaria: {}\tPrecision Detalle: {}\n".format(promedioPrecisionBinaria,promedioPrecisionDetalle,promedioSensibilidad,promedioEspecificidad)+bcolors.ENDC)
f.write("Promedio\t\t\t{}\t{}\t{}\t{}\n".format(promedioPrecisionBinaria,promedioPrecisionDetalle,promedioSensibilidad,promedioEspecificidad))
f.close()
f2.close()