#!/usr/bin/python
#coding=utf-8

#########################################################################################
#	Calcula ROC a partir de datos de clasificacion de cada modelo (RNP, RND, RNH y MPH) #
#	utiliza distintos valores de umbrales de clasificacion para generar grafico			#
#########################################################################################

import os
import gc
import sys
import numpy
import h5py
import math
from sklearn import metrics
from pandas import read_csv
import matplotlib.pyplot as plt

def curva_roc(clasificaciones_reales,valores,intervalo_umbral):
	iteraciones = int(5/intervalo_umbral)
	#print("{} iteraciones".format(iteraciones))
	esp = []
	ses = []
	for i in range(iteraciones):
		clasificaciones_pred = numpy.array([valores])
		clasificaciones_pred[clasificaciones_pred<=i*intervalo_umbral] = 1
		clasificaciones_pred[clasificaciones_pred>i*intervalo_umbral] = 0
		clasificaciones_pred = numpy.reshape(clasificaciones_pred,(numpy.shape(clasificaciones_pred)[1],1))
		matriz_confusion = metrics.confusion_matrix(clasificaciones_reales, clasificaciones_pred)
		SES = matriz_confusion[1,1]/float(matriz_confusion[1,0]+matriz_confusion[1,1])
		ESP = matriz_confusion[0,0]/float(matriz_confusion[0,0]+matriz_confusion[0,1])
		if len(esp) == 0:
			esp = [1.0-ESP]
		else:
			esp.extend([1.0-ESP])

		if len(ses) == 0:
			ses = [SES]
		else:
			ses.extend([SES])
	ses.extend([1])
	esp.extend([1])
	return esp,ses


numpy.set_printoptions(precision=12,threshold=12000000,suppress=True)

resultados_promedio = read_csv('../../resultados/ROC/datosROCpromedio2.tsv', engine = 'python',names = ['ratio_prediccion','ratio_real','clasificacion_prediccion','clasificacion_real'], sep = "\t")
resultados_paraderos = read_csv('../../resultados/ROC/datosROCparaderos2.tsv', engine = 'python',names = ['ratio_prediccion','ratio_real','clasificacion_prediccion','clasificacion_real'], sep = "\t")
resultados_distancias = read_csv('../../resultados/ROC/datosROCdistancias2.tsv', engine = 'python',names = ['ratio_prediccion','ratio_real','clasificacion_prediccion','clasificacion_real'], sep = "\t")
resultados_headway = read_csv('../../resultados/ROC/datosROCheadway2.tsv', engine = 'python',names = ['ratio_prediccion','ratio_real','clasificacion_prediccion','clasificacion_real'], sep = "\t")

print("max ratio MPH: {}".format(max(resultados_promedio['ratio_prediccion'])))
print("max ratio RNP: {}".format(max(resultados_paraderos['ratio_prediccion'])))
print("max ratio RND: {}".format(max(resultados_distancias['ratio_prediccion'])))
print("max ratio RNH: {}\n".format(max(resultados_headway['ratio_prediccion'])))


############## MPH ################
#fpr_promedio, tpr_promedio, _ = metrics.roc_curve(numpy.logical_not(resultados_promedio['clasificacion_real']).astype(int),resultados_promedio['ratio_prediccion'],pos_label=0)
fpr_promedio, tpr_promedio = curva_roc(resultados_promedio['clasificacion_real'],resultados_promedio['ratio_prediccion'],0.05)
#auc_promedio = metrics.roc_auc_score(resultados_promedio['clasificacion_real'],resultados_promedio['ratio_prediccion'])
auc_promedio = metrics.auc(fpr_promedio, tpr_promedio)

print("\n===============\nReporte MPH\n===============")
print(metrics.classification_report(resultados_promedio['clasificacion_real'], resultados_promedio['clasificacion_prediccion'], target_names=["No bunching","Bunching"],digits=6))
matriz_promedio = metrics.confusion_matrix(resultados_promedio['clasificacion_real'], resultados_promedio['clasificacion_prediccion'])
print("TN\tFP\tFN\tTP")
print("{}\t{}\t{}\t{}".format(matriz_promedio[0,0],matriz_promedio[0,1],matriz_promedio[1,0],matriz_promedio[1,1]))
print("\nSES\tESP\tACC")
SES_promedio = matriz_promedio[1,1]/float(matriz_promedio[1,0]+matriz_promedio[1,1])
ESP_promedio = matriz_promedio[0,0]/float(matriz_promedio[0,0]+matriz_promedio[0,1])
ACC_promedio = float(matriz_promedio[0,0]+matriz_promedio[1,1])/float(matriz_promedio[0,1]+matriz_promedio[1,0]+matriz_promedio[0,0]+matriz_promedio[1,1])
print("{}\t{}\t{}".format(SES_promedio,ESP_promedio,ACC_promedio))
fpr_promedio = numpy.trim_zeros(fpr_promedio)
tpr_promedio = numpy.trim_zeros(tpr_promedio)
fpr_promedio.insert(0,0)
tpr_promedio.insert(0,0)
fpr_promedio.insert(1,1-ESP_promedio)
tpr_promedio.insert(1,SES_promedio)
plt.plot(fpr_promedio,tpr_promedio,label="MPH, AUC="+str(auc_promedio),color='darkorange',lw=2)

############### RNP #################
#fpr_paraderos, tpr_paraderos, _ = metrics.roc_curve(numpy.logical_not(resultados_paraderos['clasificacion_real']).astype(int),resultados_paraderos['ratio_prediccion'],pos_label=0)
fpr_paraderos, tpr_paraderos = curva_roc(resultados_paraderos['clasificacion_real'],resultados_paraderos['ratio_prediccion'],0.05)
#auc_paraderos = metrics.roc_auc_score(resultados_paraderos['clasificacion_real'],resultados_paraderos['ratio_prediccion'])
auc_paraderos = metrics.auc(fpr_paraderos, tpr_paraderos)

print("\n===============\nReporte RNP\n===============")
print(metrics.classification_report(resultados_paraderos['clasificacion_real'], resultados_paraderos['clasificacion_prediccion'], target_names=["No bunching","Bunching"],digits=6))
matriz_paraderos = metrics.confusion_matrix(resultados_paraderos['clasificacion_real'], resultados_paraderos['clasificacion_prediccion'])
print("TN\tFP\tFN\tTP")
print("{}\t{}\t{}\t{}".format(matriz_paraderos[0,0],matriz_paraderos[0,1],matriz_paraderos[1,0],matriz_paraderos[1,1]))
print("\nSES\tESP\tACC")
SES_paraderos = matriz_paraderos[1,1]/float(matriz_paraderos[1,0]+matriz_paraderos[1,1])
ESP_paraderos = matriz_paraderos[0,0]/float(matriz_paraderos[0,0]+matriz_paraderos[0,1])
ACC_paraderos = float(matriz_paraderos[0,0]+matriz_paraderos[1,1])/float(matriz_paraderos[0,1]+matriz_paraderos[1,0]+matriz_paraderos[0,0]+matriz_paraderos[1,1])
print("{}\t{}\t{}".format(SES_paraderos,ESP_paraderos,ACC_paraderos))
fpr_paraderos = numpy.trim_zeros(fpr_paraderos)
tpr_paraderos = numpy.trim_zeros(tpr_paraderos)
fpr_paraderos.insert(0,0)
tpr_paraderos.insert(0,0)
fpr_paraderos.insert(1,1-ESP_paraderos)
tpr_paraderos.insert(1,SES_paraderos)
plt.plot(fpr_paraderos,tpr_paraderos,label="RNP, AUC="+str(auc_paraderos),color='green',lw=2)

############### RND #################
#fpr_distancias, tpr_distancias, _ = metrics.roc_curve(numpy.logical_not(resultados_distancias['clasificacion_real']).astype(int),resultados_distancias['ratio_real'],pos_label=0)
fpr_distancias, tpr_distancias = curva_roc(resultados_distancias['clasificacion_real'],resultados_distancias['ratio_prediccion'],0.05)
#auc_distancias = metrics.roc_auc_score(resultados_distancias['clasificacion_real'].astype('int32', errors='ignore'),resultados_distancias['ratio_real'])
auc_distancias = metrics.auc(fpr_distancias, tpr_distancias)

print("\n===============\nReporte RND\n===============")
print(metrics.classification_report(resultados_distancias['clasificacion_real'], resultados_distancias['clasificacion_prediccion'], target_names=["No bunching","Bunching"],digits=6))
matriz_distancias = metrics.confusion_matrix(resultados_distancias['clasificacion_real'], resultados_distancias['clasificacion_prediccion'])
print("TN\tFP\tFN\tTP")
print("{}\t{}\t{}\t{}".format(matriz_distancias[0,0],matriz_distancias[0,1],matriz_distancias[1,0],matriz_distancias[1,1]))
print("\nSES\tESP\tACC")
SES_distancias = matriz_distancias[1,1]/float(matriz_distancias[1,0]+matriz_distancias[1,1])
ESP_distancias = matriz_distancias[0,0]/float(matriz_distancias[0,0]+matriz_distancias[0,1])
ACC_distancias = float(matriz_distancias[0,0]+matriz_distancias[1,1])/float(matriz_distancias[0,1]+matriz_distancias[1,0]+matriz_distancias[0,0]+matriz_distancias[1,1])
print("{}\t{}\t{}".format(SES_distancias,ESP_distancias,ACC_distancias))
fpr_distancias = numpy.trim_zeros(fpr_distancias)
tpr_distancias = numpy.trim_zeros(tpr_distancias)
fpr_distancias.insert(0,0)
tpr_distancias.insert(0,0)
fpr_distancias.insert(1,1-ESP_distancias)
tpr_distancias.insert(1,SES_distancias)
plt.plot(fpr_distancias,tpr_distancias,label="RND, AUC="+str(auc_distancias),color='blue',lw=2)


############### RNH #################
#fpr_headway, tpr_headway, _ = metrics.roc_curve(numpy.logical_not(resultados_headway['clasificacion_real']).astype(int),resultados_headway['ratio_real'],pos_label=0)
fpr_headway, tpr_headway = curva_roc(resultados_headway['clasificacion_real'],resultados_headway['ratio_prediccion'],0.05)
#auc_headway = metrics.roc_auc_score(resultados_headway['clasificacion_real'].astype('int32', errors='ignore'),resultados_headway['ratio_real'])
auc_headway = metrics.auc(fpr_headway, tpr_headway)

print("\n===============\nReporte RNH\n===============")
print(metrics.classification_report(resultados_headway['clasificacion_real'], resultados_headway['clasificacion_prediccion'], target_names=["No bunching","Bunching"],digits=6))
matriz_headway = metrics.confusion_matrix(resultados_headway['clasificacion_real'], resultados_headway['clasificacion_prediccion'])
print("TN\tFP\tFN\tTP")
print("{}\t{}\t{}\t{}".format(matriz_headway[0,0],matriz_headway[0,1],matriz_headway[1,0],matriz_headway[1,1]))
print("\nSES\tESP\tACC")
SES_headway = matriz_headway[1,1]/float(matriz_headway[1,0]+matriz_headway[1,1])
ESP_headway = matriz_headway[0,0]/float(matriz_headway[0,0]+matriz_headway[0,1])
ACC_headway = float(matriz_headway[0,0]+matriz_headway[1,1])/float(matriz_headway[0,1]+matriz_headway[1,0]+matriz_headway[0,0]+matriz_headway[1,1])
print("{}\t{}\t{}".format(SES_headway,ESP_headway,ACC_headway))
fpr_headway = numpy.trim_zeros(fpr_headway)
tpr_headway = numpy.trim_zeros(tpr_headway)
fpr_headway.insert(0,0)
tpr_headway.insert(0,0)
fpr_headway.insert(1,1-ESP_headway)
tpr_headway.insert(1,SES_headway)
plt.plot(fpr_headway,tpr_headway,label="RNH, AUC="+str(auc_headway),color='red',lw=2)

############## Settings ######################
plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
plt.legend(loc=4)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1 - Especificidad')
plt.title('Comparación de modelos propuestos en curva ROC'.decode('utf-8'))
plt.ylabel('Sensibilidad')
plt.plot((1.0-ESP_promedio),SES_promedio,'o',color='darkorange')
plt.plot((1.0-ESP_paraderos),SES_paraderos,'go')
plt.plot((1.0-ESP_distancias),SES_distancias,'bo')
plt.plot((1.0-ESP_headway),SES_headway,'ro')
plt.show()
