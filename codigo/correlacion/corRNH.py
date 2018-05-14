#!/usr/bin/python
#coding=utf-8

#############################################
# Analisis de datos de headway:   			#
# - Analisis de Normalidad					#
# - Histogramas								#
# - Autocorrelacion para series de tiempo	#
# - Graficos Q-Q 							#
#############################################

import os
import gc
import numpy
import math
from scipy.stats import norm
from scipy.stats import shapiro
from scipy.stats import kstest
from scipy.stats.mstats import normaltest
from scipy.stats import probplot
import matplotlib.pyplot as plt
import pandas
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler

features = 4
sizeSeq = 10

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

# Fija la semilla del random
numpy.random.seed(7)

# Carga el dataset de RNH version 4
df = read_csv('../../datos/datasetDefinitivoModelo2v4.tsv', engine = 'python', names = ['ID','i','dw','t','h_t','dh_t+1'], sep = "\t")
df2 = df[df.columns[4:6]]
corr = df2.corr()

print df2.min()
print df2.max()
print corr

plt.scatter(df["h_t"],df["dh_t+1"],s=0.3)
plt.xlabel(r'$h_{t}$')
plt.ylabel(r'$\Delta h_{t+1}$')
plt.title(r'Correlación entre $\Delta h_{t+1}$ y $h_t$ (RNH)'.decode('utf-8'))
plt.show()

df1 = df.drop(df.index[len(df)-1])
df2 = df.drop(df.index[0])
df2.index = range(len(df2))
df3 = pandas.concat([df1["dh_t+1"],df2["dh_t+1"]], axis=1, ignore_index=True)
print df3.corr()
plt.scatter(df1["dh_t+1"],df2["dh_t+1"],s=0.3)
plt.xlabel(r'$\Delta h_{t}$ [s]')
plt.ylabel(r'$\Delta h_{t+1}$ [s]')
plt.title(r'Autocorrelación $\mathit{headway}$ (RNH)'.decode('utf-8'))
plt.show()

plt.title(r'Histograma $\mathit{headway}$ (RNH)')
plt.xlabel(r'$h_t$ [s]')
plt.ylabel("Frecuencia relativa")
n, bins, patches = plt.hist(df["h_t"], 150, normed=True, facecolor='blue', alpha=0.5)
promedio, desv_estandar = norm.fit(df["h_t"])
print("h_t: promedio: {}  desv_estandar: {}".format(promedio,desv_estandar))
xmin, xmax = plt.xlim()
x = numpy.linspace(xmin, xmax, 200)
p = norm.pdf(x, promedio, desv_estandar)
plt.plot(x, p, 'k', linewidth=1)
plt.show()

plt.title(r"Histograma $\Delta \mathit{headway}$ (RNH)")
plt.xlabel(r'$\Delta h_t$ [s]')
plt.ylabel("Frecuencia relativa")
n, bins, patches = plt.hist(df["dh_t+1"], 250, normed=True, facecolor='blue', alpha=0.5)
promedio, desv_estandar = norm.fit(df["dh_t+1"])
print("dh_t+1: promedio: {}  desv_estandar: {}".format(promedio,desv_estandar))
xmin, xmax = plt.xlim()
x2 = numpy.linspace(xmin, xmax, 200)
p2 = norm.pdf(x2, promedio, desv_estandar)
plt.xlim(-200,200)
plt.plot(x2, p2, 'k', linewidth=1)
plt.show()

plt.title("Histograma día de la semana (RNH)".decode('utf-8'))
plt.xlabel(r"$dw$")
plt.ylabel("Frecuencia relativa")
n, bins, patches = plt.hist(df["dw"], 5, facecolor='blue', alpha=0.5)
plt.show()

print(n)
print(bins)
print(patches)

plt.title("Histograma horarios (RNH)")
plt.xlabel(r"$t$")
plt.ylabel("Frecuencia relativa")
n, bins, patches = plt.hist(df["t"], 7, facecolor='blue', alpha=0.5)
plt.show()

print(n)
print(bins)
print(patches)

# Autocorrelacion headway
df2 = read_csv('../../datos/datasetDefinitivoModelo2.tsv', engine = 'python', names = ['ID','1','2','h_t','4','5','6','h_t+1'], sep = "\t")
df3 = df2[['h_t','h_t+1']]
corr = df3.corr()

print df3.min()
print df3.max()
print corr

plt.scatter(df2["h_t"],df2["h_t+1"],s=0.3)
plt.xlabel(r"$h_t$")
plt.ylabel(r"$h_{t+1}$")
plt.title(r"Autocorrelación $\mathit{headway}$ (RNH)".decode('utf-8'))
plt.show()

# Shappiro-Wilk
w, pValue = shapiro(df["dh_t+1"])
print "Test Shappiro-Wilk:"
print " - dh_t: {} {}".format(w,pValue)
w, pValue = shapiro(df["h_t"])
print " - h_t: {} {}".format(w,pValue)

#Kolmogorov-Smirnoff
d, pValue_ks = kstest(df["dh_t+1"], 'norm')
print "Test Kolmogorov-Smirnov:"
print " - dh_t: {} {}".format(d,pValue_ks)
d, pValue_ks = kstest(df["h_t"], 'norm')
print " - h_t: {} {}".format(d,pValue_ks)

#D'Agostino and Pearson
k2, pValue_dap = normaltest(df["dh_t+1"])
print "Test D'Agostino and Pearson:"
print " - dh_t: {} {}".format(k2,pValue_dap)
k2, pValue_dap = normaltest(df["h_t"])
print " - h_t: {} {}".format(k2,pValue_dap)

#Graficos Q-Q
probplot(df["h_t"], dist = "norm", plot = plt)
plt.title(r"Gráfico Q-Q $\mathit{Headway}$ (RNH)".decode('utf-8'))
plt.xlabel("Valores teóricos (distribución normal)".decode('utf-8'))
plt.ylabel(r"Valores $\mathit{dataset}$")
plt.show()

probplot(df["dh_t+1"], dist = "norm", plot = plt)
plt.title(r"Gráfico Q-Q $\Delta  Headway$ (RNH)".decode('utf-8'))
plt.xlabel("Valores teóricos (distribución normal)".decode('utf-8'))
plt.ylabel(r"Valores $\mathit{dataset}$")
plt.show()
