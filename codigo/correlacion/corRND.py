#!/usr/bin/python
#coding=utf-8

#############################################
# Analisis de datos de distancias:			#
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
import matplotlib.mlab as mlab
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import curve_fit
from scipy.misc import factorial
from scipy.optimize import minimize

features = 4
sizeSeq = 10

def cls():
	os.system('cls' if os.name=='nt' else 'clear')

def poisson(k, lamb):
    return (lamb**k/factorial(k)) * numpy.exp(-lamb)

def negLogLikelihood(params, data):
    """ the negative log-Likelohood-Function"""
    lnl = - numpy.sum(numpy.log(poisson(data, params[0])))
    return lnl

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

# Carga el dataset
df = read_csv('../../datos/datasetDefinitivoGPSv4.tsv', engine = 'python', names = ['ID','dw','t','i','dt','dt+1'], sep = "\t")
df2 = df[df.columns[4:6]]
corr = df2.corr()

print df2.min()
print df2.max()
print corr

plt.scatter(df["dt"],df["dt+1"],s=0.3)
plt.xlabel(r"$d_t$ [km]")
plt.ylabel(r"$d_{t+1}$ [km]")
plt.title("Autocorrelación distancias (RND)".decode('utf-8'))
plt.show()

plt.title("Histograma distancias (RND)")
plt.xlabel(r"$d_t$ [km]")
plt.ylabel("Frecuencia relativa")
n, bins, patches = plt.hist(df["dt"], bins=250, weights=numpy.zeros_like(df["dt"]) + 1. / df["dt"].count(), facecolor='blue', alpha=0.5)
#normal
promedio, desv_estandar = norm.fit(df["dt"])
print("dt: promedio: {}  desv_estandar: {}".format(promedio,desv_estandar))
xmin, xmax = plt.xlim()
x2 = numpy.linspace(xmin, xmax, 250)
p2 = norm.pdf( x2, promedio, desv_estandar)
p2 = p2/sum(p2)
plt.plot(x2, p2, 'k', linewidth=1)
plt.xlim(0,1.2)
plt.show()

plt.title("Histograma día de la semana (RND)".decode('utf-8'))
plt.xlabel(r"$dw$")
plt.ylabel("Frecuencia absoluta")
n, bins, patches = plt.hist(df["dw"], 5, facecolor='blue', alpha=0.5)
plt.show()

print(n)
print(bins)
print(patches)

plt.title("Histograma horarios (RND)")
plt.xlabel(r"$t$")
plt.ylabel("Frecuencia absoluta")
n, bins, patches = plt.hist(df["t"], 7, facecolor='blue', alpha=0.5)
plt.show()

print(n)
print(bins)
print(patches)

# Shappiro-Wilk
w, pValue_sha = shapiro(df["dt"])
print "Test Shapiro-Wilk:"
print " - d_t: {} {}".format(w,pValue_sha)

#Kolmogorov-Smirnoff
d, pValue_ks = kstest(df["dt"], 'norm')
print "Test Kolmogorov-Smirnov:"
print " - d_t: {} {}".format(d,pValue_ks)

#D'Agostino and Pearson
k2, pValue_dap = normaltest(df["dt"])
print "Test D'Agostino and Pearson:"
print " - d_t: {} {}".format(k2,pValue_dap)

#Graficos Q-Q
probplot(df["dt"], dist = "norm", plot = plt)
plt.title("Gráfico Q-Q distancias (RND)".decode('utf-8'))
plt.xlabel("Valores teóricos (distribución normal)".decode('utf-8'))
plt.ylabel(r"Valores del $\mathit{dataset}$")
plt.show()
