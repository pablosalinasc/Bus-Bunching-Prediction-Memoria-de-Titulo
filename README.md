LÉAME
======

Requisitos para de ejecución de códigos
---------------------------------------

+ Sistema operativo: distribución GNU/Linux  (recomendado Ubuntu 17.04 o superior) o Windows 10 de 64 bits (Build 1709 o superior)
+ Python (2.7.14 o superior / 3.6.1 o superior)
+ Lua (5.1.4 o superior)
+ Paquetes de Python:
	+ h5py (2.7.0 o superior)
	+ Keras (2.0.4 o superior)
	+ matplotlib (2.1.0 o superior)
	+ numpy (1.13.0 o superior)
	+ pandas (0.20.3 o superior)
	+ scikit-learn (0.19.1 o superior)
	+ scipy (0.19.1 o superior)
	+ theano (0.9.0 o superior)
	+ lapack (3.6.1 o superior)

Descripción de contenido de la memoria
------------------------------------------
```bash
├───codigo -> Códigos de implementados para realización de la memoria
│   ├───correlacion -> Códigos para análisis de datos
│   ├───entrenamiento -> Códigos para entrenamiento de redes neuronales
│   ├───generacionDatasets -> Códigos para generar dataset para cada modelo de predicción
│   ├───prediccionBunching -> Códigos para realizar pruebas de predicción de bunching
│   ├───prueba -> Códigos para realizar pruebas de rendimiento a redes entrenadas
│   └───roc -> Códigos para generar curva ROC comparativa de los modelos implementados
├───datos -> Conjuntos de datos utilizados para la realización de la memoria
│   └───datasetBrutos -> Datos extraídos directamente a partir de muestreos GPS y el Programa de Operación
└───resultados -> Almacenamiento de archivos de salida de códigos implementados
    ├───errores -> Almacenamiento de resultados de convergencia de entrenamientos de redes neuronales
    ├───modelosFinales -> Modelos finales exportados con sus respectivas pruebas de rendimiento
    │   ├───RND
    │   ├───RNH
    │   └───RNP
    ├───pruebaBunching -> Almacenamiento de resultados de predicción de bunching
    ├───pruebas -> Almacenamiento de resultados de prueba de rendimiento de modelos
    └───ROC -> Almacenamiento de resultados de clasificación para curva ROC
```
Instrucciones de ejecución
------------------------------

+ Acceder a la terminal de su respectivo sistema operativo (Bash, Powershell, Command Prompt, etc.).
+ Seleccionar el directorio del presente archivo con el comando `cd`.
+ Acceder a la carpeta `codigo` con el comando `cd ./codigo`.
+ Para la ejecución de códigos de análisis de los datos:

	1. Acceder a la carpeta `correlacion` con el comando `cd ./correlacion`.
	2. Ejecutar el código deseado con el comando `python corRND.py` o `python corRNH.py` según corresponda.
	3. Se generarán los gráficos de histograma, quantil-quantil y autocorrelación, correspondientes a la variable del dataset importado, además de imprimir resultados estadísticos de distribución y normalidad por la consola.

+ Para la ejecución de códigos de entrenamiento de redes neuronales:

	1. Acceder a la carpeta `entrenamiento` con el comando `cd ./entrenamiento`.
	2. Ejecutar el código deseado con el comando `python entrenamientoRNP.py`,  `python entrenamientoRND.py` o `python entrenamientoRNH.py` según corresponda.

	3. Es posible modificar los hiperparámetros de entrenamiento modificando las variables al inicio del código:
		- `epocas`: Cantidad de épocas de entrenamiento
		- `tasaApendizaje`: Tasa de aprendizaje utilizada por el método de entrenamiento
		- `bloquesLSTM`: Cantidad de neuronas para las capas de bloques  (RND Y RNH)
		- `porcentajeEntrenamiento`: Porcentaje de separación entre el dataset de entrenamiento y el dataset de prueba
		- `sizeSeq`: Tamaño de las secuencias que ingresan a la red neuronal
		- `neuronas`: Cantidad de neuronas para cada capa oculta del modelo (RNP)
	
	4. En la carpeta `resultados`, se exportará 1 archivo de salida por cada época de entrenamiento, con el modelo de red neuronal generado. Además en la carpeta `errores` dentro de la carpeta `resultados` se generará un archivo con el valor del error (*loss function*) del algoritmo de entrenamiento para cada predicción realizada.

+ Para la ejecución de códigos de generación de los datasets:

	1. Acceder a la carpeta `generacionDatasets` con el comando `cd ./generacionDatasets`.
	2. Ejecutar el código deseado con el comando `th generacionDatasetRNH.lua`, `th generacionDatasetsRNP-RND-MPH.lua` según corresponda. 
	3. Se generarán en la carpeta `datos` los archivos con los *datasets* correspondientes.

+ Para la ejecución de códigos de predicción de *bunching*:

	1. Acceder a la carpeta `prediccionBunching` con el comando `cd ./prediccionBunching`.

	2. Existen 3 tipos de códigos:
		- El que posee terminación `Ventana` realiza predicción de *bunching* sobre una ventana específica
		- El que posee terminación `Completo` realiza predicciones sobre todas las ventanas de prueba
		- El que posee terminación `ROC` solo exporta los valores de las clasificaciones para cada ventana de manera de poder generar la curva ROC.
	
	3. Ejecutar el código deseado con el comando `python <Nombre del archivo>.py`
	4. Se generará un archivo con los resultados de las predicciones del modelo elegido, en la carpeta `resultados`.  
	
	> Es necesario que previamente se haya realizado un proceso de entrenamiento de algún modelo de red neuronal y que se haya importado la ruta del archivo que se desea probar.

+ Para la ejecución de códigos de prueba:

	1. Acceder a la carpeta `prueba` con el comando `cd ./prueba`.
	2. Existen 2 tipos de archivo:
		- Los que poseen la terminación `RandomWalk` implementan un modelo de persistencia sobre las series de tiempo de entrenamiento y prueba de cierto modelo de red neuronal recurrente (RND o RNH).
		- Los que no posee una terminación luego de un guion realizan pruebas de rendimiento a un modelo exportado por el proceso de entrenamiento, según la red neuronal que corresponda. 
	3. Ejecutar el código deseado con el comando `python <Nombre del archivo>.py`.
	4. Se generarán archivos con los resultados de las pruebas en la carpeta "pruebas" dentro de la carpeta `resultados`.
	
	> Es necesario que previamente se haya realizado un proceso de entrenamiento de algún modelo de red neuronal y que se haya importado la ruta del archivo que se desea probar.

+ Para la ejecución de códigos para generar curva ROC: 

	1. Acceder a la carpeta `roc` con el comando `cd ./roc`.
	2. Ejecutar el código deseado con el comando `python rocComparativa.py`.
	3. Se generará el gráfico con la curva ROC comparativa entre los modelos implementados.
	
	> Es necesario que previamente se hayan generado los archivos con las muestras de clasificaciones en la predicciones de *bunching*, y se haya importado al código las rutas de los archivos correspondiente.