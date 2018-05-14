require 'math'

-- --------------------------------------------------------------------
-- GENERACION DE DATASETS A PARTIR DE DATOS BRUTOS DE GPS Y LLEGADA A PARADEROS
-- --------------------------------------------------------------------
--  Dataset RNP (segun paper de Gurmu)
--	Filas:
--	- Cada fila es una pasada por un paradero
--	- Cada viaje empieza en la pasada de primer paradero del recorrido
--	- Se consideran solo los días de semana (lunes a viernes)
--	Columnas:
--	x1 Id del Viaje 										
--	x2 Intervalo del dia (horario)
--  	0	6:00	6:30
--  	1	6:30	8:30
--  	2	8:00	9:30
--  	3	9:30	12:30
--  	4	12:30	14:00
--  	5	14:00	17:30
--  	6	17:30	21:30
--  	7	21:30	23:00
--	x3 ID parada actual i
--	x4 Tiempo de viaje desde parada 1 a i
--	x5 ID parada a consultar (j)
--	y Tiempo de viaje entre paradero i y j

--	Dataset RND
--	Columnas:
--		Metadatos:
--			x1 hora de llegada al ultimo paradero
--			x2 hora de llegada al próximo paradero
--			x3 id tramo actual
--		Secuencia:
--			x4 tiempo de viaje actual
--			x5 distancia recorrida actual
--		y Distancia a recorrer en los próximos 30 segundos


distanciaMaximaRecorrido = 18
cantidadParaderos = 43

function split(inputString, sep)
	-- permite separar un string según un separador
	if sep == nil then
		sep = "%s"
	end
	local t = {}
	local i = 1
	for str in string.gmatch(inputString,"([^" .. sep .. "]+)") do
		t[i] = str
		i = i + 1
	end
	return t
end

function insertarInicio(viajes,hora,id)
	for i=1, #viajes do
		if viajes[i].id == id then
			viajes[i].horaInicio = hora
			break
		end
	end
end

function insertarGPS(id,hora,distancia)
	for i=1, #viajes do
		if viajes[i].id == id then
			if distanciaMaximaRecorrido >= distancia then
				if viajes[i].puntosGPS ~= nil then
					table.insert(viajes[i].puntosGPS,{hora,distancia})
					break
				else
					viajes[i].puntosGPS = {}
					viajes[i].puntosGPS[1] = {hora,distancia}
					break
				end
			end
		end
	end
end

function interpolarDistanciaParadero(punto1,punto2,tiempoParadero)
	if punto1~= nil and punto2 ~= nil then	
		return punto1[2]+(punto2[2]-punto1[2])*(tiempoParadero-punto1[1])/(punto2[1]-punto1[1])
	else
		return 0
	end
end

--Los tramos no son solo respecto a la cantidad de paradas totales en un viaje específico (no representa en tramo con respecto al viaje total)
function tramoGPS(pasadas,puntoGPS)
	for i=1,#pasadas do
		if puntoGPS[1] < pasadas[i][1] then
			return i - 1
		end
	end
	return #pasadas
end

function calculaVelocidadPasada(puntosGPS,pasada,indice)
	local puntosTramo = {}
	--Se seleccionan los puntos hasta el ultimo tramo
	for i=1, #puntosGPS do
		if puntosGPS[i][3] <= indice then
			if #puntosTramo == 0 then
				puntosTramo[1] = puntosGPS[i]
			else
				table.insert(puntosTramo,puntosGPS[i])
			end
		end 
	end
	if #puntosTramo > 1 then
		return (puntosTramo[#puntosTramo][2] - puntosTramo[1][2])/(puntosTramo[#puntosTramo][1] - puntosTramo[1][1])*1000
	else
		return 0
	end
end

function calculaDistanciaPasada(puntosGPS,pasada,indice)
	
	--Busca el ultimo punto antes del tiempo de pasada y el siguiente para interpolar linealmente
	local puntoAnterior = nil
	local puntoSiguiente = nil
	
	for i=1,#puntosGPS-1 do
		if puntosGPS[i][1] < pasada[1] then
			puntoAnterior = puntosGPS[i]
			puntoSiguiente = puntosGPS[i+1]
		end
	end
	return interpolarDistanciaParadero(puntoAnterior,puntoSiguiente,pasada[1])
end

function get_day_of_week(dd, mm, yy)
 	dw = os.date('*t',os.time{year=yy,month=mm,day=dd})['wday']-1
  	if dw == 0 then
  		return 7
  	else
  		return dw
  	end
end

--Apertura del archivo de entrada
entrada = io.open('../../datos/datasetBrutos/datasetParaderosBrutoI09.txt','r')

--Guarda los viajes del dataset de pasadas
viajes = {}
viaje = nil
idAnterior = 0

for linea in entrada:lines() do

	datos = split(linea,'\t')
	
	id = tonumber(datos[1])
	patente = datos[2]
	hora = tonumber(datos[3])
	paradero = datos[4]


	--Si se continúa en el mismo viaje
	if id == idAnterior then
		table.insert(viaje.pasadas,{hora,paradero})
	--Si es un nuevo viaje
	else
		idAnterior = id
		--Cuando existen datos en viaje
		if viaje ~= nil then
			--Se inserta el viaje anterior
			table.insert(viajes,viaje)
		end
		--Se crea un nuevo viaje
		viaje = {}
		viaje.id = id
		viaje.patente = patente
		viaje.pasadas = {}
		viaje.pasadas[1] = {hora,paradero}
	end
end

if viaje ~= nil then
	--Se inserta el viaje anterior
	table.insert(viajes,viaje)
end

-- Se cierra el archivo de entrada y se libera memoria
entrada:close()
entrada = nil
collectgarbage()

print('Se lee el dataset bruto de paraderos')

--Se lee el archivo de referencia de los paraderos del servicio
entrada = io.open('../../datos/datasetBrutos/paraderosServicioI09.txt','r')

-- Lee el codigo de los paraderos del servicio elegido
paraderosServicio = {}
indice = 1

for linea in entrada:lines() do
	datos = split(linea,'\t')
	codigoParadero = datos[1]
	table.insert(paraderosServicio,{codigoParadero,indice})
	indice = indice + 1
end

-- Se cierra el archivo de entrada y se libera memoria
entrada:close()
entrada = nil
collectgarbage()

print('Se lee el archivo de paraderos del servicio')

-- Se ordenan las pasadas
for i=1,#viajes do
	--Se crea un arreglo ordenado el cual sera llenado por los datos de las pasadas de cada viaje
	pasadasOrdenadas = {}
	for j=1, #paraderosServicio do
		pasadasOrdenadas[j] = {0,paraderosServicio[j][1],j,paraderosServicio[j][2]}
	end
	-- Se agrega la hora en pasada a cada paradero
	for j=1, #viajes[i].pasadas do
		for k=1, #paraderosServicio do
			if viajes[i].pasadas[j][2] == pasadasOrdenadas[k][2] then
				pasadasOrdenadas[k][1] = viajes[i].pasadas[j][1]
			end
		end
	end

	for j=1, #pasadasOrdenadas do
		if pasadasOrdenadas[j][1] == 0 then
			--Se interpolan los tiempos a las paradas faltantes
			if pasadasOrdenadas[j+1][1] > 0 then
				pasadasOrdenadas[j][1] = math.floor((pasadasOrdenadas[j-1][1]+pasadasOrdenadas[j+1][1])/2)
			elseif pasadasOrdenadas[j+2][1] > 0 then
				pasadasOrdenadas[j][1] = math.floor(pasadasOrdenadas[j-1][1]+(pasadasOrdenadas[j+2][1]-pasadasOrdenadas[j-1][1])/3)
			elseif pasadasOrdenadas[j+3][1] > 0 then
				pasadasOrdenadas[j][1] = math.floor(pasadasOrdenadas[j-1][1]+(pasadasOrdenadas[j+3][1]-pasadasOrdenadas[j-1][1])/4)
			elseif pasadasOrdenadas[j+4][1] > 0 then
				pasadasOrdenadas[j][1] = math.floor(pasadasOrdenadas[j-1][1]+(pasadasOrdenadas[j+4][1]-pasadasOrdenadas[j-1][1])/5)
			elseif pasadasOrdenadas[j+5][1] > 0 then
				pasadasOrdenadas[j][1] = math.floor(pasadasOrdenadas[j-1][1]+(pasadasOrdenadas[j+5][1]-pasadasOrdenadas[j-1][1])/6)
			elseif pasadasOrdenadas[j+6][1] > 0 then
				pasadasOrdenadas[j][1] = math.floor(pasadasOrdenadas[j-1][1]+(pasadasOrdenadas[j+6][1]-pasadasOrdenadas[j-1][1])/7)
			elseif pasadasOrdenadas[j+7][1] > 0 then
				pasadasOrdenadas[j][1] = math.floor(pasadasOrdenadas[j-1][1]+(pasadasOrdenadas[j+7][1]-pasadasOrdenadas[j-1][1])/8)
			elseif pasadasOrdenadas[j+8][1] > 0 then
				pasadasOrdenadas[j][1] = math.floor(pasadasOrdenadas[j-1][1]+(pasadasOrdenadas[j+8][1]-pasadasOrdenadas[j-1][1])/9)
			elseif pasadasOrdenadas[j+9][1] > 0 then
				pasadasOrdenadas[j][1] = math.floor(pasadasOrdenadas[j-1][1]+(pasadasOrdenadas[j+9][1]-pasadasOrdenadas[j-1][1])/10)
			elseif pasadasOrdenadas[j+10][1] > 0 then
				pasadasOrdenadas[j][1] = math.floor(pasadasOrdenadas[j-1][1]+(pasadasOrdenadas[j+10][1]-pasadasOrdenadas[j-1][1])/11)
			elseif pasadasOrdenadas[j+11][1] > 0 then
				pasadasOrdenadas[j][1] = math.floor(pasadasOrdenadas[j-1][1]+(pasadasOrdenadas[j+11][1]-pasadasOrdenadas[j-1][1])/12)
			elseif pasadasOrdenadas[j+12][1] > 0 then
				pasadasOrdenadas[j][1] = math.floor(pasadasOrdenadas[j-1][1]+(pasadasOrdenadas[j+12][1]-pasadasOrdenadas[j-1][1])/13)
			elseif pasadasOrdenadas[j+13][1] > 0 then
				pasadasOrdenadas[j][1] = math.floor(pasadasOrdenadas[j-1][1]+(pasadasOrdenadas[j+13][1]-pasadasOrdenadas[j-1][1])/14)
			elseif pasadasOrdenadas[j+14][1] > 0 then
				pasadasOrdenadas[j][1] = math.floor(pasadasOrdenadas[j-1][1]+(pasadasOrdenadas[j+14][1]-pasadasOrdenadas[j-1][1])/15)
			elseif pasadasOrdenadas[j+15][1] > 0 then
				pasadasOrdenadas[j][1] = math.floor(pasadasOrdenadas[j-1][1]+(pasadasOrdenadas[j+15][1]-pasadasOrdenadas[j-1][1])/16)
			end
		end
	end

	viajes[i].pasadas = pasadasOrdenadas
end

print('Se ordenan las pasadas')

--Se agrega hora de inicio y datos GPS a cada viaje
entrada = io.open('../../datos/datasetBrutos/datasetGPSBrutoI09.txt','r')
horaAnterior = nil
idAnterior = nil

for linea in entrada:lines() do
	datos = split(linea,'\t')
	
	id = tonumber(datos[1])
	patente = datos[2]
	hora = tonumber(datos[3])
	distancia = tonumber(datos[4])

	insertarGPS(id,hora,distancia)

	if distancia == 0 then
		--Captura el inicio del viaje actual
		insertarInicio(viajes,hora,id)
	end

	horaAnterior = hora
	idAnterior = id
end

entrada:close()
entrada = nil
collectgarbage()

print('Se lee el dataset bruto de GPS')

--Se normalizan las listas de distancias para que esten cada 30 segundos

for i=1, #viajes do
	distanciasTemp = {}
	j=1
	distanciasTemp[j]=viajes[i].puntosGPS[j]
	j=j+1
	hora = viajes[i].puntosGPS[1][1]+30
	finViaje = viajes[i].puntosGPS[#viajes[i].puntosGPS][1]
	while hora <= finViaje do
		--Si existe una distancia que coincide con la hora
		agregado = false
		for k=1, #viajes[i].puntosGPS do
			if hora == viajes[i].puntosGPS[k][1] then
				table.insert(distanciasTemp,viajes[i].puntosGPS[k])
				agregado = true
				break
			end
		end
		if agregado == false then
			--Interpola la distancia para la hora
			for k=1, #viajes[i].puntosGPS do
				if hora < viajes[i].puntosGPS[k][1] then
					distancia = viajes[i].puntosGPS[k-1][2]+(viajes[i].puntosGPS[k][2]-viajes[i].puntosGPS[k-1][2])*(hora-viajes[i].puntosGPS[k-1][1])/(viajes[i].puntosGPS[k][1]-viajes[i].puntosGPS[k-1][1])
					table.insert(distanciasTemp,{hora,distancia})
					break
				end
			end
		end
		j=j+1
		hora = hora+30
	end
	viajes[i].puntosGPS = distanciasTemp
end

print('Se normalizan los puntos GPS')

--Ordena los puntos GPS por tramo del recorrido
for i=1, #viajes do
	for j=1, #viajes[i].puntosGPS do
		--Se conservan los datos antes del primer paradero para calcular la distancia y tiempo en primera parada
		--La tercera columna de cada punto GPS va a ser el tramo
		viajes[i].puntosGPS[j] = {viajes[i].puntosGPS[j][1],viajes[i].puntosGPS[j][2],tramoGPS(viajes[i].pasadas,viajes[i].puntosGPS[j])}
	end
	viajes[i].offset = 0
	for j=1, #viajes[i].pasadas do
		--Se calcula la distancia recorrida hasta la pasada actual en metros
		distanciaTemp = calculaDistanciaPasada(viajes[i].puntosGPS,viajes[i].pasadas[j],j)
		if j==1 then
			viajes[i].pasadas[j][4] = 0
			viajes[i].offset = distanciaTemp
		else
			viajes[i].pasadas[j][4] = distanciaTemp - viajes[i].offset
		end
	end

	--Se calcula el dia de la semana para cada viaje
	yy = tonumber(string.sub(string.format('%s',viajes[i].id),1,4))
	mm = tonumber(string.sub(string.format('%s',viajes[i].id),5,6))
	dd = tonumber(string.sub(string.format('%s',viajes[i].id),7,8))
	viajes[i].diaSemana = get_day_of_week(dd, mm, yy)
end

print('Se agregan los paraderos en los puntos GPS')

--Lee el archivo con los intervalos programados
entrada = io.open('../../datos/datasetBrutos/intervalosTiempoServicioI09.txt','r')

indice = 1
intervalosProgramados = {}
intervalosProgramados[1] = {}
intervalosProgramados[2] = {}
intervalosProgramados[3] = {}

for linea in entrada:lines() do
	datos = split(linea,'\t')
	if #datos == 1 then
		indice = indice + 1
	else	
		inicioHorario = tonumber(datos[1])
	 	intervalo = tonumber(datos[2])
		if #intervalosProgramados[indice] == 0 then
			intervalosProgramados[indice][1] = {inicioHorario,intervalo}
		else
			table.insert(intervalosProgramados[indice],{inicioHorario,intervalo})
		end
	end
end

print('Se lee el archivo con horarios y headways')

--Se entrega el indice de intervalo del dia a cada viaje segun su hora de inicio
for i=1,#viajes do
	if viajes[i].diaSemana < 6 then
		for j=2, #intervalosProgramados[1] do
			if intervalosProgramados[1][j][1] > viajes[i].horaInicio then
				viajes[i].intervalo = j-1
				break
			end
		end
	elseif viajes[i].diaSemana == 6 then
		for j=2, #intervalosProgramados[2] do
			if intervalosProgramados[2][j][1] > viajes[i].horaInicio then
				viajes[i].intervalo = j-1
				break
			end
		end
	else
		for j=2, #intervalosProgramados[3] do
			if intervalosProgramados[3][j][1] > viajes[i].horaInicio then
				viajes[i].intervalo = j-1
				break
			end	
		end
	end
end

entrada:close()
entrada = nil
collectgarbage()

print('Se agrega el indentificador del horario a cada viaje')

viajesMalos = {}
for i=1,#viajes do
	for j=2, #viajes[i].puntosGPS do
		if viajes[i].puntosGPS[j][2] - viajes[i].puntosGPS[j-1][2] > 1.2 then
			table.insert(viajesMalos, i)
			break
		end
	end
end

table.sort(viajesMalos)
 
for i=#viajesMalos, 1,-1 do
	table.remove(viajes,viajesMalos[i])
end

print('Se eliminan los viajes con puntos GPS conflictivos')

salida = io.open('../../datos/datasetDefinitivoParaderosI09.tsv','w')
for i=1, #viajes do
	--Va entregando el indice de los próximos paraderos por cada paradero del recorrido (excepto el ultimo) 
	if viajes[i].diaSemana < 6 and viajes[i].intervalo ~=nil then
		for j=1, #viajes[i].pasadas-1 do
			for k=j+1, #viajes[i].pasadas do
				if j == 1 then
					tiempoPrediccion = viajes[i].pasadas[k][1] - viajes[i].pasadas[j][1]
					salida:write(viajes[i].id..'\t'..viajes[i].intervalo..'\t'..viajes[i].pasadas[j][3]..'\t'..'0'..'\t'..viajes[i].pasadas[k][3]..'\t'..tiempoPrediccion..'\n')
				else
					tiempoTranscurrido = viajes[i].pasadas[j][1] - viajes[i].pasadas[1][1]
					tiempoPrediccion = viajes[i].pasadas[k][1] - viajes[i].pasadas[j][1]
					if tiempoPrediccion > 0 then
						salida:write(viajes[i].id..'\t'..viajes[i].intervalo..'\t'..viajes[i].pasadas[j][3]..'\t'..tiempoTranscurrido..'\t'..viajes[i].pasadas[k][3]..'\t'..tiempoPrediccion..'\n')
					end
				end
			end
		end
	end
end

salida:close()
salida = nil
collectgarbage()

print('Se termina de generar el dataset de paraderos')

salida = io.open('../../datos/datasetDefinitivoGPSv4.tsv','w')
for i=1, #viajes do
	--Va entregando el indice de los próximos paraderos por cada paradero del recorrido (excepto el ultimo) 
	if viajes[i].diaSemana < 6 and viajes[i].intervalo ~=nil then
		for j=1, #viajes[i].pasadas-1 do
			for k=2, #viajes[i].puntosGPS-1 do
				if viajes[i].puntosGPS[k][3] == j then
					--tiempoTranscurrido = viajes[i].puntosGPS[k][1] - viajes[i].pasadas[1][1]
					--latencia = viajes[i].puntosGPS[k][1] - viajes[i].puntosGPS[k-1][1]
					distanciaRecorrida = viajes[i].puntosGPS[k][2] - viajes[i].puntosGPS[k-1][2]
					--Se arregla la salida
					--distanciaProximoMuestreo = viajes[i].puntosGPS[k+1][2] - viajes[i].offset
					distanciaProximoMuestreo = viajes[i].puntosGPS[k+1][2] - viajes[i].puntosGPS[k][2]
					salida:write(viajes[i].id..'\t'..viajes[i].diaSemana..'\t'..viajes[i].intervalo..'\t'..j..'\t'..distanciaRecorrida..'\t'..distanciaProximoMuestreo..'\n')
				end
			end
		end
	end
end

salida:close()
salida = nil
collectgarbage()

print('Se termina de generar el dataset de GPS')

salida = io.open('../../datos/datasetBrutoI09Modelo2.tsv','w')
for i=1, #viajes do
	--Va entregando el indice de los próximos paraderos por cada paradero del recorrido (excepto el ultimo) 
	if viajes[i].diaSemana < 6 and viajes[i].intervalo ~=nil then
		for j=1, #viajes[i].pasadas do
			salida:write(viajes[i].id..'\t'..viajes[i].intervalo..'\t'..viajes[i].pasadas[j][3]..'\t'..viajes[i].pasadas[j][1]..'\n')
		end
	end
end

viajes = nil
salida:close()
salida = nil
collectgarbage()

salida = io.open('../../datos/datasetModeloPromedio.tsv','w')
for i=1, #viajes do
	--Va entregando el indice de los próximos paraderos por cada paradero del recorrido (excepto el ultimo) 
	if viajes[i].diaSemana < 6 and viajes[i].intervalo ~=nil then
		for j=1, #viajes[i].pasadas do
			salida:write(viajes[i].id..'\t'..viajes[i].diaSemana..'\t'..viajes[i].intervalo..'\t'..viajes[i].pasadas[j][3]..'\t'..viajes[i].pasadas[j][1]..'\n')
		end
	end
end

viajes = nil
salida:close()
salida = nil
collectgarbage()