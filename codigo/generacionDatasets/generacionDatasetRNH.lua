-- --------------------------------------------------------------------
-- GENERACION DE DATASET para RNH
-- --------------------------------------------------------------------
-- Dataset RNH:
--		Metadatos del viaje
-- 			x1: headway de inicio un par de viajes (h1)
--			x2: dia de la semana (dia)
-- 			x3: tramo del dia (id)
--		Serie de tiempo (tentativa)
-- 			x4: headway ultimo paradero común (hj)
--			x5: porcentaje del viaje (j/n)
--			x6 tiempo de viaje 1 (TT1,j)
--			x7: tiempo de viaje 2 (TT2,j)
--		y1: diferencia entre el headway proximo y el actual (hj+1 - hj)	
--

require 'math'

cantidadParaderos = 43
cantMaxBuses = 3
tiempoMaxBuses = 3600


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


function get_day_of_week(dd, mm, yy)
 	dw = os.date('*t',os.time{year=yy,month=mm,day=dd})['wday']-1
  	if dw == 0 then
  		return 7
  	else
  		return dw
  	end
end

function retornarHoras(viaje1,viaje2,indice)
	hora1 = 0
	for i=1, #viaje1.pasadas do
		if viaje1.pasadas[i][2] == indice then
			-- Se obtiene el punto directamente
			hora1=viaje1.pasadas[i][1]
			break
		elseif viaje1.pasadas[i][2] > indice then
			-- Se interpola el punto
			t1 = viaje1.pasadas[i-1][1]
			t2 = viaje1.pasadas[i][1]
			i1 = viaje1.pasadas[i-1][2]
			i2 = viaje1.pasadas[i][2]
			hora1 = t1+(t2-t1)*(indice-i1)/(i2-i1)
			break
		end
	end 
	hora2 = 0
	for i=1, #viaje2.pasadas do
		if viaje2.pasadas[i][2] == indice then
			-- Se obtiene el punto directamente
			hora2 = viaje2.pasadas[i][1]
			break
		elseif viaje2.pasadas[i][2] > indice then
			-- Se interpola el punto
			t1 = viaje2.pasadas[i-1][1]
			t2 = viaje2.pasadas[i][1]
			i1 = viaje2.pasadas[i-1][2]
			i2 = viaje2.pasadas[i][2]
			hora2 = t1+(t2-t1)*(indice-i1)/(i2-i1)
			break
		end
	end
	return hora1,hora2
end

--Apertura del archivo de entrada
entrada = io.open('../../datos/datasetBrutoI09Modelo2.tsv','r')

--Guarda los viajes del dataset de pasadas
viajes = {}
viaje = nil
idAnterior = 0

for linea in entrada:lines() do

	datos = split(linea,'\t')
	
	id = datos[1]
	tramo = tonumber(datos[2])
	paradero = tonumber(datos[3])
	hora = tonumber(datos[4])

	--Si se continúa en el mismo viaje
	if id == idAnterior then
		table.insert(viaje.pasadas,{hora,paradero,tramo})
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
		viaje.horaInicio = hora
		viaje.pasadas = {}
		--Se calcula el dia de la semana para cada viaje
		yy = tonumber(string.sub(string.format('%s',viaje.id),1,4))
		mm = tonumber(string.sub(string.format('%s',viaje.id),5,6))
		dd = tonumber(string.sub(string.format('%s',viaje.id),7,8))
		viaje.diaSemana = get_day_of_week(dd, mm, yy)
		viaje.pasadas[1] = {hora,paradero,tramo}
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

print('Lee el dataset bruto de paraderos')

salida = io.open('../../datos/datasetDefinitivoModelo2v4.tsv','w')
for i=1, #viajes do 
	for j=i+1, math.min(i+cantMaxBuses,#viajes) do
		-- Si los viajes a estudiar son del mismo día
		-- y no tienen una separacion horaria demasiado grande (no tiene sentido)
		if viajes[i].diaSemana == viajes[j].diaSemana and (viajes[j].horaInicio - viajes[i].horaInicio) < tiempoMaxBuses then
			viaje = {}
			valido = true
			for k=1, cantidadParaderos-1 do
				id = string.format('%s%s',viajes[i].id,string.sub(viajes[j].id,9,11))
				
				h1 = viajes[j].horaInicio - viajes[i].horaInicio
				tramo = viajes[i].pasadas[1][3]
				diaSemana = viajes[i].diaSemana

				hora1,hora2 = retornarHoras(viajes[i],viajes[j],k)
				
				hj = hora2 - hora1
				idParadero = k
				TT1 = hora1 - viajes[i].horaInicio
				TT2 = hora2 - viajes[j].horaInicio
				horaSgte1,horaSgte2 = retornarHoras(viajes[i],viajes[j],k+1)
				deltaH = horaSgte2 - horaSgte1 - hj
				
				if hj > tiempoMaxBuses then
					valido = false
				end

				table.insert( viaje , {id,idParadero,diaSemana,tramo,hj,deltaH})

				--salida:write(id..'\t'..idParadero..'\t'..diaSemana..'\t'..tramo..'\t'..hj..'\t'..deltaH..'\n')
			end
			if valido == true then
				for k=1,cantidadParaderos-1 do
					salida:write(viaje[k][1]..'\t'..viaje[k][2]..'\t'..viaje[k][3]..'\t'..viaje[k][4]..'\t'..viaje[k][5]..'\t'..viaje[k][6]..'\n')
				end
			end
		end
	end
end

salida:close()
salida = nil
collectgarbage()

print('Exporta el dataset del modelo 2')