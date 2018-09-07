# -*- coding: utf-8 -*-

from flask import Flask, request
import MySQLdb
import json

DB_HOST = 'localhost'
DB_USER = 'root'
DB_PASS = ''
DB_NAME = 'myapp'

app = Flask(__name__)

# Obtener todos los eventos de una ciudad
def allEvents():

    # OBTENER PARAMETROS GET
    dataGET = request.args
    if dataGET and "city" in dataGET:
        search_city = dataGET["city"]
    else:
        return "Falta parametro para obtener eventos"

    conn = MySQLdb.connect(DB_HOST, DB_USER, DB_PASS, DB_NAME)  # Conectar a la base de datos
    cursor = conn.cursor()  # Crear un cursor

    query = "SELECT name, location, date, day, time, type, description, photo " \
            "FROM event_data WHERE city = '%s'" % search_city

    cursor.execute(query)  # Ejecutar consulta pasando los parámetros deseados
    events_data = list()

    for (name, location, date, day, time, type, description, photo) in cursor:
        events_data.append({'Nombre': name, 'Lugar': location, 'Fecha': date, 'Dia': day, 'Hora': time,
                            'Categoria': type, 'Descripcion': description, 'photo': photo})

        # print("Nombre:{}\nLugar:{}\nFecha:{}\nHora:{}\n
        # Descripcion:{}\n".format(name, location, date, time, description))

    json_string = json.dumps(events_data)

    cursor.close()  # Cerrar el cursor
    conn.close()  # Cerrar la conexión

    return json_string


# Obtener eventos en una ciudad y de un tipo en concreto
def events():

    # OBTENER PARAMETROS GET
    dataGET = request.args
    if dataGET and "city" and "type" in dataGET:
        search_city = dataGET["city"]
        type_event = dataGET["type"]
    else:
        return "Falta parametro para obtener eventos"

    conn = MySQLdb.connect(DB_HOST, DB_USER, DB_PASS, DB_NAME)  # Conectar a la base de datos
    cursor = conn.cursor()  # Crear un cursor

    query = "SELECT name, location, date, day, time, description, photo " \
            "FROM event_data WHERE city = %s AND type = %s"

    cursor.execute(query, (search_city, type_event))  # Ejecutar consulta pasando los parámetros deseados
    events_data = list()

    for (name, location, date, day, time, description, photo) in cursor:
        events_data.append({'Nombre': name, 'Lugar': location, 'Fecha': date, 'Dia': day,
                            'Hora': time, 'Descripcion': description, 'photo': photo})

        # print("Nombre:{}\nLugar:{}\nFecha:{}\nHora:{}\n
        # Descripcion:{}\n".format(name, location, date, time, description))

    json_string = json.dumps(events_data)

    cursor.close()  # Cerrar el cursor
    conn.close()  # Cerrar la conexión

    return json_string


if __name__ == '__main__':
    app.run(debug=True)