# -*- coding: utf-8 -*-
from flask import Flask
import AlgoritmoPrediccion
import Events


app = Flask(__name__)

"""###################################
       ALGORITMO Prediccion
"""###################################


# Entrada de API REST para entrenamiento.
""" USO

Tipo de petición: POST

Llamar a: direccion:puerto/entrenamiento?nombreModelo=nombre

ENTRADA:
    Parámetros GET:  Enviar por parámetro GET nombreModelo, el cual es la identidad del modelo que se guardará en el sistema. 
                    Si se utiliza el mismo nombre que un modelo anterior se sobreescribe.

    Parámetros POST: File: Archivo en formato CSV. Estos serán los datos de entrenamiento.
                            En este fichero CSV, la última columna debe ser el target a predecir, en este caso el interés de asistir a un evento.
                            La primera fila deben ser los nombres de las columnas, los cuales son muy importantes porque deben reutilizarse en la predicción.

SALIDA:
    Se almacena en la carpeta modelos, en una carpeta llamada como el nombreModelo aportado, el modelo generado para poder predecir con el así como todos los datos necesarios.

    La petición POST devuelve un mensaje de éxito o de error.

"""


@app.route('/entrenamiento', methods=['POST', 'PUT'])
def algApiEntrenamiento():

    return AlgoritmoPrediccion.apiEntrenamiento()


# Entrada de API REST para predicción.
""" USO

Tipo de petición: POST

Llamar a: direccion:puerto/prediccion?nombreModelo=nombre

ENTRADA:
    Parámetros GET:  Enviar por parámetro GET nombreModelo, el cual es la identidad del modelo que debe haber sido entrenado previamente.

    Parámetros POST: File: Archivo en formato CSV o JSON (desde la app). Estos serán los datos que se utilizarán para predecir la probabilidad de que a un usuario le interese un evento.
                            En este archivo no debe estar la columna target, ya que es desconocida y es de la que queremos averiguar la probabilidad.
                            Debe contener el resto de columnas utilizadas en entrenamiento. Importante que la primera fila es el nombre de las columnas y  debe contener todas aquellas utilizadas en entrenamiento manteniendo los nombres.
                            En caso de no mantenerlos el código los rellena a ceros y no da error actualmente, pero es un error.
SALIDA:
    La petición POST devuelve un mensaje de error o texto en formato JSON con el valor de la predicción para ese evento.

"""


@app.route('/prediccion', methods=['POST', 'PUT'])
def algApiPrediccion():

    return AlgoritmoPrediccion.apiPrediccion()


# Entrada de API REST para obtener todos los eventos de una ciudad en concreto
""" USO

Tipo de petición: GET

Llamar a: direccion:puerto/eventos?city=ciudad

ENTRADAS:
    Parámetros GET:  Enviar por parámetro GET city, el cual es la ciudad de la que queremos obtener todos los eventos.

SALIDAS:
    Se devuelve por POST error o en formato JSON una lista de los eventos que tendrán lugar en esa ciudad, con los datos de cada uno.
"""


@app.route('/eventos', methods=['GET'])
def getEvents():

    return Events.allEvents()


if __name__ == '__main__':
    app.run(debug=True)


