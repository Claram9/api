# -*- coding: utf-8 -*-
from flask import Flask, request
from werkzeug.utils import secure_filename
import os
import StringIO
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge, Lasso, ElasticNet, Lars
from sklearn.ensemble import RandomForestRegressor as rf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.neural_network import MLPRegressor
from sklearn.externals import joblib
from utiles import make_sure_path_exists, listToCSV
from pandas import read_csv
import csv, json
import io
from sklearn.model_selection import cross_val_score
import datetime

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

rutaModelos = os.path.join(APP_ROOT, "algPred", "modelos")

make_sure_path_exists(rutaModelos)

n_jobs = -1     # Seleccionando máximo número de cores
verbose = 10    # Seleccionando máximo nivel de mensajes


MODELOS = (
        RandomForestRegressor(verbose=True,n_jobs=n_jobs),
        GradientBoostingRegressor(verbose=True),
        AdaBoostRegressor(n_estimators=100),
        MLPRegressor(verbose=True),
        LinearSVR(verbose=True),
        LinearRegression(n_jobs=n_jobs),
        Ridge(),
        Lasso(),
        ElasticNet(),
        Lars(verbose=verbose),
        KNeighborsRegressor(),
        SGDRegressor(verbose=True),
        DecisionTreeRegressor(),
       # SVR(probability=True),                     # Orden exponencial, para muestras grandes tarda demasiado, O(n^3)  tiempo
       # GaussianProcessRegressor(n_jobs=n_jobs),  # Orden exponencial, para muestras grandes tarda demasiado, O(n^3)  tiempo
       )


# Detecta el delimitador que se está utilizando en un fichero CSV
def detectarDelimitador(csv_file):

    delimitador = csv.Sniffer().sniff(csv_file.read(1024))
    csv_file.seek(0)
    return delimitador.delimiter


# Clasifica la petición/request en CSV o JSON o ninguno (Lanza un error)
def tipoRequest(request):

    if 'file' in request.files:
        #CSV
        return "csv"
    elif request.data:
        #JSON
        return "json"
    else:
        raise ValueError('No se han encontrado datos a tratar en la petición enviada')


def devolverCsvAPI(nombreCSV):
    ### OBTENER PARAMETROS GET
    dataGET = request.args

    if dataGET and "nombreModelo" in dataGET:
        nombreModelo = dataGET["nombreModelo"]
    else:
        return "Falta parametro de data dump modelo", 400

    rutaFichero = os.path.join(rutaModelos, nombreModelo, nombreCSV)

    # Comprobando que hemos entrenado el modelo
    if not os.path.exists(os.path.join(rutaModelos, nombreModelo)):
        return "No se ha entrenado modelo para la id: {0}".format(nombreModelo), 400

    resultado = ""

    try:
        with open(rutaFichero, 'rb') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in spamreader:
                 resultado += ', '.join(row) + "\n"
    except IOError as e:
        print str(e)
        return "Error, no existe el fichero en: {0}".format(str(rutaFichero)), 400

    return resultado


# Funciona para controlar logging
def log(mensaje):
    print str(datetime.datetime.now().time()) + " - " + mensaje


# Evalua una serie de modelos para los datos y devuelve el mejor de todos ellos
def obtenerMejorModelo(rutaIdModelo, x, y):
    log("Obteniendo mejor modelo")

    #calcular para cada modelo probado un valor de fiabilidad
    probabilidadesModelos = []
    modelosOrden = []
    numModelos = len(MODELOS)
    i=0
    for modelo in MODELOS:
        i+=1
        log("Valorando modelo {0}/{1}: {2}".format(i, numModelos,str(modelo.__class__.__name__)))
        scores = cross_val_score(modelo, x, y.values.ravel(), cv=5, verbose=verbose, n_jobs=n_jobs)

        probabilidadesModelos.append((modelo.__class__.__name__, scores.mean()))  
        modelosOrden.append((modelo, scores.mean()))

    # Almacenar resultados ordenados en variables
    rutaFichero = os.path.join(rutaIdModelo, 'modelScores.csv')

    # Ordenando modelos por importancia
    listToCSV(sorted(probabilidadesModelos, key=lambda tup: tup[1], reverse=True), rutaFichero)
    modelosOrden = sorted(modelosOrden, key=lambda tup: tup[1], reverse=True), rutaFichero

    # Devolver el modelo que mejor se adapte a los datos
    modelo = modelosOrden[0][0][0]

    return modelo


# Calcula la importancia de las variables y almacena los datos en un fichero
def calcularImportanciaVariables(rutaIdModelo, x, y):
    modelo = rf()
    modelo.fit(x, y.values.ravel())

    rutaFicheroImportancia = os.path.join(rutaIdModelo, 'importanciaVariables.csv')
    listToCSV(sorted(zip(x.columns, modelo.feature_importances_), key=lambda tup: tup[1], reverse=True), rutaFicheroImportancia)


# Entrena los datos eligiendo los mejores modelos despues de valorar varios.
# Guarda todos los archivos necesarios en el sistema.
def entrenar(fichero, nombreModelo, numModelo=None):
    # Protegiendo el sistema de posibles nombres de fichero que contengan paths dañinos.
    # http://flask.pocoo.org/docs/0.12/patterns/fileuploads/
    nombreModelo = secure_filename(nombreModelo)

    # generar una carpeta en la que guardar todos los datos basada en la identidad
    # que se le ha dado al modelo de forma que tenga una estructura funcional
    rutaIdModelo = os.path.join(rutaModelos, nombreModelo)
    make_sure_path_exists(rutaIdModelo)

    # Leer CSV obtenido
    df_leido = pd.read_csv(fichero)

    # PREPROCESAR
    # Rellenando NaN a 0 en variables no categoricas
    categoricals = []
    for col, col_type in df_leido.dtypes.iteritems():
        if col_type == 'O':
            categoricals.append(col)
        else:
            df_leido[col].fillna(0, inplace=True)

    # Obtener nombre de la columna a predecir, que sera la ultima
    nombreColumnaClase = df_leido.columns[-1:]

    # Tranformando variables categoricas a numericas de forma que no haya problemas de ordinalidad
    df_preprocesado = pd.get_dummies(df_leido, dummy_na=True)

    # colocando aparte la columna de clasificacion. Definida como la ultima columna
    x = df_preprocesado[df_preprocesado.columns.difference(nombreColumnaClase)]
    y = df_leido[nombreColumnaClase]

    # print y

    # ENTRENAMIENTO DEL MODELO
    # Seleccionando modelo
    # print numModelo
    if numModelo!=None:
        modelo = MODELOS[numModelo]

        log("Valorando modelo: {0}".format(str(modelo.__class__.__name__)))
        scores = cross_val_score(modelo, x, y.values.ravel(), cv=5, verbose=verbose, n_jobs=n_jobs)

        # Almacenar resultado
        rutaFichero = os.path.join(rutaIdModelo, 'modelScores.csv')
        listToCSV((modelo.__class__.__name__, scores.mean()), rutaFichero)
    else:
        # Obteniendo modelo que se ajuste mejor a los datos
        modelo = obtenerMejorModelo(rutaIdModelo, x, y)

    # Entrenando modelo
    modelo.fit(x, y.values.ravel())

    # GUARDAR DATOS NECESARIOS

    # guardar fichero csv de datos de entrenamiento, por si se quieren revisar o reutilizar
    df_leido.to_csv(os.path.join(rutaIdModelo, "datosEntrenamiento.csv"), index=False)

    # guardar preprocesado de datos para poder despues procesar en prediccion los datos
    # que se nos aporten manteniendo el mismo formato de conversion
    # ( en el caso de OHE es rellenar a ceros las columnas de los valores que no nos aporten en prediccion )
    joblib.dump(list(x.columns), os.path.join(rutaIdModelo, 'columnas_preprocesado.pkl'))

    # guardar preprocesado en formato csv para poder consultar sin utilizar la API REST
    listToCSV([list(x.columns)], os.path.join(rutaIdModelo, 'columnas_preprocesado.csv'))

    # guardar estructura de las columnas para recordarlo durante la prediccion
    listToCSV([df_leido.columns[:-1].tolist()], os.path.join(rutaIdModelo, "columnas.csv"))

    # Calculando importancia variables
    calcularImportanciaVariables(rutaIdModelo, x, y)

    # guardar modelo
    joblib.dump(modelo, os.path.join(rutaIdModelo, "modelo.pkl"))

    return "Entrenamiento realizado con exito"


# Predice la probabilidad de las salidas (en este caso la probabilidad de que le interese un evento).
def predecir(dataset, nombreModelo):

    # genera ruta donde se ha guardado el modelo que se ha elegido
    rutaIdModelo = os.path.join(rutaModelos, nombreModelo)

    # Comprobando que hemos entrenado el modelo
    if not os.path.exists(rutaIdModelo):
        return "No se ha entrenado modelo para la id: {0}".format(nombreModelo), 400

    # Creamos copia del dataset
    df_leido = dataset.copy()
    # Se utiliza sólo si queremos comprobar el rmse ya que añadiríamos al data set la columna
    # a predecir para ver el error en la prediccion.
    # nombreColumnaClase = df_leido.columns[-1:]
    # y = df_leido[nombreColumnaClase]

    # PREPROCESAR
    # Rellenando NaN a 0 en variables no categoricas
    categoricals = []
    for col, col_type in df_leido.dtypes.iteritems():
        if col_type == 'O':
            categoricals.append(col)
        else:
            df_leido[col].fillna(0, inplace=True)

    # Tranformando variables categoricas a numericas de forma que no haya problemas de ordinalidad
    df_preprocesado = pd.get_dummies(df_leido, dummy_na=True)

    # cargar preprocesado hecho en entrenamiento y utilizar el mismo para generar columnas vacias
    # para aquellos valores que no nos hayan aportado en prediccion
    rutaFicheroColumnasPrep = os.path.join(rutaIdModelo, 'columnas_preprocesado.pkl')
    try:
        columnas_preprocesado = joblib.load(rutaFicheroColumnasPrep)
    except IOError as e:
        print str(e)
        return "Error, no existe el fichero en: {0}".format(str(rutaFicheroColumnasPrep)), 400

    for col in columnas_preprocesado:
        if col not in df_preprocesado.columns:
            df_preprocesado[col] = 0

    # Ordenar las columnas de la misma forma que en el entrenamiento
    df_preprocesado=df_preprocesado[columnas_preprocesado]

    # CARGAR MODELO
    rutaFicheroModelo = os.path.join(rutaIdModelo, "modelo.pkl")
    try:
        modelo = joblib.load(rutaFicheroModelo)
    except IOError as e:
        print str(e)
        return "Error, no existe el fichero en: {0}".format(str(rutaFicheroModelo)), 400

    # PREDICCION
    # obtener una prediccion
    prediccion = modelo.predict(df_preprocesado)
    # Para calcular el rmse
    # rmse = math.sqrt(mean_squared_error(y, prediccion))
    # print('Test Score: %.2f RMSE' % (rmse))
    # print prediccion
    # obtener y devolver el valor predecido
    datos = list()
    datos.append({'prediccion': prediccion[0]})
    prediccion_json = json.dumps(datos)
    # print prediccion_json

    return prediccion_json


# Entrada de API REST para entrenamiento.
""" USO

Tipo de petición: POST

Llamar a: direccion:puerto/entrenamiento?nombreModelo=nombre

ENTRADA:
    Parámetros GET:  Enviar por parámetro GET nombreModelo, el cual es la identidad del modelo que se salvará en el sistema. Si se utiliza el mismo nombre que un modelo anterior se sobreescribe.

    Parámetros POST: File: Archivo en formato CSV estructurado como los que se encuentran en la carpeta datos. Estos serán los datos de entrenamiento.
                            En este fichero CSV, la última columna debe ser el target a predecir, en el caso de estados es el fallo o éxito.
                            La primera fila deben ser los nombres de las columnas, los cuales son muy importantes porque deben reutilizarse en la predicción.

SALIDA:
    Se almacena en la carpeta modelos, en una carpeta llamada como el nombreModelo aportado, el modelo generado para poder predecir con el así como todos los datos necesarios.

    La petición POST devuelve un mensaje de éxito o de error.

"""
@app.route('/alg2dis/entrenamiento', methods=['POST', 'PUT'])
def apiEntrenamiento():
    # OBTENER PARAMETROS GET
    dataGET = request.args

    if dataGET:
        if "nombreModelo" in dataGET:
            nombreModelo = dataGET["nombreModelo"]
        else:
            return "Falta parametro nombreModelo", 400
        if "numModelo" in dataGET:
            numModelo = int(dataGET["numModelo"])
            if numModelo<0 or numModelo>=len(MODELOS):
                return "numModelo no contiene un valor adecuado", 400
        else:
            numModelo = None
    else:
        return "Falta parametro nombreModelo", 400

    # OBTENER FICHERO DE DATOS PARA ENTRENAMIENTO

    # Comprobacion de envio de fichero
    if not 'file' in request.files:
        return "No se ha enviado un archivo como 'file'", 400

    # obtener fichero
    file = request.files['file']

    # PROCESAR CONTENIDO DEL FICHERO
    # contenido a string
    file_contents_string = file.stream.read().decode("utf-8")

    # string a fichero simulado
    file_string_IO = StringIO.StringIO(file_contents_string)

    # ENTRENAR Y OBTENER RESULTADOS
    # entrenamiento
    return entrenar(file_string_IO, nombreModelo, numModelo)


# Entrada de API REST para predicción.
""" USO

Tipo de petición: POST

Llamar a: direccion:puerto/prediccion?nombreModelo=nombre

ENTRADA:
    Parámetros GET:  Enviar por parámetro GET nombreModelo, el cual es la identidad del modelo que debe haber sido entrenado previamente.

    Parámetros POST: File: Archivo en formato CSV estructurado como los que se encuentran en la carpeta datos. Estos serán los datos que se utilizarán de los estados para predecir la probabilidad de fallo.
                            En este CSV no debe estar la columna target, ya que es desconocida y es de la que queremos averiguar la probabilidad.
                            Debe contener el resto de columnas utilizadas en entrenamiento. Importante que la primera fila es el nombre de las columnas y este nombre de columnas debe contener todas aquellas utilizadas en entrenamiento manteniendo los nombres.
                            En caso de no mantenerlos el código los rellena a ceros y no da error actualmente, pero es un error.
SALIDA:
    La petición POST devuelve un mensaje de error o texto en formato CSV donde la primera fila son los valores de la columna a predecir y cada una de las siguientes filas es la probabilidad de error para esos valores de la primera.

"""
@app.route('/alg2dis/prediccion', methods=['POST', 'PUT'])
def apiPrediccion():
    # OBTENER PARAMETROS GET
    dataGET = request.args

    if dataGET and "nombreModelo" in dataGET:
        nombreModelo = dataGET["nombreModelo"]
    else:
        return "Falta parametro de data dump modelo", 400

    # Comprobaremos si es csv o json ya que para las pruebas se usó csv
    # pero desde la app se envian los datos en formato json
    if tipoRequest(request) == "csv":

        # OBTENER FICHERO DE DATOS PARA PREDICCION
        file = request.files['file']

        # contenido a string
        stream = io.StringIO(file.stream.read().decode("utf-8-sig"), newline=None)

        # string a fichero simulado
        # file_string_IO = StringIO.StringIO(file_contents_string)
        sep=detectarDelimitador(stream)
        dataset = read_csv(stream,sep=sep)

    elif tipoRequest(request) == "json":
        dataJson = request.get_json()
        dataset = pd.DataFrame(dataJson,index=[0])

    else:
        return "No se han recibido datos"

    # PREDECIR Y OBTENER RESULTADOS
    # prediccion
    prediccion = predecir(dataset, nombreModelo)

    return prediccion


if __name__ == '__main__':
    app.run(debug=True)


