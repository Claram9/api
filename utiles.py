import os, errno
import csv

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def listToCSV(lista, filename):
    lista = lista
    with open(filename, "wb") as f:
        writer = csv.writer(f,delimiter =';')
        writer.writerows(lista)

def unicode2utf8(lista):
    return [x.encode('utf-8') if isinstance(x, unicode) else x for x in lista]