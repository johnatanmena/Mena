import logging
import azure.functions as func
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import json
import pickle
import joblib 
import pandas as pd
import pyodbc 


def main(req: func.HttpRequest) -> func.HttpResponse:
    req_body = req.get_json()
    variable1 = req_body.get('variable1')
    azuredriver = "ODBC Driver 17 for SQL Server"
    azurebase = "EEG"
    usuario = "admi_sensores_1" 
    password = "ITM_2020"
    server = "sensores-2020.database.windows.net"
    connStr = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+azurebase+';UID='+usuario+';PWD='+ password)
    cursor = connStr.cursor()
    SQL_Script = "SELECT * FROM dbo.EEG_DATA_FINAL_2"
    df = pd.io.sql.read_sql(SQL_Script,connStr)
    connStr.close()
    Datos= df.to_numpy()
    X=Datos[:,:-1]
    Y=Datos[:,-1]
    m,n = X.shape
    Y = Y.reshape((m,1))
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=42)
    modelo = SVC(kernel='linear')
    modelo.fit(X_train, Y_train)
    predicciones = modelo.predict(X_test)
    AccActual = accuracy_score(Y_test,predicciones)
    json_response = json.dumps(classification_report(Y_test, predicciones),indent=2)
    if variable1 < 10:
        return func.HttpResponse(json_response)
    else:
        return func.HttpResponse("NUBE Puede que se ingresara in valor mal en el postman pero la funcion se ejecuto meleramente",status_code=200)
