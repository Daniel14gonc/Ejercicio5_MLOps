import os 
import logging

from rest_framework import status 
from urllib.parse import urlparse

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle


base_path = os.getcwd() 
data_path=os.path.normpath(base_path+os.sep+'data')
pickle_path=os.path.normpath(base_path+os.sep+'models')
log_path=os.path.normpath(base_path+os.sep+'logs')

if not os.path.exists(log_path):
    os.makedirs(log_path)

log_file = os.path.join(log_path, 'training.log')
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

class Training: 

    def accuracy_measures(self, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"Precisión del modelo: {accuracy * 100:.2f}%")

        # Mostrar el reporte de clasificación
        logging.info("Reporte de clasificación:")
        logging.info(classification_report(y_test, y_pred))

        # Mostrar la matriz de confusión
        logging.info("Matriz de confusión:")
        logging.info(confusion_matrix(y_test, y_pred))

        return accuracy   
    
    def best_features(self, pipeline, X):
        selector = pipeline.named_steps['selector']
        mask = selector.get_support()  # Array booleano de las características seleccionadas
        selected_features = X.columns[mask]
        logging.info("Características seleccionadas: %s", selected_features)

    def train(self, request): 
        return_dict=dict()
        try: 
            data = pd.read_csv(data_path + '/train_data.csv')
            data['categoria_precio'] = pd.qcut(data['Price'], q=4, labels=['Bajo', 'Medio Bajo', 'Medio Alto', 'Alto'])
            df_numerico = data.select_dtypes(include=['number']).dropna(axis=1)
            df_numerico = df_numerico.drop(columns=['Longtitude', 'Lattitude'])
            X = df_numerico.drop(columns=['Price'])
            y = data['categoria_precio']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            pipeline = Pipeline([
                ('scaler', StandardScaler()),  
                ('selector', SelectKBest(score_func=f_classif, k=7)),  
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))  
            ])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            accuracy = self.accuracy_measures(y_test, y_pred)
            self.best_features(pipeline, X)
            
            pickle_file = os.path.normpath(pickle_path+os.sep+'model.pkl')
            pickle.dump(pipeline, open(pickle_file, 'wb'))

            selector = pipeline.named_steps['selector']
            selected_columns = X.columns[selector.get_support()]
            columns_file = os.path.normpath(pickle_path + os.sep + 'selected_columns.pkl')
            with open(columns_file, 'wb') as f:
                pickle.dump(selected_columns.tolist(), f)

            return_dict['response'] = 'Model Trained Successfully'
            return_dict['status']=status.HTTP_200_OK
            return return_dict 

        except Exception as e: 
            return_dict['response']="Exception when training the module: "+str(e.__str__)
            return_dict['status']=status.HTTP_500_INTERNAL_SERVER_ERROR
            return return_dict 