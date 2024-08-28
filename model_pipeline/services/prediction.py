# prediction.py
import os
import pickle
import pandas as pd

base_path = os.getcwd()
pickle_path = os.path.normpath(base_path + os.sep + 'models')

class Predictor:
    def __init__(self):
        # Cargar el modelo y las columnas seleccionadas al inicializar la clase
        self.model = self.load_model()
        self.selected_columns = self.load_selected_columns()

    def load_model(self):
        try:
            pickle_file = os.path.normpath(pickle_path + os.sep + 'model.pkl')
            with open(pickle_file, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")

    def load_selected_columns(self):
        try:
            columns_file = os.path.normpath(pickle_path + os.sep + 'selected_columns.pkl')
            with open(columns_file, 'rb') as f:
                selected_columns = pickle.load(f)
            return selected_columns
        except Exception as e:
            raise Exception(f"Error loading selected columns: {str(e)}")

    def predict(self, data):
        """
        Método para predecir una o muchas observaciones.
        `data` puede ser un diccionario de una observación o una lista de diccionarios.
        """
        try:
            if isinstance(data, dict):  # Si es un solo diccionario, convertir a DataFrame
                input_data = pd.DataFrame([data])
            elif isinstance(data, list):  # Si es una lista de diccionarios, convertir a DataFrame
                input_data = pd.DataFrame(data)
            else:
                raise ValueError("Invalid data format. Must be a dict or list of dicts.")

            # Usar solo las columnas seleccionadas para la predicción
            input_data = input_data[self.selected_columns]

            # Realizar la predicción
            predictions = self.model.predict(input_data)
            return predictions.tolist()  # Devolver la predicción como lista para mayor flexibilidad
        except Exception as e:
            raise Exception(f"Error during prediction: {str(e)}")
