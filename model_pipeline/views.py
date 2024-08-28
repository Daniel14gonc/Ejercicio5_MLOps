from django.shortcuts import render
from model_pipeline.services.training import Training
from model_pipeline.services.prediction import Predictor
from rest_framework.response import Response 
from rest_framework import status
from rest_framework.views import APIView

predictor = Predictor()

class TrainModel(APIView):
    def get(self, request):
        training = Training()
        response_dict = training.train(request)
        print(response_dict)
        return Response(response_dict)

class PredictView(APIView):
    def post(self, request):
        try:
            # Obtener los datos del request
            data = request.data

            # Realizar la predicción utilizando el servicio de predicción
            predictions = predictor.predict(data)
            
            response = {
                'predictions': predictions
            }

            return Response(response, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)