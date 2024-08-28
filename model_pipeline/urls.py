from django.urls import path
from .views import TrainModel, PredictView

urlpatterns = [
    path('train/', TrainModel.as_view(), name='train'),
    path('predict/', PredictView.as_view(), name='predict')
]