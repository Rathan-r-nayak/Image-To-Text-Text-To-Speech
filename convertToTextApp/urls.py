from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.convertImageToText, name='convertImageToText'),
    path('extractText', views.extractText, name='extractText')
]
