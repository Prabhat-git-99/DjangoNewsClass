from django.urls import path
import classification.views as views

urlpatterns = [
    path('classify/', views.News_Classification.as_view(), name = 'api_classify'),
]