from django.urls import path
from . import views
from .views import stats_view, clear_history

urlpatterns = [
    path('', views.home, name='home'),  # Page d'accueil principale
    path('analyze/', views.analyze_local_video, name='analyze_local'),  # Page d'analyse
    path('map/', views.map_view, name='map_view'),  # Carte
    path('stats/', views.stats_view, name='stats'),  # Statistiques
    path('upload-video/', views.upload_video, name='upload_video'),  # API pour l'upload de vidéo
    path('api/traffic-data/', views.get_traffic_data, name='traffic_data'),
    path('clear-history/', views.clear_history, name='clear_history'),
    path('deep-clean/', views.force_deep_clean, name='force_deep_clean'),
    path('api/accidents/', views.get_accidents, name='get_accidents'),
    path('api/accident-alert/', views.check_accident_alert, name='check_accident_alert'),
    path('api/reset-accident-alerts/', views.reset_accident_alerts, name='reset_accident_alerts'),
    path('webcam-feed/', views.webcam_feed, name='webcam_feed'),
]
