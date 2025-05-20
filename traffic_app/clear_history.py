"""
Script pour nettoyer complètement l'historique des données de trafic et d'accident
Ce script efface toutes les données des tables principales pour un nouveau départ.
"""

import os
import sys
import django

# Configurer l'environnement Django
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "traffic_project.settings")
django.setup()

from traffic_app.models import VideoUpload, TrafficData, Vehicle, Accident
from django.db import transaction

def clear_all_history():
    """Effacer complètement toutes les données liées au trafic et aux accidents"""
    try:
        with transaction.atomic():
            # Effacer d'abord les tables dépendantes
            print("Suppression des données d'accidents...")
            accident_count = Accident.objects.count()
            Accident.objects.all().delete()
            
            print("Suppression des données de véhicules...")
            vehicle_count = Vehicle.objects.count()
            Vehicle.objects.all().delete()
            
            print("Suppression des données de trafic...")
            traffic_count = TrafficData.objects.count()
            TrafficData.objects.all().delete()
            
            print("Suppression des uploads de vidéos...")
            upload_count = VideoUpload.objects.count()
            VideoUpload.objects.all().delete()
            
            print(f"Nettoyage terminé avec succès!")
            print(f"Éléments supprimés: {accident_count} accidents, {vehicle_count} véhicules, "
                  f"{traffic_count} données de trafic, {upload_count} vidéos")
            
            return True
    except Exception as e:
        print(f"Erreur lors du nettoyage: {e}")
        return False

if __name__ == "__main__":
    print("Début du nettoyage de l'historique...")
    result = clear_all_history()
    print("Terminé" if result else "Échec du nettoyage") 