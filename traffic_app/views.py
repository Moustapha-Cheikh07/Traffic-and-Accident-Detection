from django.shortcuts import render
from django.http import JsonResponse, FileResponse
from .models import TrafficData, Vehicle, VideoUpload, Accident, RoadEvent
from .detection.video_processing import VideoProcessor, process_video_feed
from django.views.decorators.csrf import csrf_exempt
import json
import folium
from datetime import datetime, timedelta
import os
import logging
from django.conf import settings
import shutil
import time
from django.db.models import Count, Q, Sum, Avg, F
from django.db.models.functions import ExtractHour
from django.db import transaction
import base64
import numpy as np
import cv2

logger = logging.getLogger(__name__)

def index(request):
    return render(request, 'index.html')

def home(request):
    """Vue de la page d'accueil"""
    return render(request, 'Acceuil.html')

from django.shortcuts import render
from django.http import JsonResponse
import os
import shutil
import logging
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from .models import TrafficData  # Assurez-vous d'importer votre modèle TrafficData

logger = logging.getLogger(__name__)

# Liste des villes avec leurs coordonnées (latitude, longitude)
CITIES = {
    "Nouakchott": [18.0735, -15.9582],
    "Nouadhibou": [20.9333, -17.0333],
    "Kiffa": [16.6228, -11.4058],
    "Kaédi": [16.1500, -13.5000],
    "Sélibaby": [15.1594, -12.1847],
    "Atar": [20.5170, -13.0486],
    "Zouerate": [22.7333, -12.4667],
    "Rosso": [16.5138, -15.8050],
    "Néma": [16.6167, -7.2667],
    "Aleg": [17.0536, -13.9094],
    "Boutilimit": [17.5500, -14.6833],
}

@csrf_exempt
def analyze_local_video(request):
    if request.method == 'GET':
        return render(request, 'analyze_local.html', {'cities': CITIES})
    elif request.method == 'POST':
        video_file = request.FILES.get('video')
        latitude = request.POST.get('latitude')
        longitude = request.POST.get('longitude')

        logger.info(f"Tentative d'analyse de la vidéo.")

        # Vérification du fichier vidéo
        if not video_file:
            return JsonResponse({'error': 'Aucune vidéo sélectionnée.'})

        # Vérifier si cette vidéo a déjà été traitée récemment (dans les 10 dernières minutes)
        video_filename = video_file.name
        file_size = video_file.size  # Obtenir la taille du fichier
        
        # Chercher les TrafficData avec le même nom de fichier exactement
        recent_videos = TrafficData.objects.filter(
            video_source__endswith=video_filename,
            timestamp__gte=datetime.now() - timedelta(minutes=10)
        )
        
        if recent_videos.exists():
            recent_video = recent_videos.first()
            # Vérifier si le fichier source existe
            if os.path.exists(recent_video.video_source):
                # Vérifier si la taille du fichier est similaire (pour éviter les faux positifs)
                stored_size = os.path.getsize(recent_video.video_source)
                if abs(stored_size - file_size) < 1024:  # Tolérance de 1KB
                    # Si nous avons une correspondance exacte, retourner les résultats de l'analyse précédente
                    
                    # Récupérer l'URL de la vidéo traitée
                    output_video = os.path.basename(recent_video.video_source)
                    video_url = f'/media/processed_videos/{output_video}'
                    
                    # Récupérer les types de véhicules pour ce traffic_data
                    vehicle_types = {}
                    for vehicle in Vehicle.objects.filter(traffic_data=recent_video):
                        vehicle_types[vehicle.vehicle_type] = getattr(vehicle, 'count', 1)
                    
                    vehicle_type_summary = ", ".join([f"{count} {vtype}" for vtype, count in vehicle_types.items()])
                    
                    return JsonResponse({
                        'success': True,
                        'message': 'Cette vidéo a déjà été analysée récemment',
                        'results': {
                            'vehicle_count': recent_video.vehicle_count,
                            'vehicle_types': vehicle_types,
                            'vehicle_type_summary': vehicle_type_summary,
                            'congestion_level': recent_video.congestion_level,
                            'video_url': video_url
                        }
                    })

        # Sauvegarder la vidéo téléchargée
        video_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, video_file.name)

        with open(video_path, 'wb+') as destination:
            for chunk in video_file.chunks():
                destination.write(chunk)

        logger.info(f"Vidéo reçue et sauvegardée : {video_path}")

        if not latitude or not longitude:
            return JsonResponse({'error': 'Latitude et Longitude requises pour l\'analyse.'})

        try:
            processor = VideoProcessor()
            logger.info("Début du traitement de la vidéo...")

            # Traiter la vidéo
            result = processor.process_video_feed(video_path)
            if not result:
                return JsonResponse({'error': 'Échec du traitement de la vidéo'})

            # Copier la vidéo traitée dans le dossier media
            media_dir = os.path.join(settings.MEDIA_ROOT, 'processed_videos')
            os.makedirs(media_dir, exist_ok=True)
            output_video = os.path.basename(result['output_video'])
            media_path = os.path.join(media_dir, output_video)
            shutil.copy2(result['output_video'], media_path)

            # Obtenir les résultats améliorés du traitement
            unique_vehicle_count = result.get('unique_vehicle_count', 0)
            max_vehicles_per_frame = result.get('max_vehicles_per_frame', 0)
            vehicle_types_data = result.get('vehicle_types', {})
            
            # Calculer le niveau moyen de congestion
            congestion_level = 0
            if result.get('results'):
                congestion_level = sum(r['congestion_level'] for r in result['results']) / len(result['results'])

            # Enregistrer les résultats dans la base de données
            traffic_data = TrafficData.objects.create(
                vehicle_count=unique_vehicle_count,
                congestion_level=congestion_level,
                latitude=float(latitude),
                longitude=float(longitude),
                video_source=video_path
            )

            # Enregistrer les véhicules détectés selon les types
            for vehicle_type, count in vehicle_types_data.items():
                Vehicle.objects.create(
                    traffic_data=traffic_data,
                    vehicle_type=vehicle_type,
                    confidence_score=1.0,  # Valeur par défaut
                    count=count  # Ajouter ce champ au modèle Vehicle si nécessaire
                )
                
            # Also create a VideoUpload record to ensure consistency with the stats page
            from django.core.files import File
            with open(video_path, 'rb') as f:
                file_object = File(f, name=os.path.basename(video_path))
                video_upload = VideoUpload.objects.create(
                    video_file=file_object,
                    location=f"{latitude}, {longitude}",
                    status='completed',
                    processed=True,
                    vehicle_count=unique_vehicle_count,
                    congestion_level=congestion_level,
                    processing_time=(time.time() - start_time)
                )
                
            logger.info(f"[STATS DEBUG] Created matching records: TrafficData ID={traffic_data.id}, VideoUpload ID={video_upload.id}")

            logger.info(f"Analyse terminée avec succès. {unique_vehicle_count} véhicules uniques détectés.")

            # Créer l'URL de la vidéo traitée
            video_url = f'/media/processed_videos/{output_video}'
            
            # Créer un résumé des types de véhicules pour l'interface utilisateur
            vehicle_type_summary = ", ".join([f"{count} {vtype}" for vtype, count in vehicle_types_data.items()])

            return JsonResponse({
                'success': True,
                'results': {
                    'vehicle_count': unique_vehicle_count,
                    'max_vehicles_per_frame': max_vehicles_per_frame,
                    'vehicle_types': vehicle_types_data,
                    'vehicle_type_summary': vehicle_type_summary,
                    'congestion_level': congestion_level,
                    'video_url': video_url
                }
            })

        except Exception as e:
            logger.error(f"Erreur lors de l'analyse de la vidéo: {str(e)}")
            return JsonResponse({'error': f'Erreur lors de l\'analyse de la vidéo: {str(e)}'})

    return render(request, 'analyze_local.html', {'cities': CITIES})




import folium
from django.shortcuts import render
from django.db.models import Q
from datetime import datetime
from .models import TrafficData

import folium
from django.shortcuts import render
from django.db.models import Q
from datetime import datetime
from .models import TrafficData

import folium
from django.shortcuts import render
from django.db.models import Q
from datetime import datetime
from .models import TrafficData

# Définition des coordonnées des villes
CITIES = {
    "Nouakchott": [18.0735, -15.9582],
    "Nouadhibou": [20.9333, -17.0333],
    "Kiffa": [16.6228, -11.4058],
    "Kaédi": [16.1500, -13.5000],
    "Sélibaby": [15.1594, -12.1847],
    "Atar": [20.5170, -13.0486],
    "Zouerate": [22.7333, -12.4667],
    "Rosso": [16.5138, -15.8050],
    "Néma": [16.6167, -7.2667],
    "Aleg": [17.0536, -13.9094],
    "Boutilimit": [17.5500, -14.6833],
}

import folium
from django.shortcuts import render
from django.db.models import Q
from datetime import datetime
from .models import TrafficData

def map_view(request):
    available_dates = TrafficData.objects.dates('timestamp', 'day', order='DESC')
    available_locations = TrafficData.objects.values_list('location_name', flat=True).distinct()

    selected_city = request.GET.get('selected_city', 'Nouakchott')
    selected_location = request.GET.get('selected_location', '')
    selected_date = request.GET.get('selected_date', '')
    show_accidents = request.GET.get('show_accidents', 'true') == 'true'

    traffic_filters = Q()

    if selected_city in CITIES:
        traffic_filters &= Q(location_name__icontains=selected_city)

    if selected_location:
        traffic_filters &= Q(location_name__icontains=selected_location)

    if selected_date:
        try:
            selected_date_obj = datetime.strptime(selected_date, "%Y-%m-%d").date()
            traffic_filters &= Q(timestamp__date=selected_date_obj)
        except ValueError:
            selected_date = ''

    traffic_data = TrafficData.objects.filter(traffic_filters).order_by('-timestamp')

    # Filtrage pour les accidents
    accident_filters = Q()
    if selected_date:
        try:
            selected_date_obj = datetime.strptime(selected_date, "%Y-%m-%d").date()
            accident_filters &= Q(timestamp__date=selected_date_obj)
        except ValueError:
            pass
    if selected_location:
        accident_filters &= Q(traffic_data__location_name__icontains=selected_location)
    if selected_city in CITIES:
        accident_filters &= Q(traffic_data__location_name__icontains=selected_city)
    
    # Récupérer les accidents si demandé
    accidents = []
    if show_accidents:
        accidents = Accident.objects.filter(accident_filters).order_by('-timestamp')

    # Centrer la carte sur la première localisation trouvée ou la ville par défaut
    if traffic_data.exists():
        first_data = traffic_data.first()
        map_center = [first_data.latitude, first_data.longitude]
    else:
        map_center = CITIES.get(selected_city, [18.0735, -15.9582])

    # Si focus sur un accident spécifique
    accident_focus = request.GET.get('accident', 'false') == 'true'
    zoom_start = 13
    if accident_focus and request.GET.get('lat') and request.GET.get('lng'):
        try:
            lat = float(request.GET.get('lat'))
            lng = float(request.GET.get('lng'))
            map_center = [lat, lng]
            zoom_start = 18  # Zoom plus proche pour la vue d'accident
        except ValueError:
            zoom_start = 13

    traffic_map = folium.Map(location=map_center, zoom_start=zoom_start)

    # Ajouter tous les marqueurs pour le trafic
    for data in traffic_data:
        if data.congestion_level >= 0.8:
            color = 'red'
            icon = 'exclamation-triangle'
        elif data.congestion_level >= 0.5:
            color = 'orange'
            icon = 'car'
        else:
            color = 'green'
            icon = 'check-circle'

        popup_info = f'''
        <b>Localisation :</b> {data.location_name} <br>
        <b>Véhicules :</b> {data.vehicle_count} <br>
        <b>Congestion :</b> {data.congestion_level:.1%} <br>
        <b>Heure :</b> {data.timestamp.strftime('%H:%M')} <br>
        '''

        if data.congestion_level >= 0.8:
            popup_info += "<b style='color:red;'>🚨 EMBOUTEILLAGE 🚨</b>"

        folium.Marker(
            location=[data.latitude, data.longitude],
            popup=folium.Popup(popup_info, max_width=300),
            icon=folium.Icon(color=color, icon=icon, prefix='fa'),
            tooltip=data.location_name
        ).add_to(traffic_map)
    
    # Ajouter les marqueurs d'accidents
    for accident in accidents:
        # Utiliser une couleur différente selon la gravité
        if accident.severity == 3:  # Sévère
            accident_color = 'darkred'
        elif accident.severity == 2:  # Modéré
            accident_color = 'red'
        else:  # Mineur
            accident_color = 'orange'
            
        accident_popup = f'''
        <b style='color:red;font-size:14px;'>⚠️ ACCIDENT</b><br>
        <b>Date :</b> {accident.timestamp.strftime('%d/%m/%Y %H:%M')}<br>
        <b>Sévérité :</b> {accident.get_severity_display()}<br>
        <b>Confiance :</b> {accident.confidence_score:.1%}<br>
        <b>Localisation :</b> {accident.traffic_data.location_name}<br>
        '''
        
        folium.Marker(
            location=[accident.latitude, accident.longitude],
            popup=folium.Popup(accident_popup, max_width=300),
            icon=folium.Icon(color=accident_color, icon='ambulance', prefix='fa'),
            tooltip="Accident"
        ).add_to(traffic_map)

    context = {
        'map': traffic_map._repr_html_(),  # Assurez-vous que la carte est bien rendue
        'available_dates': available_dates,
        'available_locations': available_locations,
        'selected_date': selected_date,
        'selected_city': selected_city,
        'selected_location': selected_location,
        'cities': CITIES.keys(),
        'show_accidents': show_accidents,
        'accident_count': accidents.count() if show_accidents else 0
    }
    return render(request, 'map.html', context)







from django.shortcuts import render
from django.db.models import Q
from datetime import datetime
from .models import TrafficData, VideoUpload, Vehicle

from django.shortcuts import render
from django.db.models import Q
from datetime import datetime
from .models import TrafficData, VideoUpload, Vehicle

from django.shortcuts import render
from django.db.models import Q
from datetime import datetime
from .models import TrafficData, VideoUpload, Vehicle

from django.shortcuts import render
from django.db.models import Q
from datetime import datetime
from .models import TrafficData, VideoUpload, Vehicle

def stats_view(request):
    try:
        # Force a cache refresh using the current timestamp
        cache_buster = str(datetime.now().timestamp())
        print(f"[DEBUG] Loading stats with cache buster: {cache_buster}")
        
        # Log current database state for debugging
        total_videos = VideoUpload.objects.count()
        total_traffic_data = TrafficData.objects.count()
        total_vehicles = Vehicle.objects.count()
        total_accidents = Accident.objects.count()
        print(f"[DEBUG] Database state: {total_videos} videos, {total_traffic_data} traffic data, {total_vehicles} vehicles, {total_accidents} accidents")
        
        # Print some sample records for debugging
        print("[DEBUG] Recent VideoUploads:")
        for upload in VideoUpload.objects.all().order_by('-upload_date')[:5]:
            print(f"  - ID: {upload.id}, File: {upload.video_file}, Date: {upload.upload_date}, Count: {upload.vehicle_count}")
        
        print("[DEBUG] Recent TrafficData:")
        for td in TrafficData.objects.all().order_by('-timestamp')[:5]:
            print(f"  - ID: {td.id}, Source: {td.video_source}, Date: {td.timestamp}, Count: {td.vehicle_count}")
        
        # Clear any potential Django cache for this view
        from django.core.cache import cache
        cache.clear()
        
        # Force a database connection refresh to ensure we get the latest data
        from django.db import connection
        connection.close()
        connection.connect()
        
        # Récupérer les dates et localisations uniques pour les filtres - avec requêtes fraîches
        from django.db.models import Max
        available_dates = VideoUpload.objects.dates('upload_date', 'day', order='DESC')
        available_locations = VideoUpload.objects.values_list('location', flat=True).distinct()
        
        # Obtenir les filtres de date et localisation
        selected_date = request.GET.get('selected_date', '')
        selected_location = request.GET.get('selected_location', '')
        
        # Construire les filtres
        upload_filters = Q()
        traffic_filters = Q()
        accident_filters = Q()
        
        if selected_date:
            try:
                selected_date_obj = datetime.strptime(selected_date, "%Y-%m-%d").date()
                upload_filters &= Q(upload_date__date=selected_date_obj)
                traffic_filters &= Q(timestamp__date=selected_date_obj)
                accident_filters &= Q(timestamp__date=selected_date_obj)
            except ValueError:
                pass
                
        if selected_location:
            upload_filters &= Q(location__icontains=selected_location)
            traffic_filters &= Q(location_name__icontains=selected_location)
            accident_filters &= Q(traffic_data__location_name__icontains=selected_location)
        
        # IMPORTANT: Get both VideoUpload and TrafficData records to ensure we have the complete picture
        uploads = list(VideoUpload.objects.filter(upload_filters).order_by('-upload_date').all())
        traffic_data_records = list(TrafficData.objects.filter(traffic_filters).order_by('-timestamp').all())
        
        # Merge the data to ensure we count everything - there may be data in TrafficData not linked to VideoUpload
        upload_vehicle_count = sum(upload.vehicle_count or 0 for upload in uploads)
        traffic_data_vehicle_count = sum(td.vehicle_count or 0 for td in traffic_data_records)
        
        # Use the larger count to be safe
        total_videos = len(uploads)
        total_vehicles = max(upload_vehicle_count, traffic_data_vehicle_count)
        
        # Print what we're using for stats
        print(f"[DEBUG] Using for stats: {total_videos} videos, {total_vehicles} vehicles")
        print(f"[DEBUG] From uploads: {upload_vehicle_count} vehicles")
        print(f"[DEBUG] From traffic data: {traffic_data_vehicle_count} vehicles")
        
        # Convertir les niveaux de congestion en pourcentage pour l'affichage
        avg_congestion = 0
        
        # Get congestion from both sources and use the most reliable
        if uploads:
            # Ensure that we're handling null values correctly and calculating percentage
            valid_congestion_levels = [upload.congestion_level for upload in uploads if upload.congestion_level is not None]
            if valid_congestion_levels:
                upload_avg_congestion = sum(valid_congestion_levels) / len(valid_congestion_levels) * 100
                avg_congestion = upload_avg_congestion
        
        if traffic_data_records and avg_congestion == 0:
            valid_td_congestion = [td.congestion_level for td in traffic_data_records if td.congestion_level is not None]
            if valid_td_congestion:
                td_avg_congestion = sum(valid_td_congestion) / len(valid_td_congestion) * 100
                # Use traffic data congestion if available
                avg_congestion = td_avg_congestion
        
        # Déterminer l'heure de pointe
        try:
            # Try both VideoUpload and TrafficData for peak hour
            upload_hour_data = VideoUpload.objects.filter(upload_filters).annotate(
                hour=ExtractHour('upload_date')
            ).values('hour').annotate(
                count=Count('id'),
                avg_vehicles=Avg('vehicle_count')
            ).order_by('-avg_vehicles')
            
            traffic_hour_data = TrafficData.objects.filter(traffic_filters).annotate(
                hour=ExtractHour('timestamp')
            ).values('hour').annotate(
                count=Count('id'),
                avg_vehicles=Avg('vehicle_count')
            ).order_by('-avg_vehicles')
            
            peak_hour = None
            if upload_hour_data.exists():
                peak_hour = upload_hour_data.first()['hour']
            elif traffic_hour_data.exists():
                peak_hour = traffic_hour_data.first()['hour']
        except Exception as e:
            print(f"[DEBUG] Error calculating peak hour: {str(e)}")
            peak_hour = None
        
        # Format des données pour les graphiques
        try:
            analysis_history = list(VideoUpload.objects.filter(upload_filters).annotate(
                date=F('upload_date__date')
            ).values('id', 'upload_date', 'location', 'vehicle_count', 'congestion_level', 'processing_time', 'status').all())
        except Exception as e:
            print(f"[DEBUG] Error retrieving analysis history: {str(e)}")
            analysis_history = []
        
        # Tendances du trafic - force fresh query
        try:
            traffic_trend = list(VideoUpload.objects.filter(upload_filters).values('upload_date__date').annotate(
                date=F('upload_date__date'),
                avg_vehicles=Avg('vehicle_count'),
                avg_congestion=Avg('congestion_level')
            ).order_by('date').all())
            
            traffic_labels = [entry['date'].strftime('%d/%m/%Y') for entry in traffic_trend]
            traffic_data_values = [float(entry['avg_vehicles'] or 0) for entry in traffic_trend]
        except Exception as e:
            print(f"[DEBUG] Error calculating traffic trends: {str(e)}")
            traffic_labels = []
            traffic_data_values = []
        
        # Répartition des types de véhicules - force fresh query with connection reset
        connection.close()
        connection.connect()
        
        # IMPORTANT CHANGE: Use both VideoUpload and TrafficData to get vehicle counts
        # First, collect all the Traffic Data IDs
        related_traffic_data_ids = []
        
        # 1. Get IDs directly from TrafficData
        related_traffic_data_ids.extend(td.id for td in traffic_data_records)
        
        # 2. Also try to match VideoUploads to TrafficData by filename
        video_filenames = VideoUpload.objects.filter(
            upload_filters
        ).values_list('video_file', flat=True)
        
        for filename in video_filenames:
            if filename:
                base_filename = os.path.basename(filename)
                traffic_data_ids = TrafficData.objects.filter(
                    video_source__endswith=base_filename
                ).values_list('id', flat=True)
                related_traffic_data_ids.extend(list(traffic_data_ids))
        
        # Remove duplicates
        related_traffic_data_ids = list(set(related_traffic_data_ids))
        
        # Log ID mappings for debugging
        print(f"[DEBUG] Found {len(related_traffic_data_ids)} TrafficData records for vehicle analysis")
        
        # Then use these TrafficData IDs to fetch related vehicles
        vehicle_distribution = []
        vehicle_types = []
        vehicle_counts = []
        
        try:
            # Use coalesce to handle NULL count values by converting them to 1
            from django.db.models.functions import Coalesce
            from django.db.models import Value
            
            vehicle_distribution = list(Vehicle.objects.filter(
                traffic_data__id__in=related_traffic_data_ids
            ).values('vehicle_type').annotate(
                count=Sum(Coalesce('count', Value(1)))
            ).order_by('-count').all())
            
            print(f"[DEBUG] Vehicle distribution query successful: found {len(vehicle_distribution)} types")
            
            vehicle_types = [item['vehicle_type'] for item in vehicle_distribution]
            vehicle_counts = [item['count'] for item in vehicle_distribution]
        except Exception as e:
            print(f"[DEBUG] Error in vehicle distribution query: {str(e)}")
            # Fallback to basic query
            try:
                vehicle_distribution = list(Vehicle.objects.filter(
                    traffic_data__id__in=related_traffic_data_ids
                ).values('vehicle_type').annotate(
                    count=Count('id')
                ).order_by('-count').all())
                print(f"[DEBUG] Fallback query returned {len(vehicle_distribution)} vehicle types")
                
                vehicle_types = [item['vehicle_type'] for item in vehicle_distribution]
                vehicle_counts = [item['count'] for item in vehicle_distribution]
            except Exception as e2:
                print(f"[DEBUG] Error in fallback vehicle query: {str(e2)}")
                vehicle_types = []
                vehicle_counts = []
        
        # Statistiques d'accidents - force fresh query
        connection.close()
        connection.connect()
        
        try:
            accidents = list(Accident.objects.filter(accident_filters).all())
            accident_count = len(accidents)
            
            # Obtenir les points noirs d'accidents (endroits avec le plus d'accidents)
            accident_hotspots = list(Accident.objects.filter(accident_filters).values('traffic_data__location_name').annotate(
                count=Count('id')
            ).order_by('-count')[:5].all())
            
            # S'assurer que nous avons des données de points noirs, sinon créer des valeurs par défaut
            if not accident_hotspots:
                accident_hotspot_labels = ["Aucune donnée"]
                accident_hotspot_counts = [0]
            else:
                accident_hotspot_labels = [spot['traffic_data__location_name'] or "Lieu inconnu" for spot in accident_hotspots]
                accident_hotspot_counts = [spot['count'] for spot in accident_hotspots]
            
            # Répartition des accidents par sévérité - force fresh query
            connection.close()
            connection.connect()
            severity_counts = list(Accident.objects.filter(accident_filters).values('severity').annotate(
                count=Count('id')
            ).order_by('severity').all())
            
            severity_labels = ["Mineur", "Modéré", "Sévère"]
            severity_data = [0, 0, 0]  # Initialisation avec des zéros
            
            for item in severity_counts:
                if 1 <= item['severity'] <= 3:  # Vérifier que l'indice est valide
                    severity_data[item['severity']-1] = item['count']
        except Exception as e:
            print(f"[DEBUG] Error processing accident data: {str(e)}")
            accident_count = 0
            accident_hotspot_labels = ["Aucune donnée"]
            accident_hotspot_counts = [0]
            severity_labels = ["Mineur", "Modéré", "Sévère"]
            severity_data = [0, 0, 0]
        
        # Final log of the data that will be sent to the template
        print(f"[DEBUG] Stats summary: {total_videos} videos, {total_vehicles} vehicles, {accident_count} accidents")
        print(f"[DEBUG] Vehicle distribution: {vehicle_types} / {vehicle_counts}")
        
        context = {
            'total_vehicles': total_vehicles,
            'total_videos': total_videos,
            'peak_hour': peak_hour,
            'avg_congestion': round(avg_congestion, 2),  # Format en pourcentage avec 2 décimales
            'traffic_labels': traffic_labels,
            'traffic_data': traffic_data_values,
            'vehicle_types': vehicle_types,
            'vehicle_counts': vehicle_counts,
            'analysis_history': analysis_history,
            'available_dates': available_dates,
            'available_locations': available_locations,
            'selected_date': selected_date,
            'selected_location': selected_location,
            # Données pour les accidents
            'accident_count': accident_count,
            'accident_hotspot_labels': json.dumps(accident_hotspot_labels),
            'accident_hotspot_counts': json.dumps(accident_hotspot_counts),
            'severity_labels': json.dumps(severity_labels),
            'severity_data': json.dumps(severity_data),
            # Add cache buster to force browser to reload
            'cache_buster': cache_buster
        }
    
        # Set cache control headers in the response
        response = render(request, 'stats.html', context)
        response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response['Pragma'] = 'no-cache'
        response['Expires'] = '0'
        return response
    except Exception as e:
        print(f"[DEBUG] Unhandled error in stats_view: {str(e)}")
        # Return a simple error page
        return render(request, 'error.html', {'error_message': f"Une erreur s'est produite lors du chargement des statistiques: {str(e)}"})



from django.http import JsonResponse
from django.shortcuts import redirect
from .models import VideoUpload, TrafficData, Vehicle, Accident
from django.db import transaction
import logging

logger = logging.getLogger(__name__)

def clear_history(request):
    if request.method == "POST":
        try:
            with transaction.atomic():
                # Supprimer toutes les entrées des tables dans l'ordre correct
                logger.info("Suppression des données d'accidents...")
                accident_count = Accident.objects.count()
                Accident.objects.all().delete()
                
                logger.info("Suppression des données de véhicules...")
                vehicle_count = Vehicle.objects.count()
                Vehicle.objects.all().delete()
                
                logger.info("Suppression des données de trafic...")
                traffic_count = TrafficData.objects.count()
                TrafficData.objects.all().delete()
                
                logger.info("Suppression des uploads de vidéos...")
                upload_count = VideoUpload.objects.count()
                VideoUpload.objects.all().delete()
                
                total_deleted = accident_count + vehicle_count + traffic_count + upload_count
                logger.info(f"Nettoyage terminé : {total_deleted} éléments supprimés")
                
                return JsonResponse({
                    'success': True, 
                    'message': f"Historique supprimé avec succès ({total_deleted} éléments)"
                })
        except Exception as e:
            logger.error(f"Erreur lors de la suppression de l'historique: {str(e)}")
            return JsonResponse({
                'success': False, 
                'message': f"Erreur lors de la suppression: {str(e)}"
            }, status=500)
    
    return JsonResponse({'success': False, 'message': "Requête invalide"}, status=400)




@csrf_exempt
def process_video(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video_file = request.FILES['video']
        
        # Vérifier si cette vidéo a déjà été traitée récemment (dans les 10 dernières minutes)
        video_filename = video_file.name
        file_size = video_file.size
        
        # Chercher les vidéos avec le même nom exactement
        recent_uploads = VideoUpload.objects.filter(
            video_file__endswith=video_filename,
            upload_date__gte=datetime.now() - timedelta(minutes=10)
        )
        
        if recent_uploads.exists():
            recent_upload = recent_uploads.first()
            # Vérifier la taille du fichier
            if abs(recent_upload.video_file.size - file_size) < 1024:  # Tolérance de 1KB
                # Récupérer les types de véhicules
                traffic_data = TrafficData.objects.filter(
                    video_source__endswith=video_filename,
                    timestamp__gte=datetime.now() - timedelta(minutes=10)
                ).first()
                
                vehicle_types = {}
                if traffic_data:
                    for vehicle in Vehicle.objects.filter(traffic_data=traffic_data):
                        vehicle_types[vehicle.vehicle_type] = getattr(vehicle, 'count', 1)
                
                # Si nous avons une correspondance exacte, retourner les résultats du traitement précédent
                return JsonResponse({
                    'success': True,
                    'message': 'Cette vidéo a déjà été analysée récemment',
                    'results': {
                        'vehicle_count': recent_upload.vehicle_count,
                        'vehicle_types': vehicle_types,
                        'congestion_level': recent_upload.congestion_level
                    }
                })
        
        # Traiter la vidéo
        processor = VideoProcessor()
        result = processor.process_video_feed(video_file)
        
        # Enrichir les résultats avec les informations supplémentaires
        if isinstance(result, dict) and 'success' in result and result['success'] and 'vehicle_types' not in result:
            result['vehicle_types'] = result.get('vehicle_types', {})
            result['max_vehicles_per_frame'] = result.get('max_vehicles_per_frame', 0)
            result['unique_vehicle_count'] = result.get('unique_vehicle_count', 0)
        
        return JsonResponse(result)
    return JsonResponse({'error': 'Invalid request'}, status=400)

def get_traffic_data(request):
    data = TrafficData.objects.all().order_by('-timestamp')[:100]
    traffic_info = [{
        'timestamp': item.timestamp,
        'vehicle_count': item.vehicle_count,
        'congestion_level': item.congestion_level,
        'latitude': item.latitude,
        'longitude': item.longitude
    } for item in data]
    return JsonResponse({'traffic_data': traffic_info})

@csrf_exempt
def upload_video(request):
    if request.method == 'POST' and request.FILES.get('video'):
        try:
            start_time = time.time()
            video_file = request.FILES['video']
            location = request.POST.get('location', 'Nouakchott, Mauritanie')
            
            # Log the start of a new analysis process for debugging
            logger.info(f"[STATS DEBUG] Starting new analysis for video: {video_file.name}")
            
            # Calculer une signature unique pour le fichier
            video_filename = video_file.name
            file_size = video_file.size
            
            # Vérifier si cette vidéo a déjà été traitée récemment (dans les 10 dernières minutes)
            # avec une correspondance exacte du nom et de la taille du fichier
            recent_uploads = VideoUpload.objects.filter(
                video_file__endswith=video_filename,
                upload_date__gte=datetime.now() - timedelta(minutes=10)
            ).order_by('-upload_date')[:10]  # Récupérer les 10 plus récents pour vérification
            
            # Vérification stricte des doublons
            for recent_upload in recent_uploads:
                # Vérifier la taille et le nom du fichier
                same_filename = recent_upload.video_file.name.endswith(video_filename)
                size_matches = abs(recent_upload.video_file.size - file_size) < 1024  # Tolérance de 1KB
                
                if same_filename and size_matches:
                    logger.info(f"[STATS DEBUG] Vidéo déjà analysée: {video_filename} (ID: {recent_upload.id})")
                    
                    # Récupérer les détails des véhicules détectés
                    traffic_data = TrafficData.objects.filter(
                        video_source__endswith=os.path.basename(recent_upload.video_file.name),
                        timestamp__gte=datetime.now() - timedelta(minutes=15)
                    ).first()
                    
                    # Obtenir le décompte des véhicules par type
                    vehicle_types = {}
                    if traffic_data:
                        for vehicle in Vehicle.objects.filter(traffic_data=traffic_data):
                            vehicle_types[vehicle.vehicle_type] = vehicle.count
                    
                    # Créer un résumé textuel des types de véhicules
                    vehicle_type_summary = ", ".join([f"{count} {vtype}" for vtype, count in vehicle_types.items()])
                    
                    # Vérifier s'il y a des accidents associés
                    has_accident = Accident.objects.filter(
                        traffic_data=traffic_data
                    ).exists() if traffic_data else False
                    
                    # Retourner les résultats précédents sans retraiter la vidéo
                    return JsonResponse({
                        'success': True,
                        'message': 'Cette vidéo a déjà été analysée récemment',
                        'duplicate': True,
                        'results': {
                            'vehicle_count': recent_upload.vehicle_count,
                            'max_vehicles_per_frame': recent_upload.vehicle_count,  # Fallback
                            'vehicle_type_summary': vehicle_type_summary,
                            'vehicle_types': vehicle_types,
                            'congestion_level': recent_upload.congestion_level,
                            'processing_time': recent_upload.processing_time,
                            'processed_video_url': recent_upload.video_file.url,
                            'has_accident': has_accident
                        }
                    })
            
            # Vérifier aussi directement les TrafficData pour des vidéos avec le même nom
            existing_traffic_data = TrafficData.objects.filter(
                video_source__endswith=video_filename,
                timestamp__gte=datetime.now() - timedelta(minutes=15)
            ).first()
            
            if existing_traffic_data:
                logger.info(f"[STATS DEBUG] TrafficData déjà existant pour cette vidéo: {video_filename}")
                # Récupérer les types de véhicules
                vehicle_types = {}
                for vehicle in Vehicle.objects.filter(traffic_data=existing_traffic_data):
                    vehicle_types[vehicle.vehicle_type] = vehicle.count
                
                # Créer un résumé textuel des types de véhicules
                vehicle_type_summary = ", ".join([f"{count} {vtype}" for vtype, count in vehicle_types.items()])
                
                # Vérifier s'il y a des accidents associés
                has_accident = Accident.objects.filter(
                    traffic_data=existing_traffic_data
                ).exists()
                
                # Récupérer la VideoUpload associée
                associated_upload = VideoUpload.objects.filter(
                    video_file__endswith=video_filename,
                    upload_date__lte=existing_traffic_data.timestamp + timedelta(minutes=1),
                    upload_date__gte=existing_traffic_data.timestamp - timedelta(minutes=1)
                ).first()
                
                if associated_upload:
                    return JsonResponse({
                        'success': True,
                        'message': 'Cette vidéo a déjà été analysée récemment (TrafficData existant)',
                        'duplicate': True,
                        'results': {
                            'vehicle_count': existing_traffic_data.vehicle_count,
                            'vehicle_type_summary': vehicle_type_summary,
                            'vehicle_types': vehicle_types,
                            'congestion_level': existing_traffic_data.congestion_level,
                            'processed_video_url': associated_upload.video_file.url if associated_upload else None,
                            'has_accident': has_accident
                        }
                    })
            
            # Si on arrive ici, la vidéo n'est pas un doublon, donc on l'analyse
            logger.info(f"[STATS DEBUG] Nouvelle vidéo à traiter: {video_filename}")
            
            # Créer l'entrée VideoUpload sans transaction d'abord
            video_upload = VideoUpload.objects.create(
                video_file=video_file,
                location=location,
                status='processing'
            )
            
            # Obtenir le chemin du fichier uploadé
            video_path = video_upload.video_file.path
            
            # Traiter la vidéo en dehors de la transaction
            processor = VideoProcessor()
            result = processor.process_video_feed(video_path)
            
            if result:
                # Calculer le temps de traitement
                processing_time = time.time() - start_time
                
                # Utiliser les données améliorées du résultat
                unique_vehicle_count = result.get('unique_vehicle_count', 0)
                max_vehicles_per_frame = result.get('max_vehicles_per_frame', 0)
                vehicle_types_data = result.get('vehicle_types', {})
                
                # Calculer le niveau de congestion (soit depuis result, soit moyen)
                congestion_level = 0
                results_list = result.get('results', [])
                if results_list:
                    congestion_level = sum(r.get('congestion_level', 0) for r in results_list) / len(results_list)
                
                # N'utiliser les transactions que pour les opérations d'écriture en base
                with transaction.atomic():
                    # Mettre à jour VideoUpload avec les résultats
                    video_upload.processed = True
                    video_upload.status = 'completed'
                    video_upload.vehicle_count = unique_vehicle_count
                    video_upload.congestion_level = congestion_level
                    video_upload.processing_time = processing_time
                    video_upload.save()
                    
                    logger.info(f"[STATS DEBUG] VideoUpload mis à jour: ID={video_upload.id}, Véhicules={unique_vehicle_count}")
                    
                    # Vérifier si un TrafficData existe déjà pour cette vidéo
                    current_time = datetime.now()
                    existing_traffic = TrafficData.objects.filter(
                        video_source__endswith=os.path.basename(video_path),
                        timestamp__gte=current_time - timedelta(minutes=1)
                    ).first()
                    
                    if existing_traffic:
                        traffic_data = existing_traffic
                        logger.info(f"[STATS DEBUG] Utilisation d'un TrafficData existant: ID={traffic_data.id}")
                    else:
                        # Créer un nouvel enregistrement TrafficData
                        traffic_data = TrafficData.objects.create(
                            vehicle_count=unique_vehicle_count,
                            congestion_level=congestion_level,
                            latitude=18.0735,  # Coordonnées de Nouakchott
                            longitude=-15.9582,
                            video_source=video_path,
                            location_name=location,
                            timestamp=current_time  # Utiliser l'heure actuelle
                        )
                        logger.info(f"[STATS DEBUG] Nouveau TrafficData créé: ID={traffic_data.id}, Véhicules={unique_vehicle_count}")
                    
                    # IMPORTANT: Log the connection between VideoUpload and TrafficData for debugging
                    logger.info(f"[STATS DEBUG] Connection: VideoUpload ID={video_upload.id} <-> TrafficData ID={traffic_data.id}")
                    
                    # Make sure the vehicle count is consistent between VideoUpload and TrafficData
                    if traffic_data.vehicle_count != video_upload.vehicle_count:
                        logger.info(f"[STATS DEBUG] Synchronizing vehicle counts: TrafficData={traffic_data.vehicle_count} -> {video_upload.vehicle_count}")
                        traffic_data.vehicle_count = video_upload.vehicle_count
                        traffic_data.save()
                
                # Séparer la création des véhicules dans une transaction distincte pour éviter le blocage
                with transaction.atomic():
                    # Enregistrer les véhicules détectés selon les types (éviter les doublons)
                    for vehicle_type, count in vehicle_types_data.items():
                        # Vérifier si ce type de véhicule existe déjà pour ce TrafficData
                        existing_vehicle = Vehicle.objects.filter(
                            traffic_data=traffic_data,
                            vehicle_type=vehicle_type
                        ).first()
                        
                        if existing_vehicle:
                            # Mettre à jour le compteur existant si nécessaire
                            if existing_vehicle.count != count:
                                existing_vehicle.count = count
                                existing_vehicle.save()
                                logger.info(f"[STATS DEBUG] Véhicule mis à jour: type={vehicle_type}, count={count}")
                        else:
                            # Créer un nouvel enregistrement de véhicule
                            vehicle = Vehicle.objects.create(
                                traffic_data=traffic_data,
                                vehicle_type=vehicle_type,
                                confidence_score=1.0,
                                count=count
                            )
                            logger.info(f"[STATS DEBUG] Nouveau véhicule enregistré: type={vehicle_type}, count={count}")
                
                # Séparer la création des accidents dans une transaction distincte
                has_accidents = False
                with transaction.atomic():
                    # Enregistrer les accidents détectés (éviter les doublons)
                    for accident in result.get('accidents', []):
                        if accident.get('detected', False) and accident.get('confirmed', False):
                            # Vérifier si cet accident a déjà été enregistré
                            if 'location' in accident and accident['location']:
                                lat = accident['location'][0]
                                lon = accident['location'][1]
                                
                                existing_accident = Accident.objects.filter(
                                    traffic_data=traffic_data,
                                    latitude=lat,
                                    longitude=lon
                                ).first()
                                
                                if existing_accident:
                                    has_accidents = True
                                    continue
                            
                            # Déterminer la sévérité en fonction de la confiance et des facteurs
                            confidence = accident.get('confidence', 0.0)
                            severity = 1  # Par défaut: Mineur
                            
                            # Facteurs aggravants
                            details = accident.get('details', {})
                            involved_vehicles = len(accident.get('involved_vehicles', []))
                            
                            # Ajuster la sévérité selon les critères
                            if confidence > 0.8 or involved_vehicles > 3:
                                severity += 1
                            if confidence > 0.9 or involved_vehicles > 5 or details.get('overlap_score', 0) > 0.7:
                                severity += 1
                            
                            # Limiter à 3 (Sévère)
                            severity = min(severity, 3)
                            
                            # Créer l'entrée d'accident
                            accident_obj = Accident.objects.create(
                                traffic_data=traffic_data,
                                latitude=accident.get('location', [18.0735, -15.9582])[0] if accident.get('location') else 18.0735,
                                longitude=accident.get('location', [18.0735, -15.9582])[1] if accident.get('location') else -15.9582,
                                confidence_score=confidence,
                                details={
                                    **details,
                                    'involved_vehicles': accident.get('involved_vehicles', [])
                                },
                                severity=severity,
                                viewed=False  # Définir comme non vu pour permettre les notifications
                            )
                            logger.info(f"[STATS DEBUG] Nouvel accident enregistré: ID={accident_obj.id}, Sévérité={severity}")
                            has_accidents = True
                
                # Créer un résumé des types de véhicules pour l'interface utilisateur
                vehicle_type_summary = ", ".join([f"{count} {vtype}" for vtype, count in vehicle_types_data.items()])
                
                # Invalider les caches qui pourraient affecter la page de stats
                from django.core.cache import cache
                cache.clear()
                
                # Loguer l'état final de la base de données pour le débogage
                total_videos = VideoUpload.objects.count()
                total_traffic_data = TrafficData.objects.count()
                total_vehicles = Vehicle.objects.count()
                total_accidents = Accident.objects.count()
                logger.info(f"[STATS DEBUG] État final de la base de données: {total_videos} videos, {total_traffic_data} traffic data, {total_vehicles} vehicles, {total_accidents} accidents")
                
                return JsonResponse({
                    'success': True,
                    'message': 'Vidéo traitée avec succès',
                    'duplicate': False,
                    'results': {
                        'vehicle_count': unique_vehicle_count,
                        'max_vehicles_per_frame': max_vehicles_per_frame,
                        'vehicle_type_summary': vehicle_type_summary,
                        'vehicle_types': vehicle_types_data,
                        'congestion_level': congestion_level,
                        'processing_time': processing_time,
                        'processed_video_url': video_upload.video_file.url,
                        'has_accident': has_accidents
                    }
                })
            else:
                video_upload.status = 'failed'
                video_upload.error_message = 'Erreur lors du traitement de la vidéo'
                video_upload.save()
                return JsonResponse({
                    'success': False,
                    'error': 'Erreur lors du traitement de la vidéo'
                })
                
        except Exception as e:
            if 'video_upload' in locals():
                video_upload.status = 'failed'
                video_upload.error_message = str(e)
                video_upload.save()
            logger.error(f"Erreur lors du traitement de la vidéo: {str(e)}")
            return JsonResponse({
                'success': False,
                'error': f"Erreur: {str(e)}"
            })
    
    return JsonResponse({
        'success': False,
        'error': 'Méthode non autorisée ou fichier manquant'
    })

def get_accidents(request):
    """API endpoint pour récupérer les données d'accidents"""
    # Obtenir les filtres de la requête
    date_filter = request.GET.get('date')
    location_filter = request.GET.get('location')
    
    # Construire la requête
    query = Q()
    if date_filter:
        query &= Q(timestamp__date=datetime.strptime(date_filter, '%Y-%m-%d').date())
    if location_filter:
        query &= Q(traffic_data__location_name__icontains=location_filter)
    
    # Obtenir les accidents avec filtre
    accidents = Accident.objects.filter(query).order_by('-timestamp')[:50]
    
    # Formater pour la réponse API
    accident_data = [{
        'id': acc.id,
        'timestamp': acc.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        'latitude': acc.latitude,
        'longitude': acc.longitude,
        'severity': acc.get_severity_display(),
        'confidence': acc.confidence_score
    } for acc in accidents]
    
    return JsonResponse({'accidents': accident_data})

def check_accident_alert(request):
    """Vérifier s'il y a de nouvelles alertes d'accident à afficher"""
    # Obtenir le cookie qui indique si l'alerte automatique a déjà été vue
    accident_alerts_shown = request.COOKIES.get('accident_alerts_shown', 'false')
    
    # Si les alertes ont déjà été montrées, ne rien afficher
    if accident_alerts_shown == 'true':
        return JsonResponse({'has_alert': False})
    
    # Ne pas générer de nouvelles alertes, seulement montrer celles qui existent et n'ont pas été vues
    latest_accident = Accident.objects.filter(
        viewed=False
    ).order_by('-timestamp').first()
    
    if latest_accident:
        # Marquer comme vu dans la base de données
        latest_accident.viewed = True
        latest_accident.save()
        
        # Marquer tous les autres accidents de la même vidéo comme vus également
        # (un seul rapport d'accident par vidéo)
        if latest_accident.traffic_data:
            Accident.objects.filter(
                traffic_data=latest_accident.traffic_data,
                viewed=False
            ).update(viewed=True)
        
        response = JsonResponse({
            'has_alert': True,
            'accident': {
                'id': latest_accident.id,
                'timestamp': latest_accident.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'location': latest_accident.traffic_data.location_name,
                'latitude': latest_accident.latitude,
                'longitude': latest_accident.longitude,
                'severity': latest_accident.get_severity_display()
            }
        })
        
        # Définir un cookie pour ne plus montrer d'alertes automatiques
        response.set_cookie('accident_alerts_shown', 'true', max_age=86400)  # 24 heures
        return response
    
    return JsonResponse({'has_alert': False})

@csrf_exempt
def reset_accident_alerts(request):
    """Réinitialiser les alertes d'accident"""
    if request.method == 'POST':
        # Reset all viewed flags in the database
        Accident.objects.all().update(viewed=False)
        
        # Créer une réponse pour supprimer le cookie
        response = JsonResponse({'success': True})
        response.delete_cookie('accident_alerts_shown')
        
        return response
    
    return JsonResponse({'success': False, 'error': 'Méthode non autorisée'}, status=405)

@csrf_exempt
def force_deep_clean(request):
    """Force a deep clean of all database tables and cache"""
    if request.method == "POST":
        try:
            from django.core.cache import cache
            from django.db import connection
            
            logger.info("[DEEP CLEAN] Starting complete database cleanup...")
            
            # Clear Django cache
            cache.clear()
            
            # First count all records for verification
            accident_count_before = Accident.objects.count()
            vehicle_count_before = Vehicle.objects.count()
            traffic_count_before = TrafficData.objects.count()
            upload_count_before = VideoUpload.objects.count()
            
            total_before = accident_count_before + vehicle_count_before + traffic_count_before + upload_count_before
            logger.info(f"[DEEP CLEAN] Records before delete: {total_before} ({accident_count_before} accidents, {vehicle_count_before} vehicles, {traffic_count_before} traffic records, {upload_count_before} uploads)")
            
            # First try Django ORM deletion to respect foreign keys
            try:
                Accident.objects.all().delete()
                Vehicle.objects.all().delete()
                TrafficData.objects.all().delete()
                VideoUpload.objects.all().delete()
                logger.info("[DEEP CLEAN] Deleted all records via Django ORM")
            except Exception as e:
                logger.error(f"[DEEP CLEAN] Error during ORM deletion: {str(e)}")
            
            # Execute raw SQL to make sure all data is deleted
            with connection.cursor() as cursor:
                # Disable foreign key checks temporarily
                cursor.execute("PRAGMA foreign_keys = OFF;")
                
                # Delete all data from all relevant tables
                cursor.execute("DELETE FROM traffic_app_accident;")
                cursor.execute("DELETE FROM traffic_app_vehicle;")
                cursor.execute("DELETE FROM traffic_app_trafficdata;")
                cursor.execute("DELETE FROM traffic_app_videoupload;")
                
                # Reset all primary key sequences
                cursor.execute("DELETE FROM sqlite_sequence WHERE name='traffic_app_accident';")
                cursor.execute("DELETE FROM sqlite_sequence WHERE name='traffic_app_vehicle';")
                cursor.execute("DELETE FROM sqlite_sequence WHERE name='traffic_app_trafficdata';")
                cursor.execute("DELETE FROM sqlite_sequence WHERE name='traffic_app_videoupload';")
                
                # Re-enable foreign key checks
                cursor.execute("PRAGMA foreign_keys = ON;")
                
                logger.info("[DEEP CLEAN] Executed raw SQL deletion")
            
            # Vacuum the database to reclaim space
            with connection.cursor() as cursor:
                cursor.execute("VACUUM;")
                logger.info("[DEEP CLEAN] Vacuumed database")
            
            # Clear Django's cache again to ensure clean state
            connection.close()
            connection.connect()
            cache.clear()
            
            # Verify deletion was successful
            accident_count_after = Accident.objects.count()
            vehicle_count_after = Vehicle.objects.count()
            traffic_count_after = TrafficData.objects.count()
            upload_count_after = VideoUpload.objects.count()
            
            total_after = accident_count_after + vehicle_count_after + traffic_count_after + upload_count_after
            
            if total_after > 0:
                logger.error(f"[DEEP CLEAN] Failed to delete all records! Remaining: {total_after}")
                
                # Try one more time with alternative method
                with connection.cursor() as cursor:
                    cursor.execute("DELETE FROM django_session;")
                    cursor.execute("DELETE FROM traffic_app_accident;")
                    cursor.execute("DELETE FROM traffic_app_vehicle;")
                    cursor.execute("DELETE FROM traffic_app_trafficdata;")
                    cursor.execute("DELETE FROM traffic_app_videoupload;")
                    cursor.execute("VACUUM;")
                
                accident_count_final = Accident.objects.count()
                vehicle_count_final = Vehicle.objects.count()
                traffic_count_final = TrafficData.objects.count()
                upload_count_final = VideoUpload.objects.count()
                
                total_final = accident_count_final + vehicle_count_final + traffic_count_final + upload_count_final
                success = total_final == 0
                
                if not success:
                    logger.error(f"[DEEP CLEAN] Final count after second attempt: {total_final}")
                    return JsonResponse({
                        'success': False, 
                        'message': f"Nettoyage profond incomplet. {total_final} éléments n'ont pas pu être supprimés."
                    })
            else:
                logger.info("[DEEP CLEAN] Successfully deleted all records")
            
            return JsonResponse({
                'success': True, 
                'message': f"Nettoyage profond effectué avec succès. {total_before} éléments ont été supprimés."
            })
        except Exception as e:
            logger.error(f"[DEEP CLEAN] Error during deep clean: {str(e)}")
            return JsonResponse({
                'success': False, 
                'message': f"Erreur lors du nettoyage profond: {str(e)}"
            }, status=500)
    
    return JsonResponse({'success': False, 'message': "Requête invalide"}, status=400)

@csrf_exempt
def webcam_feed(request):
    """Vue pour traiter le flux de la webcam"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            frame = data.get('image')
            
            if not frame:
                return JsonResponse({'success': False, 'error': 'No image data received'})
            
            # Traiter la frame avec le processeur vidéo existant
            processor = VideoProcessor()
            results = processor.process_frame(frame)
            
            # Préparer la réponse
            response = {
                'success': True,
                'vehicle_count': results.get('vehicle_count', 0),
                'vehicle_types': results.get('vehicle_types', {}),
                'congestion_level': results.get('congestion_level', 0),
                'has_accident': results.get('has_accident', False),
                'confidence': results.get('confidence', 0)
            }
            
            return JsonResponse(response)
            
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
            
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

def send_event_notifications(events):
    """Envoie des notifications pour les événements détectés"""
    for event in events:
        notification_text = ""
        if event['type'] == 'accident':
            notification_text = f"⚠️ ACCIDENT DÉTECTÉ! Sévérité: {event['severity']}"
        elif event['type'] == 'roadwork':
            notification_text = f"🚧 TRAVAUX EN COURS! Impact: {event['severity']}"
        elif event['type'] == 'police':
            notification_text = f"👮 PRÉSENCE POLICIÈRE DÉTECTÉE!"
        
        # Envoyer la notification (à implémenter selon vos besoins)
        print(f"[NOTIFICATION] {notification_text}")
