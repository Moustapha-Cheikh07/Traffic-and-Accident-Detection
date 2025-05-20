from django.db import models
from django.utils import timezone

class VideoUpload(models.Model):
    video_file = models.FileField(upload_to='uploads/')
    upload_date = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)
    location = models.CharField(max_length=255, default="Nouakchott, Mauritanie")
    vehicle_count = models.IntegerField(default=0)
    congestion_level = models.FloatField(default=0.0)
    processing_time = models.FloatField(default=0.0)  # en secondes
    status = models.CharField(max_length=50, default='pending')
    error_message = models.TextField(blank=True, null=True)

    def __str__(self):
        return f"Vidéo {self.id} - {self.upload_date.strftime('%d/%m/%Y %H:%M')}"

class TrafficData(models.Model):
    timestamp = models.DateTimeField(default=timezone.now)
    vehicle_count = models.IntegerField(default=0)
    congestion_level = models.FloatField(default=0.0)
    latitude = models.FloatField(default=18.0735)  # Coordonnées par défaut de Nouakchott
    longitude = models.FloatField(default=-15.9582)
    video_source = models.CharField(max_length=255)
    location_name = models.CharField(max_length=255, default="Nouakchott")
    weather_condition = models.CharField(max_length=100, blank=True, default="")
    temperature = models.FloatField(null=True, blank=True)
    peak_hour = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"Trafic à {self.location_name} - {self.timestamp.strftime('%d/%m/%Y %H:%M')}"

class Vehicle(models.Model):
    VEHICLE_TYPES = [
        ('car', 'Voiture'),
        ('truck', 'Camion'),
        ('bus', 'Bus'),
        ('motorcycle', 'Moto'),
    ]
    
    traffic_data = models.ForeignKey(TrafficData, on_delete=models.CASCADE, related_name='vehicles')
    vehicle_type = models.CharField(max_length=50, choices=VEHICLE_TYPES, default='car')
    detection_time = models.DateTimeField(auto_now_add=True)
    confidence_score = models.FloatField(default=0.0)
    trajectory = models.JSONField(null=True, blank=True)
    count = models.IntegerField(default=1)  # Nombre de véhicules de ce type
    
    def __str__(self):
        count_display = f" ({self.count})" if self.count > 1 else ""
        return f"{self.vehicle_type}{count_display} détecté le {self.detection_time.strftime('%d/%m/%Y %H:%M')}"

class Accident(models.Model):
    SEVERITY_CHOICES = [
        (1, 'Mineur'),
        (2, 'Modéré'),
        (3, 'Sévère'),
    ]
    
    traffic_data = models.ForeignKey(TrafficData, on_delete=models.CASCADE, related_name='accidents')
    timestamp = models.DateTimeField(default=timezone.now)
    latitude = models.FloatField()
    longitude = models.FloatField()
    confidence_score = models.FloatField(default=0.0)
    severity = models.IntegerField(choices=SEVERITY_CHOICES, default=1)
    details = models.JSONField(null=True, blank=True)
    verified = models.BooleanField(default=False)  # Pour vérification manuelle pour améliorer le modèle
    viewed = models.BooleanField(default=False)  # Pour le système de notification
    
    # Champs supplémentaires pour l'apprentissage par renforcement
    false_positive = models.BooleanField(null=True, blank=True)  # Pour suivre les détections incorrectes
    contributing_factors = models.JSONField(null=True, blank=True)  # Analyse supplémentaire
    
    def __str__(self):
        return f"Accident à {self.latitude}, {self.longitude} le {self.timestamp.strftime('%d/%m/%Y %H:%M')}"
        
    class Meta:
        ordering = ['-timestamp']

class RoadEvent(models.Model):
    EVENT_TYPES = [
        ('accident', 'Accident'),
        ('roadwork', 'Travaux'),
        ('police', 'Présence Police'),
    ]

    SEVERITY_LEVELS = [
        (1, 'Faible'),
        (2, 'Modéré'),
        (3, 'Sévère'),
    ]

    traffic_data = models.ForeignKey('TrafficData', on_delete=models.CASCADE)
    event_type = models.CharField(max_length=20, choices=EVENT_TYPES)
    timestamp = models.DateTimeField(default=timezone.now)
    latitude = models.FloatField()
    longitude = models.FloatField()
    severity = models.IntegerField(choices=SEVERITY_LEVELS, default=1)
    description = models.TextField(blank=True)
    confidence_score = models.FloatField(default=0.0)
    is_active = models.BooleanField(default=True)
    notified = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.get_event_type_display()} - {self.timestamp.strftime('%Y-%m-%d %H:%M')}"
