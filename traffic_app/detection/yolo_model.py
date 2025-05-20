from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Dict, Tuple
import torch
from datetime import datetime
import os
import torch.serialization
import torch.hub

class TrafficDetector:
    def __init__(self):
        try:
            # Charger directement le modèle depuis Ultralytics
            print("Chargement du modèle YOLOv8n...")
            self.model = YOLO('yolov8n')
            print("Modèle YOLO chargé avec succès")
            
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")
            raise

        # Classes pertinentes pour la détection du trafic
        self.target_classes = {
            2: 'car',      # voiture
            3: 'motorcycle',  # moto
            5: 'bus',      # bus
            7: 'truck'     # camion
        }
        
        # Configuration pour la détection d'accidents
        self.accident_patterns = {
            'vehicle_proximity': 0.4,    # Seuil pour véhicules anormalement proches
            'unusual_orientation': 25,    # Seuil en degrés pour orientation inhabituelle
            'debris_detection': 0.45,    # Seuil de confiance pour la détection de débris
            'sudden_trajectory_change': 0.45,  # Détection de changement brusque de trajectoire
            'vehicle_overlap': 0.3,      # Détection de chevauchement de véhicules
            'stopped_vehicles': 0.4,     # Seuil pour la détection de véhicules arrêtés
            'density_anomaly': 0.35,     # Seuil pour la détection d'anomalies de densité
            'min_confidence': 0.35,      # Seuil minimum de confiance pour déclarer un accident
            'min_significant_factors': 2  # Nombre minimum de facteurs significatifs requis
        }
        
        # Historique des données de détection pour analyse temporelle
        self.prev_frame_detections = []
        self.detection_history = []
        self.history_max_frames = 15
        self.false_positive_count = 0
        self.confirmed_incidents = set()
        self.accident_history = {}
    
    def process_single_frame(self, frame):
        """Traite une seule frame pour la détection en temps réel"""
        try:
            # Utiliser la fonction detect_vehicles au lieu de detect_objects
            detections, annotated_frame = self.detect_vehicles(frame)
            
            # Initialiser les compteurs
            vehicle_count = len(detections)  # Compter directement depuis les détections
            vehicle_types = {}
            has_accident = False
            accident_confidence = 0.0
            accident_severity = 0
            
            # Traiter les détections
            if detections:
                # Compter les types de véhicules
                for detection in detections:
                    vehicle_type = detection['class']
                    vehicle_types[vehicle_type] = vehicle_types.get(vehicle_type, 0) + 1
                
                # Vérifier les collisions potentielles
                if len(detections) >= 2:
                    # Utiliser la fonction detect_accidents existante
                    accident_result = self.detect_accidents(frame, detections)
                    has_accident = accident_result.get('detected', False)
                    accident_confidence = accident_result.get('confidence', 0.0)
                    if 'details' in accident_result:
                        accident_severity = len(accident_result['details'].get('vehicle_types', []))
            
            # Calculer le niveau de congestion basé sur le nombre de véhicules
            # et la taille de l'image
            frame_area = frame.shape[0] * frame.shape[1]
            vehicle_density = vehicle_count / (frame_area / 100000)  # Normaliser par rapport à la taille
            congestion_level = min(1.0, vehicle_density / 2.0)  # Réduire le seuil à 2 véhicules par 100000 pixels
            
            print(f"[DEBUG] Frame analysis - Vehicles: {vehicle_count}, Types: {vehicle_types}, Congestion: {congestion_level:.2f}, Accident: {has_accident}")
            
            return {
                'success': True,
                'vehicle_count': vehicle_count,
                'vehicle_types': vehicle_types,
                'congestion_level': congestion_level,
                'has_accident': has_accident,
                'accident_confidence': accident_confidence,
                'accident_severity': accident_severity,
                'annotated_frame': annotated_frame  # Retourner l'image annotée
            }
            
        except Exception as e:
            print(f"Erreur lors du traitement de la frame: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }

    def detect_vehicles(self, frame: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        if frame is None or frame.size == 0:
            print("Frame invalide reçue")
            return [], np.array([])
            
        # Vérifier la qualité de l'image
        if frame.mean() < 5:  # Réduire encore le seuil de luminosité
            print("Image trop sombre pour analyse")
            return [], frame
            
        # Faire une copie du frame pour le dessin
        display_frame = frame.copy()
        
        try:
            # Prédiction avec YOLOv8 avec un seuil de confiance plus bas
            results = self.model(frame, conf=0.25, verbose=True)  # Activer le mode verbose pour le débogage
            detections = []
            
            if results is None or len(results) == 0:
                print("Aucun résultat de détection")
                return [], display_frame
            
            # Traiter chaque détection
            for r in results:
                boxes = r.boxes
                print(f"[DEBUG] Nombre total de détections: {len(boxes)}")
                
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    print(f"[DEBUG] Classe détectée: {cls}, Confiance: {conf:.2f}")
                    
                    # Vérifier si la classe est un véhicule que nous voulons détecter
                    if cls in self.target_classes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        
                        detection = {
                            'class': self.target_classes[cls],
                            'confidence': conf,
                            'bbox': xyxy.tolist()
                        }
                        detections.append(detection)
                        
                        # Dessiner la boîte sur l'image
                        x1, y1, x2, y2 = map(int, xyxy)
                        color = self._get_color_for_class(self.target_classes[cls])
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{self.target_classes[cls]} {conf:.2f}"
                        cv2.putText(display_frame, label, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Ajouter un compteur de véhicules en haut de l'image
            count_text = f"Véhicules détectés: {len(detections)}"
            cv2.putText(display_frame, count_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            print(f"[DEBUG] Détections finales: {len(detections)} véhicules trouvés")
            if len(detections) > 0:
                print(f"[DEBUG] Types de véhicules: {[d['class'] for d in detections]}")
            
            return detections, display_frame
            
        except Exception as e:
            print(f"Erreur lors de la détection: {e}")
            import traceback
            traceback.print_exc()
            return [], display_frame

    def _get_color_for_class(self, class_name):
        """Retourne une couleur différente pour chaque type de véhicule"""
        colors = {
            'car': (0, 255, 0),      # Vert
            'motorcycle': (255, 0, 0),  # Rouge
            'bus': (0, 0, 255),      # Bleu
            'truck': (255, 255, 0)   # Jaune
        }
        return colors.get(class_name, (0, 255, 0))  # Vert par défaut

    def calculate_congestion(self, detections: List[Dict]) -> float:
        if not detections:
            return 0.0
            
        # Calculer le niveau de congestion basé sur le nombre de véhicules
        # et leur proximité
        vehicle_count = len(detections)
        
        # Seuil de base pour la congestion
        base_threshold = 5  # Considéré comme trafic normal
        max_threshold = 15  # Considéré comme congestion maximale
        
        # Calculer le niveau de congestion normalisé
        congestion_level = min(1.0, vehicle_count / max_threshold)
        
        return congestion_level
        
    def detect_accidents(self, frame: np.ndarray, detections: List[Dict]) -> Dict:
        """Détecter les accidents potentiels dans la frame vidéo avec une analyse avancée"""
        if not detections or len(detections) < 2:
            return {'detected': False, 'confidence': 0.0}

        try:
            # Extraire les positions des véhicules
            vehicle_positions = []
            for detection in detections:
                bbox = detection['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                vehicle_positions.append({
                    'position': (center_x, center_y),
                    'bbox': bbox,
                    'type': detection['class']
                })

            # Calculer les scores pour différents facteurs d'accident
            proximity_score = self._calculate_proximity_score(vehicle_positions)
            overlap_score = self._calculate_vehicle_overlap(vehicle_positions)
            orientation_score = self._analyze_vehicle_orientation(frame, detections)

            # Calculer le score final
            accident_factors = {
                'proximity': proximity_score,
                'overlap': overlap_score,
                'orientation': orientation_score
            }

            # Compter combien de facteurs dépassent leur seuil
            significant_factors = sum(1 for score in accident_factors.values() 
                                   if score > self.accident_patterns['min_confidence'])

            # Calculer la confiance globale
            confidence = max(accident_factors.values())

            # Déterminer s'il y a un accident
            is_accident = (significant_factors >= self.accident_patterns['min_significant_factors'] and 
                         confidence > self.accident_patterns['min_confidence'])

            if is_accident:
                # Obtenir le centroïde de l'accident
                accident_location = self._get_accident_centroid(vehicle_positions)
                
                # Identifier les véhicules impliqués
                involved_vehicles = []
                for pos in vehicle_positions:
                    dist = ((pos['position'][0] - accident_location[0])**2 + 
                           (pos['position'][1] - accident_location[1])**2)**0.5
                    if dist < 100:  # Distance en pixels
                        involved_vehicles.append(pos['type'])

                return {
                    'detected': True,
                    'confidence': confidence,
                    'location': accident_location,
                    'details': {
                        'factors': accident_factors,
                        'vehicle_types': involved_vehicles,
                        'significant_factors': significant_factors
                    }
                }

            return {'detected': False, 'confidence': confidence}

        except Exception as e:
            print(f"Erreur lors de la détection d'accidents: {e}")
            import traceback
            traceback.print_exc()
            return {'detected': False, 'confidence': 0.0}

    def _calculate_proximity_score(self, vehicle_positions):
        """Calcule un score basé sur la proximité anormale entre véhicules"""
        if len(vehicle_positions) < 2:
            return 0.0

        min_distance = float('inf')
        for i in range(len(vehicle_positions)):
            for j in range(i + 1, len(vehicle_positions)):
                pos1 = vehicle_positions[i]['position']
                pos2 = vehicle_positions[j]['position']
                distance = ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
                min_distance = min(min_distance, distance)

        # Normaliser la distance (plus la distance est petite, plus le score est élevé)
        proximity_threshold = 100  # Distance en pixels considérée comme trop proche
        score = max(0.0, 1.0 - (min_distance / proximity_threshold))
        return score

    def _calculate_vehicle_overlap(self, vehicle_positions):
        """Calcule un score basé sur le chevauchement des véhicules"""
        if len(vehicle_positions) < 2:
            return 0.0

        max_overlap = 0.0
        for i in range(len(vehicle_positions)):
            for j in range(i + 1, len(vehicle_positions)):
                bbox1 = vehicle_positions[i]['bbox']
                bbox2 = vehicle_positions[j]['bbox']

                # Calculer l'intersection
                x_left = max(bbox1[0], bbox2[0])
                y_top = max(bbox1[1], bbox2[1])
                x_right = min(bbox1[2], bbox2[2])
                y_bottom = min(bbox1[3], bbox2[3])

                if x_right > x_left and y_bottom > y_top:
                    intersection = (x_right - x_left) * (y_bottom - y_top)
                    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
                    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
                    overlap = intersection / min(area1, area2)
                    max_overlap = max(max_overlap, overlap)

        return max_overlap

    def _analyze_vehicle_orientation(self, frame, detections):
        """Analyse l'orientation des véhicules pour détecter des positions anormales"""
        try:
            # Utiliser les contours pour estimer l'orientation
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            max_angle_diff = 0.0

            for detection in detections:
                bbox = detection['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                
                # Extraire la région du véhicule
                roi = gray[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                # Trouver les contours
                _, thresh = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    # Trouver le plus grand contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    if len(largest_contour) >= 5:  # Minimum 5 points nécessaires
                        # Calculer l'ellipse englobante
                        ellipse = cv2.fitEllipse(largest_contour)
                        angle = ellipse[2]
                        
                        # Normaliser l'angle entre 0 et 90 degrés
                        angle = angle % 90
                        expected_angle = 0 if angle < 45 else 90
                        angle_diff = abs(angle - expected_angle) / 90.0
                        max_angle_diff = max(max_angle_diff, angle_diff)

            return max_angle_diff
        except Exception as e:
            print(f"Erreur lors de l'analyse de l'orientation: {e}")
            return 0.0

    def _get_accident_centroid(self, vehicle_positions):
        """Calcule le centre de l'accident basé sur les positions des véhicules impliqués"""
        if not vehicle_positions:
            return (0, 0)

        x_sum = sum(pos['position'][0] for pos in vehicle_positions)
        y_sum = sum(pos['position'][1] for pos in vehicle_positions)
        count = len(vehicle_positions)
        return (x_sum / count, y_sum / count)

    def _detect_stopped_vehicles(self):
        """Détecte les véhicules qui se sont arrêtés soudainement ou qui sont immobiles"""
        if len(self.detection_history) < 3:
            return 0.0
            
        stopped_score = 0.0
        
        # Obtenir les dernières frames d'historique
        recent_history = self.detection_history[-3:]
        
        # Suivre les véhicules à travers les frames
        vehicle_tracks = {}
        
        # Pour chaque frame dans l'historique récent
        for frame_idx, frame_info in enumerate(recent_history):
            detections = frame_info['detections']
            
            for detection in detections:
                bbox = detection['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                
                # Identifier le véhicule par sa position dans la première frame
                best_match = None
                min_dist = float('inf')
                
                # Trouver le véhicule le plus proche dans les trames précédentes
                if frame_idx > 0:
                    for vehicle_id, positions in vehicle_tracks.items():
                        if len(positions) != frame_idx:
                            continue
                            
                        prev_x, prev_y = positions[-1]
                        dist = ((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2) ** 0.5
                        
                        if dist < min_dist and dist < 50:  # Seuil de distance
                            min_dist = dist
                            best_match = vehicle_id
                
                if best_match is not None:
                    # Ajouter la position à ce véhicule existant
                    vehicle_tracks[best_match].append((center_x, center_y))
                else:
                    # Nouveau véhicule détecté
                    vehicle_id = f"v{len(vehicle_tracks)}"
                    vehicle_tracks[vehicle_id] = [(center_x, center_y)]
        
        # Analyser les pistes pour détecter les véhicules arrêtés
        stopped_vehicles = 0
        moving_vehicles = 0
        
        for vehicle_id, positions in vehicle_tracks.items():
            if len(positions) < 3:
                continue  # Pas assez de données pour ce véhicule
                
            # Calculer les distances entre positions consécutives
            distances = []
            for i in range(1, len(positions)):
                x1, y1 = positions[i-1]
                x2, y2 = positions[i]
                dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                distances.append(dist)
            
            # Un véhicule est considéré comme arrêté si sa distance moyenne est très petite
            avg_distance = sum(distances) / len(distances)
            if avg_distance < 5.0:  # Seuil très bas de mouvement
                stopped_vehicles += 1
            else:
                moving_vehicles += 1
        
        # Calculer un score basé sur la proportion de véhicules arrêtés
        total_tracked = stopped_vehicles + moving_vehicles
        if total_tracked > 0:
            stopped_score = stopped_vehicles / total_tracked
            
            # Augmenter le score si beaucoup de véhicules sont arrêtés
            if stopped_vehicles > 2 and stopped_score > 0.3:
                stopped_score *= 1.5
                
            stopped_score = min(1.0, stopped_score)  # Limiter à 1.0
        
        return stopped_score
    
    def _detect_density_anomaly(self, vehicle_positions):
        """Détecte les anomalies de densité des véhicules (regroupements anormaux)"""
        if len(vehicle_positions) < 3:
            return 0.0
            
        # Calculer les distances entre toutes les paires de véhicules
        all_distances = []
        for i in range(len(vehicle_positions)):
            for j in range(i+1, len(vehicle_positions)):
                pos1 = vehicle_positions[i]
                pos2 = vehicle_positions[j]
                
                # Calculer la distance euclidienne entre les centres
                dist = ((pos1['position'][0] - pos2['position'][0]) ** 2 + (pos1['position'][1] - pos2['position'][1]) ** 2) ** 0.5
                all_distances.append(dist)
        
        if not all_distances:
            return 0.0
        
        # Calculer la distribution des distances
        avg_distance = sum(all_distances) / len(all_distances)
        std_distance = (sum((d - avg_distance) ** 2 for d in all_distances) / len(all_distances)) ** 0.5
        
        # Calculer le coefficient de variation (CV = écart-type / moyenne)
        # Un CV élevé indique une distribution inégale des véhicules
        if avg_distance > 0:
            cv = std_distance / avg_distance
        else:
            cv = 0.0
        
        # Normaliser le coefficient de variation pour obtenir un score
        # Plus le CV est élevé, plus la distribution est irrégulière (ce qui peut indiquer un accident)
        density_score = min(1.0, cv / 2.0)  # Normaliser à [0,1]
        
        return density_score
