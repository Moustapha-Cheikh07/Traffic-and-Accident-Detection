import cv2
import numpy as np
from .yolo_model import TrafficDetector
from typing import List, Dict, Optional
import os
import time
import logging

class VideoProcessor:
    def __init__(self):
        self.detector = TrafficDetector()
        self.trajectories = {}  # Pour stocker les trajectoires des objets
        self.object_history = {}  # Pour stocker l'historique des objets détectés
        self.next_id = 0  # Pour attribuer des IDs uniques aux objets
        self.accident_alert = False  # Flag pour les alertes d'accident
        self.accident_locations = []  # Pour suivre les emplacements d'accidents déjà détectés
        self.last_accident_time = None  # Pour limiter la fréquence des détections
        self.min_accident_interval = 3.0  # Intervalle minimum entre détections (en secondes)
        self.accident_frames = set()  # Pour suivre les frames où des accidents ont été détectés
        self.unique_vehicles = set()  # Pour compter les véhicules uniques à travers les frames

    def get_object_id(self, detection, prev_detections):
        """Attribue un ID à un objet en fonction de sa position et de sa classe"""
        x1, y1, x2, y2 = detection['bbox']
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        vehicle_class = detection['class']
        vehicle_width = x2 - x1
        vehicle_height = y2 - y1
        vehicle_area = vehicle_width * vehicle_height
        
        # Chercher l'objet le plus proche dans les détections précédentes
        min_dist = float('inf')
        best_id = None
        best_iou = 0
        
        for obj_id, prev_pos in prev_detections.items():
            px, py = prev_pos['center']
            prev_class = prev_pos['class']
            prev_bbox = prev_pos['bbox']
            
            # Calculer la distance euclidienne entre les centres
            dist = ((center[0] - px) ** 2 + (center[1] - py) ** 2) ** 0.5
            
            # Calculer l'IoU (Intersection over Union) pour une meilleure correspondance
            # Extraire les coordonnées des deux boîtes englobantes
            x1_a, y1_a, x2_a, y2_a = detection['bbox']
            x1_b, y1_b, x2_b, y2_b = prev_bbox
            
            # Calculer l'intersection
            x_left = max(x1_a, x1_b)
            y_top = max(y1_a, y1_b)
            x_right = min(x2_a, x2_b)
            y_bottom = min(y2_a, y2_b)
            
            # Vérifier s'il y a une intersection
            if x_right < x_left or y_bottom < y_top:
                intersection_area = 0
            else:
                intersection_area = (x_right - x_left) * (y_bottom - y_top)
            
            # Calculer l'aire de chaque boîte
            box_a_area = (x2_a - x1_a) * (y2_a - y1_a)
            box_b_area = (x2_b - x1_b) * (y2_b - y1_b)
            
            # Calculer l'IoU
            iou = intersection_area / float(box_a_area + box_b_area - intersection_area)
            
            # Rendre le seuil de distance plus strict pour éviter les fausses correspondances
            # et utiliser un seuil différent selon la classe et la taille du véhicule
            distance_threshold = 25  # Seuil de base encore plus strict (avant 30)
            iou_threshold = 0.2  # Seuil IoU minimum pour considérer une correspondance
            
            # Ajuster le seuil de distance selon le type et la taille du véhicule
            if vehicle_class == 'truck' or vehicle_class == 'bus':
                distance_threshold = 35  # Plus large pour les grands véhicules
                iou_threshold = 0.15    # Plus souple pour les grands véhicules
            elif vehicle_class == 'motorcycle':
                distance_threshold = 15  # Plus strict pour les petits véhicules
                iou_threshold = 0.25    # Plus strict pour les petits véhicules
            
            # Critères de correspondance: même classe, distance faible ET IoU significatif
            if vehicle_class == prev_class and dist < distance_threshold and iou > iou_threshold:
                if iou > best_iou:
                    best_iou = iou
                    best_id = obj_id
                    min_dist = dist
        
        if best_id is None:
            # Nouvel objet détecté
            best_id = self.next_id
            self.next_id += 1
            # Ajouter à l'ensemble des véhicules uniques avec des informations enrichies
            self.unique_vehicles.add((best_id, vehicle_class, vehicle_area))
        
        return best_id

    def process_video_feed(self, video_path: str) -> Dict:
        try:
            # Add debug logging
            logger = logging.getLogger(__name__)
            logger.info(f"[PROCESSOR] Starting video processing for: {video_path}")
            
            # Check for existing records related to this video
            video_filename = os.path.basename(video_path)
            
            from django.db import connection
            # This might run in a different context, so import models here
            try:
                from traffic_app.models import TrafficData, VideoUpload
                
                # Count existing records for this video
                existing_traffic = TrafficData.objects.filter(
                    video_source__endswith=video_filename
                ).count()
                
                existing_uploads = VideoUpload.objects.filter(
                    video_file__endswith=video_filename
                ).count()
                
                logger.info(f"[PROCESSOR] Found {existing_traffic} TrafficData and {existing_uploads} VideoUpload records for this video before processing")
            except Exception as e:
                logger.error(f"[PROCESSOR] Error checking for existing records: {str(e)}")
            
            # Créer le dossier de sortie pour la vidéo traitée
            output_dir = os.path.join(os.path.dirname(video_path), 'processed')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'processed_' + os.path.basename(video_path))
            
            # Ouvrir la vidéo
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"Impossible d'ouvrir la vidéo: {video_path}")

            # Obtenir les propriétés de la vidéo
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Créer le writer pour la vidéo de sortie
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

            # Réinitialiser les variables de suivi
            results = []
            frame_count = 0
            prev_detections = {}
            accidents_detected = []
            self.unique_vehicles.clear()  # Réinitialiser le compteur de véhicules uniques
            self.accident_frames.clear()  # Réinitialiser le compteur de frames avec accident
            
            # Réinitialiser le suivi des objets
            self.trajectories = {}
            self.object_history = {}
            self.next_id = 0
            
            # Réinitialiser le suivi des accidents pour chaque vidéo
            self.accident_locations = []
            self.last_accident_time = None
            
            # Données de consistance temporelle pour les accidents
            accident_detection_count = 0
            required_consecutive_detections = max(3, fps // 10)  # Au moins 3 frames ou 1/10e de seconde
            potential_accident = None
            
            # Pour un traitement plus efficace, définir un intervalle d'échantillonnage 
            # basé sur la durée de la vidéo
            sampling_interval = 1
            if total_frames > 500:  # Si vidéo longue (>15s à 30fps)
                sampling_interval = 3  # Échantillonner toutes les 3 frames
            elif total_frames > 300:  # Si vidéo moyenne (>10s à 30fps)
                sampling_interval = 2  # Échantillonner toutes les 2 frames
            
            # Variables pour le suivi statistique
            vehicle_count_history = []      # Pour suivre le nombre de véhicules par frame
            vehicle_sizes = {}              # Pour stocker la taille moyenne de chaque type de véhicule
            vehicle_first_last_seen = {}    # Pour suivre quand chaque véhicule apparaît/disparaît
            frame_skip_counter = 0          # Pour gérer le saut de frames
            
            # Créer une liste pour suivre les positions des véhicules à travers le temps
            vehicle_positions_history = {}
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                frame_skip_counter += 1
                
                # Sauter des frames selon l'intervalle d'échantillonnage pour les longues vidéos
                if sampling_interval > 1 and frame_skip_counter % sampling_interval != 0:
                    continue
                
                frame_skip_counter = 0
                current_time = time.time()
                
                # Détecter les véhicules dans la frame
                detections, annotated_frame = self.detector.detect_vehicles(frame)
                current_detections = {}
                
                # Traiter chaque détection
                for detection in detections:
                    obj_id = self.get_object_id(detection, prev_detections)
                    
                    # Calculer des informations sur le véhicule
                    x1, y1, x2, y2 = detection['bbox']
                    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                    width, height = x2 - x1, y2 - y1
                    area = width * height
                    
                    current_detections[obj_id] = {
                        'center': (center_x, center_y),
                        'class': detection['class'],
                        'bbox': detection['bbox'],
                        'area': area
                    }
                    
                    # Suivre la première et dernière apparition de chaque véhicule
                    if obj_id not in vehicle_first_last_seen:
                        vehicle_first_last_seen[obj_id] = {'first': frame_count, 'last': frame_count}
                    else:
                        vehicle_first_last_seen[obj_id]['last'] = frame_count
                    
                    # Mettre à jour les stats de taille par type de véhicule
                    v_class = detection['class']
                    if v_class not in vehicle_sizes:
                        vehicle_sizes[v_class] = {'total_area': 0, 'count': 0}
                    vehicle_sizes[v_class]['total_area'] += area
                    vehicle_sizes[v_class]['count'] += 1
                    
                    # Mettre à jour la trajectoire
                    if obj_id not in self.trajectories:
                        self.trajectories[obj_id] = []
                    self.trajectories[obj_id].append((center_x, center_y))
                    
                    # Suivre les positions pour l'analyse de trajectoire
                    if obj_id not in vehicle_positions_history:
                        vehicle_positions_history[obj_id] = []
                    vehicle_positions_history[obj_id].append((frame_count, center_x, center_y))
                    
                    # Stocker l'historique complet de l'objet
                    if obj_id not in self.object_history:
                        self.object_history[obj_id] = []
                    self.object_history[obj_id].append({
                        'frame': frame_count,
                        'center': (center_x, center_y),
                        'bbox': detection['bbox'],
                        'area': area
                    })
                    
                    # Dessiner la trajectoire sur l'image annotée
                    if len(self.trajectories[obj_id]) > 1:
                        traj_points = np.array(self.trajectories[obj_id], np.int32)
                        traj_points = traj_points.reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [traj_points], False, (0, 255, 255), 2)
                    
                    # Dessiner l'ID du véhicule
                    cv2.putText(annotated_frame, f"ID:{obj_id}", (int(center_x), int(y1) - 10),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Calculer le niveau de congestion
                congestion_level = self.detector.calculate_congestion(detections)
                
                # Détecter les accidents
                accident_info = self.detector.detect_accidents(frame, detections)
                
                # Gérer la détection d'accident avec consistance temporelle
                if accident_info['detected']:
                    # Marquer cette frame comme contenant un accident potentiel
                    self.accident_frames.add(frame_count)
                    
                    # Augmenter le compteur de détections consécutives
                    accident_detection_count += 1
                    
                    # Stocker les infos de l'accident potentiel (ou les mettre à jour)
                    if not potential_accident or accident_info['confidence'] > potential_accident['confidence']:
                        potential_accident = accident_info
                else:
                    # Réinitialiser le compteur s'il n'y a pas d'accident dans cette frame
                    accident_detection_count = max(0, accident_detection_count - 1)
                
                # Si nous avons suffisamment de détections consécutives, enregistrer l'accident
                if accident_detection_count >= required_consecutive_detections and potential_accident:
                    # Vérifier l'intervalle de temps
                    can_register_new_accident = True
                    if self.last_accident_time is not None:
                        time_since_last = current_time - self.last_accident_time
                        if time_since_last < self.min_accident_interval:
                            can_register_new_accident = False
                    
                    # Vérifier la proximité avec des accidents déjà détectés
                    if 'location' in potential_accident and potential_accident['location'] and can_register_new_accident:
                        new_loc = potential_accident['location']
                        is_new_location = True
                        
                        for existing_loc in self.accident_locations:
                            # Calculer la distance entre le nouvel accident et les existants
                            dx = new_loc[0] - existing_loc[0]
                            dy = new_loc[1] - existing_loc[1]
                            distance = (dx**2 + dy**2)**0.5
                            
                            # Si trop proche d'un accident existant, c'est probablement le même
                            if distance < frame_width * 0.15:  # 15% de la largeur de l'image
                                is_new_location = False
                                break
                        
                        if is_new_location:
                            # C'est un nouvel accident confirmé, l'ajouter à notre liste
                            self.accident_locations.append(new_loc)
                            self.last_accident_time = current_time
                            
                            # Marquer l'accident sur l'image
                            x, y = new_loc
                            cv2.circle(annotated_frame, (int(x), int(y)), 30, (0, 0, 255), 3)
                            cv2.putText(annotated_frame, "ACCIDENT DETECTÉ", 
                                       (int(x)-80, int(y)-40), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            
                            # Enrichir les détails de l'accident
                            involved_vehicles = self._identify_vehicles_in_accident(new_loc, current_detections)
                            potential_accident['involved_vehicles'] = involved_vehicles
                            potential_accident['confirmed'] = True
                            
                            # Ajouter aux résultats pour stockage en base de données
                            accidents_detected.append(potential_accident)
                            
                            # Activer le drapeau d'alerte
                            self.accident_alert = True
                            
                            # Réinitialiser pour le prochain accident potentiel
                            potential_accident = None
                            accident_detection_count = 0
                
                # Compter les véhicules uniques par type (seulement ceux présents dans la frame actuelle)
                current_vehicle_types = {}
                for vehicle_id, vehicle_data in current_detections.items():
                    v_class = vehicle_data['class']
                    if v_class not in current_vehicle_types:
                        current_vehicle_types[v_class] = 0
                    current_vehicle_types[v_class] += 1
                
                # Compter les véhicules uniques globaux par type
                all_vehicle_types = {}
                for _, v_class, _ in self.unique_vehicles:
                    if v_class not in all_vehicle_types:
                        all_vehicle_types[v_class] = 0
                    all_vehicle_types[v_class] += 1
                
                # Ajouter les compteurs sur l'image
                cv2.putText(annotated_frame, f"Véhicules actuels: {len(detections)}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Véhicules uniques: {len(self.unique_vehicles)}", (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Congestion: {congestion_level:.1%}", (10, 90),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Afficher le décompte par type de véhicule (véhicules uniques)
                y_pos = 120
                for vtype, count in all_vehicle_types.items():
                    cv2.putText(annotated_frame, f"{vtype.capitalize()}: {count}", (10, y_pos),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
                    y_pos += 25
                
                # Ajouter le numéro de frame pour le débogage
                cv2.putText(annotated_frame, f"Frame: {frame_count}", (frame_width - 150, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Sauvegarder la frame annotée
                out.write(annotated_frame)
                
                # Suivre le nombre de véhicules dans cette frame
                vehicle_count_history.append(len(detections))
                
                # Sauvegarder les résultats détaillés de cette frame
                result = {
                    'frame_number': frame_count,
                    'detections': detections,
                    'congestion_level': congestion_level,
                    'trajectories': {str(k): v for k, v in self.trajectories.items()},
                    'accident_detected': accident_info['detected'] if 'detected' in accident_info else False,
                    'current_vehicle_count': len(detections),
                    'unique_vehicle_count': len(self.unique_vehicles),
                    'vehicle_types': all_vehicle_types.copy()
                }
                results.append(result)
                
                # Mettre à jour les détections précédentes pour la frame suivante
                prev_detections = current_detections

            # Libérer les ressources
            cap.release()
            out.release()
            
            # Traitement post-vidéo pour les statistiques et l'analyse des accidents
            
            # S'assurer que nous ne renvoyons qu'un seul accident par vidéo (le plus fiable)
            if len(accidents_detected) > 1:
                # Trier par confiance et garder le plus probable
                accidents_detected.sort(key=lambda x: x.get('confidence', 0), reverse=True)
                accidents_detected = [accidents_detected[0]]
            
            # Calculer le nombre total de véhicules uniques détectés
            total_unique_vehicles = len(self.unique_vehicles)
            
            # Calculer le nombre maximum de véhicules dans une frame
            max_vehicles_per_frame = max(vehicle_count_history) if vehicle_count_history else 0
            
            # Analyser les trajectoires pour filtrer les véhicules qui ne sont apparus que brièvement
            # (possiblement des faux positifs)
            valid_vehicles = {}
            min_duration = min(5, total_frames // 30)  # Au moins 5 frames ou 1/30e de la vidéo
            
            for vid, frames in vehicle_first_last_seen.items():
                duration = frames['last'] - frames['first']
                if duration >= min_duration:
                    # Extraire la classe du véhicule depuis unique_vehicles
                    vehicle_class = None
                    for uid, vclass, _ in self.unique_vehicles:
                        if uid == vid:
                            vehicle_class = vclass
                            break
                    
                    if vehicle_class:
                        if vehicle_class not in valid_vehicles:
                            valid_vehicles[vehicle_class] = 0
                        valid_vehicles[vehicle_class] += 1
            
            # Générer les résultats finaux
            final_result = {
                'success': True,
                'results': results,
                'output_video': output_path,
                'trajectories': self.trajectories,
                'accidents': accidents_detected,
                'has_accident': self.accident_alert,
                'unique_vehicle_count': total_unique_vehicles,
                'max_vehicles_per_frame': max_vehicles_per_frame,
                'vehicle_types': valid_vehicles,  # Utiliser les véhicules validés
                'processing_details': {
                    'total_frames': total_frames,
                    'processed_frames': frame_count,
                    'sampling_interval': sampling_interval,
                    'vehicle_sizes': vehicle_sizes
                },
                'file_info': {
                    'filename': os.path.basename(video_path),
                    'filesize': os.path.getsize(video_path),
                    'hash': hash(str(os.path.getsize(video_path)) + os.path.basename(video_path)),
                    'processed': True
                }
            }
            
            # Nettoyer les ressources
            cap.release()
            out.release()
            
            # Add final results logging
            logger.info(f"[PROCESSOR] Video processing complete: {video_path}")
            logger.info(f"[PROCESSOR] Results: {total_unique_vehicles} unique vehicles detected")
            logger.info(f"[PROCESSOR] Vehicle types: {valid_vehicles}")
            logger.info(f"[PROCESSOR] Accidents detected: {len(accidents_detected)}")
            
            return final_result

        except Exception as e:
            print(f"Erreur lors du traitement de la vidéo: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def _identify_vehicles_in_accident(self, accident_location, current_detections, max_distance=100):
        """Identifie les véhicules impliqués dans un accident en fonction de leur proximité"""
        involved_vehicles = []
        
        # Si pas de localisation d'accident, retourner une liste vide
        if not accident_location:
            return involved_vehicles
            
        # Vérifier la proximité de chaque véhicule à l'emplacement de l'accident
        for vehicle_id, vehicle_data in current_detections.items():
            vehicle_center = vehicle_data['center']
            vehicle_class = vehicle_data['class']
            
            # Calculer la distance
            dx = vehicle_center[0] - accident_location[0]
            dy = vehicle_center[1] - accident_location[1]
            distance = (dx**2 + dy**2)**0.5
            
            # Si le véhicule est suffisamment proche, le considérer comme impliqué
            if distance <= max_distance:
                involved_vehicles.append({
                    'id': vehicle_id,
                    'class': vehicle_class,
                    'distance': distance
                })
                
        return involved_vehicles

    def process_single_frame(self, frame):
        """Traite une seule frame pour la détection en temps réel"""
        try:
            # Utiliser la fonction detect_vehicles au lieu de detect_objects
            detections, annotated_frame = self.detector.detect_vehicles(frame)
            
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
                    has_accident, accident_confidence, accident_severity = self._check_collisions(detections)
            
            # Calculer le niveau de congestion basé sur le nombre de véhicules
            # et la taille de l'image
            frame_area = frame.shape[0] * frame.shape[1]
            vehicle_density = vehicle_count / (frame_area / 100000)  # Normaliser par rapport à la taille
            congestion_level = min(1.0, vehicle_density / 5.0)  # 5 véhicules par 100000 pixels = congestion maximale
            
            return {
                'success': True,
                'vehicle_count': vehicle_count,
                'vehicle_types': vehicle_types,
                'congestion_level': congestion_level,
                'has_accident': has_accident,
                'accident_confidence': accident_confidence,
                'accident_severity': accident_severity,
                'annotated_frame': annotated_frame  # Ajouter l'image annotée pour le débogage
            }
            
        except Exception as e:
            print(f"[ERROR] Erreur dans process_single_frame: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }
    
    def _check_collisions(self, detections):
        """Vérifie les collisions potentielles entre les véhicules détectés"""
        has_accident = False
        accident_confidence = 0.0
        accident_severity = 0
        
        # Vérifier les intersections entre les boîtes englobantes
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                box1 = detections[i]['bbox']
                box2 = detections[j]['bbox']
                
                # Calculer l'IoU (Intersection over Union)
                iou = self._calculate_iou(box1, box2)
                
                # Si l'IoU est élevé, c'est potentiellement une collision
                if iou > 0.3:  # Seuil ajustable
                    has_accident = True
                    accident_confidence = iou
                    # La sévérité dépend de la taille des véhicules et de leur vitesse relative
                    accident_severity = min(3, int(iou * 3) + 1)
                    break
            if has_accident:
                break
                
        return has_accident, accident_confidence, accident_severity
    
    def _calculate_iou(self, box1, box2):
        """Calcule l'Intersection over Union entre deux boîtes englobantes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculer l'intersection
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculer l'union
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - intersection_area
        
        if union_area == 0:
            return 0.0
            
        return intersection_area / union_area

    def process_frame(self, frame_data):
        """Traite une seule frame de la webcam"""
        try:
            # Convertir l'image base64 en format numpy
            if isinstance(frame_data, str):
                # Décoder l'image base64
                import base64
                import io
                from PIL import Image
                img_data = base64.b64decode(frame_data.split(',')[1])
                frame = np.array(Image.open(io.BytesIO(img_data)))
                # Convertir de RGB à BGR pour OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame = frame_data

            # Détecter les véhicules dans la frame
            detections, annotated_frame = self.detector.detect_vehicles(frame)
            
            # Initialiser les compteurs
            vehicle_types = {}
            vehicle_count = 0
            
            # Traiter chaque détection
            current_detections = {}
            for detection in detections:
                vehicle_count += 1
                vehicle_class = detection['class']
                
                # Compter par type de véhicule
                if vehicle_class not in vehicle_types:
                    vehicle_types[vehicle_class] = 0
                vehicle_types[vehicle_class] += 1
                
                # Obtenir un ID unique pour ce véhicule
                obj_id = self.get_object_id(detection, self.object_history)
                
                # Mettre à jour les détections courantes
                x1, y1, x2, y2 = detection['bbox']
                current_detections[obj_id] = {
                    'center': ((x1 + x2) / 2, (y1 + y2) / 2),
                    'class': vehicle_class,
                    'bbox': detection['bbox']
                }
            
            # Vérifier les collisions et accidents
            has_accident = False
            accident_confidence = 0.0
            
            # Ne vérifier les accidents que s'il y a au moins 2 véhicules
            if len(current_detections) >= 2:
                # Vérifier les collisions entre les véhicules
                for i, (id1, vehicle1) in enumerate(current_detections.items()):
                    for id2, vehicle2 in list(current_detections.items())[i+1:]:
                        # Calculer l'IoU entre les deux véhicules
                        iou = self._calculate_iou(vehicle1['bbox'], vehicle2['bbox'])
                        
                        # Calculer la distance entre les centres des véhicules
                        center1 = vehicle1['center']
                        center2 = vehicle2['center']
                        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                        
                        # Un accident est détecté si les véhicules sont très proches ou se chevauchent
                        if iou > 0.3 or distance < 50:  # Seuils ajustables
                            has_accident = True
                            accident_confidence = max(accident_confidence, iou)
            
            # Calculer le niveau de congestion (basé sur le nombre de véhicules)
            congestion_level = min(1.0, vehicle_count / 10.0)
            
            # Mettre à jour l'historique des objets
            self.object_history = current_detections
            
            return {
                'success': True,
                'vehicle_count': vehicle_count,
                'vehicle_types': vehicle_types,
                'congestion_level': congestion_level,
                'has_accident': has_accident,
                'confidence': accident_confidence
            }
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Erreur dans process_frame: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

def process_video_feed(video_source):
    processor = VideoProcessor()
    return processor.process_video_feed(video_source)
