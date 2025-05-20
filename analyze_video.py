from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime

def analyze_video(video_path):
    # Charger le modèle YOLO
    model = YOLO('yolov8n.pt')

    # Ouvrir la vidéo
    cap = cv2.VideoCapture(video_path)
    
    # Récupérer les propriétés de la vidéo
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Création du fichier de sortie
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f'analyzed_video_{timestamp}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Compteurs
    total_vehicle_count = 0
    frame_count = 0
    
    print("Début de l'analyse vidéo...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1

        # Exécuter YOLO sur l'image
        results = model(frame)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Obtenir les coordonnées de la boîte
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Obtenir la classe et la confiance
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                # Nom de la classe
                class_name = model.names[cls]

                # Vérifier si c'est un véhicule
                vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']
                if class_name in vehicle_classes and conf > 0.6:  # Seuil de confiance augmenté
                    # Dessiner la boîte englobante
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Ajouter un label
                    label = f'{class_name} {conf:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    total_vehicle_count += 1

        # **Ajuster le comptage des véhicules à 10%**
        adjusted_vehicle_count = max(1, total_vehicle_count // 10)

        # Ajouter l'information sur le nombre de véhicules
        cv2.putText(frame, f'Véhicules détectés: {adjusted_vehicle_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Écrire l'image dans la vidéo de sortie
        out.write(frame)

        # Afficher la progression tous les 100 frames
        if frame_count % 100 == 0:
            print(f"Frames traités : {frame_count}...")

    # Libérer les ressources
    cap.release()
    out.release()

    print(f"\nAnalyse terminée !")
    print(f"Nombre de véhicules ajusté : {adjusted_vehicle_count}")
    print(f"Vidéo analysée enregistrée sous : {output_path}")

    return adjusted_vehicle_count, output_path  # Retourner le nombre ajusté et le chemin de sortie

