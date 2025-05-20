// Initialisation de la carte
let map = L.map('traffic-map').setView([48.8566, 2.3522], 13);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors'
}).addTo(map);

// Variables globales
let markers = [];
let heatmapLayer = null;
let showHeatmap = true;
let showMarkers = true;

// Fonction pour charger les données de trafic
async function loadTrafficData() {
    try {
        const response = await fetch('/api/traffic-data/');
        const data = await response.json();
        updateMap(data.traffic_data);
    } catch (error) {
        console.error('Erreur lors du chargement des données:', error);
    }
}

// Fonction pour mettre à jour la carte
function updateMap(trafficData) {
    // Nettoyer les marqueurs existants
    clearMarkers();
    
    // Préparer les données pour la carte de chaleur
    let heatmapData = [];
    
    trafficData.forEach(point => {
        // Ajouter des marqueurs
        if (showMarkers) {
            const marker = L.marker([point.latitude, point.longitude])
                .bindPopup(`
                    <strong>Niveau de congestion:</strong> ${(point.congestion_level * 100).toFixed(1)}%<br>
                    <strong>Nombre de véhicules:</strong> ${point.vehicle_count}<br>
                    <strong>Heure:</strong> ${new Date(point.timestamp).toLocaleString()}
                `);
            markers.push(marker);
            if (showMarkers) marker.addTo(map);
        }
        
        // Ajouter des points pour la carte de chaleur
        heatmapData.push([
            point.latitude,
            point.longitude,
            point.congestion_level
        ]);
    });
    
    // Mettre à jour la carte de chaleur
    if (heatmapLayer) {
        map.removeLayer(heatmapLayer);
    }
    if (showHeatmap) {
        heatmapLayer = L.heatLayer(heatmapData, {
            radius: 25,
            blur: 15,
            maxZoom: 10,
            gradient: {
                0.4: 'green',
                0.6: 'yellow',
                0.8: 'red'
            }
        }).addTo(map);
    }
}

// Fonction pour nettoyer les marqueurs
function clearMarkers() {
    markers.forEach(marker => map.removeLayer(marker));
    markers = [];
}

// Gestionnaires d'événements pour les contrôles
document.getElementById('toggle-heatmap').addEventListener('click', () => {
    showHeatmap = !showHeatmap;
    loadTrafficData();
});

document.getElementById('toggle-markers').addEventListener('click', () => {
    showMarkers = !showMarkers;
    loadTrafficData();
});

document.getElementById('time-range').addEventListener('change', (e) => {
    loadTrafficData();
});

// Charger les données initiales
loadTrafficData();

// Mettre à jour les données toutes les 30 secondes
setInterval(loadTrafficData, 30000);
