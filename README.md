
<h1 align="center">📊 Plateforme de suivi de l'indice des prix à la consommation en Mauritanie</h1>

<p align="center">
  <b>Une plateforme interactive pour analyser l'évolution de l'INPC en Mauritanie, comprendre les dynamiques économiques et prendre des décisions éclairées.</b>  
</p>

<p align="center">
  🔍 <i>Suivi des prix</i> | 📈 <i>Visualisation des tendances économiques</i> | 💰 <i>Indicateurs économiques clés</i>  
</p>

---

## 🌍 Table des Matières
🔹 [📖 Description du Projet](#-description-du-projet)  
🔹 [🎯 Objectif](#-objectif)  
🔹 [⚙️ Technologies Utilisées](#%EF%B8%8F-technologies-utilisées)  
🔹 [🚀 Fonctionnalités Clés](#-fonctionnalités-clés)  
🔹 [📸 Aperçu Visuel](#-aperçu-visuel)  
🔹 [📦 Installation & Utilisation](#-installation--utilisation)   
🔹 [📬 Contact](#-contact)  

---

## 📖 Description du Projet  
**Suivi de l'Indice des Prix à la Consommation en Mauritanie** est un outil interactif conçu pour suivre et analyser les fluctuations des prix des biens et services en Mauritanie. Grâce à des graphiques interactifs et des filtres avancés, cette plateforme vous permet de comprendre en profondeur les dynamiques économiques du pays.

- **📊 Interface intuitive** pour explorer les variations des prix dans différentes catégories de produits.
- **📉 Analyse détaillée** des tendances économiques avec des visualisations de données dynamiques.
- **🛒 Comparaison** des prix entre régions et catégories, permettant une vue d'ensemble complète des tendances des prix.

--- 

## 🎯 Objectif  
Le principal objectif de ce projet est de fournir une plateforme **dynamique** et **précise** pour l’analyse des prix à la consommation en Mauritanie. En offrant des outils interactifs, ce projet vise à :

- **Faciliter l'accès** à l'information économique pour les chercheurs, économistes, décideurs, et citoyens.
- **Soutenir la prise de décisions stratégiques** dans le domaine économique, notamment pour les politiques publiques et les entreprises.
- **Proposer une plateforme évolutive** qui pourra intégrer d'autres indicateurs économiques au fur et à mesure.

🎯 **Pour qui ?**  
- **Économistes & chercheurs** 📊  
- **Gouvernements & institutions** 📈  
- **Entrepreneurs & investisseurs** 💰  
- **Citoyens curieux des tendances économiques** 🏠  

---

## ⚙️ Technologies Utilisées  

| 🛠️ Technologie | 🚀 Rôle |
|----------------|--------|
| 🐍 **Python**   | Backend et traitement des données |
| 🌍 **Django**   | Développement du serveur web |
| 🗄️ **PostgreSQL** | Base de données |
| 🎨 **HTML / CSS / JS** | Conception et développement de l'interface utilisateur |
| 📊 **Chart.js / D3.js** | Visualisation interactive des données |
| 🐳 **Docker**   | Conteneurisation et déploiement |

---

## 🚀 Fonctionnalités Clés  
✅ **📊 Tableau de bord interactif** – Suivi des prix par produit et région en temps réel.  
✅ **🔍 Filtres avancés** – Sélection de catégories, périodes spécifiques, et comparaisons géographiques.  
✅ **📈 Visualisation dynamique** – Graphiques interactifs comme les courbes de tendance, heatmaps, histogrammes, et plus encore.  
✅ **📡 API RESTful** – Accès aux données sous forme de JSON pour les intégrateurs et développeurs.  
✅ **🔐 Mode Admin** – Interface dédiée pour gérer et modifier les données via Django Admin.  
✅ **🐳 Déploiement Dockerisé** – Facilité d'exécution et de déploiement sans configuration locale complexe.

---

## 📸 Aperçu Visuel  

### Vue d'une partie du tableau de bord :
![Tableau de bord](images/a.png)

### Visualisation de quelques produit :
![Graphiques dynamiques](images/b.png)

---

📬 Contact
Pour toute question, suggestion ou collaboration, n'hésitez pas à me contacter :

LinkedIn : www.linkedin.com/in/amadou-diallo-ing04

Email : 23217@esp.mr


## 📦 Installation & Utilisation  

Suivez ces étapes pour installer et lancer le projet sur votre machine locale :

```bash
# 1️⃣ Cloner le projet depuis GitHub
git clone https://github.com/AmadouMamadouDiallo/Suivi-de-I-indice-des-prix-a-la-consommation-en-Mauritanie.git
cd Suivi-de-I-indice-des-prix-a-la-consommation-en-Mauritanie

# 2️⃣ Créer un environnement virtuel et installer les dépendances
python -m venv venv
source venv/bin/activate  # (Sous Windows: venv\Scripts\activate)
pip install -r requirements.txt

# 3️⃣ Appliquer les migrations de la base de données
python manage.py migrate

# 4️⃣ Lancer le serveur Django
python manage.py runserver

# 5️⃣ Accéder à l'application via votre navigateur
http://127.0.0.1:8000/

