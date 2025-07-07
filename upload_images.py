#!/usr/bin/env python3
"""
OneDrive Upload Script für Travel Hunter Bilder
Benötigt: pip install requests msal
"""

import os
import requests
from pathlib import Path

def upload_to_onedrive():
    """
    Hinweis: Dieses Skript benötigt Microsoft Graph API Authentifizierung
    
    Für die einfache Nutzung:
    1. Installiere OneDrive Desktop App
    2. Führe upload_to_onedrive.sh aus
    
    Für API-Upload:
    1. Registriere App in Azure Portal
    2. Hole Client ID und Secret
    3. Implementiere OAuth Flow
    """
    
    images_path = Path("data_acquisition/destination_images")
    
    if not images_path.exists():
        print("❌ Bilder-Ordner nicht gefunden!")
        return
    
    image_files = list(images_path.glob("*"))
    print(f"📊 Gefunden: {len(image_files)} Bilder")
    print(f"📁 Ordner: {images_path.absolute()}")
    
    print("\n🔧 Für OneDrive Upload:")
    print("1. Nutze die OneDrive Desktop App")
    print("2. Oder führe ./upload_to_onedrive.sh aus")
    print("3. Oder ziehe den Ordner manuell in den Browser")
    
    onedrive_link = "https://zhaw-my.sharepoint.com/:f:/r/personal/kryezleo_students_zhaw_ch/Documents/Travel%20Hunter/Image/Wikipedia_destination"
    print(f"\n🔗 OneDrive Link: {onedrive_link}")

if __name__ == "__main__":
    upload_to_onedrive()
