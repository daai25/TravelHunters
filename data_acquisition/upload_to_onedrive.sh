#!/bin/bash
# OneDrive Upload Script
# Kopiert Bilder zum lokalen OneDrive Ordner

# Finde OneDrive Ordner
ONEDRIVE_PATH="$HOME/OneDrive - ZHAW"
TARGET_PATH="$ONEDRIVE_PATH/Travel Hunter/Image/Wikipedia_destination"

# Erstelle Zielordner falls nicht vorhanden
mkdir -p "$TARGET_PATH"

# Kopiere alle Bilder
echo "🚀 Kopiere Bilder zu OneDrive..."
rsync -av --progress data_acquisition/destination_images/ "$TARGET_PATH/"

echo "✅ Upload abgeschlossen!"
echo "📁 Bilder sind jetzt in: $TARGET_PATH"
echo "🔄 OneDrive synchronisiert automatisch..."
