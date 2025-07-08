#!/usr/bin/env python3
"""
Script zum Generieren einer massiven Wikipedia-Destination-Liste
Erstellt eine neue massive wikipedia_destination_links.py mit 1500+ Destinationen
"""

import sys
import os
from wikipedia_link_discoverer import MassiveWikipediaLinkDiscoverer

def generate_massive_destination_file():
    """Generiert eine neue massive Destination-Datei"""
    print("🚀 Generiere massive Wikipedia-Destination-Liste...")
    
    # Entdecke massive Links
    discoverer = MassiveWikipediaLinkDiscoverer()
    massive_urls = discoverer.discover_massive_links(target_count=1500)
    
    # Generiere Python-Code für die neue Datei
    file_content = f'''#!/usr/bin/env python3
"""
Wikipedia Destination Links - MASSIVE EDITION
Automatisch generierte Sammlung von {len(massive_urls)} Wikipedia-URLs für weltweite Destinationen
Generiert am: {discoverer.session.headers.get('User-Agent', 'TravelHunters')}
"""

# Kleinere Testliste (Original 15 Destinationen)
wikipedia_destination_urls = [
    # EUROPA - Top Städte (5)
    "https://en.wikipedia.org/wiki/Paris",
    "https://en.wikipedia.org/wiki/London", 
    "https://en.wikipedia.org/wiki/Rome",
    "https://en.wikipedia.org/wiki/Barcelona",
    "https://en.wikipedia.org/wiki/Amsterdam",
    
    # ASIEN - Top Städte (5)
    "https://en.wikipedia.org/wiki/Tokyo",
    "https://en.wikipedia.org/wiki/Bangkok",
    "https://en.wikipedia.org/wiki/Singapore",
    "https://en.wikipedia.org/wiki/Dubai",
    "https://en.wikipedia.org/wiki/Seoul",
    
    # AMERIKA - Top Städte (3)
    "https://en.wikipedia.org/wiki/New_York_City",
    "https://en.wikipedia.org/wiki/Los_Angeles",
    "https://en.wikipedia.org/wiki/Toronto",
    
    # AFRIKA & OZEANIEN (2)
    "https://en.wikipedia.org/wiki/Sydney",
    "https://en.wikipedia.org/wiki/Cape_Town",
]

# MASSIVE EDITION - {len(massive_urls)} Destinationen weltweit
wikipedia_destination_urls_extended = [
'''
    
    # Füge alle URLs hinzu
    for i, url in enumerate(massive_urls):
        # URL formatieren mit sauberen Anführungszeichen
        clean_url = url.replace('"', '\\"')  # Escape quotes
        file_content += f'    "{clean_url}",\n'
        
        # Kommentare alle 50 URLs für bessere Übersicht
        if (i + 1) % 50 == 0:
            file_content += f'    # --- {i + 1} Destinationen ---\n'
    
    # Datei abschließen
    file_content += f''']

# Statistiken
MASSIVE_STATS = {{
    "total_destinations": {len(massive_urls)},
    "discovery_method": "Automatisch von Wikipedia-Listen gesammelt",
    "sources": [
        "National capitals",
        "Largest cities worldwide", 
        "Tourist destinations",
        "Regional city lists",
        "Transport hubs (airports, ports)"
    ],
    "filter_applied": "Gültige Destinationen, keine Codes/Templates",
    "target_count": 1500
}}

# Funktionen
def get_destination_urls():
    """Gibt die kleine Testliste zurück"""
    return wikipedia_destination_urls

def get_extended_destination_urls():
    """Gibt die MASSIVE Liste zurück"""
    return wikipedia_destination_urls_extended

def get_massive_stats():
    """Gibt Statistiken über die massive Liste zurück"""
    return MASSIVE_STATS

if __name__ == "__main__":
    print("Wikipedia Destination Links - MASSIVE EDITION")
    print("=" * 60)
    print(f"🌍 Total destinations: {{MASSIVE_STATS['total_destinations']}}")
    print(f"🎯 Target achieved: {{len(wikipedia_destination_urls_extended)}} destinations")
    print(f"📊 Discovery sources: {{len(MASSIVE_STATS['sources'])}}")
    print()
    print("🔥 READY FOR MASSIVE SCRAPING! 🔥")
    print("Verwende wikipedia_destination_urls_extended für das komplette Scraping")
'''

    # Speichere die neue Datei
    output_file = "wikipedia_destination_links_massive.py"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(file_content)
    
    print(f"✅ Massive Destination-Datei erstellt: {output_file}")
    print(f"📊 {len(massive_urls)} Destinationen bereit!")
    print(f"🚀 Bereit für massive Scraping-Operation!")
    
    return output_file, len(massive_urls)

if __name__ == "__main__":
    output_file, count = generate_massive_destination_file()
    print(f"\\n🎯 Nächster Schritt: Ersetze die ursprüngliche wikipedia_destination_links.py")
    print(f"   oder verwende direkt {output_file} für {count} Destinationen!")
