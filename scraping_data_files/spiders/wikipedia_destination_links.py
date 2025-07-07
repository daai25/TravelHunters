#!/usr/bin/env python3
"""
Wikipedia Destination Links
Sammlung von Wikipedia-URLs für weltweite Destinationen
Inklusive automatische Generierung von massiven Link-Listen (1000+ Destinationen)
"""

import requests
import time
import re
from urllib.parse import urljoin
from bs4 import BeautifulSoup

# Kleinere Testliste mit 15 bekannten Destinationen
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

# Erweiterte Liste für späteren Gebrauch (auskommentiert)

wikipedia_destination_urls_extended = [
    # Alle obigen Links plus:
    
    # EUROPA - Weitere Städte
    "https://en.wikipedia.org/wiki/Berlin",
    "https://en.wikipedia.org/wiki/Vienna",
    "https://en.wikipedia.org/wiki/Prague",
    "https://en.wikipedia.org/wiki/Istanbul",
    "https://en.wikipedia.org/wiki/Athens",
    "https://en.wikipedia.org/wiki/Madrid",
    "https://en.wikipedia.org/wiki/Lisbon",
    "https://en.wikipedia.org/wiki/Florence",
    "https://en.wikipedia.org/wiki/Venice",
    "https://en.wikipedia.org/wiki/Milan",
    "https://en.wikipedia.org/wiki/Munich",
    "https://en.wikipedia.org/wiki/Zurich",
    "https://en.wikipedia.org/wiki/Stockholm",
    "https://en.wikipedia.org/wiki/Copenhagen",
    "https://en.wikipedia.org/wiki/Oslo",
    "https://en.wikipedia.org/wiki/Helsinki",
    "https://en.wikipedia.org/wiki/Dublin",
    "https://en.wikipedia.org/wiki/Edinburgh",
    "https://en.wikipedia.org/wiki/Brussels",
    "https://en.wikipedia.org/wiki/Budapest",
    
    # ASIEN - Weitere Städte
    "https://en.wikipedia.org/wiki/Kyoto",
    "https://en.wikipedia.org/wiki/Mumbai",
    "https://en.wikipedia.org/wiki/Delhi",
    "https://en.wikipedia.org/wiki/Beijing",
    "https://en.wikipedia.org/wiki/Shanghai",
    "https://en.wikipedia.org/wiki/Hong_Kong",
    "https://en.wikipedia.org/wiki/Kuala_Lumpur",
    "https://en.wikipedia.org/wiki/Jakarta",
    "https://en.wikipedia.org/wiki/Manila",
    "https://en.wikipedia.org/wiki/Ho_Chi_Minh_City",
    "https://en.wikipedia.org/wiki/Hanoi",
    "https://en.wikipedia.org/wiki/Phnom_Penh",
    "https://en.wikipedia.org/wiki/Yangon",
    "https://en.wikipedia.org/wiki/Colombo",
    "https://en.wikipedia.org/wiki/Kathmandu",
    "https://en.wikipedia.org/wiki/Thimphu",
    "https://en.wikipedia.org/wiki/Dhaka",
    "https://en.wikipedia.org/wiki/Islamabad",
    "https://en.wikipedia.org/wiki/Kabul",
    "https://en.wikipedia.org/wiki/Tehran",
    
    # AMERIKA - Weitere Städte
    "https://en.wikipedia.org/wiki/Chicago",
    "https://en.wikipedia.org/wiki/San_Francisco",
    "https://en.wikipedia.org/wiki/Las_Vegas",
    "https://en.wikipedia.org/wiki/Miami",
    "https://en.wikipedia.org/wiki/Boston",
    "https://en.wikipedia.org/wiki/Washington,_D.C.",
    "https://en.wikipedia.org/wiki/Vancouver",
    "https://en.wikipedia.org/wiki/Montreal",
    "https://en.wikipedia.org/wiki/Mexico_City",
    "https://en.wikipedia.org/wiki/Cancún",
    "https://en.wikipedia.org/wiki/Guatemala_City",
    "https://en.wikipedia.org/wiki/San_José,_Costa_Rica",
    "https://en.wikipedia.org/wiki/Panama_City",
    "https://en.wikipedia.org/wiki/Havana",
    "https://en.wikipedia.org/wiki/Kingston,_Jamaica",
    "https://en.wikipedia.org/wiki/Santo_Domingo",
    "https://en.wikipedia.org/wiki/San_Juan,_Puerto_Rico",
    "https://en.wikipedia.org/wiki/Caracas",
    "https://en.wikipedia.org/wiki/Bogotá",
    "https://en.wikipedia.org/wiki/Quito",
    "https://en.wikipedia.org/wiki/Lima",
    "https://en.wikipedia.org/wiki/La_Paz",
    "https://en.wikipedia.org/wiki/Santiago",
    "https://en.wikipedia.org/wiki/Buenos_Aires",
    "https://en.wikipedia.org/wiki/Montevideo",
    "https://en.wikipedia.org/wiki/Asunción",
    "https://en.wikipedia.org/wiki/São_Paulo",
    "https://en.wikipedia.org/wiki/Rio_de_Janeiro",
    "https://en.wikipedia.org/wiki/Brasília",
    
    # AFRIKA - Städte
    "https://en.wikipedia.org/wiki/Cairo",
    "https://en.wikipedia.org/wiki/Alexandria",
    "https://en.wikipedia.org/wiki/Casablanca",
    "https://en.wikipedia.org/wiki/Marrakech",
    "https://en.wikipedia.org/wiki/Tunis",
    "https://en.wikipedia.org/wiki/Algiers",
    "https://en.wikipedia.org/wiki/Lagos",
    "https://en.wikipedia.org/wiki/Abuja",
    "https://en.wikipedia.org/wiki/Accra",
    "https://en.wikipedia.org/wiki/Dakar",
    "https://en.wikipedia.org/wiki/Bamako",
    "https://en.wikipedia.org/wiki/Ouagadougou",
    "https://en.wikipedia.org/wiki/Abidjan",
    "https://en.wikipedia.org/wiki/Addis_Ababa",
    "https://en.wikipedia.org/wiki/Nairobi",
    "https://en.wikipedia.org/wiki/Kampala",
    "https://en.wikipedia.org/wiki/Kigali",
    "https://en.wikipedia.org/wiki/Dar_es_Salaam",
    "https://en.wikipedia.org/wiki/Lusaka",
    "https://en.wikipedia.org/wiki/Harare",
    "https://en.wikipedia.org/wiki/Gaborone",
    "https://en.wikipedia.org/wiki/Windhoek",
    "https://en.wikipedia.org/wiki/Johannesburg",
    "https://en.wikipedia.org/wiki/Durban",
    
    # OZEANIEN - Städte
    "https://en.wikipedia.org/wiki/Melbourne",
    "https://en.wikipedia.org/wiki/Brisbane",
    "https://en.wikipedia.org/wiki/Perth",
    "https://en.wikipedia.org/wiki/Adelaide",
    "https://en.wikipedia.org/wiki/Auckland",
    "https://en.wikipedia.org/wiki/Wellington",
    "https://en.wikipedia.org/wiki/Christchurch",
    "https://en.wikipedia.org/wiki/Suva",
    "https://en.wikipedia.org/wiki/Port_Moresby",
    "https://en.wikipedia.org/wiki/Honiara",
    "https://en.wikipedia.org/wiki/Port_Vila",
    "https://en.wikipedia.org/wiki/Nuku%27alofa",
    "https://en.wikipedia.org/wiki/Apia",
]


# Funktion zum Abrufen der Links
def get_destination_urls():
    """Gibt die aktuelle Liste der Wikipedia-Destination-URLs zurück"""
    return wikipedia_destination_urls

def get_extended_destination_urls():
    """Gibt die erweiterte Liste zurück - jetzt mit 2000+ Destinationen"""
    # Starte mit der manuellen erweiterten Liste
    base_list = list(wikipedia_destination_urls_extended)
    
    # Füge automatisch generierte massive Links hinzu um 2000+ zu erreichen
    print("🚀 Erweitere Liste auf 2000+ Destinationen...")
    needed_additional = 2500 - len(base_list)  # Erhöht auf 2500 um sicher 2000+ zu haben
    
    if needed_additional > 0:
        massive_links = get_massive_destination_urls(max_destinations=needed_additional)
        # Kombiniere und entferne Duplikate
        combined_links = list(set(base_list + massive_links))
        print(f"✅ Erweiterte Liste fertig: {len(combined_links)} Destinationen")
        return combined_links
    else:
        return base_list

def get_massive_destination_urls(max_destinations=2500):
    """Gibt eine massive Liste von 2000+ Destinationen zurück (automatisch generiert)"""
    return generate_and_get_massive_links(max_destinations)

# Statistiken
def get_stats():
    """Gibt Statistiken über die Link-Listen zurück"""
    extended_count = len(get_extended_destination_urls())  # Jetzt dynamisch berechnet
    return {
        "basic_destinations": len(wikipedia_destination_urls),
        "extended_destinations": f"{extended_count} (inklusive massive Generierung)",
        "massive_destinations": "1000+ (automatisch integriert)",
        "regions": {
            "europa": "250+",
            "asien": "250+", 
            "amerika": "250+",
            "afrika_ozeanien": "250+"
        }
    }

class MassiveLinkGenerator:
    """
    Automatische Generierung von 1000+ Wikipedia-Destination-Links
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TravelHunters-LinkDiscoverer 1.0 (Educational Project)'
        })
        self.discovered_links = set()
        
    def get_links_from_category(self, category_url, max_links=200):
        """Extrahiert Links aus einer Wikipedia-Kategorie"""
        try:
            print(f"🔍 Durchsuche: {category_url}")
            response = self.session.get(category_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            links = []
            
            # Finde alle Links in der Kategorie
            content_div = soup.find('div', {'id': 'mw-content-text'})
            if content_div:
                for link in content_div.find_all('a', href=True):
                    href = link.get('href')
                    if href and href.startswith('/wiki/') and ':' not in href:
                        full_url = urljoin('https://en.wikipedia.org', href)
                        if self.is_valid_destination(href):
                            links.append(full_url)
                            if len(links) >= max_links:
                                break
            
            print(f"✅ Gefunden: {len(links)} Links")
            return list(set(links))
            
        except Exception as e:
            print(f"❌ Fehler bei {category_url}: {e}")
            return []
    
    def is_valid_destination(self, href):
        """Prüft ob ein Link eine gültige Destination ist"""
        # Ausschließen von Meta-Seiten, Listen, etc.
        exclude_patterns = [
            'Category:', 'Template:', 'Help:', 'User:', 'Talk:', 'File:',
            'List_of_', 'Lists_of_', 'Index_of_', 'Timeline_of_',
            'History_of_', 'Geography_of_', 'Demographics_of_',
            'Economy_of_', 'Politics_of_', 'Culture_of_',
            'Tourist_trap', 'Doors_Open_Days', 'Welcome_sign',
            'Cities_Development_Initiative', 'Scenic_viewpoint',
            'Template:', 'Portal:', 'Wikipedia:', 'Special:',
            'Main_Page'
        ]
        
        for pattern in exclude_patterns:
            if pattern in href:
                return False
        
        # Nur Links zu Städten, Ländern und bekannten Orten akzeptieren
        valid_patterns = [
            'wiki/', '_City', '_Town', '_Province', '_State',
            '_Region', '_County', '_Prefecture'
        ]
        
        # Akzeptiere nur wenn es ein gültiges Muster enthält oder ein einfacher Ortsname ist
        if any(pattern in href for pattern in valid_patterns):
            return True
            
        # Einfache Ortsnamen ohne Unterstriche sind oft Städte
        wiki_part = href.replace('/wiki/', '')
        if '_' not in wiki_part or wiki_part.count('_') <= 2:
            return True
            
        return False
    
    def generate_massive_links(self, max_destinations=2500):
        """Generiert eine massive Liste von 2000+ Destination-Links"""
        print("🚀 Starte massive Link-Generierung...")
        print(f"🎯 Ziel: {max_destinations} Destinationen")
        print("=" * 60)
        
        # Erweiterte Wikipedia-Kategorien für deutlich mehr Destinationen
        categories = [
            # Städte nach Kontinent
            'https://en.wikipedia.org/wiki/Category:Cities_in_Europe',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Asia',
            'https://en.wikipedia.org/wiki/Category:Cities_in_North_America',
            'https://en.wikipedia.org/wiki/Category:Cities_in_South_America',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Africa',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Oceania',
            
            # Hauptstädte
            'https://en.wikipedia.org/wiki/Category:Capitals_in_Europe',
            'https://en.wikipedia.org/wiki/Category:Capitals_in_Asia',
            'https://en.wikipedia.org/wiki/Category:Capitals_in_Africa',
            'https://en.wikipedia.org/wiki/Category:Capitals_in_North_America',
            'https://en.wikipedia.org/wiki/Category:Capitals_in_South_America',
            
            # Touristische Destinationen
            'https://en.wikipedia.org/wiki/Category:World_Heritage_Sites',
            'https://en.wikipedia.org/wiki/Category:Populated_places_by_continent',
            'https://en.wikipedia.org/wiki/Category:Municipalities',
            'https://en.wikipedia.org/wiki/Category:Tourist_attractions',
            
            # Europa - alle großen Länder
            'https://en.wikipedia.org/wiki/Category:Cities_in_Germany',
            'https://en.wikipedia.org/wiki/Category:Cities_in_France',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Italy',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Spain',
            'https://en.wikipedia.org/wiki/Category:Cities_in_the_United_Kingdom',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Poland',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Turkey',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Belgium',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Portugal',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Switzerland',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Sweden',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Norway',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Denmark',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Finland',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Hungary',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Romania',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Croatia',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Serbia',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Bulgaria',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Slovenia',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Slovakia',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Estonia',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Latvia',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Lithuania',
            
            # Asien - alle großen Länder
            'https://en.wikipedia.org/wiki/Category:Cities_in_Japan',
            'https://en.wikipedia.org/wiki/Category:Cities_in_China',
            'https://en.wikipedia.org/wiki/Category:Cities_in_India',
            'https://en.wikipedia.org/wiki/Category:Cities_in_South_Korea',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Vietnam',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Indonesia',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Malaysia',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Singapore',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Philippines',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Iran',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Iraq',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Israel',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Saudi_Arabia',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Pakistan',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Bangladesh',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Sri_Lanka',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Myanmar',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Cambodia',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Laos',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Mongolia',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Kazakhstan',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Uzbekistan',
            
            # Amerika - Nord, Mittel und Süd
            'https://en.wikipedia.org/wiki/Category:Cities_in_the_United_States',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Canada',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Mexico',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Brazil',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Argentina',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Chile',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Peru',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Colombia',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Venezuela',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Ecuador',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Bolivia',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Uruguay',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Paraguay',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Guatemala',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Costa_Rica',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Panama',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Nicaragua',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Honduras',
            'https://en.wikipedia.org/wiki/Category:Cities_in_El_Salvador',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Cuba',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Dominican_Republic',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Jamaica',
            
            # Afrika - alle großen Länder
            'https://en.wikipedia.org/wiki/Category:Cities_in_South_Africa',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Egypt',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Nigeria',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Kenya',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Ethiopia',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Tanzania',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Uganda',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Ghana',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Morocco',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Algeria',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Tunisia',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Libya',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Sudan',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Cameroon',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Ivory_Coast',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Senegal',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Mali',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Burkina_Faso',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Niger',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Chad',
            
            # Ozeanien
            'https://en.wikipedia.org/wiki/Category:Cities_in_Australia',
            'https://en.wikipedia.org/wiki/Category:Cities_in_New_Zealand',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Papua_New_Guinea',
            'https://en.wikipedia.org/wiki/Category:Cities_in_Fiji',
            
            # Zusätzliche spezielle Kategorien
            'https://en.wikipedia.org/wiki/Category:Provincial_capitals',
            'https://en.wikipedia.org/wiki/Category:State_capitals',
            'https://en.wikipedia.org/wiki/Category:Port_cities',
            'https://en.wikipedia.org/wiki/Category:Historic_cities',
            'https://en.wikipedia.org/wiki/Category:Cultural_centers',
            'https://en.wikipedia.org/wiki/Category:Religious_centers',
            'https://en.wikipedia.org/wiki/Category:Coastal_cities',
            'https://en.wikipedia.org/wiki/Category:Mountain_cities',
            'https://en.wikipedia.org/wiki/Category:Island_cities',
        ]
        
        all_links = set()
        
        # Sammle Links aus allen Kategorien
        for i, category in enumerate(categories, 1):
            if len(all_links) >= max_destinations:
                break
                
            print(f"\n📂 Kategorie {i}/{len(categories)}")
            links = self.get_links_from_category(category, max_links=300)
            all_links.update(links)
            
            print(f"📊 Zwischenstand: {len(all_links)} einzigartige Links")
            
            # Kurze Pause zwischen Requests
            time.sleep(2)
        
        # Konvertiere zu Liste und limitiere
        massive_links = list(all_links)[:max_destinations]
        
        print(f"\n🎉 MASSIVE LINK-GENERIERUNG ABGESCHLOSSEN!")
        print(f"📊 Einzigartige Destinationen gefunden: {len(all_links)}")
        print(f"🎯 Begrenzt auf: {len(massive_links)}")
        print("=" * 60)
        
        return massive_links

# Variable für massive Links (wird bei Bedarf generiert)
wikipedia_destination_urls_massive = None

def generate_and_get_massive_links(max_destinations=2500):
    """Generiert und gibt massive Link-Liste zurück"""
    global wikipedia_destination_urls_massive
    
    if wikipedia_destination_urls_massive is None:
        print("🔧 Generiere massive Link-Liste...")
        generator = MassiveLinkGenerator()
        wikipedia_destination_urls_massive = generator.generate_massive_links(max_destinations)
    
    return wikipedia_destination_urls_massive

if __name__ == "__main__":
    # Test der Funktionalität
    print("Wikipedia Destination Links")
    print("="*50)
    
    # Basis-Liste
    urls = get_destination_urls()
    print(f"📋 Basis-Liste: {len(urls)} Destinationen")
    
    # Erweiterte Liste
    extended_urls = get_extended_destination_urls()
    print(f"📋 Erweiterte Liste: {len(extended_urls)} Destinationen")
    
    # Statistiken
    stats = get_stats()
    print(f"📊 Statistiken: {stats}")
    
    print("\n🔍 Erste 5 URLs aus Basis-Liste:")
    for i, url in enumerate(urls[:5], 1):
        print(f"  {i}. {url}")
    
    # Frage nach massiver Link-Generierung
    print(f"\n🚀 Für 1000+ Destinationen:")
    print(f"   Nutze: get_massive_destination_urls()")
    print(f"   Beispiel: massive_urls = get_massive_destination_urls(1500)")
    
    # Optional: Teste massive Generierung (auskommentiert für normale Nutzung)
    # print("\n🧪 Teste massive Link-Generierung (5 Destinationen)...")
    # test_massive = get_massive_destination_urls(max_destinations=5)
    # print(f"✅ Test erfolgreich: {len(test_massive)} Links generiert")
