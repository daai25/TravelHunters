#!/usr/bin/env python3
"""
Wikipedia Destination Links
Sammlung von Wikipedia-URLs für weltweite Destinationen
"""

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
    """Gibt die erweiterte Liste zurück"""
    return wikipedia_destination_urls_extended

# Statistiken
def get_stats():
    """Gibt Statistiken über die Link-Liste zurück"""
    return {
        "total_destinations": len(wikipedia_destination_urls),
        "regions": {
            "europa": 5,
            "asien": 5, 
            "amerika": 3,
            "afrika_ozeanien": 2
        }
    }

if __name__ == "__main__":
    # Test der Funktionalität
    urls = get_destination_urls()
    stats = get_stats()
    
    print("Wikipedia Destination Links")
    print("="*50)
    print(f"Total destinations: {stats['total_destinations']}")
    print(f"Regions: {stats['regions']}")
    print("\nFirst 5 URLs:")
    for i, url in enumerate(urls[:5], 1):
        print(f"{i}. {url}")
    print("...")
