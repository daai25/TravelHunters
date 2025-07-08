"""
Data Loading Utilities for TravelHunters
Loads hotel data from SQLite database and prepares for ML models
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Optional, Tuple

class HotelDataLoader:
    def __init__(self, db_path: str = None):
        """Initialize data loader with database path"""
        # Set project root
        self.project_root = Path(__file__).parent.parent.parent
        
        if db_path is None:
            # Default path to database
            self.db_path = self.project_root / "database" / "travelhunters.db"
        else:
            self.db_path = Path(db_path)
    
    def load_hotels(self) -> pd.DataFrame:
        """Load hotel data from JSON files or database, fallback to mock data"""
        try:
            # First try to load from SQLite database
            print("✅ Loading hotel data from SQLite database...")
            hotels_df = self._load_from_database()
            if not hotels_df.empty:
                print(f"✅ Loaded {len(hotels_df)} hotels from database")
                return hotels_df
        except Exception as e:
            print(f"❌ Error loading from database: {e}")

        try:
            # Fallback to JSON files (real booking.com data)
            json_paths = [
                self.project_root / "data_acquisition" / "json_final" / "booking_worldwide_enriched.json",
                self.project_root / "data_acquisition" / "json_final" / "booking_worldwide.json"
            ]
            
            for json_path in json_paths:
                if json_path.exists():
                    print(f"✅ Loading hotel data from: {json_path.name}")
                    with open(json_path, 'r', encoding='utf-8') as f:
                        hotel_data = json.load(f)
                    
                    # Convert JSON data to DataFrame with standardized columns
                    df = self._convert_json_to_dataframe(hotel_data)
                    if not df.empty:
                        print(f"✅ Loaded {len(df)} hotels from JSON file")
                        return df
            
            # Fallback to old database structure
            conn = sqlite3.connect(self.db_path)
            query = """
            SELECT 
                hotel_id as id,
                image_url
            FROM hotel
            WHERE hotel_id IS NOT NULL
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if not df.empty:
                print(f"✅ Loaded {len(df)} hotels from old database structure")
                return self._enrich_hotel_data(df)
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
        
        # Final fallback to mock data
        print("🔄 Creating mock data for demonstration...")
        return self._create_mock_hotel_data()
    
    def _load_from_database(self) -> pd.DataFrame:
        """Load hotel data from the booking_worldwide table in SQLite database"""
        # Correct path to database
        db_path = self.project_root / "data_acquisition" / "database" / "travelhunters.db"
        
        if not db_path.exists():
            raise FileNotFoundError(f"Database file not found: {db_path}")
        
        conn = sqlite3.connect(db_path)
        
        # Query the booking_worldwide table
        query = """
        SELECT 
            id,
            name,
            link,
            rating,
            price,
            location,
            description,
            image_url,
            latitude,
            longitude
        FROM booking_worldwide
        WHERE name IS NOT NULL AND price IS NOT NULL
        """
        
        try:
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                return pd.DataFrame()
            
            # Convert data types
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(3.5)
            df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(100)
            df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce').fillna(0)
            df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce').fillna(0)
            
            # Generate missing fields for ML compatibility
            df['review_count'] = np.random.randint(10, 2000, len(df))
            df['distance_to_center'] = np.random.uniform(0.5, 15.0, len(df))
            
            # Extract amenities from description
            df['amenities'] = df['description'].apply(self._extract_amenities_from_text)
            
            # Convert amenities to JSON string format for consistency
            df['amenities'] = df['amenities'].apply(lambda x: json.dumps(x) if isinstance(x, list) else '[]')
            
            return df
            
        except Exception as e:
            conn.close()
            raise e
    
    def _convert_json_to_dataframe(self, hotel_data: List[Dict]) -> pd.DataFrame:
        """Convert JSON hotel data to standardized DataFrame"""
        hotels = []
        
        for i, hotel in enumerate(hotel_data):
            try:
                # Extract basic information
                hotel_id = i + 1  # Use index as ID
                name = hotel.get('name', 'Unknown Hotel')
                location = hotel.get('location', 'Unknown Location')
                
                # Extract and clean price
                price_str = hotel.get('price', '0')
                try:
                    # Remove any non-numeric characters except decimal point
                    price = float(''.join(c for c in str(price_str) if c.isdigit() or c == '.'))
                    if price == 0:
                        price = np.random.uniform(50, 300)  # Random price if missing
                except:
                    price = np.random.uniform(50, 300)
                
                # Extract and clean rating
                rating_str = hotel.get('rating', '0')
                try:
                    rating = float(rating_str)
                    if rating < 1 or rating > 10:
                        rating = np.random.uniform(3.5, 5.0)
                    elif rating > 5:  # Convert 10-point scale to 5-point scale
                        rating = rating / 2
                except:
                    rating = np.random.uniform(3.5, 5.0)
                
                # Generate review count based on rating
                base_reviews = max(10, int(np.random.exponential(200)))
                if rating >= 4.5:
                    review_count = int(base_reviews * np.random.uniform(1.5, 3.0))
                elif rating >= 4.0:
                    review_count = int(base_reviews * np.random.uniform(1.0, 2.0))
                else:
                    review_count = int(base_reviews * np.random.uniform(0.5, 1.5))
                
                # Extract images
                images = hotel.get('images', [])
                image_url = images[0] if images else f"https://example.com/hotel_images/{hotel_id}.jpg"
                
                # Extract description
                description = hotel.get('description', '')
                if not description:
                    description = f"Beautiful hotel in {location} offering comfortable accommodation."
                
                # Generate amenities based on hotel name and description
                amenities = self._extract_amenities_from_text(name + ' ' + description)
                
                # Calculate distance to center (mock data)
                distance_to_center = round(np.random.exponential(3) + 0.5, 1)
                
                # Generate coordinates (mock data based on location)
                latitude, longitude = self._generate_coordinates_for_location(location)
                
                hotels.append({
                    'id': hotel_id,
                    'name': name,
                    'location': location,
                    'price': round(price, 0),
                    'rating': round(rating, 1),
                    'review_count': review_count,
                    'image_url': image_url,
                    'amenities': json.dumps(amenities),
                    'description': description,
                    'latitude': latitude,
                    'longitude': longitude,
                    'distance_to_center': distance_to_center
                })
                
            except Exception as e:
                print(f"⚠️ Error processing hotel {i}: {str(e)}")
                continue
        
        if hotels:
            return pd.DataFrame(hotels)
        else:
            return pd.DataFrame()
    
    def _extract_amenities_from_text(self, text: str) -> List[str]:
        """Extract amenities from hotel name and description"""
        text = text.lower()
        amenities = []
        
        # Basic amenities
        amenities.append('wifi')  # Assume all hotels have wifi
        
        # Check for specific amenities
        if any(word in text for word in ['pool', 'swimming', 'spa']):
            amenities.append('pool')
        if any(word in text for word in ['gym', 'fitness', 'exercise']):
            amenities.append('gym')
        if any(word in text for word in ['parking', 'garage']):
            amenities.append('parking')
        if any(word in text for word in ['breakfast', 'buffet']):
            amenities.append('breakfast')
        if any(word in text for word in ['spa', 'wellness', 'massage']):
            amenities.append('spa')
        if any(word in text for word in ['restaurant', 'dining', 'food']):
            amenities.append('restaurant')
        if any(word in text for word in ['bar', 'lounge', 'drinks']):
            amenities.append('bar')
        if any(word in text for word in ['room service', 'service']):
            amenities.append('room_service')
        if any(word in text for word in ['air conditioning', 'air-conditioning', 'a/c']):
            amenities.append('air_conditioning')
        
        # Add some random amenities based on hotel quality indicators
        if any(word in text for word in ['luxury', 'grand', 'palace', 'resort']):
            amenities.extend(['spa', 'restaurant', 'room_service', 'gym'])
        if any(word in text for word in ['business', 'executive']):
            amenities.extend(['gym', 'restaurant'])
        if any(word in text for word in ['family', 'kids']):
            amenities.extend(['pool', 'restaurant'])
        
        return list(set(amenities))  # Remove duplicates
    
    def _generate_coordinates_for_location(self, location: str) -> Tuple[float, float]:
        """Generate approximate coordinates based on location string"""
        # Simple mapping for major cities (mock coordinates)
        city_coords = {
            'paris': (48.8566, 2.3522),
            'london': (51.5074, -0.1278),
            'new york': (40.7128, -74.0060),
            'tokyo': (35.6762, 139.6503),
            'berlin': (52.5200, 13.4050),
            'rome': (41.9028, 12.4964),
            'barcelona': (41.3851, 2.1734),
            'amsterdam': (52.3676, 4.9041),
            'vienna': (48.2082, 16.3738),
            'prague': (50.0755, 14.4378),
            'zurich': (47.3769, 8.5417),
        }
        
        location_lower = location.lower()
        for city, coords in city_coords.items():
            if city in location_lower:
                # Add some random variation
                lat = coords[0] + np.random.uniform(-0.1, 0.1)
                lon = coords[1] + np.random.uniform(-0.1, 0.1)
                return round(lat, 6), round(lon, 6)
        
        # Default random coordinates if city not found
        return round(np.random.uniform(-60, 70), 6), round(np.random.uniform(-180, 180), 6)
    
    def _create_mock_hotel_data(self) -> pd.DataFrame:
        """Create realistic mock hotel data for demonstration"""
        np.random.seed(42)  # For reproducible results
        
        # Hotel names and locations
        hotel_names = [
            "Grand Palace Hotel", "City Center Inn", "Luxury Resort & Spa", "Budget Traveler Lodge",
            "Business Executive Hotel", "Family Friendly Resort", "Boutique Heritage Hotel", 
            "Eco Green Lodge", "Sunset Beach Resort", "Mountain View Hotel", "Downtown Plaza",
            "Airport Express Hotel", "Romantic Getaway Resort", "Adventure Base Camp", 
            "Cultural Heritage Inn", "Modern Design Hotel", "Riverside Lodge", "Urban Loft Hotel",
            "Wellness Retreat Center", "Historic Castle Hotel", "Beachfront Paradise Resort",
            "City Walk Hotel", "Garden Oasis Resort", "Skyline Tower Hotel", "Cozy Corner Inn",
            "Luxury Peninsula Hotel", "Backpacker Hostel Plus", "Corporate Center Hotel",
            "Spa & Wellness Resort", "Charming Village Inn", "Metropolitan Grand Hotel",
            "Seaside Escape Resort", "Alpine Mountain Lodge", "Cultural District Hotel",
            "Luxury Boutique Suites", "Family Adventure Resort", "Business Park Hotel",
            "Romantic Castle Resort", "Eco Sustainable Lodge", "Urban Chic Hotel"
        ]
        
        locations = [
            "New York, NY", "London, UK", "Paris, France", "Tokyo, Japan", "Sydney, Australia",
            "Berlin, Germany", "Rome, Italy", "Barcelona, Spain", "Amsterdam, Netherlands",
            "Vienna, Austria", "Prague, Czech Republic", "Budapest, Hungary", "Zurich, Switzerland",
            "Copenhagen, Denmark", "Stockholm, Sweden", "Oslo, Norway", "Helsinki, Finland",
            "Dublin, Ireland", "Edinburgh, Scotland", "Brussels, Belgium", "Lisbon, Portugal",
            "Athens, Greece", "Istanbul, Turkey", "Dubai, UAE", "Singapore", "Hong Kong",
            "Bangkok, Thailand", "Bali, Indonesia", "Mumbai, India", "Cairo, Egypt",
            "Cape Town, South Africa", "São Paulo, Brazil", "Buenos Aires, Argentina",
            "Mexico City, Mexico", "Toronto, Canada", "Los Angeles, CA", "Miami, FL",
            "Chicago, IL", "San Francisco, CA", "Las Vegas, NV"
        ]
        
        # Generate hotel data
        n_hotels = min(len(hotel_names), len(locations))
        hotels_data = []
        
        for i in range(n_hotels):
            # Basic info
            hotel_id = i + 1
            name = hotel_names[i]
            location = locations[i]
            
            # Price (realistic distribution)
            if 'luxury' in name.lower() or 'grand' in name.lower() or 'palace' in name.lower():
                price_base = np.random.uniform(200, 500)
            elif 'budget' in name.lower() or 'hostel' in name.lower() or 'lodge' in name.lower():
                price_base = np.random.uniform(30, 100)
            else:
                price_base = np.random.uniform(80, 250)
            
            price = round(price_base, 0)
            
            # Rating (biased towards higher ratings)
            rating = round(np.random.beta(7, 2) * 4 + 1, 1)  # Skewed towards 4-5
            rating = min(5.0, max(1.0, rating))
            
            # Review count (correlated with rating)
            review_base = max(10, int(np.random.exponential(200)))
            if rating >= 4.5:
                review_count = int(review_base * np.random.uniform(1.5, 3.0))
            elif rating >= 4.0:
                review_count = int(review_base * np.random.uniform(1.0, 2.0))
            else:
                review_count = int(review_base * np.random.uniform(0.5, 1.5))
            
            # Distance to center (km)
            distance_to_center = round(np.random.exponential(5) + 0.5, 1)
            
            # Amenities based on hotel type and price
            base_amenities = ["wifi", "reception"]
            if price > 150:
                base_amenities.extend(["pool", "gym", "spa", "restaurant", "room_service"])
            if price > 80:
                base_amenities.extend(["breakfast", "parking", "air_conditioning"])
            if 'family' in name.lower():
                base_amenities.extend(["pool", "playground", "family_rooms"])
            if 'business' in name.lower():
                base_amenities.extend(["wifi", "meeting_rooms", "business_center"])
            if 'spa' in name.lower() or 'wellness' in name.lower():
                base_amenities.extend(["spa", "gym", "pool", "massage"])
            
            amenities = list(set(base_amenities))  # Remove duplicates
            
            # Description based on hotel characteristics
            description_parts = [f"Welcome to {name}, located in the heart of {location}."]
            
            if price > 200:
                description_parts.append("Experience luxury and elegance with premium amenities and exceptional service.")
            elif price < 80:
                description_parts.append("Affordable accommodation with comfortable rooms and essential amenities.")
            else:
                description_parts.append("Modern hotel offering comfortable stays with excellent facilities.")
            
            if 'spa' in name.lower():
                description_parts.append("Relax and rejuvenate at our world-class spa and wellness center.")
            if 'family' in name.lower():
                description_parts.append("Perfect for families with children, featuring family-friendly amenities.")
            if 'business' in name.lower():
                description_parts.append("Ideal for business travelers with meeting facilities and high-speed internet.")
            if distance_to_center <= 2:
                description_parts.append("Conveniently located near city center attractions and shopping.")
            
            description = " ".join(description_parts)
            
            # Coordinates (approximate)
            latitude = round(np.random.uniform(-60, 70), 6)
            longitude = round(np.random.uniform(-180, 180), 6)
            
            hotels_data.append({
                'id': hotel_id,
                'name': name,
                'location': location,
                'price': price,
                'rating': rating,
                'review_count': review_count,
                'image_url': f"https://example.com/hotel_images/{hotel_id}.jpg",
                'amenities': json.dumps(amenities),
                'description': description,
                'latitude': latitude,
                'longitude': longitude,
                'distance_to_center': distance_to_center
            })
        
        df = pd.DataFrame(hotels_data)
        print(f"✅ Created {len(df)} mock hotels for demonstration")
        return df
    
    def _enrich_hotel_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich minimal hotel data with additional mock fields"""
        # Add missing columns if they don't exist
        if 'name' not in df.columns:
            df['name'] = [f"Hotel {i+1}" for i in range(len(df))]
        
        if 'price' not in df.columns:
            df['price'] = np.random.uniform(50, 300, len(df)).round(0)
        
        if 'rating' not in df.columns:
            df['rating'] = (np.random.beta(7, 2, len(df)) * 4 + 1).round(1)
        
        # Add other missing fields similarly...
        return df
    
    def load_user_interactions(self) -> pd.DataFrame:
        """Load or create synthetic user-hotel interactions"""
        # For now, create synthetic data
        # Later, this could come from actual user behavior
        
        hotels_df = self.load_hotels()
        if hotels_df.empty:
            return pd.DataFrame()
        
        # Create synthetic users and interactions
        n_users = 1000
        n_interactions = 5000
        
        user_ids = range(1, n_users + 1)
        
        # Generate random interactions
        interactions = []
        for _ in range(n_interactions):
            user_id = np.random.choice(user_ids)
            hotel_id = np.random.choice(hotels_df['id'].values)
            rating = np.random.choice([3, 4, 5], p=[0.2, 0.3, 0.5])  # Mostly positive
            interactions.append({
                'user_id': user_id,
                'hotel_id': hotel_id,
                'rating': rating,
                'interaction_type': 'rating'
            })
        
        interactions_df = pd.DataFrame(interactions)
        print(f"✅ Generated {len(interactions_df)} synthetic user interactions")
        
        return interactions_df
    
    def get_data_summary(self) -> Dict:
        """Get summary statistics of the data"""
        hotels_df = self.load_hotels()
        interactions_df = self.load_user_interactions()
        
        summary = {
            'n_hotels': len(hotels_df),
            'n_users': len(interactions_df['user_id'].unique()) if not interactions_df.empty else 0,
            'n_interactions': len(interactions_df),
            'avg_rating': hotels_df['rating'].mean() if not hotels_df.empty else 0,
            'price_range': {
                'min': hotels_df['price'].min() if not hotels_df.empty else 0,
                'max': hotels_df['price'].max() if not hotels_df.empty else 0,
                'mean': hotels_df['price'].mean() if not hotels_df.empty else 0
            }
        }
        
        return summary

if __name__ == "__main__":
    # Test the data loader
    loader = HotelDataLoader()
    
    # Load data
    hotels = loader.load_hotels()
    interactions = loader.load_user_interactions()
    summary = loader.get_data_summary()
    
    print("\n📊 Data Summary:")
    print(f"Hotels: {summary['n_hotels']}")
    print(f"Users: {summary['n_users']}")
    print(f"Interactions: {summary['n_interactions']}")
    print(f"Avg Rating: {summary['avg_rating']:.2f}")
    print(f"Price Range: ${summary['price_range']['min']:.0f} - ${summary['price_range']['max']:.0f}")
    
    if not hotels.empty:
        print(f"\n🏨 Hotels Preview:")
        print(hotels[['name', 'location', 'price', 'rating']].head())
