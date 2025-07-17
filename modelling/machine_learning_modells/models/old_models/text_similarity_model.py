"""
Advanced Text-Based Hotel Recommender using Enhanced NLP and Similarity Matching
Recommends hotels based on sophisticated text similarity analysis with multi-language support
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import re
from typing import List, Dict, Tuple, Optional
import joblib
import warnings
warnings.filterwarnings('ignore')

class TextBasedRecommender:
    """Advanced text-based hotel recommender with enhanced NLP capabilities and semantic understanding"""
    
    def enhanced_tokenizer(self, text):
        """Verbesserte Tokenisierung mit Lemmatization und Synonym-Erweiterung (ohne multiprocessing)"""
        # Fallback auf einfache Tokenisierung, falls spaCy nicht verfügbar ist
        try:
            doc = self.nlp(text.lower().strip())
            tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha and len(token) > 1]
        except Exception:
            tokens = re.sub(r'[^a-z0-9äöüßáéíóúñç ]', '', text.lower()).split()
        # Synonym-Erweiterung (kleines Mapping, kann erweitert werden)
        synonym_map = {
            'pool': ['swimmingpool', 'schwimmbad'],
            'beach': ['strand', 'sea', 'ocean'],
            'family': ['kids', 'children', 'familie', 'kinder'],
            'spa': ['wellness', 'relaxation'],
            'city': ['stadt', 'downtown', 'zentrum'],
            'luxury': ['premium', 'highend', 'deluxe', 'luxus'],
            'cheap': ['budget', 'affordable', 'günstig'],
            'restaurant': ['dining', 'essen'],
            'view': ['aussicht', 'blick'],
        }
        expanded = []
        for t in tokens:
            expanded.append(t)
            for key, syns in synonym_map.items():
                if t == key or t in syns:
                    expanded.extend([key] + syns)
        return list(set(expanded))
    
    def __init__(self, max_features: int = 2000, use_lsa: bool = True, lsa_components: int = 150, 
                 enable_clustering: bool = True, debug_mode: bool = False):
        """
        Initialize the advanced text-based recommender
        
        Args:
            max_features: Maximum number of TF-IDF features
            use_lsa: Whether to use Latent Semantic Analysis (SVD)
            lsa_components: Number of LSA components
            enable_clustering: Whether to enable hotel clustering for better recommendations
            debug_mode: Whether to show debug information
        """
        self.max_features = max_features
        self.use_lsa = use_lsa
        self.lsa_components = lsa_components
        self.enable_clustering = enable_clustering
        self.debug_mode = debug_mode

        # spaCy-Modell nur einmal pro Instanz laden
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except Exception as e:
            print(f"⚠️ spaCy konnte nicht geladen werden: {e}")
            self.nlp = None

        # Enhanced TF-IDF vectorizer with advanced settings
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=8000,  # Mehr Features für bessere Unterscheidung
            stop_words='english',
            ngram_range=(1, 5),  # Bis 5-grams für bessere Kontext-Erfassung
            min_df=2,  # Seltene Terme raus
            max_df=0.90,  # Sehr häufige Terme raus
            lowercase=True,
            strip_accents='unicode',
            analyzer='word',
            tokenizer=self.enhanced_tokenizer,
            norm='l2',
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True
        )
        
        # Enhanced LSA for better semantic understanding
        self.lsa_model = TruncatedSVD(
            n_components=lsa_components, 
            algorithm='randomized', 
            random_state=42, 
            n_iter=10  # More iterations for better convergence
        ) if use_lsa else None
        
        # Hotel clustering for diversity
        self.hotel_clusterer = KMeans(
            n_clusters=min(20, max_features // 100),
            random_state=42,
            n_init=10,
            algorithm='lloyd'  # Kein multiprocessing
        ) if enable_clustering else None
        
        # Storage for processed data and models
        self.hotel_texts = []
        self.hotel_ids = []
        self.hotel_clusters = None
        self.tfidf_matrix = None
        self.lsa_matrix = None
        self.is_fitted = False
        self.query_cache = {}  # Cache for frequently used queries
        
    def prepare_hotel_texts(self, hotels_df: pd.DataFrame) -> List[str]:
        """
        Prepare text representations of hotels
        
        Args:
            hotels_df: Hotels dataframe with text features
            
        Returns:
            List of text representations
        """
        hotel_texts = []
        
        for idx, hotel in hotels_df.iterrows():
            text_parts = []
            
            # Hotel name (high importance)
            if pd.notna(hotel['name']):
                name = str(hotel['name']).strip()
                text_parts.extend([name] * 5)  # Erhöhte Wiederholung für mehr Bedeutung
            
            # Location (wichtig für geografische Suchen)
            if pd.notna(hotel['location']):
                location = str(hotel['location']).strip()
                text_parts.extend([location] * 3)  # Mehr Gewicht für den Standort
                
                # Extrahiere Land oder Stadt aus dem Standort
                location_parts = location.split(',')
                if len(location_parts) > 1:
                    # Füge Stadt und Land getrennt hinzu für besseres Matching
                    for part in location_parts:
                        part = part.strip()
                        if part:
                            text_parts.extend([part] * 2)  # Wiederhole für wichtige Orte
                
                # Füge Schlüsselwörter für Strandhotels hinzu
                location_lower = location.lower()
                if any(beach_term in location_lower for beach_term in ['beach', 'strand', 'sea', 'meer', 'ocean', 'coast', 'küste', 'island', 'insel']):
                    text_parts.extend(['beach hotel', 'beach resort', 'sea view', 'oceanfront', 'strand hotel'] * 2)
                
                # Füge Schlüsselwörter für Stadthotels hinzu
                if any(city_term in location_lower for city_term in ['city', 'stadt', 'downtown', 'zentrum', 'central', 'metropolitan']):
                    text_parts.extend(['city hotel', 'downtown hotel', 'central location', 'urban hotel'] * 2)
            
            # Description (most important)
            if pd.notna(hotel['description']):
                description = str(hotel['description']).strip()
                # Bereinige und erweitere die Beschreibung
                description = self._clean_text(description)
                text_parts.extend([description] * 3)  # Mehr Wiederholungen für Emphasis
                
                # Extrahiere spezifische Merkmale und füge sie hinzu
                family_features = self._extract_family_features(description)
                if family_features:
                    text_parts.extend([family_features] * 3)  # Stärker gewichten
                    
                relaxation_features = self._extract_relaxation_features(description)
                if relaxation_features:
                    text_parts.extend([relaxation_features] * 3)  # Stärker gewichten
                    
                # Extrahiere Aktivitäten und Annehmlichkeiten
                activity_features = self._extract_activity_features(description)
                if activity_features:
                    text_parts.extend([activity_features] * 2)
                    
                # Extrahiere Merkmale für Geschäftsreisen
                business_features = self._extract_business_features(description)
                if business_features:
                    text_parts.extend([business_features] * 2)
                
                # Suche nach wichtigen Schlüsselwörtern und verstärke sie
                key_features = []
                for feature in ['pool', 'swimming pool', 'spa', 'beach', 'kids club', 'family', 'wifi', 
                               'restaurant', 'breakfast', 'luxury', 'suite', 'view', 'central', 'resort',
                               'schwimmbad', 'strand', 'familie', 'kinder', 'frühstück', 'luxus', 'aussicht']:
                    if feature in description.lower():
                        key_features.append(feature)
                
                if key_features:
                    text_parts.extend([' '.join(key_features)] * 3)
            
            # Amenities (wichtig für Filterfunktionen)
            if pd.notna(hotel.get('amenities')):
                amenities = self._process_amenities_text(hotel['amenities'])
                text_parts.extend([amenities] * 3)  # Mehr Gewicht für Annehmlichkeiten
                
                # Extrahiere und verstärke wichtige Annehmlichkeiten
                amenities_lower = amenities.lower()
                amenity_highlights = []
                
                for amenity in ['wifi', 'swimming pool', 'spa', 'breakfast', 'restaurant', 'bar', 'gym', 
                               'parking', 'air conditioning', 'beach', 'children', 'family']:
                    if amenity in amenities_lower:
                        amenity_highlights.append(amenity)
                
                if amenity_highlights:
                    text_parts.extend([' '.join(amenity_highlights)] * 3)
            
            # Price category (for better text matching)
            if pd.notna(hotel.get('price')):
                price_category = self._categorize_price(hotel['price'])
                text_parts.append(price_category)
                
                # Explizit Budget/Luxury-Begriffe hinzufügen
                if hotel['price'] <= 100:
                    text_parts.extend(['budget hotel', 'affordable hotel', 'cheap hotel', 'inexpensive', 'low cost'] * 2)
                elif hotel['price'] >= 300:
                    text_parts.extend(['luxury hotel', 'premium hotel', 'high-end hotel', 'exclusive'] * 2)
            
            # Rating category (for text matching of quality terms)
            if pd.notna(hotel.get('rating')):
                rating_category = self._categorize_rating(hotel['rating'])
                text_parts.append(rating_category)
                
                # Explizite Qualitätsbegriffe basierend auf Bewertung
                if hotel['rating'] >= 9.0:
                    text_parts.extend(['excellent hotel', 'top rated', 'best hotel', 'exceptional quality'] * 2)
                elif hotel['rating'] >= 8.0:
                    text_parts.extend(['very good hotel', 'high quality', 'well rated'] * 2)
            
            # Hotel type inference from name
            hotel_type = self._infer_hotel_type(str(hotel['name']) if pd.notna(hotel['name']) else "")
            if hotel_type:
                text_parts.extend([hotel_type] * 2)
            
            # Combine all text parts and ensure we don't have too much repetition
            combined_text = ' '.join(text_parts)
            # Wir begrenzen die Länge des Textes, um zu viel Wiederholung zu vermeiden
            max_length = 5000  # Maximal 5000 Zeichen
            if len(combined_text) > max_length:
                combined_text = combined_text[:max_length]
                
            hotel_texts.append(self._clean_text(combined_text))
        
        return hotel_texts
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S*@\S*\s?', '', text)
        
        # Remove special characters but keep spaces and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\-]', ' ', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _process_amenities_text(self, amenities: str) -> str:
        """Process amenities into clean text"""
        try:
            import json
            # Try parsing as JSON
            if amenities.startswith('[') or amenities.startswith('{'):
                amenity_list = json.loads(amenities)
                if isinstance(amenity_list, list):
                    return ' '.join(amenity_list)
                else:
                    return str(amenity_list)
            else:
                # Clean string representation
                amenities = amenities.replace('[', '').replace(']', '').replace('"', '').replace("'", "")
                return amenities
        except:
            # Fallback to string processing
            amenities = str(amenities).replace('[', '').replace(']', '').replace('"', '').replace("'", "")
            return amenities
    
    def _categorize_price(self, price: float) -> str:
        """Categorize price into text description"""
        if price <= 50:
            return "budget cheap affordable economy low-cost"
        elif price <= 100:
            return "mid-range moderate standard reasonable"
        elif price <= 200:
            return "premium upscale quality higher-end comfort"
        elif price <= 350:
            return "luxury expensive high-end exclusive upmarket premium"
        else:
            return "ultra-luxury deluxe top-tier five-star exclusive luxury premium"
    
    def _categorize_rating(self, rating: float) -> str:
        """Categorize rating into text description"""
        if rating <= 3.5:
            return "basic decent acceptable standard"
        elif rating <= 4.0:
            return "good quality recommended reliable pleasant"
        elif rating <= 4.5:
            return "very good excellent superior high-quality outstanding"
        elif rating <= 4.7:
            return "exceptional superb fantastic premium top-quality excellent"
        else:
            return "outstanding exceptional perfect top-rated world-class luxury"
    
    def _infer_hotel_type(self, name: str) -> str:
        """Infer hotel type from name"""
        name_lower = name.lower()
        
        hotel_types = {
            'resort': 'resort vacation leisure spa relaxation family kids children',
            'inn': 'inn cozy intimate boutique',
            'suite': 'suite apartment extended-stay family kids children spacious',
            'hotel': 'hotel standard accommodation',
            'motel': 'motel budget roadside',
            'hostel': 'hostel budget backpacker social',
            'b&b': 'bed-breakfast intimate personal',
            'lodge': 'lodge rustic nature outdoor',
            'villa': 'villa luxury private exclusive high-end',
            'palace': 'palace luxury elegant exclusive royalty high-end',
            'grand': 'grand luxury elegant exclusive spacious high-end',
            'premium': 'premium luxury high-end exclusive',
            'family': 'family children kids child-friendly playground',
            'spa': 'spa wellness relaxation relaxing retreat',
            'beach': 'beach seaside ocean coastal water relaxing'
        }
        
        result_keywords = []
        
        for keyword, description in hotel_types.items():
            if keyword in name_lower:
                result_keywords.append(description)
        
        # Kombiniere alle gefundenen Schlüsselwörter
        if result_keywords:
            return ' '.join(result_keywords)
            
        # Wenn kein spezifischer Typ erkannt wurde, Standard zurückgeben
        return "hotel accommodation"
    
    def fit(self, hotels_df: pd.DataFrame):
        """
        Fit the text-based model on hotel data
        
        Args:
            hotels_df: Hotels dataframe
        """
        # Debug-Informationen aktivieren
        debug = False  # Reduziere Debug-Ausgaben
        
        try:
            print(f"🔍 Bereite Hoteltexte vor...")
            self.hotel_texts = self.prepare_hotel_texts(hotels_df)
            self.hotel_ids = hotels_df['id'].tolist()
            
            if debug:
                # Analysiere die erstellten Texte
                text_lengths = [len(text) for text in self.hotel_texts]
                avg_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
                print(f"DEBUG: Durchschnittliche Textlänge: {avg_length:.1f} Zeichen")
                print(f"DEBUG: Kürzester Text: {min(text_lengths)} Zeichen")
                print(f"DEBUG: Längster Text: {max(text_lengths)} Zeichen")
                
                # Zeige ein Beispiel an
                if self.hotel_texts:
                    sample_idx = min(5, len(self.hotel_texts) - 1)
                    sample_text = self.hotel_texts[sample_idx][:500] + "..." if len(self.hotel_texts[sample_idx]) > 500 else self.hotel_texts[sample_idx]
                    print(f"DEBUG: Beispiel für einen vorbereiteten Hoteltext: {sample_text}")
                    
                    # Zeige die Tokens des Beispieltextes
                    sample_tokens = self.tfidf_vectorizer.build_tokenizer()(self.hotel_texts[sample_idx][:200])
                    print(f"DEBUG: Beispiel für Tokens: {sample_tokens[:20]}...")
            
            print(f"🔍 Passe TF-IDF Vektorisierer an...")
            # Fit TF-IDF vectorizer
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.hotel_texts)
            
            if debug:
                # TF-IDF-Matrix analysieren
                print(f"DEBUG: TF-IDF Matrix Form: {self.tfidf_matrix.shape}")
                print(f"DEBUG: TF-IDF Matrix Dichte: {self.tfidf_matrix.nnz / (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1]):.4f}")
                
                # Zeige die häufigsten Terme im Vokabular
                feature_names = self.tfidf_vectorizer.get_feature_names_out()
                print(f"DEBUG: Vokabulargröße: {len(feature_names)}")
                
                # Analysiere die 20 wichtigsten Terme (mit höchster IDF)
                if hasattr(self.tfidf_vectorizer, 'idf_'):
                    idf_values = self.tfidf_vectorizer.idf_
                    word_idf = list(zip(feature_names, idf_values))
                    top_words = sorted(word_idf, key=lambda x: x[1], reverse=True)[:20]
                    print(f"DEBUG: Top 20 Terme nach IDF: {top_words}")
            
            # Apply LSA if enabled
            if self.use_lsa and self.lsa_model:
                print(f"🔍 Wende LSA-Transformation an...")
                try:
                    # Überprüfen, ob die TF-IDF-Matrix nicht leer ist
                    if self.tfidf_matrix.nnz == 0:
                        print(f"⚠️ WARNUNG: TF-IDF-Matrix ist leer! LSA wird nicht angewendet.")
                        self.use_lsa = False
                    else:
                        self.lsa_matrix = self.lsa_model.fit_transform(self.tfidf_matrix)
                        
                        if debug:
                            # LSA-Matrix analysieren
                            print(f"DEBUG: LSA Matrix Form: {self.lsa_matrix.shape}")
                            print(f"DEBUG: Erklärte Varianz: {sum(self.lsa_model.explained_variance_ratio_):.4f}")
                            
                            # Zeige die 5 wichtigsten Komponenten
                            components = self.lsa_model.components_
                            for i, component in enumerate(components[:5]):
                                top_features_idx = component.argsort()[-10:]
                                top_features = [feature_names[idx] for idx in top_features_idx]
                                print(f"DEBUG: Top-Features für LSA-Komponente {i}: {top_features}")
                except Exception as e:
                    print(f"⚠️ Fehler bei LSA-Transformation: {e}")
                    self.use_lsa = False
                    # Fallback auf TF-IDF ohne LSA
                    self.lsa_matrix = None
            
            self.is_fitted = True
            
            print(f"✅ Text model fitted successfully!")
            print(f"Hotels processed: {len(self.hotel_texts)}")
            print(f"TF-IDF features: {self.tfidf_matrix.shape[1]}")
            if self.use_lsa:
                print(f"LSA components: {self.lsa_matrix.shape[1]}")
            else:
                print(f"LSA nicht verwendet - verwende direkt TF-IDF-Ähnlichkeit")
                
        except Exception as e:
            print(f"❌ Fehler beim Anpassen des Text-Models: {e}")
            raise
    
    def search_hotels(self, query: str, top_k: int = 10, similarity_threshold: float = 0.01) -> pd.DataFrame:
        """
        Search hotels based on text query
        
        Args:
            query: User search query
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            DataFrame with hotel recommendations and similarity scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before searching")
        
        # Debug-Ausgabe aktivieren
        debug = False  # Reduziere Debug-Ausgaben
        
        # Clean, enhance with synonyms, and vectorize query
        cleaned_query = self._clean_text(query)
        enhanced_query = self._enhance_query(cleaned_query)
        
        if debug:
            print(f"DEBUG: Original query: '{query}'")
            print(f"DEBUG: Cleaned query: '{cleaned_query}'")
            print(f"DEBUG: Enhanced query: '{enhanced_query}'")
        
        # Überprüfen, ob die Vektorisierung funktioniert
        try:
            # Finde die Top-Begriffe im Vokabular, die zur Abfrage passen
            query_vector = self.tfidf_vectorizer.transform([enhanced_query])
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # Feature-Namen und ihre TF-IDF-Werte für die Abfrage drucken
            if debug:
                query_terms = query_vector.toarray()[0]
                query_terms_dict = {feature_names[i]: query_terms[i] for i in query_terms.nonzero()[0]}
                sorted_terms = sorted(query_terms_dict.items(), key=lambda x: x[1], reverse=True)
                print(f"DEBUG: Top query terms: {sorted_terms[:10]}")
            
            # Use LSA if available
            if self.use_lsa and self.lsa_model:
                query_lsa = self.lsa_model.transform(query_vector)
                similarity_scores = cosine_similarity(query_lsa, self.lsa_matrix).flatten()
                if debug:
                    print(f"DEBUG: Using LSA similarity. Max score: {max(similarity_scores):.4f}")
            else:
                similarity_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                if debug:
                    print(f"DEBUG: Using TF-IDF similarity. Max score: {max(similarity_scores):.4f}")
            
            # Create results dataframe
            results_df = pd.DataFrame({
                'hotel_id': self.hotel_ids,
                'similarity_score': similarity_scores,
                'hotel_text': self.hotel_texts
            })
            
            # Drastisch reduzierte Schwelle verwenden - wir wollen mehr Ergebnisse bekommen
            # und später nach Gesamtscore sortieren
            min_threshold = 0.001  # Sehr niedrige Schwelle
            
            # Speichere die 50 besten Ergebnisse, unabhängig von der Schwelle
            top_results = results_df.sort_values('similarity_score', ascending=False).head(max(50, top_k * 5))
            
            # Wenn es noch immer Hotels mit Ähnlichkeitswert > 0 gibt, filtere damit
            positive_scores = top_results[top_results['similarity_score'] > 0]
            if len(positive_scores) >= max(3, top_k // 2):
                filtered_results = positive_scores
            else:
                # Sonst verwende die Top-50-Ergebnisse, ungeachtet des Scores
                filtered_results = top_results
            
            if debug:
                print(f"DEBUG: Nach Ähnlichkeitsfilterung: {len(filtered_results)} Hotels")
                if not filtered_results.empty:
                    print(f"DEBUG: Max similarity: {filtered_results['similarity_score'].max():.4f}")
                    print(f"DEBUG: Min similarity: {filtered_results['similarity_score'].min():.4f}")
                    print(f"DEBUG: Avg similarity: {filtered_results['similarity_score'].mean():.4f}")
            
            # Entferne Duplikate nach hotel_id (behalte höchste Bewertung)
            results_df = filtered_results.sort_values('similarity_score', ascending=False).drop_duplicates(subset=['hotel_id'], keep='first')
            
            # Sortieren nach Ähnlichkeitswert
            results_df = results_df.sort_values('similarity_score', ascending=False)
            
            # Mehr Ergebnisse zurückgeben, um nachfolgende Filterung zu ermöglichen
            return results_df.head(max(top_k * 3, 50))
            
        except Exception as e:
            print(f"ERROR in search_hotels: {str(e)}")
            # Im Fehlerfall ein leeres DataFrame zurückgeben
            return pd.DataFrame(columns=['hotel_id', 'similarity_score', 'hotel_text'])
    
    def recommend_hotels(self, query: str, hotels_df: pd.DataFrame, user_preferences: Dict = None, 
                        top_k: int = 10) -> pd.DataFrame:
        """
        Recommend hotels based on text query and preferences
        
        Args:
            query: User search query
            hotels_df: Original hotels dataframe
            user_preferences: Additional preference filters
            top_k: Number of recommendations
            
        Returns:
            DataFrame with detailed hotel recommendations
        """
        # Verbessere die Such-Pipeline für robustere Ergebnisse
        original_top_k = top_k
        dynamic_top_k = top_k * 3  # Mehr Hotels suchen, um mehr Filter zu überstehen
        min_result_count = max(3, top_k // 2)  # Mindestanzahl an zurückzugebenden Hotels
        
        # First apply user preference filters to hotels_df before text search
        filtered_hotels_df = hotels_df.copy()
        strict_filter = True  # Anfangs strenge Filter anwenden
        
        # Erste Filterrunde mit strengen Kriterien
        if user_preferences:
            # Mindestens 30 Hotels nach Filter behalten, sonst Filter lockern
            min_hotels_after_filter = 30
            
            # Apply price filter
            if 'max_price' in user_preferences:
                max_price = user_preferences['max_price']
                filtered_hotels_df = filtered_hotels_df[filtered_hotels_df['price'] <= max_price]
                print(f"🔍 Nach Preisfilter (≤{max_price}): {len(filtered_hotels_df)} Hotels")
                
                # Wenn zu wenige Hotels übrig sind, Filter lockern
                if len(filtered_hotels_df) < min_hotels_after_filter:
                    relaxed_max_price = max_price * 1.2  # 20% höheres Budget
                    filtered_hotels_df = hotels_df[hotels_df['price'] <= relaxed_max_price]
                    strict_filter = False
                    print(f"🔄 Filter gelockert: Preisgrenze auf {relaxed_max_price:.2f} erhöht: {len(filtered_hotels_df)} Hotels")
            
            # Apply rating filter
            if 'min_rating' in user_preferences:
                min_rating = user_preferences['min_rating']
                filtered_hotels_df = filtered_hotels_df[filtered_hotels_df['rating'] >= min_rating]
                print(f"🔍 Nach Bewertungsfilter (≥{min_rating}): {len(filtered_hotels_df)} Hotels")
                
                # Wenn zu wenige Hotels übrig sind, Filter lockern
                if len(filtered_hotels_df) < min_hotels_after_filter:
                    relaxed_min_rating = max(0, min_rating - 1)  # 1 Punkt niedriger (min. 0)
                    filtered_hotels_df = filtered_hotels_df[filtered_hotels_df['rating'] >= relaxed_min_rating]
                    strict_filter = False
                    print(f"🔄 Filter gelockert: Mindestbewertung auf {relaxed_min_rating} gesenkt: {len(filtered_hotels_df)} Hotels")
            
            # Apply amenities as preference-based scoring (not hard filtering)
            # This is handled in the scoring phase, not as a filter
            # if 'preferred_amenities' in user_preferences and user_preferences['preferred_amenities']:
            #     # Amenities are now handled as preferences in scoring, not as hard filters
            #     pass
        
        # Wenn keine Hotels mehr übrig sind, verwende Original-Dataframe mit Preis- und Rating-Filter
        if filtered_hotels_df.empty or len(filtered_hotels_df) < 5:
            print("⚠️ Zu wenige Hotels mit strengen Filtern. Verwende gelockerte Filter...")
            filtered_hotels_df = hotels_df.copy()
            strict_filter = False
            
            # Grundlegende Preis- und Rating-Filter
            if user_preferences:
                if 'max_price' in user_preferences:
                    relaxed_max_price = user_preferences['max_price'] * 1.5  # 50% höheres Budget
                    filtered_hotels_df = filtered_hotels_df[filtered_hotels_df['price'] <= relaxed_max_price]
                
                if 'min_rating' in user_preferences:
                    relaxed_min_rating = max(0, user_preferences['min_rating'] - 2)  # 2 Punkte niedriger
                    filtered_hotels_df = filtered_hotels_df[filtered_hotels_df['rating'] >= relaxed_min_rating]
        
        if filtered_hotels_df.empty:
            print("❌ Keine Hotels entsprechen den Filterkriterien, selbst mit gelockerten Filtern")
            return pd.DataFrame()
        
        # Erstelle eine temporäre Zuordnung für gefilterte Hotels
        filtered_hotel_ids = set(filtered_hotels_df['id'].tolist())
        
        # Textbasierte Ähnlichkeiten nur für gefilterte Hotels abrufen
        # Erhöhe Top-K, um genügend Hotels für nachfolgende Filter zu haben
        text_results = self.search_hotels(query, top_k=min(len(filtered_hotels_df), dynamic_top_k), 
                                          similarity_threshold=0.05 if strict_filter else 0.01)
        
        # Filtere Textergebnisse, um nur Hotels einzuschließen, die die Präferenzfilter bestanden haben
        text_results = text_results[text_results['hotel_id'].isin(filtered_hotel_ids)]
        
        if len(text_results) < min_result_count and len(filtered_hotels_df) > min_result_count:
            print(f"⚠️ Zu wenige relevante Text-Ergebnisse ({len(text_results)}). Erhöhe Ergebnisumfang...")
            
            # Reduziere die Ähnlichkeitsschwelle für mehr Ergebnisse
            text_results = self.search_hotels(query, top_k=min(len(filtered_hotels_df), dynamic_top_k * 2),
                                             similarity_threshold=0.01)
            text_results = text_results[text_results['hotel_id'].isin(filtered_hotel_ids)]
        
        # Mit Hotel-Details zusammenführen
        detailed_results = text_results.merge(
            filtered_hotels_df[['id', 'name', 'location', 'price', 'rating', 'description', 'amenities']], 
            left_on='hotel_id', 
            right_on='id', 
            how='inner'
        )
        
        # Duplikate nach Hotelnamen und Standort entfernen (das gleiche Hotel mit verschiedenen IDs)
        detailed_results = detailed_results.drop_duplicates(subset=['name', 'location'], keep='first')
        
        # Zusätzlich Duplikate nach Hotel-ID entfernen
        detailed_results = detailed_results.drop_duplicates(subset=['hotel_id'], keep='first')
        
        print(f"🔍 Nach Duplikatentfernung: {len(detailed_results)} einzigartige Hotels")
        
        # Wenn immer noch zu wenige Hotels, erweitere die Suche
        if len(detailed_results) < min_result_count and not filtered_hotels_df.empty:
            print(f"⚠️ Zu wenige relevante Ergebnisse ({len(detailed_results)}). Erweitere Suche...")
            
            # Nehme die Top-Hotels basierend auf Rating und Preis
            backup_hotels = filtered_hotels_df.sort_values('rating', ascending=False).head(top_k)
            
            # Füge einen Hinweis hinzu, dass diese nicht text-basiert sind
            backup_hotels['similarity_score'] = 0.01
            backup_hotels['hotel_text'] = ""
            backup_hotels['hotel_id'] = backup_hotels['id']
            
            # Füge sie zu den Ergebnissen hinzu und entferne Duplikate
            detailed_results = pd.concat([detailed_results, backup_hotels])
            detailed_results = detailed_results.drop_duplicates(subset=['id'], keep='first')
            detailed_results = detailed_results.drop_duplicates(subset=['name', 'location'], keep='first')
        
        # Präferenz-Gewichtung anwenden
        if user_preferences:
            detailed_results = self._apply_text_weighting(detailed_results, user_preferences)
        else:
            detailed_results['final_score'] = detailed_results['similarity_score']
        
        # Sortieren und Top-K zurückgeben
        final_results = detailed_results.sort_values('final_score', ascending=False).head(original_top_k)
        
        print(f"✅ {len(final_results)} Empfehlungen für '{query}' mit Ihren Filtern")
        
        # Detaillierte Ergebnisse zurückgeben
        result_columns = ['hotel_id', 'name', 'final_score', 'similarity_score', 
                         'price', 'rating', 'location', 'description', 'amenities']
        
        # Nur vorhandene Spalten zurückgeben
        available_columns = [col for col in result_columns if col in final_results.columns]
        return final_results[available_columns]
    
    def _apply_text_filters(self, results_df: pd.DataFrame, preferences: Dict) -> pd.DataFrame:
        """Apply preference filters to text search results"""
        filtered_df = results_df.copy()
        
        # Price filter
        if 'max_price' in preferences:
            filtered_df = filtered_df[filtered_df['price'] <= preferences['max_price']]
        
        # Rating filter
        if 'min_rating' in preferences:
            filtered_df = filtered_df[filtered_df['rating'] >= preferences['min_rating']]
        
        return filtered_df
    
    def _apply_text_weighting(self, results_df: pd.DataFrame, preferences: Dict) -> pd.DataFrame:
        """Apply preference weighting to combine text similarity with other factors"""
        weighted_df = results_df.copy()
        
        # Debug-Ausgabe aktivieren
        debug = False  # Reduziere Debug-Ausgaben
        
        try:
            # Default weights
            text_weight = preferences.get('text_importance', 0.6)
            price_weight = preferences.get('price_importance', 0.2)
            rating_weight = preferences.get('rating_importance', 0.2)
            
            if debug:
                print(f"DEBUG: Angewandte Gewichte - Text: {text_weight:.2f}, Preis: {price_weight:.2f}, Bewertung: {rating_weight:.2f}")
                print(f"DEBUG: Originale Similarity Scores - Min: {weighted_df['similarity_score'].min():.6f}, Max: {weighted_df['similarity_score'].max():.6f}")
            
            # Normalize scores
            if len(weighted_df) > 1:
                # Textähnlichkeitswerte anpassen - auch wenn sie alle 0 sind, wir erstellen eine relative Rangfolge
                similarity_scores = weighted_df['similarity_score'].values
                min_sim = similarity_scores.min()
                max_sim = similarity_scores.max()
                
                # Wenn alle Ähnlichkeitswerte gleich sind (z.B. alle 0), erstellen wir künstliche Werte
                # Damit die Sortierung nach anderen Faktoren erfolgen kann
                if abs(max_sim - min_sim) < 0.00001:
                    if debug:
                        print(f"DEBUG: Alle Ähnlichkeitswerte sind nahezu identisch ({min_sim:.6f}). Erstelle künstliche Werte.")
                    
                    # Erstelle künstliche Textähnlichkeitswerte basierend auf Rating und Preis
                    # Da wir keine echte Textübereinstimmung haben, nehmen wir an, dass höher bewertete Hotels besser sind
                    normalized_ratings = (weighted_df['rating'] - weighted_df['rating'].min()) / (weighted_df['rating'].max() - weighted_df['rating'].min() + 0.001)
                    normalized_prices = 1 - (weighted_df['price'] - weighted_df['price'].min()) / (weighted_df['price'].max() - weighted_df['price'].min() + 0.001)
                    
                    # Künstliche Werte zwischen 0.01 und 0.2 (niedrig, aber nicht 0)
                    weighted_df['text_score'] = 0.01 + (normalized_ratings * 0.6 + normalized_prices * 0.4) * 0.19
                else:
                    # Min-Max-Normalisierung der Ähnlichkeitswerte
                    weighted_df['text_score'] = (weighted_df['similarity_score'] - min_sim) / (max_sim - min_sim)
                    
                    # Wenn die höchste Ähnlichkeit immer noch niedrig ist, verstärken wir die Unterschiede
                    if max_sim < 0.1:
                        # Wende Potenzierung an, um kleine Unterschiede zu verstärken
                        weighted_df['text_score'] = weighted_df['text_score'].apply(lambda x: x ** 0.5)
                    
                    # Sigmoid-Funktion mit angepasstem Schwellenwert für niedrige Werte
                    weighted_df['text_score'] = weighted_df['text_score'].apply(
                        lambda x: 1 / (1 + np.exp(-12 * (x - 0.5))) if x > 0 else 0.01
                    )
                
                if debug:
                    print(f"DEBUG: Normalisierte Text Scores - Min: {weighted_df['text_score'].min():.4f}, Max: {weighted_df['text_score'].max():.4f}")
                
                # Price score (lower price = higher score) with log scale for better distribution
                max_price = weighted_df['price'].max()
                min_price = weighted_df['price'].min()
                price_range = max_price - min_price
                
                if price_range > 0:
                    # Logarithmische Transformation für bessere Verteilung
                    weighted_df['price_score'] = weighted_df['price'].apply(
                        lambda p: 1 - np.log1p(p - min_price) / np.log1p(price_range) if p > min_price else 1.0
                    )
                else:
                    weighted_df['price_score'] = 1.0
                
                # Rating score (10-point scale) mit verstärkter Kurve
                weighted_df['rating_score'] = weighted_df['rating'] / 10.0
                
                # Verstärkte Gewichtung für Ratings mit Potenzfunktion
                # Bewertungen > 8 werden stärker gewichtet, < 6 werden abgewertet
                weighted_df['rating_score'] = weighted_df['rating_score'].apply(
                    lambda r: r ** 0.7 if r > 0.8 else (r ** 1.3 if r < 0.6 else r)
                )
                
                # Bonus für sehr gute Bewertungen
                weighted_df['rating_score'] = weighted_df['rating_score'].apply(
                    lambda r: r * 1.25 if r > 0.85 else r
                )
                
                if debug:
                    print(f"DEBUG: Price Scores - Min: {weighted_df['price_score'].min():.4f}, Max: {weighted_df['price_score'].max():.4f}")
                    print(f"DEBUG: Rating Scores - Min: {weighted_df['rating_score'].min():.4f}, Max: {weighted_df['rating_score'].max():.4f}")
            else:
                # Für einzelne Hotels einfache Normalisierung verwenden
                weighted_df['text_score'] = max(weighted_df['similarity_score'].iloc[0], 0.1)  # Mindestens 0.1
                weighted_df['price_score'] = 1.0
                weighted_df['rating_score'] = weighted_df['rating'] / 10.0
            
            # Gewichtete Gesamtpunktzahl berechnen
            weighted_df['final_score'] = (
                text_weight * weighted_df['text_score'] +
                price_weight * weighted_df['price_score'] +
                rating_weight * weighted_df['rating_score']
            )
            
            # Amenity scoring - add bonus for preferred amenities
            if 'preferred_amenities' in preferences and preferences['preferred_amenities']:
                amenity_scores = self._calculate_amenity_score(weighted_df, preferences)
                amenity_weight = preferences.get('amenity_importance', 0.1)
                weighted_df['amenity_score'] = amenity_scores
                weighted_df['final_score'] = (
                    (text_weight * weighted_df['text_score'] +
                     price_weight * weighted_df['price_score'] +
                     rating_weight * weighted_df['rating_score']) * (1 - amenity_weight) +
                    amenity_weight * weighted_df['amenity_score']
                )
                if debug:
                    print(f"DEBUG: Amenity Scores - Min: {weighted_df['amenity_score'].min():.4f}, Max: {weighted_df['amenity_score'].max():.4f}")
            else:
                weighted_df['amenity_score'] = 0.5  # Neutral score when no amenities specified
            
            # Boni für besondere Merkmale
            # 1. Bonus für Hotels mit sehr hoher Textähnlichkeit (falls vorhanden)
            high_similarity_mask = weighted_df['similarity_score'] > 0.1
            if high_similarity_mask.any():
                weighted_df.loc[high_similarity_mask, 'final_score'] *= 1.2
                if debug:
                    print(f"DEBUG: Bonus für hohe Textähnlichkeit angewendet auf {high_similarity_mask.sum()} Hotels")
            
            # 2. Bonus für außergewöhnlich gute Bewertungen
            excellent_rating_mask = weighted_df['rating'] >= 9.0
            if excellent_rating_mask.any():
                weighted_df.loc[excellent_rating_mask, 'final_score'] *= 1.15
                if debug:
                    print(f"DEBUG: Bonus für exzellente Bewertungen angewendet auf {excellent_rating_mask.sum()} Hotels")
            
            # 3. Bonus für sehr gutes Preis-Leistungs-Verhältnis
            # Definiere gutes Preis-Leistungs-Verhältnis als hohe Bewertung bei moderatem Preis
            good_value_mask = (weighted_df['rating'] >= 8.0) & (weighted_df['price'] <= preferences.get('max_price', 1000) * 0.6)
            if good_value_mask.any():
                weighted_df.loc[good_value_mask, 'final_score'] *= 1.1
                if debug:
                    print(f"DEBUG: Bonus für gutes Preis-Leistungs-Verhältnis angewendet auf {good_value_mask.sum()} Hotels")
            
            # Füge Erklärbarkeit hinzu
            weighted_df['text_contribution'] = text_weight * weighted_df['text_score'] * (1 - preferences.get('amenity_importance', 0.0))
            weighted_df['price_contribution'] = price_weight * weighted_df['price_score'] * (1 - preferences.get('amenity_importance', 0.0))
            weighted_df['rating_contribution'] = rating_weight * weighted_df['rating_score'] * (1 - preferences.get('amenity_importance', 0.0))
            weighted_df['amenity_contribution'] = preferences.get('amenity_importance', 0.0) * weighted_df.get('amenity_score', 0.5)
            
            # Analysiere die endgültigen Punktzahlen
            if debug:
                print(f"DEBUG: Final Scores - Min: {weighted_df['final_score'].min():.4f}, Max: {weighted_df['final_score'].max():.4f}")
            
            return weighted_df
            
        except Exception as e:
            print(f"ERROR in _apply_text_weighting: {str(e)}")
            # Im Fehlerfall die ursprünglichen Daten zurückgeben
            weighted_df['final_score'] = weighted_df['similarity_score']
            weighted_df['text_contribution'] = weighted_df['similarity_score']
            weighted_df['price_contribution'] = 0
            weighted_df['rating_contribution'] = 0
            weighted_df['amenity_contribution'] = 0
            return weighted_df
    
    def get_query_keywords(self, query: str, top_k: int = 10) -> List[str]:
        """Extract top keywords from query based on TF-IDF"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        cleaned_query = self._clean_text(query)
        query_vector = self.tfidf_vectorizer.transform([cleaned_query])
        
        # Get feature names and scores
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        scores = query_vector.toarray()[0]
        
        # Get top keywords
        keyword_scores = list(zip(feature_names, scores))
        keyword_scores = [(word, score) for word, score in keyword_scores if score > 0]
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [word for word, score in keyword_scores[:top_k]]
    
    def save_model(self, filepath: str):
        """Save the fitted model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        # Speichere grundlegende Vektorisierungsparameter (ohne nicht-serialisierbare Komponenten)
        vectorizer_params = {
            'max_features': self.tfidf_vectorizer.max_features,
            'stop_words': self.tfidf_vectorizer.stop_words,
            'ngram_range': self.tfidf_vectorizer.ngram_range,
            'min_df': self.tfidf_vectorizer.min_df,
            'max_df': self.tfidf_vectorizer.max_df,
            'lowercase': self.tfidf_vectorizer.lowercase,
            'strip_accents': self.tfidf_vectorizer.strip_accents,
            'analyzer': self.tfidf_vectorizer.analyzer,
            'norm': self.tfidf_vectorizer.norm,
            'use_idf': self.tfidf_vectorizer.use_idf,
            'smooth_idf': self.tfidf_vectorizer.smooth_idf,
            'sublinear_tf': self.tfidf_vectorizer.sublinear_tf
        }
        
        # Speichere Vokabular explizit als Dictionary für bessere Kompatibilität
        if hasattr(self.tfidf_vectorizer, 'vocabulary_'):
            vectorizer_params['vocabulary'] = dict(self.tfidf_vectorizer.vocabulary_)
        
        # Speichere IDF-Werte als Liste für bessere Serialisierbarkeit
        if hasattr(self.tfidf_vectorizer, 'idf_'):
            vectorizer_params['idf_values'] = self.tfidf_vectorizer.idf_.tolist()
        
        # Erstelle ein sauberes Model-Daten-Dictionary ohne problematische Objekte
        model_data = {
            'vectorizer_params': vectorizer_params,
            'hotel_texts': self.hotel_texts,
            'hotel_ids': self.hotel_ids,
            'max_features': self.max_features,
            'use_lsa': self.use_lsa,
            'lsa_components': self.lsa_components,
            'enable_clustering': self.enable_clustering,
            'debug_mode': self.debug_mode,
            'is_fitted': True
        }
        
        # Füge LSA-Modell hinzu, falls vorhanden (aber ohne komplexe Matrix-Objekte)
        if self.use_lsa and self.lsa_model is not None:
            try:
                # Speichere nur die LSA-Komponenten und Parameter
                lsa_params = {
                    'n_components': self.lsa_model.n_components,
                    'algorithm': self.lsa_model.algorithm,
                    'n_iter': self.lsa_model.n_iter,
                    'random_state': self.lsa_model.random_state
                }
                
                # Speichere die Komponenten als Listen für bessere Serialisierbarkeit
                if hasattr(self.lsa_model, 'components_'):
                    lsa_params['components'] = [comp.tolist() for comp in self.lsa_model.components_]
                
                if hasattr(self.lsa_model, 'explained_variance_ratio_'):
                    lsa_params['explained_variance_ratio'] = self.lsa_model.explained_variance_ratio_.tolist()
                
                model_data['lsa_params'] = lsa_params
            except Exception as lsa_err:
                print(f"⚠️ LSA-Modellkomponenten konnten nicht gespeichert werden: {lsa_err}")
                model_data['lsa_params'] = None
        
        try:
            # Versuche das bereinigte Modell zu speichern
            joblib.dump(model_data, filepath)
            print(f"✅ Text model saved to {filepath}")
            return True
        except Exception as e:
            print(f"⚠️ Fehler beim Speichern des Text-Modells: {e}")
            print(f"Details: {str(type(e))}: {str(e)}")
            
            # Fallback: Versuche ein minimales Modell zu speichern
            try:
                minimal_data = {
                    'hotel_texts': self.hotel_texts,
                    'hotel_ids': self.hotel_ids,
                    'max_features': self.max_features,
                    'use_lsa': self.use_lsa,
                    'lsa_components': self.lsa_components,
                    'enable_clustering': self.enable_clustering,
                    'debug_mode': self.debug_mode,
                    'is_fitted': True
                }
                joblib.dump(minimal_data, filepath)
                print(f"✅ Minimales Text-Modell gespeichert nach {filepath}")
                return True
            except Exception as e2:
                print(f"❌ Konnte auch kein minimales Modell speichern: {e2}")
                print(f"Details: {str(type(e2))}: {str(e2)}")
                return False
    
    def load_model(self, filepath: str):
        """Load a fitted model"""
        try:
            # Versuche das Modell zu laden und fange alle Fehler ab
            try:
                model_data = joblib.load(filepath)
                print(f"📂 Modell-Datei erfolgreich geladen: {filepath}")
            except Exception as load_error:
                print(f"⚠️ Fehler beim Laden der Modell-Datei: {str(load_error)}")
                print("⚠️ Initialisiere ein neues Text-Modell...")
                self.is_fitted = False
                return False
            
            # Lade grundlegende Konfigurationsparameter
            self.max_features = model_data.get('max_features', self.max_features)
            self.use_lsa = model_data.get('use_lsa', self.use_lsa)
            self.lsa_components = model_data.get('lsa_components', self.lsa_components)
            self.enable_clustering = model_data.get('enable_clustering', self.enable_clustering)
            self.debug_mode = model_data.get('debug_mode', self.debug_mode)
            
            # Lade Daten
            self.hotel_texts = model_data.get('hotel_texts', [])
            self.hotel_ids = model_data.get('hotel_ids', [])
            self.is_fitted = model_data.get('is_fitted', False)
            
            # Überprüfe, ob wir ein minimales oder vollständiges Modell haben
            minimal_model = 'is_fitted' in model_data and 'vectorizer_params' not in model_data
            if minimal_model:
                print(f"📄 Minimales Text-Modell geladen von {filepath}")
                
                # Neu erstellen des TF-IDF Vektorisierers und Training
                if self.hotel_texts and len(self.hotel_texts) > 0:
                    print("🔄 Führe schnelles Neutraining des TF-IDF-Vektorisierers durch...")
                    # Erstelle den Vectorizer mit den Standardparametern
                    self.tfidf_vectorizer = TfidfVectorizer(
                        max_features=self.max_features,
                        stop_words='english',
                        ngram_range=(1, 4),
                        tokenizer=self.enhanced_tokenizer,  # Benutze die statische Methode
                        min_df=1, max_df=0.95,
                        lowercase=True,
                        strip_accents='unicode',
                        analyzer='word',
                        norm='l2',
                        use_idf=True,
                        smooth_idf=True,
                        sublinear_tf=True
                    )
                    # Training durchführen
                    self._retrain_model_quick()
                
                return True
            
            # Vollständiges Modell laden
            if 'vectorizer_params' in model_data:
                print("📄 Vollständiges Text-Modell gefunden, rekonstruiere TF-IDF-Vektorisierer...")
                # TfidfVectorizer mit denselben Parametern und Vokabular neu erstellen
                params = model_data['vectorizer_params']
                
                try:
                    # Wörterbuch aus gespeichertem Vokabular extrahieren
                    vocabulary = None
                    if 'vocabulary' in params:
                        vocabulary = params['vocabulary']
                    
                    # Erstelle den TfidfVectorizer mit denselben Parametern
                    self.tfidf_vectorizer = TfidfVectorizer(
                        max_features=params.get('max_features', self.max_features),
                        stop_words=params.get('stop_words', 'english'),
                        ngram_range=params.get('ngram_range', (1, 4)),
                        min_df=params.get('min_df', 1),
                        max_df=params.get('max_df', 0.95),
                        lowercase=params.get('lowercase', True),
                        strip_accents=params.get('strip_accents', 'unicode'),
                        analyzer=params.get('analyzer', 'word'),
                        tokenizer=self.enhanced_tokenizer,  # Verwende die statische Tokenizer-Methode
                        norm=params.get('norm', 'l2'),
                        use_idf=params.get('use_idf', True),
                        smooth_idf=params.get('smooth_idf', True),
                        sublinear_tf=params.get('sublinear_tf', True),
                        vocabulary=vocabulary
                    )
                    
                    # IDF-Werte wiederherstellen, falls vorhanden
                    if 'idf_values' in params:
                        import numpy as np
                        self.tfidf_vectorizer.idf_ = np.array(params['idf_values'])
                except Exception as vec_error:
                    print(f"⚠️ Fehler beim Rekonstruieren des Vektorisierers: {str(vec_error)}")
                    print("⚠️ Erstelle neuen Vektorisierer...")
                    self.tfidf_vectorizer = TfidfVectorizer(
                        max_features=self.max_features,
                        stop_words='english',
                        ngram_range=(1, 4),
                        tokenizer=self.enhanced_tokenizer,  # Statische Methode
                        min_df=1, max_df=0.95,
                        lowercase=True,
                        strip_accents='unicode',
                        analyzer='word',
                        norm='l2',
                        use_idf=True,
                        smooth_idf=True,
                        sublinear_tf=True
                    )
            else:
                # Fallback, falls vectorizer_params nicht vorhanden sind
                print("📄 Keine Vektorisierer-Parameter gefunden. Erstelle neu...")
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=self.max_features,
                    stop_words='english',
                    ngram_range=(1, 4),
                    tokenizer=self.enhanced_tokenizer,  # Statische Methode
                    min_df=1, max_df=0.95,
                    lowercase=True,
                    strip_accents='unicode',
                    analyzer='word',
                    norm='l2',
                    use_idf=True,
                    smooth_idf=True,
                    sublinear_tf=True
                )
            
            # Rekonstruiere LSA-Modell, falls Parameter vorhanden sind
            if 'lsa_params' in model_data and model_data['lsa_params'] and self.use_lsa:
                try:
                    lsa_params = model_data['lsa_params']
                    from sklearn.decomposition import TruncatedSVD
                    
                    self.lsa_model = TruncatedSVD(
                        n_components=lsa_params.get('n_components', self.lsa_components),
                        algorithm=lsa_params.get('algorithm', 'randomized'),
                        n_iter=lsa_params.get('n_iter', 10),
                        random_state=lsa_params.get('random_state', 42)
                    )
                    
                    # Wenn Komponenten vorhanden sind, setze sie
                    if 'components' in lsa_params:
                        import numpy as np
                        self.lsa_model.components_ = np.array(lsa_params['components'])
                    
                    # Wenn Varianzerklärung vorhanden ist, setze sie
                    if 'explained_variance_ratio' in lsa_params:
                        self.lsa_model.explained_variance_ratio_ = np.array(lsa_params['explained_variance_ratio'])
                    
                    print("📄 LSA-Modell rekonstruiert")
                except Exception as lsa_error:
                    print(f"⚠️ Fehler beim Rekonstruieren des LSA-Modells: {str(lsa_error)}")
                    self.lsa_model = None
            else:
                # Kein LSA-Modell in den Daten
                self.lsa_model = None
                if self.use_lsa:
                    print("📄 LSA-Modellparameter nicht gefunden, aber use_lsa=True. LSA wird neu initialisiert.")
            
            # Matrix-Objekte müssen neu berechnet werden
            self.tfidf_matrix = None
            self.lsa_matrix = None
            self.hotel_clusterer = None  # Cluster müssen neu berechnet werden
            
            # Wenn Texte vorhanden sind, berechne die Matrizen neu
            if self.hotel_texts and len(self.hotel_texts) > 0:
                print("🔄 Berechne TF-IDF und LSA-Matrizen neu...")
                try:
                    self._retrain_model_quick()
                except Exception as train_error:
                    print(f"⚠️ Fehler beim Neutraining: {str(train_error)}")
                    print("⚠️ Modell ist möglicherweise nicht vollständig funktionsfähig.")
            else:
                print("⚠️ Keine Hotel-Texte gefunden. Das Modell ist noch nicht trainiert.")
                self.is_fitted = False
            
            print(f"✅ Text-Modell erfolgreich geladen von {filepath}")
            return True
            
        except Exception as e:
            print(f"⚠️ Fehler beim Laden des Text-Modells: {str(e)}")
            print(f"Details: {str(type(e))}: {str(e)}")
            print("⚠️ Initialisiere ein neues Text-Modell...")
            self.is_fitted = False
            return False
            
    def _retrain_model_quick(self):
        """Schnelles Neutraining des TF-IDF-Modells mit vorhandenen Texten"""
        # Prüfe, ob Hotel-Texte verfügbar sind
        if not self.hotel_texts or len(self.hotel_texts) == 0:
            print("⚠️ Keine Hotel-Texte für schnelles Neutraining verfügbar!")
            return False
        
        try:
            # Prüfe, ob der TF-IDF-Vektorisierer existiert
            if not hasattr(self, 'tfidf_vectorizer') or self.tfidf_vectorizer is None:
                print("⚠️ TF-IDF-Vektorisierer nicht vorhanden, erstelle neu...")
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=self.max_features,
                    stop_words='english',
                    ngram_range=(1, 4),
                    tokenizer=self.enhanced_tokenizer,  # Verwende die statische Methode
                    min_df=1, max_df=0.95,
                    lowercase=True,
                    strip_accents='unicode',
                    analyzer='word',
                    norm='l2',
                    use_idf=True,
                    smooth_idf=True,
                    sublinear_tf=True
                )
            
            print(f"🔄 Trainiere TF-IDF neu mit {len(self.hotel_texts)} Hotel-Texten...")
            
            # TF-IDF-Matrix berechnen
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.hotel_texts)
            
            # Ausgabe der Matrix-Dimensionen zur Überprüfung
            print(f"📊 TF-IDF-Matrix erstellt: {self.tfidf_matrix.shape[0]} Hotels × {self.tfidf_matrix.shape[1]} Features")
            
            # LSA anwenden, wenn aktiviert
            if self.use_lsa:
                # Erstelle neues LSA-Modell, falls keines existiert
                if not self.lsa_model:
                    # Sichere Berechnung der Komponenten
                    max_components = min(
                        self.lsa_components,
                        self.tfidf_matrix.shape[1] - 1 if self.tfidf_matrix.shape[1] > 1 else 1,
                        self.tfidf_matrix.shape[0] - 1 if self.tfidf_matrix.shape[0] > 1 else 1,
                        500  # Absolute Obergrenze
                    )
                    
                    # Mindestens 2 Komponenten
                    max_components = max(2, max_components)
                    
                    self.lsa_model = TruncatedSVD(
                        n_components=max_components, 
                        algorithm='randomized', 
                        random_state=42,
                        n_iter=10  # Mehr Iterationen für stabilere Ergebnisse
                    )
                
                # LSA-Matrix berechnen
                try:
                    self.lsa_matrix = self.lsa_model.fit_transform(self.tfidf_matrix)
                    print(f"📊 LSA-Transformation durchgeführt: {self.lsa_matrix.shape[1]} Komponenten")
                except Exception as lsa_error:
                    print(f"⚠️ Fehler bei LSA-Berechnung: {str(lsa_error)}")
                    print("⚠️ Fortfahren ohne LSA...")
                    self.lsa_matrix = None
                    self.use_lsa = False
            
            # Hotel-Clustering, falls aktiviert
            if self.enable_clustering and len(self.hotel_texts) >= 20:
                try:
                    import numpy as np
                    
                    # Benutze die passende Matrix für Clustering
                    matrix_for_clustering = self.lsa_matrix if self.use_lsa and self.lsa_matrix is not None else self.tfidf_matrix
                    
                    # Erstelle neues Cluster-Modell, falls keines existiert
                    if not self.hotel_clusterer:
                        n_clusters = min(20, len(self.hotel_texts) // 10)  # Nicht mehr Cluster als 1/10 der Hotels
                        n_clusters = max(2, n_clusters)  # Mindestens 2 Cluster
                        self.hotel_clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    
                    # Clustering durchführen
                    # Für KMeans mit sparsen Matrizen müssen wir zu einem dichten Array konvertieren
                    if isinstance(matrix_for_clustering, np.ndarray):
                        self.hotel_clusters = self.hotel_clusterer.fit_predict(matrix_for_clustering)
                    else:
                        self.hotel_clusters = self.hotel_clusterer.fit_predict(matrix_for_clustering.toarray())
                    
                    print(f"📊 Hotel-Clustering durchgeführt")
                except Exception as cluster_error:
                    print(f"⚠️ Fehler beim Hotel-Clustering: {str(cluster_error)}")
                    self.hotel_clusters = None
            
            self.is_fitted = True
            print("✅ Schnelles Neutraining erfolgreich abgeschlossen")
            return True
        
        except Exception as e:
            print(f"❌ Fehler beim schnellen Neutraining: {str(e)}")
            self.tfidf_matrix = None
            self.lsa_matrix = None
            self.is_fitted = False
            return False
    
    def _enhance_query(self, query: str) -> str:
        """Query-Expansion: Synonyme und verwandte Begriffe ergänzen"""
        # Synonym-Mapping wie im Tokenizer
        synonym_map = {
            'pool': ['swimmingpool', 'schwimmbad'],
            'beach': ['strand', 'sea', 'ocean'],
            'family': ['kids', 'children', 'familie', 'kinder'],
            'spa': ['wellness', 'relaxation'],
            'city': ['stadt', 'downtown', 'zentrum'],
            'luxury': ['premium', 'highend', 'deluxe', 'luxus'],
            'cheap': ['budget', 'affordable', 'günstig'],
            'restaurant': ['dining', 'essen'],
            'view': ['aussicht', 'blick'],
        }
        tokens = query.lower().split()
        expanded = []
        for t in tokens:
            expanded.append(t)
            for key, syns in synonym_map.items():
                if t == key or t in syns:
                    expanded.extend([key] + syns)
        return ' '.join(list(set(expanded)))
        """
        Verbessert die Benutzeranfrage durch Synonyme und Rechtschreibkorrekturen
        
        Args:
            query: Ursprüngliche Benutzeranfrage
            
        Returns:
            Verbesserte Abfrage mit Synonymen und Korrekturen
        """
        # Debug-Modus
        debug = True
        if debug:
            print(f"DEBUG ENHANCE: Originale Query: '{query}'")
            
        # Die ursprüngliche Abfrage immer beibehalten
        original_query = query
            
        # Erweiterte Rechtschreibkorrekturen für häufige Fehler in Deutsch und Englisch
        spelling_corrections = {
            # Englische Korrekturen
            'luxory': 'luxury luxurious',
            'luxurios': 'luxury luxurious',
            'luxurious': 'luxury luxurious',
            'relaxing': 'relaxing relaxation relax',
            'relax': 'relax relaxing relaxation',
            'kids': 'kids children family child-friendly',
            'children': 'children kids family child-friendly',
            'family': 'family families children kids child-friendly',
            'swiming': 'swimming',
            'swimmingpool': 'swimming pool',
            'pool': 'pool swimming swimming-pool water',
            'breakfast': 'breakfast morning-meal buffet food',
            'breakfest': 'breakfast',
            'wiifi': 'wifi',
            'wlan': 'wifi internet wireless',
            'see': 'sea ocean beach',
            'strand': 'beach strand sea ocean',
            'meer': 'sea ocean beach meer',
            'gym': 'gym fitness workout',
            'romantic': 'romantic couples honeymoon',
            'buisness': 'business',
            'bussiness': 'business',
            'center': 'centre center downtown',
            'hotell': 'hotel',
            'resturant': 'restaurant',
            'restarant': 'restaurant',
            'restrant': 'restaurant',
            'appartment': 'apartment',
            'appartments': 'apartments',
            'appartement': 'apartment',
            'accomodation': 'accommodation',
            'acommodation': 'accommodation',
            'freindly': 'friendly',
            'confortable': 'comfortable',
            'comftable': 'comfortable',
            'veiw': 'view',
            'oceanview': 'ocean view',
            'seaview': 'sea view',
            'mountian': 'mountain',
            'mountin': 'mountain',
            'cheap': 'cheap affordable budget inexpensive low-cost',
            'expensive': 'expensive luxury premium high-end',
            'good': 'good quality excellent great nice',
            'great': 'great excellent outstanding exceptional',
            'nice': 'nice good pleasant lovely',
            'big': 'big large spacious roomy',
            'small': 'small cozy compact',
            'clean': 'clean tidy neat spotless hygienic',
            
            # Deutsche Korrekturen
            'günstig': 'günstig billig preiswert affordable cheap budget',
            'guenstig': 'günstig billig preiswert affordable cheap budget',
            'billig': 'billig günstig preiswert affordable cheap budget',
            'preiswert': 'preiswert günstig billig affordable cheap budget',
            'teuer': 'teuer luxury expensive premium hochwertig',
            'luxus': 'luxus luxury luxurious premium hochwertig',
            'luxuriös': 'luxuriös luxus luxury luxurious premium',
            'luxurioes': 'luxuriös luxus luxury luxurious premium',
            'gut': 'gut good quality hochwertig excellent',
            'gutes': 'gut good quality hochwertig excellent',
            'schön': 'schön beautiful nice pretty lovely attraktiv',
            'schoen': 'schön beautiful nice pretty lovely attraktiv',
            'gross': 'groß big large spacious geräumig',
            'groß': 'groß big large spacious geräumig',
            'klein': 'klein small cozy compact gemütlich',
            'sauber': 'sauber clean tidy neat spotless hygienic',
            'stadt': 'stadt city urban downtown metropolitan innenstadt zentrum',
            'innenstadt': 'innenstadt downtown city zentrum stadt urban',
            'zentrum': 'zentrum center centre innenstadt downtown central',
            'strand': 'strand beach sea ocean küste meer',
            'meer': 'meer sea ocean strand beach küste',
            'schwimmbad': 'schwimmbad pool swimming',
            'fruehstueck': 'frühstück breakfast buffet',
            'frühstück': 'frühstück breakfast buffet',
            'wifi': 'wifi wlan internet wireless',
            'wlan': 'wlan wifi internet wireless',
            'internet': 'internet wifi wlan wireless',
            'kinder': 'kinder children kids familie family',
            'familie': 'familie family kinder children',
            'familienfreundlich': 'familienfreundlich family-friendly kinderfreundlich children',
            'spa': 'spa wellness massage entspannung relaxation',
            'wellness': 'wellness spa massage entspannung relaxation',
            'entspannung': 'entspannung relaxation erholung wellness spa',
            'ruhig': 'ruhig quiet peaceful calm entspannend',
            'geschäft': 'geschäft business arbeit work',
            'geschaeft': 'geschäft business arbeit work',
            'arbeit': 'arbeit work business geschäft',
            'konferenz': 'konferenz conference meeting geschäft',
            'am': 'am at on near in bei neben',
            'in': 'in at on inside within',
            'bei': 'bei at near by',
            'mit': 'mit with including inklusive',
            'ohne': 'ohne without',
            'für': 'für for',
            'fuer': 'für for'
        }
        
        # Erweiterte Synonymwörterbuch für Hotelbegriffe
        synonym_dict = {
            # Allgemeine Hoteltypen
            'hotel': 'hotel accommodation lodging stay unterkunft',
            'resort': 'resort vacation retreat leisure destination urlaub ferienresort',
            'apartment': 'apartment flat condo suite appartement wohnung',
            'hostel': 'hostel backpacker budget affordable jugendherberge',
            'lodge': 'lodge cabin chalet cottage hütte',
            'inn': 'inn bed breakfast cozy gasthof',
            
            # Lage/Umgebung
            'luxury': 'luxury luxurious high-end exclusive premium upscale fancy elegant luxus luxuriös',
            'budget': 'budget cheap affordable economy low-cost inexpensive günstig preiswert billig',
            'beach': 'beach seaside waterfront ocean sea coastal shore strand meer küste',
            'city': 'city urban downtown metropolitan central stadt stadtmitte innenstadt',
            'quiet': 'quiet peaceful tranquil calm serene silent relaxing ruhig friedlich still',
            'mountain': 'mountain hill mountainside alpine highlands berg gebirge alpen',
            'countryside': 'countryside rural nature scenic land ländlich natur landschaft',
            'downtown': 'downtown central city center heart zentrum innenstadt mitte',
            'airport': 'airport transportation transit flughafen',
            
            # Eigenschaften
            'family': 'family kid kids child children child-friendly playground familie kinder familienfreundlich spielplatz',
            'business': 'business corporate work conference meeting professional geschäft arbeit konferenz geschäftsreise',
            'romantic': 'romantic couples honeymoon intimate romantisch paar flitterwochen',
            'modern': 'modern contemporary sleek minimalist stylish modern zeitgenössisch stilvoll',
            'historic': 'historic heritage traditional old classic historisch erbe traditionell alt klassisch',
            'clean': 'clean hygienic spotless tidy sauber hygienisch makellos ordentlich',
            'friendly': 'friendly welcoming hospitable warm kind freundlich einladend gastfreundlich warm nett',
            'comfortable': 'comfortable cozy cosy snug pleasant homely komfortabel gemütlich angenehm',
            'spacious': 'spacious roomy large big generous geräumig groß großzügig',
            'relaxing': 'relaxing peaceful calm serene restful soothing tranquil entspannend friedlich ruhig gelassen erholsam',
            
            # Annehmlichkeiten
            'pool': 'pool swimming swimming-pool aquatic schwimmbad schwimmen wasser',
            'spa': 'spa wellness massage therapy relaxation wellness massage therapie entspannung',
            'wifi': 'wifi internet connection wireless wlan internet verbindung drahtlos',
            'breakfast': 'breakfast morning-meal buffet continental frühstück morgenmahl büffet',
            'restaurant': 'restaurant dining food cuisine meal restaurant essen speisen küche mahlzeit',
            'bar': 'bar drinks alcohol cocktails getränke alkohol cocktails',
            'gym': 'gym fitness workout exercise health fitnessstudio fitness training übung gesundheit',
            'parking': 'parking car garage parkplatz auto garage',
            'view': 'view scenic panoramic overlook vista aussicht landschaftlich panoramisch überblick',
            'balcony': 'balcony terrace patio outdoor balkon terrasse patio',
            'garden': 'garden green outdoor nature garten grün draußen natur',
            'kitchen': 'kitchen kitchenette cooking self-catering küche kochnische kochen selbstverpflegung',
            'pet': 'pet dog cat animal friendly haustier hund katze tier freundlich'
        }
        
        # Konzeptbezogene Begriffserweiterungen
        concept_groups = {
            'family_vacation': 'family kids children playground activities kid-friendly familie kinder spielplatz aktivitäten kinderfreundlich familienfreundlich familienurlaub',
            'beach_holiday': 'beach ocean sea sand shore swimming sunbathing strand meer ozean sand ufer schwimmen sonnenbaden strandurlaub',
            'city_break': 'city downtown sightseeing shopping culture attractions stadt innenstadt besichtigung einkaufen kultur attraktionen städtereise',
            'luxury_stay': 'luxury premium exclusive high-end five-star elegant luxus premium exklusiv hochwertig fünf-sterne elegant',
            'business_trip': 'business work wifi conference meeting professional geschäft arbeit wlan konferenz treffen professionell geschäftsreise',
            'romantic_getaway': 'romantic couple honeymoon intimate privacy romantik paar flitterwochen intim privatsphäre',
            'wellness_retreat': 'wellness spa massage relaxation therapy wellness spa massage entspannung therapie',
            'budget_travel': 'budget affordable cheap inexpensive economy value budget erschwinglich günstig preiswert ökonomisch wert'
        }
        
        # Query normalisieren (Kleinschreibung, Entfernung von Sonderzeichen)
        normalized_query = query.lower()
        normalized_query = re.sub(r'[^a-z0-9äöüßáéíóúñ\s]', ' ', normalized_query)
        normalized_query = re.sub(r'\s+', ' ', normalized_query).strip()
        
        if debug:
            print(f"DEBUG ENHANCE: Normalisierte Query: '{normalized_query}'")
        
        # Wörter in der Abfrage identifizieren
        query_words = normalized_query.split()
        
        if debug:
            print(f"DEBUG ENHANCE: Query Wörter: {query_words}")
        
        # Verbesserte Wörter sammeln
        enhanced_words = []
        
        # Original-Query immer behalten und verstärken
        enhanced_words.append(original_query.lower())  # Originale Query immer beibehalten
        enhanced_words.append(normalized_query)
        
        # Füge spezielle Hotel-Suchbegriffe hinzu, wenn die Abfrage zu allgemein ist
        if len(query_words) < 3:
            enhanced_words.extend(['hotel', 'accommodation', 'lodging', 'stay', 'unterkunft'])
        
        # Prüfen auf Konzeptübereinstimmungen für ganze Konzepte
        for concept, terms in concept_groups.items():
            # Überprüfen, ob die Anfrage mit Konzepten übereinstimmt
            concept_words = set(terms.split())
            query_set = set(query_words)
            
            # Wenn mindestens 1 Wort aus dem Konzept in der Anfrage vorkommt
            matching_words = concept_words.intersection(query_set)
            if matching_words:
                # Wenn mehr Übereinstimmungen, dann stärkere Gewichtung
                if len(matching_words) >= 2:
                    enhanced_words.extend([terms] * 2)  # Doppelte Gewichtung bei mehreren Matches
                else:
                    enhanced_words.append(terms)
        
        # Rechtschreibkorrekturen und Synonyme für jedes Wort
        for word in query_words:
            # Ursprüngliches Wort hinzufügen
            enhanced_words.append(word)
            
            # Rechtschreibkorrektur anwenden
            if word in spelling_corrections:
                enhanced_words.append(spelling_corrections[word])
                enhanced_words.append(spelling_corrections[word])  # Doppelt für mehr Gewicht
            
            # Synonyme hinzufügen
            if word in synonym_dict:
                enhanced_words.append(synonym_dict[word])
                
                # Für sehr wichtige Begriffe wie 'luxury', 'family' usw. stärker gewichten
                important_terms = ['luxury', 'family', 'beach', 'city', 'business', 'spa', 
                                  'pool', 'wifi', 'breakfast', 'luxus', 'familie', 'strand', 
                                  'stadt', 'geschäft', 'schwimmbad', 'frühstück']
                if word in important_terms:
                    enhanced_words.append(synonym_dict[word])  # Nochmals hinzufügen für mehr Gewicht
        
        # Finde n-Gramme (Phrasen) in der Anfrage und füge sie auch hinzu
        n_grams = []
        for n in range(2, min(4, len(query_words) + 1)):  # 2-3-Gramme
            for i in range(len(query_words) - n + 1):
                n_gram = ' '.join(query_words[i:i+n])
                n_grams.append(n_gram)
        
        # Füge die n-Gramme hinzu (mit Gewichtung für längere n-Gramme)
        for n_gram in n_grams:
            enhanced_words.append(n_gram)
            # Längere n-Gramme stärker gewichten
            if len(n_gram.split()) > 2:
                enhanced_words.append(n_gram)
        
        # Verbesserte Abfrage erstellen
        enhanced_query = ' '.join(enhanced_words)
        
        # Abfrage könnte sehr lang werden, also begrenzen
        max_length = 2000  # Maximal 2000 Zeichen
        if len(enhanced_query) > max_length:
            enhanced_query = enhanced_query[:max_length]
        
        if debug:
            print(f"DEBUG ENHANCE: Erweiterte Query: '{enhanced_query[:100]}...'")
            print(f"DEBUG ENHANCE: Erweiterte Query Länge: {len(enhanced_query)} Zeichen")
        
        # Direkte Übereinstimmung mit den originalen Worten priorisieren
        final_query = f"{original_query.lower()} {original_query.lower()} {enhanced_query}"
            
        return final_query

    def _extract_family_features(self, description: str) -> str:
        """Extrahiert familienfreundliche Features aus der Hotelbeschreibung"""
        family_keywords = [
            'family', 'families', 'kid', 'kids', 'children', 'child', 'play', 'playground',
            'game room', 'activities for children', 'child-friendly', 'family-friendly',
            'babysit', 'babysitting', 'cribs', 'crib', 'baby', 'toddler', 'youth',
            'family suite', 'connecting rooms', 'animation', 'waterslide', 'mini club',
            'kids club', 'children pool', 'children\'s pool', 'family package'
        ]
        
        family_features = []
        description_lower = description.lower()
        
        for keyword in family_keywords:
            if keyword in description_lower:
                family_features.append(keyword)
        
        if family_features:
            return "family-friendly " + " ".join(family_features) + " children kids"
        
        return ""

    def _extract_relaxation_features(self, description: str) -> str:
        """Extrahiert entspannungsbezogene Features aus der Hotelbeschreibung"""
        relaxation_keywords = [
            'spa', 'sauna', 'massage', 'wellness', 'relaxation', 'relaxing', 'tranquil', 'peaceful',
            'quiet', 'serene', 'retreat', 'jacuzzi', 'hot tub', 'pool', 'thermal', 'meditation',
            'yoga', 'garden', 'lounge', 'terrace', 'balcony', 'view', 'beach', 'lakeside',
            'nature', 'calm', 'comfort', 'cozy', 'luxury', 'pamper', 'treat'
        ]
        
        relaxation_features = []
        description_lower = description.lower()
        
        for keyword in relaxation_keywords:
            if keyword in description_lower:
                relaxation_features.append(keyword)
        
        if relaxation_features:
            return "relaxing " + " ".join(relaxation_features) + " relaxation comfort"
        
        return ""

    def _extract_activity_features(self, description: str) -> str:
        """Extrahiert aktivitätsbezogene Features aus der Hotelbeschreibung"""
        activity_keywords = [
            'hiking', 'trekking', 'biking', 'cycling', 'swimming', 'fishing', 'skiing', 'snowboarding',
            'golf', 'tennis', 'water sports', 'diving', 'snorkeling', 'surfing', 'sailing',
            'cooking class', 'dance class', 'yoga class', 'art class', 'music class',
            'workshop', 'seminar', 'conference', 'event', 'meeting',
            'team building', 'retreat', 'wellness', 'spa', 'massage',
            'relaxation', 'leisure', 'entertainment', 'nightlife', 'shopping',
            'sightseeing', 'tour', 'excursion', 'adventure', 'exploration'
        ]
        
        activity_features = []
        description_lower = description.lower()
        
        for keyword in activity_keywords:
            if keyword in description_lower:
                activity_features.append(keyword)
        
        if activity_features:
            return "activities " + " ".join(activity_features)
        
        return ""

    def _extract_business_features(self, description: str) -> str:
        """Extrahiert geschäftsreiserelevante Features aus der Hotelbeschreibung"""
        business_keywords = [
            'business', 'conference', 'meeting', 'seminar', 'workshop', 'event', 'team building',
            'presentation', 'negotiation', 'corporate', 'executive', 'professional',
            'office', 'business center', 'meeting room', 'conference room', 'boardroom',
            'projector', 'videoconference', 'whiteboard', 'flip chart', 'high-speed internet',
            'wifi', 'catering', 'coffee break', 'lunch', 'dinner',
            'business lounge', 'executive floor', 'VIP', 'concierge', 'secretarial',
            'translation', 'transcription', 'printing', 'copying', 'fax',
            'car rental', 'airport transfer', 'shuttle service', 'parking'
        ]
        
        business_features = []
        description_lower = description.lower()
        
        for keyword in business_keywords:
            if keyword in description_lower:
                business_features.append(keyword)
        
        if business_features:
            return "business-friendly " + " ".join(business_features)
        
        return ""
    
    def _calculate_amenity_score(self, hotels_df: pd.DataFrame, preferences: Dict) -> pd.Series:
        """
        Calculate amenity score based on preferred amenities
        
        Args:
            hotels_df: Hotels DataFrame
            preferences: User preferences including preferred_amenities
            
        Returns:
            Series of amenity scores (0-1 scale)
        """
        amenity_scores = pd.Series([0.5] * len(hotels_df), index=hotels_df.index)
        
        preferred_amenities = preferences.get('preferred_amenities', [])
        if not preferred_amenities:
            return amenity_scores
        
        # Calculate score based on how many preferred amenities the hotel has
        for i, (idx, hotel) in enumerate(hotels_df.iterrows()):
            matching_amenities = 0
            total_amenities = len(preferred_amenities)
            
            # Check amenities in the description/amenities field
            amenities_text = str(hotel.get('amenities', '')) + ' ' + str(hotel.get('description', ''))
            
            for amenity in preferred_amenities:
                if amenity.lower() in amenities_text.lower():
                    matching_amenities += 1
                # Also check specific amenity columns if they exist
                amenity_col = f'has_{amenity}'
                if amenity_col in hotel.index and hotel[amenity_col] == 1:
                    matching_amenities += 1
            
            if total_amenities > 0:
                # Score from 0.3 (no matches) to 1.0 (all matches)
                base_score = 0.3
                bonus = 0.7 * (matching_amenities / total_amenities)
                amenity_scores.iloc[i] = base_score + bonus
        
        return amenity_scores

if __name__ == "__main__":
    # Test the text-based recommender
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
    from data_preparation.load_data import HotelDataLoader
    
    # Load data
    loader = HotelDataLoader()
    hotels_df = loader.load_hotels()
    
    if not hotels_df.empty:
        # Create and fit text model
        text_recommender = TextBasedRecommender(max_features=500, use_lsa=True)
        text_recommender.fit(hotels_df)
        
        # Test text search
        test_queries = [
            "relaxing in luxury hotel",
            "cheap family friendly hotel with pool",
            "business hotel with wifi",
            "spa resort with wellness"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            
            # Get keywords
            keywords = text_recommender.get_query_keywords(query, top_k=5)
            print(f"📝 Keywords: {keywords}")
            
            # Test with different budgets
            budgets = [200, 400, 600]
            
            for budget in budgets:
                user_prefs = {'max_price': budget}
                
                recommendations = text_recommender.recommend_hotels(
                    query, hotels_df, user_prefs, top_k=3
                )
                
                print(f"\n💰 Budget €{budget}:")
                if len(recommendations) > 0:
                    for _, hotel in recommendations.iterrows():
                        print(f"  - {hotel['name'][:50]:50} | €{hotel['price']:6.0f} | {hotel['rating']}/10⭐")
                else:
                    print("  ❌ Keine Hotels in dieser Preiskategorie gefunden")
        
        print(f"\n✅ Text-Modell erfolgreich getestet!")
    else:
        print("❌ Keine Hotel-Daten geladen")
