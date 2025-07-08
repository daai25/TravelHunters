"""
Matrix Builder für Empfehlungssysteme
Erstellt sparse Matrizen und implementiert spezielle Aufteilungsmethoden für Empfehlungssysteme
"""

import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, Union, List, Optional

class RecommendationMatrixBuilder:
    """Erstellt und verarbeitet Matrizen für Empfehlungssysteme"""
    
    def __init__(self):
        """Initialisiert den Matrix Builder"""
        self.user_index = {}  # Mapping von user_id zu Matrix-Index
        self.item_index = {}  # Mapping von hotel_id zu Matrix-Index
        self.reverse_user_index = {}  # Mapping von Matrix-Index zu user_id
        self.reverse_item_index = {}  # Mapping von Matrix-Index zu hotel_id
    
    def build_interaction_matrix(self, interactions_df: pd.DataFrame, 
                                user_col: str = 'user_id', 
                                item_col: str = 'hotel_id',
                                rating_col: str = 'rating') -> sp.csr_matrix:
        """
        Erstellt eine sparse Interaktionsmatrix aus einem DataFrame
        
        Args:
            interactions_df: DataFrame mit User-Item-Interaktionen
            user_col: Name der Spalte mit User-IDs
            item_col: Name der Spalte mit Item-IDs
            rating_col: Name der Spalte mit Bewertungen
            
        Returns:
            Eine CSR sparse Matrix mit Benutzer-Item-Interaktionen
        """
        # Einzigartige Benutzer und Items identifizieren
        unique_users = interactions_df[user_col].unique()
        unique_items = interactions_df[item_col].unique()
        
        # Indizes erstellen
        self.user_index = {user_id: i for i, user_id in enumerate(unique_users)}
        self.item_index = {item_id: i for i, item_id in enumerate(unique_items)}
        
        # Umgekehrte Indizes erstellen
        self.reverse_user_index = {i: user_id for user_id, i in self.user_index.items()}
        self.reverse_item_index = {i: item_id for item_id, i in self.item_index.items()}
        
        # Matrix-Dimensionen
        n_users = len(unique_users)
        n_items = len(unique_items)
        
        # Sparse-Matrix-Daten vorbereiten
        user_indices = [self.user_index[user] for user in interactions_df[user_col]]
        item_indices = [self.item_index[item] for item in interactions_df[item_col]]
        ratings = interactions_df[rating_col].values
        
        # Sparse-Matrix erstellen
        interaction_matrix = sp.csr_matrix((ratings, (user_indices, item_indices)), 
                                        shape=(n_users, n_items))
        
        print(f"✅ Interaktionsmatrix erstellt: {n_users} Benutzer × {n_items} Hotels")
        print(f"   Dichte: {interaction_matrix.nnz / (n_users * n_items):.5f}")
        
        return interaction_matrix
    
    def split_leave_one_out(self, interaction_matrix: sp.csr_matrix, 
                          test_size: float = 0.2,
                          random_state: int = 42) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
        """
        Teilt die Interaktionsmatrix nach der Leave-One-Out-Methode
        
        Args:
            interaction_matrix: Sparse-Interaktionsmatrix
            test_size: Anteil der für Tests zurückzuhaltenden Daten
            random_state: Seed für Reproduzierbarkeit
            
        Returns:
            (train_matrix, test_matrix): Trainings- und Testmatrizen
        """
        # Anzahl der Benutzer und Items
        n_users, n_items = interaction_matrix.shape
        
        # Datenpunkte sammeln
        interactions = []
        for user_idx in range(n_users):
            # Holen Sie sich die Indizes der Nicht-Null-Elemente für diesen Benutzer
            _, item_indices, ratings = sp.find(interaction_matrix[user_idx])
            for i, (item_idx, rating) in enumerate(zip(item_indices, ratings)):
                interactions.append((user_idx, item_idx, rating))
        
        # Konvertieren zu DataFrame für einfachere Verarbeitung
        interactions_df = pd.DataFrame(interactions, columns=['user_idx', 'item_idx', 'rating'])
        
        # Interaktionen nach Benutzer gruppieren und aufteilen
        train_interactions = []
        test_interactions = []
        
        np.random.seed(random_state)
        for user_idx, group in interactions_df.groupby('user_idx'):
            # Nur aufteilen, wenn der Benutzer mehr als eine Interaktion hat
            if len(group) > 1:
                # Zufällig auswählen, wie viele Interaktionen für Tests verwendet werden
                n_test = max(1, int(len(group) * test_size))
                
                # Zufällige Auswahl für Tests
                test_indices = np.random.choice(group.index, size=n_test, replace=False)
                test_rows = group.loc[test_indices]
                train_rows = group.drop(test_indices)
                
                train_interactions.append(train_rows)
                test_interactions.append(test_rows)
            else:
                # Wenn nur eine Interaktion, zu Training hinzufügen
                train_interactions.append(group)
        
        # Zu DataFrames zusammenführen
        train_df = pd.concat(train_interactions)
        test_df = pd.concat(test_interactions) if test_interactions else pd.DataFrame(columns=['user_idx', 'item_idx', 'rating'])
        
        # Zu sparse Matrizen konvertieren
        train_matrix = sp.csr_matrix(
            (train_df['rating'], (train_df['user_idx'], train_df['item_idx'])),
            shape=(n_users, n_items)
        )
        
        test_matrix = sp.csr_matrix(
            (test_df['rating'], (test_df['user_idx'], test_df['item_idx'])),
            shape=(n_users, n_items)
        )
        
        print(f"✅ Leave-One-Out-Aufteilung erstellt:")
        print(f"   Trainingsdaten: {train_matrix.nnz} Interaktionen")
        print(f"   Testdaten: {test_matrix.nnz} Interaktionen")
        
        return train_matrix, test_matrix
    
    def split_time_based(self, interactions_df: pd.DataFrame, 
                       timestamp_col: str,
                       test_ratio: float = 0.2,
                       user_col: str = 'user_id', 
                       item_col: str = 'hotel_id',
                       rating_col: str = 'rating') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Teilt die Interaktionen zeitbasiert auf
        
        Args:
            interactions_df: DataFrame mit User-Item-Interaktionen
            timestamp_col: Name der Spalte mit Zeitstempeln
            test_ratio: Anteil der neueren Interaktionen für Tests
            user_col: Name der Spalte mit User-IDs
            item_col: Name der Spalte mit Item-IDs
            rating_col: Name der Spalte mit Bewertungen
            
        Returns:
            (train_df, test_df): Trainings- und Test-DataFrames
        """
        if timestamp_col not in interactions_df.columns:
            raise ValueError(f"Zeitstempelspalte {timestamp_col} nicht im DataFrame gefunden")
        
        # Sortieren nach Zeitstempel
        sorted_df = interactions_df.sort_values(by=timestamp_col)
        
        # Aufteilungspunkt bestimmen
        split_idx = int(len(sorted_df) * (1 - test_ratio))
        
        # Aufteilen
        train_df = sorted_df.iloc[:split_idx].copy()
        test_df = sorted_df.iloc[split_idx:].copy()
        
        print(f"✅ Zeitbasierte Aufteilung erstellt:")
        print(f"   Trainingsdaten: {len(train_df)} Interaktionen")
        print(f"   Testdaten: {len(test_df)} Interaktionen")
        print(f"   Trainingsperiode: {sorted_df[timestamp_col].iloc[0]} bis {sorted_df[timestamp_col].iloc[split_idx-1]}")
        print(f"   Testperiode: {sorted_df[timestamp_col].iloc[split_idx]} bis {sorted_df[timestamp_col].iloc[-1]}")
        
        return train_df, test_df
    
    def convert_to_dataframe(self, matrix: sp.csr_matrix) -> pd.DataFrame:
        """
        Konvertiert eine sparse Matrix zurück zu einem DataFrame
        
        Args:
            matrix: CSR sparse Matrix
            
        Returns:
            DataFrame mit user_id, hotel_id und rating
        """
        # Matrix-Elemente extrahieren
        user_indices, item_indices, ratings = sp.find(matrix)
        
        # Zu ursprünglichen IDs zurückkonvertieren
        user_ids = [self.reverse_user_index[idx] for idx in user_indices]
        item_ids = [self.reverse_item_index[idx] for idx in item_indices]
        
        # DataFrame erstellen
        df = pd.DataFrame({
            'user_id': user_ids,
            'hotel_id': item_ids,
            'rating': ratings
        })
        
        return df
    
    def get_item_popularity(self, interaction_matrix: sp.csr_matrix) -> Dict[int, int]:
        """
        Berechnet die Popularität jedes Items (Anzahl der Interaktionen)
        
        Args:
            interaction_matrix: Sparse-Interaktionsmatrix
            
        Returns:
            Dictionary mit Item-Index als Schlüssel und Popularität als Wert
        """
        # Summe der Nicht-Null-Elemente pro Spalte
        popularity = {}
        
        # Für jede Spalte (Item) die Anzahl der Nicht-Null-Elemente zählen
        _, n_items = interaction_matrix.shape
        for item_idx in range(n_items):
            popularity[item_idx] = interaction_matrix[:, item_idx].nnz
        
        return popularity
    
    def get_popular_items(self, interaction_matrix: sp.csr_matrix, top_k: int = 10) -> List[int]:
        """
        Gibt die Top-K beliebtesten Items zurück
        
        Args:
            interaction_matrix: Sparse-Interaktionsmatrix
            top_k: Anzahl der zurückzugebenden Items
            
        Returns:
            Liste der Indizes der beliebtesten Items
        """
        popularity = self.get_item_popularity(interaction_matrix)
        sorted_items = sorted(popularity.items(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_items[:top_k]]
