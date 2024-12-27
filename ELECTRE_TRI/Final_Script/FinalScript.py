import pandas as pd
import numpy as np
import requests
import json
import csv
import time
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import os

class DataProcessor:
    """Classe pour le traitement et la conversion des données"""
    
    @staticmethod
    def json_to_csv(json_file_path, csv_file_path, encoding='utf-8'):
        """Convertit un fichier JSON en CSV avec les colonnes désirées"""
        # Reference to original code in converterCsv.py
        """
        startLine: 42
        endLine: 70
        """
        
    @staticmethod
    def filter_csv_columns(csv_data, columns):
        """Filtre les colonnes spécifiées d'un fichier CSV"""
        filtered_data = [{col: row[col] for col in columns if col in row} for row in csv_data]
        return filtered_data

    @staticmethod
    def refine_csv_output(input_csv_file, output_csv_file, columns):
        """Raffine le fichier CSV en ne gardant que les colonnes spécifiées"""
        with open(input_csv_file, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            csv_data = list(reader)

        filtered_data = DataProcessor.filter_csv_columns(csv_data, columns)

        with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            writer.writerows(filtered_data)

class OpenFoodFactsClient:
    """Classe pour interagir avec l'API OpenFoodFacts"""
    
    def __init__(self):
        self.base_url = "https://world.openfoodfacts.net/api/v2/search"
        self.headers = {
            "User-Agent": "Python Script - Educational Purpose"
        }
        self.required_fields = (
            "code,product_name,brands,nutriscore_grade,ecoscore_grade,"
            "categories_tags_en,ingredients_text,nutriments,additives_tags,"
            "image_url,image_small_url,image_front_url,image_front_small_url,labels"
        )

    def _fetch_page(self, nutriscore, page=1):
        """Récupère une page de produits avec un nutriscore spécifique"""
        params = {
            "categories_tags_en": "Pastas",
            "nutrition_grades_tags": nutriscore,
            "fields": self.required_fields,
            "page": page,
            "page_size": 50
        }
        
        try:
            response = requests.get(
                self.base_url,
                params=params,
                headers=self.headers
            )
            response.raise_for_status()
            products = response.json().get('products', [])
            
            for product in products:
                additives_tags = product.get('additives_tags', [])
                product['additives_count'] = len(additives_tags)
                labels = product.get('labels', '').split(',')
                product['is_bio'] = 'Bio' in labels
            
            return products
        except requests.exceptions.RequestException as e:
            print(f"Erreur lors de la récupération de la page {page} pour le nutriscore {nutriscore}: {e}")
            return []

    def _has_valid_images(self, product):
        """Vérifie si le produit a des images valides"""
        image_fields = ['image_url', 'image_small_url', 'image_front_url', 'image_front_small_url']
        return any(product.get(field) for field in image_fields)

    def _has_valid_ecoscore(self, product):
        """Vérifie si le produit a un ecoscore valide"""
        return product.get('ecoscore_grade', '').lower() in ['a', 'b', 'c', 'd', 'e']

    def collect_all_products(self):
        """Collecte tous les produits selon les spécifications"""
        first_150 = []
        bio_count = 0
        target_bio = 75
        
        # Collecte des premiers 150 produits (30 de chaque grade A-E)
        for grade in ['a', 'b', 'c', 'd', 'e']:
            print(f"\nCollecte des produits avec Nutriscore {grade.upper()}...")
            products = self._collect_products_by_nutriscore(grade, 30)
            
            for product in products:
                if product['is_bio']:
                    bio_count += 1
            
            first_150.extend(products)
            print(f"Collecté {len(products)} produits avec Nutriscore {grade.upper()} (Produits Bio: {bio_count})")

        # Collecte des 150 produits restants
        remaining_150 = self._collect_remaining_products(first_150, bio_count, target_bio)
        
        total_products = first_150 + remaining_150
        final_bio_count = sum(1 for p in total_products if p['is_bio'])
        print(f"\nCollecte finale terminée:")
        print(f"Total des produits: {len(total_products)}")
        print(f"Total des produits Bio: {final_bio_count}")
        
        return total_products

    def _collect_products_by_nutriscore(self, nutriscore, target_count):
        """Collecte un nombre spécifique de produits pour un nutriscore donné"""
        collected_products = []
        page = 1
        
        while len(collected_products) < target_count and page <= 10:
            products = self._fetch_page(nutriscore, page)
            if not products:
                break
                
            for product in products:
                if (self._has_valid_images(product) and 
                    self._has_valid_ecoscore(product) and 
                    len(collected_products) < target_count):
                    collected_products.append(product)
            
            page += 1
            time.sleep(1)
            
        return collected_products

    def _collect_remaining_products(self, existing_products, current_bio_count, target_bio):
        """Collecte les produits restants en respectant les critères bio"""
        remaining_products = []
        page = 1
        
        while len(remaining_products) < 150 and page <= 20:
            for grade in ['a', 'b', 'c', 'd', 'e']:
                products = self._fetch_page(grade, page)
                
                for product in products:
                    if (self._has_valid_images(product) and 
                        self._has_valid_ecoscore(product) and 
                        product not in existing_products and 
                        product not in remaining_products and 
                        len(remaining_products) < 150):
                        
                        if current_bio_count < target_bio and product['is_bio']:
                            remaining_products.append(product)
                            current_bio_count += 1
                        elif current_bio_count >= target_bio and not product['is_bio']:
                            remaining_products.append(product)
                
                time.sleep(1)
            page += 1
        
        return remaining_products

    def save_products(self, products, filename="pasta_products.json"):
        """Sauvegarde les produits dans un fichier JSON"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(products, f, indent=2, ensure_ascii=False)
        print(f"\nProduits sauvegardés dans {filename}")

    def summarize_aliments(self, csv_file_path):
        """Génère un résumé des statistiques des aliments"""
        df = pd.read_csv(csv_file_path)
        total_count = len(df)
        
        summary = {}
        
        # Statistiques Nutriscore
        for grade in ['A', 'B', 'C', 'D', 'E']:
            count = df['nutriscore_grade'].str.upper().value_counts().get(grade, 0)
            summary[f'Nutriscore {grade}'] = f"{count}/{total_count}"
        
        # Statistiques Ecoscore
        for grade in ['A', 'B', 'C', 'D', 'E']:
            count = df['ecoscore_grade'].str.upper().value_counts().get(grade, 0)
            summary[f'Ecoscore {grade}'] = f"{count}/{total_count}"
        
        # Statistiques Bio
        bio_count = df['is_bio'].sum() if 'is_bio' in df.columns else 0
        summary["Bio"] = f"{bio_count}/{total_count}"
        
        return summary
class ElectreTri:
    """Classe pour l'analyse ELECTRE TRI"""
    
    def __init__(self, file_path, weights, lambda_value=0.6, k=5):
        """Initialisation d'ELECTRE TRI"""
        self.data = pd.read_csv(file_path)
        self.weights = weights
        self.lambda_value = lambda_value
        self.k = k
        self.profiles = None
        
        self.criteres_minimiser = [
            'energy-kcal_value', 'sugars_100g', 'saturated-fat_value', 
            'sodium_100g', 'Nbr additifs à risque', 'additives_count'
        ]
        self.criteres_maximiser = [
            'fruits-vegetables-nuts-estimate-from-ingredients_serving', 
            'proteins_100g', 
            'fiber_100g'
        ]

    def preprocess_data(self):
        """Prétraitement des données pour l'analyse"""
        if 'energy-kcal_value' in self.data.columns:
            self.data['energy_value_kcal'] = self.data.apply(
                lambda row: row['energy-kcal_value'] if pd.notnull(row['energy-kcal_value']) else (
                    row['energy_value'] * 0.239006 if row['energy_unit'] == 'kJ' else row['energy_value']
                ), axis=1
            )
        else:
            self.data['energy_value_kcal'] = self.data.apply(
                lambda row: row['energy_value'] * 0.239006 if row['energy_unit'] == 'kJ' else row['energy_value'],
                axis=1
            )
        
        self.data = self.data[
            self.criteres_minimiser + self.criteres_maximiser + ['nutriscore_grade']
        ].dropna()

    def calculate_profiles_quantiles(self):
        """Calcul des profils limites basé sur les quantiles"""
        self.preprocess_data()
        profiles = {}
        quantiles = [0.2, 0.4, 0.6, 0.8]
        
        # Profil π1 (pire cas)
        profiles['π1'] = {
            critere: self.data[critere].max() for critere in self.criteres_minimiser
        }
        profiles['π1'].update({
            critere: self.data[critere].min() for critere in self.criteres_maximiser
        })
        
        # Profil π6 (meilleur cas)
        profiles['π6'] = {
            critere: self.data[critere].min() for critere in self.criteres_minimiser
        }
        profiles['π6'].update({
            critere: self.data[critere].max() for critere in self.criteres_maximiser
        })
        
        # Profils intermédiaires (π2 à π5)
        for i, q in enumerate(quantiles, 2):
            profile = {}
            for critere in self.criteres_minimiser:
                profile[critere] = self.data[critere].quantile(1-q)
            for critere in self.criteres_maximiser:
                profile[critere] = self.data[critere].quantile(q)
            profiles[f'π{i}'] = profile
            
        self.profiles = profiles
        return profiles

    def calculate_profiles_knn(self, k=5, num_profiles=6):
        """Calcul des profils utilisant l'algorithme KNN"""
        self.preprocess_data()
        profiles = {}
        
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(self.data[self.criteres_minimiser + self.criteres_maximiser])
        
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(data_scaled)
        
        reference_points = []
        
        for i in range(num_profiles):
            ratio = i / (num_profiles - 1)
            ref_point = []
            
            for critere in self.criteres_minimiser:
                max_val = self.data[critere].max()
                min_val = self.data[critere].min()
                ref_point.append(max_val - (max_val - min_val) * ratio)
                
            for critere in self.criteres_maximiser:
                max_val = self.data[critere].max()
                min_val = self.data[critere].min()
                ref_point.append(min_val + (max_val - min_val) * ratio)
                
            reference_points.append(ref_point)
        
        reference_points_scaled = scaler.transform(reference_points)
        
        for i, ref_point in enumerate(reference_points_scaled, 1):
            distances, indices = knn.kneighbors([ref_point])
            neighbors = self.data.iloc[indices[0]]
            
            profile = {}
            for critere in self.criteres_minimiser + self.criteres_maximiser:
                profile[critere] = neighbors[critere].mean()
                
            profiles[f'π{i}'] = profile
        
        self._enforce_monotonicity(profiles)
        return profiles

    def _enforce_monotonicity(self, profiles):
        """Application des contraintes de monotonie sur les profils"""
        for i in range(1, 6):
            current_profile = profiles[f'π{i}']
            next_profile = profiles[f'π{i+1}']
            
            for critere in self.criteres_minimiser:
                if current_profile[critere] < next_profile[critere]:
                    avg = (current_profile[critere] + next_profile[critere]) / 2
                    current_profile[critere] = avg
                    next_profile[critere] = avg
            
            for critere in self.criteres_maximiser:
                if current_profile[critere] > next_profile[critere]:
                    avg = (current_profile[critere] + next_profile[critere]) / 2
                    current_profile[critere] = avg
                    next_profile[critere] = avg
    def categorize_super_nutri_score(self, quantile_class, ecoscore_grade, is_bio):
        """Catégorisation du Super Nutri Score"""
        pessimist = quantile_class.upper()
        eco_score = str(ecoscore_grade).upper()
        bio = str(is_bio).upper()

        if pd.isna(pessimist) or pd.isna(eco_score) or pd.isna(bio):
            return "Non évaluable"

        score_mapping = {
            "Excellent": [
                (pessimist == 'A' and eco_score == 'A'),
                (pessimist == 'A' and eco_score == 'B' and bio == 'TRUE'),
            ],
            "Bon": [
                (pessimist in ['B', 'C'] and eco_score in ['A', 'B']),
                (pessimist == 'A' and eco_score in ['B'] and bio == 'FALSE'),
                (pessimist in ['B', 'C'] and eco_score == 'C' and bio == 'TRUE'),
            ],
            "Médiocre": [
                (pessimist in ['B', 'C'] and eco_score == 'C' and bio == 'FALSE'),
                (pessimist in ['B', 'C'] and eco_score in ['C', 'D']),
                (pessimist == 'D' and eco_score == 'C' and bio == 'TRUE'),
            ],
            "Mauvais": [
                (pessimist in ['B', 'C'] and eco_score == 'E'),
                (pessimist == 'D' and eco_score == 'C' and bio == 'FALSE'),
                (pessimist == 'E'),
            ]
        }

        for category, conditions in score_mapping.items():
            if any(conditions):
                return category

        return "Mauvais"

    def concordance_index(self, product, profile):
        """Calcul des indices de concordance partiels"""
        concordance = {}
        
        for critere in self.criteres_minimiser:
            concordance[critere] = 1 if profile[critere] >= product[critere] else 0
            
        for critere in self.criteres_maximiser:
            concordance[critere] = 1 if product[critere] >= profile[critere] else 0
            
        return concordance

    def global_concordance(self, concordance_indices):
        """Calcul de l'indice de concordance global"""
        criteria_order = [
            'energy-kcal_value', 'saturated-fat_value', 'sugars_100g', 'sodium_100g',
            'proteins_100g', 'fiber_100g', 'fruits-vegetables-nuts-estimate-from-ingredients_serving',
            'additives_count', 'Nbr additifs à risque'
        ]
        
        concordance_sum = sum(
            concordance_indices[critere] * weight
            for critere, weight in zip(criteria_order, self.weights)
        )
        
        return concordance_sum / sum(self.weights)

    def outranks(self, product, profile):
        """Détermine si le produit surclasse le profil"""
        concordance = self.concordance_index(product, profile)
        global_concordance = self.global_concordance(concordance)
        return global_concordance >= self.lambda_value

    def pessimistic_majority_sorting(self, product, profiles):
        """Classification majoritaire pessimiste"""
        for k in range(5, 0, -1):
            if self.outranks(product, profiles[f'π{k}']):
                return chr(69 - k + 1)
        return 'E'

    def optimistic_majority_sorting(self, product, profiles):
        """Classification majoritaire optimiste"""
        for k in range(1, 6):
            if not self.outranks(product, profiles[f'π{k}']):
                return chr(69 - k + 1)
        return 'A'

    def process_csv_with_classes(self, input_file, knn_profiles, quantile_profiles, 
                               majority_sorting_method, output_file="output_with_classes.csv"):
        """Traitement du fichier CSV avec classifications"""
        self.data = pd.read_csv(input_file)
        
        knn_classes = []
        quantile_classes = []
        super_nutri_scores = []

        for _, product in self.data.iterrows():
            knn_class = majority_sorting_method(product, knn_profiles)
            quantile_class = majority_sorting_method(product, quantile_profiles)
            
            super_nutri_score = self.categorize_super_nutri_score(
                quantile_class,
                product['ecoscore_grade'],
                product['is_bio']
            )
            
            knn_classes.append(knn_class)
            quantile_classes.append(quantile_class)
            super_nutri_scores.append(super_nutri_score)
        
        self.data['knn_class'] = knn_classes
        self.data['quantile_class'] = quantile_classes
        self.data['super_nutri_score'] = super_nutri_scores
        
        # Save the file in Results folder
        output_path = os.path.join("Results", output_file)
        self.data.to_csv(output_path, index=False)
        print(f"Fichier traité sauvegardé dans {output_path}")

        # Sauvegarde des scores dans différents formats
        results_dir = "Results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        # Sauvegarde en format texte
        txt_path = os.path.join(results_dir, "super_nutri_scores.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for index, row in self.data.iterrows():
                f.write(f"Product {index + 1}: {row['super_nutri_score']}\n")
        
        # Sauvegarde en format Excel
        output_excel_file = os.path.join(results_dir, "super_nutri_scores.xlsx")
        columns_to_save = ['code', 'ecoscore_grade', 'is_bio', 'knn_class', 
                          'quantile_class', 'super_nutri_score']
        self.data[columns_to_save].to_excel(output_excel_file, index=False)
        print(f"Données sauvegardées dans {output_excel_file}")

def save_profiles_to_csv(profiles, filename="profiles_output.csv"):
    # Convert profiles dictionary to a DataFrame
    profiles_df = pd.DataFrame(profiles).T
    # Add a column for profile names
    profiles_df.insert(0, 'profile', profiles_df.index)
    # Save the profiles in Results folder
    output_path = os.path.join("Results", filename)
    profiles_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Profiles saved to {output_path}")

def main():
    """Fonction principale d'exécution"""
    file_path = "Output-Database.csv"
    weights = [2, 2, 2, 2, 1, 1, 1, 4, 4]
    lambda_values = [0.5, 0.6, 0.7]

    # Initialisation
    electre = ElectreTri(file_path, weights, None)
    
    # Génération des profils
    quantile_profiles = electre.calculate_profiles_quantiles()
    knn_profiles = electre.calculate_profiles_knn()
    save_profiles_to_csv(quantile_profiles,"quantiles_profiles_output.csv")
    save_profiles_to_csv(knn_profiles,"knn_profiles_output.csv")

    

    # Traitement pour chaque valeur lambda
    for lambda_value in lambda_values:
        electre = ElectreTri(file_path, weights, lambda_value)
        
        # Classification pessimiste
        output_file = f"output_with_classes_lambda_pessimistic_{lambda_value}.csv"
        electre.process_csv_with_classes(
            file_path, 
            knn_profiles, 
            quantile_profiles,
            electre.pessimistic_majority_sorting,
            output_file
        )
        
        # Classification optimiste
        output_file = f"output_with_classes_lambda_optimistic_{lambda_value}.csv"
        electre.process_csv_with_classes(
            file_path, 
            knn_profiles, 
            quantile_profiles,
            electre.optimistic_majority_sorting,
            output_file
        )

if __name__ == "__main__":
    main()
# def load_existing_data(csv_file_path):
#     """Charge les données existantes et calcule les statistiques nécessaires"""
#     df = pd.read_csv(csv_file_path)
#     existing_codes = set(df['code'].astype(str))
#     current_nutriscore_e = df['nutriscore_grade'].str.upper().value_counts().get('E', 0)
#     current_ecoscore_e = df['ecoscore_grade'].str.upper().value_counts().get('E', 0)
#     return existing_codes, current_nutriscore_e, current_ecoscore_e

# def merge_csv_files(input_files, output_file):
#     """Fusionne plusieurs fichiers CSV en un seul"""
#     dfs = [pd.read_csv(f) for f in input_files]
#     merged_df = pd.concat(dfs, ignore_index=True)
#     merged_df.to_csv(output_file, index=False)
#     print(f"Fichiers fusionnés sauvegardés dans {output_file}")

# if __name__ == "__main__":
#     main()

