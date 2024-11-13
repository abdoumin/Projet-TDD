import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class ElectreTri:
    def __init__(self, file_path, weights, lambda_value=0.6):
        """
        Initialisation d'ELECTRE TRI
        """
        self.data = pd.read_csv(file_path)
        self.weights = weights
        self.lambda_value = lambda_value
        self.profiles = None
        
        # Define the criteria to minimize and maximize based on instructions
        self.criteres_minimiser = [
            'energy-kcal_value', 'sugars_100g', 'saturated-fat_value', 'sodium_100g'
        ]
        self.criteres_maximiser = [
            'fruits-vegetables-nuts-estimate-from-ingredients_serving', 
            'proteins_100g', 
            'fiber_100g'
        ]
        
    def preprocess_data(self):
        """
        Preprocess the data to ensure energy values are in kcal and to prepare relevant columns.
        """
        # Ensure energy values are in kcal
        self.data['energy_value_kcal'] = self.data.apply(
            lambda row: row['energy-kcal_value'] if pd.notnull(row['energy-kcal_value']) else (
                row['energy_value'] * 0.239006 if row['energy_unit'] == 'kJ' else row['energy_value']
            ), axis=1
        )
        
        # Filter data to only include necessary columns
        self.data = self.data[
            self.criteres_minimiser + self.criteres_maximiser
        ].dropna()
        
    def calculate_profiles_quantiles(self):
        """
        Méthode pour calculer les profils limites basé sur les quantiles
        """
        self.preprocess_data()
        profiles = {}
        
        # Define quantile levels for profiles π2 to π5
        quantiles = [0.2, 0.4, 0.6, 0.8]
        
        # π1 (worst-case scenario)
        profiles['π1'] = {
            critere: self.data[critere].max() for critere in self.criteres_minimiser
        }
        profiles['π1'].update({
            critere: self.data[critere].min() for critere in self.criteres_maximiser
        })
        
        # π6 (best-case scenario)
        profiles['π6'] = {
            critere: self.data[critere].min() for critere in self.criteres_minimiser
        }
        profiles['π6'].update({
            critere: self.data[critere].max() for critere in self.criteres_maximiser
        })
        
        # Intermediate profiles (π2 to π5) using quantiles
        for i, q in enumerate(quantiles, 2):
            profile = {}
            for critere in self.criteres_minimiser:
                profile[critere] = self.data[critere].quantile(q)
            for critere in self.criteres_maximiser:
                profile[critere] = self.data[critere].quantile(1 - q)
            profiles[f'π{i}'] = profile
            
        self.profiles = profiles
        return profiles
    
    def calculate_profiles_clustering(self, num_profiles=5):
        """
        Méthode pour calculer les profils limites basé sur K-Means clustering
        """
        # Preprocess data
        self.preprocess_data()
        
        # Normalisation des données
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(self.data)
        
        # Apply KMeans to create the clusters
        kmeans = KMeans(n_clusters=num_profiles, random_state=42)
        kmeans.fit(data_scaled)
        
        # Cluster centers become the profile limits
        cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
        
        # Create a dictionary to store profiles
        profiles = {}
        for i in range(num_profiles):
            profiles[f'π{i+1}'] = dict(zip(self.data.columns, cluster_centers[i]))
        
        self.profiles = profiles
        return profiles

    def concordance_index(self, product, profile):
        """
        Calcul des indices de concordance partiels
        """
        concordance = {}
        
        for critere in self.criteres_minimiser:
            concordance[critere] = 1 if profile[critere] >= product[critere] else 0
            
        for critere in self.criteres_maximiser:
            concordance[critere] = 1 if product[critere] >= profile[critere] else 0
            
        return concordance
    
    def global_concordance(self, concordance_indices):
        """
        Calcul de l'indice de concordance global
        """
        return sum(
            concordance * weight 
            for concordance, weight in zip(
                concordance_indices.values(), 
                self.weights
            )
        ) / sum(self.weights)
    
    def outranks(self, product, profile):
        """
        Détermine si le produit surclasse le profil
        """
        concordance = self.concordance_index(product, profile)
        global_concordance = self.global_concordance(concordance)
        return global_concordance >= self.lambda_value
    
    def pessimistic_majority_sorting(self):
        """
        Procédure d'affectation pessimiste
        """
        results = []
        
        for _, product in self.data.iterrows():
            assigned = False
            for k in range(5, 0, -1):  # Assuming π5 is the best, π1 is the worst
                if self.outranks(product, self.profiles[f'π{k}']):
                    results.append(chr(64 + k))  # A=65, B=66, etc.
                    assigned = True
                    break
            if not assigned:
                results.append('E')
                
        self.data['classe_electre'] = results
        return self.data
    
    def optimistic_majority_sorting(self):
        """
        Procédure d'affectation optimiste
        """
        results = []
        
        for _, product in self.data.iterrows():
            assigned = False
            for k in range(1, 6):  # Assuming π1 is the worst, π5 is the best
                profile = self.profiles[f'π{k}']
                if (self.outranks(profile, product) and 
                    not self.outranks(product, profile)):
                    results.append(chr(64 + k - 1))
                    assigned = True
                    break
            if not assigned:
                results.append('A')
                
        self.data['classe_electre'] = results
        return self.data

# Initialize the ElectreTri class with the CSV file path and the weights
file_path = "dataPastas-refined-output.csv"
weights = [2, 2, 2, 2, 1, 1, 1, 4, 4]  # Example weights, adjust as needed

electre = ElectreTri(file_path, weights)

# Generate and display quantile-based profiles
print("Quantile-Based Profiles:")
quantile_profiles = electre.calculate_profiles_quantiles()
for profile_name, profile_values in quantile_profiles.items():
    print(f"{profile_name}:")
    for criterion, value in profile_values.items():
        print(f"  {criterion}: {value}")
    print()

# Generate and display clustering-based profiles
print("Clustering-Based Profiles:")
cluster_profiles = electre.calculate_profiles_clustering()
for profile_name, profile_values in cluster_profiles.items():
    print(f"{profile_name}:")
    for criterion, value in profile_values.items():
        print(f"  {criterion}: {value}")
    print()
