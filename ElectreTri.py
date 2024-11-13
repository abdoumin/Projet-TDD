import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class ElectreTri:
    def __init__(self, file_path, weights, lambda_value=0.6, k=5):
        """
        Initialisation d'ELECTRE TRI
        """
        self.data = pd.read_csv(file_path)
        self.weights = weights
        self.lambda_value = lambda_value
        self.k = k  # Number of nearest neighbors for KNN-based profiles
        self.profiles = None
        
        # Define the criteria to minimize and maximize based on updated instructions
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
        """
        Preprocess the data to ensure energy values are in kcal and to prepare relevant columns.
        """
        # Ensure energy values are in kcal
        if 'energy-kcal_value' in self.data.columns:
            self.data['energy_value_kcal'] = self.data.apply(
                lambda row: row['energy-kcal_value'] if pd.notnull(row['energy-kcal_value']) else (
                    row['energy_value'] * 0.239006 if row['energy_unit'] == 'kJ' else row['energy_value']
                ), axis=1
            )
        else:
            # Convert 'energy_value' to kcal if 'energy-kcal_value' is not present
            self.data['energy_value_kcal'] = self.data.apply(
                lambda row: row['energy_value'] * 0.239006 if row['energy_unit'] == 'kJ' else row['energy_value'],
                axis=1
            )
        
        # Filter data to only include necessary columns and drop missing values
        self.data = self.data[
            self.criteres_minimiser + self.criteres_maximiser + ['nutriscore_grade']
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
    
    # def calculate_profiles_knn(self):
    #     """
    #     Méthode pour calculer les profils limites basé sur K-Nearest Neighbors
    #     """
    #     self.preprocess_data()
        
    #     # Normalize data for consistent distance measurements
    #     scaler = StandardScaler()
    #     data_scaled = scaler.fit_transform(self.data[self.criteres_minimiser + self.criteres_maximiser])
    
        
    #     profiles = {}
    #     categories = self.data['nutriscore_grade'].unique()
        
    #     for category in categories:
    #         # Filter data for the current category
    #         category_data = self.data[self.data['nutriscore_grade'] == category]
    #         # Drop the 'nutriscore_grade' column to ensure only numeric data remains
    #         numeric_category_data = category_data[self.criteres_minimiser + self.criteres_maximiser]

    #         if len(numeric_category_data) < self.k:
    #             print(f"Not enough data points in category {category} for KNN with k={self.k}.")
    #             continue
            
    #         # Find k-nearest neighbors within the category
    #         knn = NearestNeighbors(n_neighbors=self.k)
    #         category_data_scaled = scaler.transform(category_data[self.criteres_minimiser + self.criteres_maximiser])
    #         knn.fit(category_data_scaled)
            
    #         # Compute the mean of the k-nearest neighbors as the profile for this category
    #         distances, indices = knn.kneighbors(category_data_scaled)
    #         profile_values = numeric_category_data.iloc[indices.flatten()].mean()
            
    #         # Store the profile in the original scale
    #         profiles[f'π_{category}'] = profile_values.to_dict()
        
    #     self.profiles = profiles
    #     return profiles
    
    def calculate_profiles_knn(self, output_file="profiles.csv"):
        """
        Calculate profiles based on existing Nutri-Score categories and save to a CSV file.
        
        Args:
            output_file (str): Path to save the calculated profiles.
        """
        
        self.preprocess_data()

        # Define an empty dictionary to store profiles
        profiles = {}
        
        # Define Nutri-Score categories (from worst 'E' to best 'A')
        classes = ['E', 'D', 'C', 'B', 'A']
        
        # Calculate the centroid for each Nutri-Score category
        for i, classe in enumerate(classes, 1):
            # Filter data for each Nutri-Score category
            mask = (self.data['nutriscore_grade'].str.upper() == classe)
            if mask.any():  # Check if there are items in this category
                # Calculate the mean values for all relevant features in this category, excluding 'nutriscore_grade'
                centroid = self.data[mask][self.criteres_minimiser + self.criteres_maximiser].mean()
                # Store the centroid as a profile, with π1 being the worst (E) and π5 the best (A)
                profiles[f'π{i}'] = centroid
        
        # Convert profiles dictionary to a DataFrame for easier saving
        profiles_df = pd.DataFrame(profiles).T  # Transpose for better layout
        
        # Save the profiles DataFrame to a CSV file
        profiles_df.to_csv(output_file, index=True)
        
        # Save profiles to the instance for future use
        self.profiles = profiles_df
        print(f"Profiles saved to {output_file}")
        return profiles_df

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

# Method to save profiles to a file with UTF-8 encoding
def save_profiles_to_file(profiles, filename="profiles_output.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        for profile_name, profile_values in profiles.items():
            f.write(f"{profile_name}:\n")
            for criterion, value in profile_values.items():
                f.write(f"  {criterion}: {value}\n")
            f.write("\n")  # Add a blank line between profiles
    print(f"Profiles saved to {filename}")


# Initialize the ElectreTri class with the CSV file path and the weights
file_path = "dataPastas-refined-output.csv"
weights = [2, 2, 2, 2, 1, 1, 1, 4, 4]  # Example weights, adjust as needed

electre = ElectreTri(file_path, weights)

# Generate quantile-based profiles and save them to a file
quantile_profiles = electre.calculate_profiles_quantiles()
save_profiles_to_file(quantile_profiles, filename="quantile_profiles_output.txt")

# Calculate KNN-based profiles
knn_profiles = electre.calculate_profiles_knn()
save_profiles_to_file(knn_profiles, filename="cluster_profiles_output.txt")
