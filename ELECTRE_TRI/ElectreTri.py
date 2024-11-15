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
                profile[critere] = self.data[critere].quantile(1-q)
            for critere in self.criteres_maximiser:
                profile[critere] = self.data[critere].quantile(q)
            profiles[f'π{i}'] = profile
            
        self.profiles = profiles
        return profiles
        
    def calculate_profiles_knn(self):
        """
        Calculate six profiles based on existing Nutri-Score categories and display them in the specified format.
        """

        # Preprocess data to ensure it's ready for profile calculation
        self.preprocess_data()

        # Define an empty dictionary to store profiles
        profiles = {}
        
        # Define Nutri-Score categories in order from worst ('E') to best ('A')
        classes = ['E', 'D', 'C', 'B', 'A']
        
        # Calculate the centroid for each Nutri-Score category
        for i, classe in enumerate(classes, 1):
            # Filter data for each Nutri-Score category
            mask = (self.data['nutriscore_grade'].str.upper() == classe)
            if mask.any():  # Check if there are items in this category
                # Calculate the mean values for all relevant features in this category, excluding 'nutriscore_grade'
                centroid = self.data[mask][self.criteres_minimiser + self.criteres_maximiser].mean()
                # Store the centroid as a profile
                profiles[f'π{i}'] = centroid

        # Create a sixth profile (π6) based on the 'A' profile with further enhanced values
        if 'π5' in profiles:
            best_profile = profiles['π5'].copy()
            for criterion in self.criteres_minimiser:
                best_profile[criterion] = max(self.data[criterion].min(), best_profile[criterion] * 0.8)  # Enhance minimization
            for criterion in self.criteres_maximiser:
                best_profile[criterion] = min(self.data[criterion].max(), best_profile[criterion] * 1.2)  # Enhance maximization
            profiles['π6'] = best_profile        
        # Save profiles to the instance for future use
        self.profiles = profiles
        return profiles
    
    def calculate_profiles_clustering(self, num_profiles=6):
        """
        Calculate profiles based on K-Means clustering and display them in the specified format.
        
        Args:
            num_profiles (int): Number of profiles to create (default is 6).
        
        Returns:
            dict: A dictionary with profiles as centroids of each cluster.
        """

        # Preprocess data to ensure it's ready for clustering
        self.preprocess_data()
        
        # Normalize the data for consistent clustering
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(self.data[self.criteres_minimiser + self.criteres_maximiser])

        # Apply K-Means clustering to create the specified number of clusters
        kmeans = KMeans(n_clusters=num_profiles, random_state=42)
        kmeans.fit(data_scaled)

        # Retrieve the cluster centers (centroids) and transform them back to the original scale
        cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
        
        # Create profiles from cluster centers and order them from worst to best
        profiles = {}
        for i, center in enumerate(sorted(cluster_centers, key=lambda x: -x[0]), 1):  # Sort by a criterion for ordering
            profiles[f'π{i}'] = dict(zip(self.data.columns, center))
        
        # Save profiles to the instance for future use
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
        Calcul de l'indice de concordance global en associant chaque critère à son poids correct
        """
        # Liste des poids correspondant aux critères
        weights = [2, 2, 2, 2, 1, 1, 1, 4, 4] # Correspondant aux critères dans l'ordre
        
        # Liste des noms des critères dans le même ordre que les poids
        criteria_order = [
            'energy-kcal_value', 'saturated-fat_value', 'sugars_100g', 'sodium_100g',
            'proteins_100g', 'fiber_100g', 'fruits-vegetables-nuts-estimate-from-ingredients_serving',
            'additives_count', 'Nbr additifs à risque'
        ]
        
        # Calcul de la somme pondérée des indices de concordance
        concordance_sum = sum(
            concordance_indices[critere] * weight
            for critere, weight in zip(criteria_order, weights)
        )
        
        # Retourner l'indice de concordance global normalisé par la somme des poids
        return concordance_sum / sum(weights)

    
    def outranks(self, product, profile):
        """
        Détermine si le produit surclasse le profil
        """
        concordance = self.concordance_index(product, profile)
        global_concordance = self.global_concordance(concordance)
        return global_concordance >= self.lambda_value
    
    def pessimistic_majority_sorting(self, product, profiles):
        """
        Perform pessimistic majority sorting for a single product and return the assigned class.
        
        Args:
            product (pd.Series): The product data.
            profiles (dict): The profiles to be used for classification (KNN or quantile-based).
        
        Returns:
            str: The assigned class for the product.
        """
        for k in range(5, 0, -1):  # Start with π5 as the best (A), π1 as the worst (E)
            if self.outranks(product, profiles[f'π{k}']):
                return chr(69 - k + 1)  # E=69, D=68, C=67, B=66, A=65
        return 'E'  # Default to 'E' if no profile is outranked
    
    def optimistic_majority_sorting(self, product, profiles):
        """
        Perform optimistic majority sorting for a single product and return the assigned class.
        
        Args:
            product (pd.Series): The product data.
            profiles (dict): The profiles to be used for classification (KNN or quantile-based).
        
        Returns:
            str: The assigned class for the product.
        """
        for k in range(1, 6):  # Start with π1 as the worst (E), moving up to π5 as the best (A)
            if not self.outranks(product,profiles[f'π{k}']):
                return chr(69 - k + 1)  # E=69, D=68, C=67, B=66, A=65
        return 'A'  # Default to 'A' if no profile outranks the product
    def process_csv_with_classes(self, input_file, knn_profiles, quantile_profiles,majority_sorting_method,output_file="output_with_classes.csv"):
        """
        Process a CSV file, classify products using KNN and quantile profiles, and add 'knn_class' and 'quantile_class' columns.
        
        Args:
            input_file (str): Path to the input CSV file.
            knn_profiles (dict): KNN-based profiles.
            quantile_profiles (dict): Quantile-based profiles.
            output_file (str): Path to save the processed CSV with added columns.
        """
        # Read the input CSV file
        self.data = pd.read_csv(input_file)
        
        # Add columns for classifications
        knn_classes = []
        quantile_classes = []
        
        # Process each product in the dataset
        for _, product in self.data.iterrows():
            # Determine the class using KNN-based profiles with pessimistic majority sorting
            knn_class = majority_sorting_method(product, knn_profiles)
            knn_classes.append(knn_class)
            
            # Determine the class using quantile-based profiles with pessimistic majority sorting
            quantile_class = majority_sorting_method(product, quantile_profiles)
            quantile_classes.append(quantile_class)
        
        # Add the classes to the DataFrame
        self.data['knn_class'] = knn_classes
        self.data['quantile_class'] = quantile_classes
        
        # Save the updated DataFrame to a new CSV file
        self.data.to_csv(output_file, index=False)
        print(f"Processed file saved to {output_file}")


# Method to save profiles to a file with UTF-8 encoding
# def save_profiles_to_file(profiles, filename="profiles_output.txt"):
#     with open(filename, "w", encoding="utf-8") as f:
#         for profile_name, profile_values in profiles.items():
#             f.write(f"{profile_name}:\n")
#             for criterion, value in profile_values.items():
#                 f.write(f"  {criterion}: {value}\n")
#             f.write("\n")  # Add a blank line between profiles
#     print(f"Profiles saved to {filename}")
    
def save_profiles_to_csv(profiles, filename="profiles_output.csv"):
    # Convert profiles dictionary to a DataFrame
    profiles_df = pd.DataFrame(profiles).T
    # Add a column for profile names
    profiles_df.insert(0, 'profile', profiles_df.index)
    # Save DataFrame to CSV
    profiles_df.to_csv(filename, index=False, encoding="utf-8")
    print(f"Profiles saved to {filename}")

# # Save profiles to CSV files
# save_profiles_to_csv(quantile_profiles, "quantiles_profiles_output.csv")
# save_profiles_to_csv(knn_profiles, "knn_profiles_output.csv")



# Initialize the ElectreTri class with the CSV file path and the weights
file_path = "Output-Database.csv"
weights = [2, 2, 2, 2, 1, 1, 1, 4, 4]  # Adjust weights as needed
lambda_values = [0.5, 0.6, 0.7]  # Lambda values to test

electre = ElectreTri(file_path, weights, None)
# Generate quantile and KNN profiles
quantile_profiles = electre.calculate_profiles_quantiles()
knn_profiles = electre.calculate_profiles_clustering()

save_profiles_to_csv(quantile_profiles,"quantiles_profiles_output.txt")
save_profiles_to_csv(knn_profiles,"knn_profiles_output.txt")
# Generate profiles and process files for each lambda value
for lambda_value in lambda_values:
    # Instantiate the ElectreTri class with the current lambda value
    
    electre = ElectreTri(file_path, weights, lambda_value)

    # Define the output file name based on the lambda value
    output_file = f"output_with_classes_lambda_pessimestic_{lambda_value}.csv"
    
    # Process the CSV and save the classified data to the Excel file
    electre.process_csv_with_classes(file_path, knn_profiles, quantile_profiles,electre.pessimistic_majority_sorting,output_file=output_file)
for lambda_value in lambda_values:
    # Instantiate the ElectreTri class with the current lambda value
    
    electre = ElectreTri(file_path, weights, lambda_value)

    # Define the output file name based on the lambda value
    output_file = f"output_with_classes_lambda_optimistic_{lambda_value}.csv"
    
    # Process the CSV and save the classified data to the Excel file
    electre.process_csv_with_classes(file_path, knn_profiles, quantile_profiles,electre.optimistic_majority_sorting,output_file=output_file)


