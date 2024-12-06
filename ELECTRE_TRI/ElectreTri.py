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
    def categorize_super_nutri_score(self, quantile_class, ecoscore_grade, is_bio):
        # Normalize input to handle potential case variations
        pessimist = quantile_class.upper()
        eco_score = str(ecoscore_grade).upper()
        bio = str(is_bio).upper()

        # Check for data completeness
        if pd.isna(pessimist) or pd.isna(eco_score) or pd.isna(bio):
            return "Non évaluable"

        # More granular scoring logic
        score_mapping = {
            "Excellent": [
                (pessimist == 'A' and eco_score == 'A'),
            (pessimist == 'A' and eco_score == 'B' and bio == 'TRUE'),
            #(pessimist == 'A' and bio == 'TRUE')
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

        # Iterate through categories in order of preference
        for category, conditions in score_mapping.items():
            if any(conditions):
                return category

        return "Mauvais"  # Default case

        
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
        
    def calculate_profiles_knn(self, k=5, num_profiles=6):
        """
        Calculates profiles using true K-Nearest Neighbors algorithm.
        
        Args:
            k: Number of nearest neighbors to consider
            num_profiles: Number of profiles to generate (default 6)
        
        Returns:
            Dictionary of ordered profiles based on KNN
        """
        self.preprocess_data()
        
        # Initialize profiles dictionary
        profiles = {}
        
        # Standardize the data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(self.data[self.criteres_minimiser + self.criteres_maximiser])
        
        # Create KNN model
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(data_scaled)
        
        # Calculate reference points for each profile
        reference_points = []
        
        # Create reference points for each profile
        for i in range(num_profiles):
            ratio = i / (num_profiles - 1)  # Creates ratios from 0 to 1
            ref_point = []
            
            # For minimize criteria: worst (high) to best (low)
            for critere in self.criteres_minimiser:
                max_val = self.data[critere].max()
                min_val = self.data[critere].min()
                ref_point.append(max_val - (max_val - min_val) * ratio)
                
            # For maximize criteria: worst (low) to best (high)
            for critere in self.criteres_maximiser:
                max_val = self.data[critere].max()
                min_val = self.data[critere].min()
                ref_point.append(min_val + (max_val - min_val) * ratio)
                
            reference_points.append(ref_point)
        
        # Scale reference points
        reference_points_scaled = scaler.transform(reference_points)
        
        # Find k nearest neighbors for each reference point
        for i, ref_point in enumerate(reference_points_scaled, 1):
            # Find indices of k nearest neighbors
            distances, indices = knn.kneighbors([ref_point])
            
            # Get the actual data points for these neighbors
            neighbors = self.data.iloc[indices[0]]
            
            # Create profile based on the average of neighbors
            profile = {}
            for critere in self.criteres_minimiser + self.criteres_maximiser:
                profile[critere] = neighbors[critere].mean()
                
            profiles[f'π{i}'] = profile
        
        # Ensure monotonicity
        self._enforce_monotonicity(profiles)
        
        return profiles

    def _enforce_monotonicity(self, profiles):
        """
        Enforces monotonicity constraints on profiles.
        
        Args:
            profiles: Dictionary of profiles to adjust
        """
        # Iterate through profiles from π1 to π5
        for i in range(1, 6):
            current_profile = profiles[f'π{i}']
            next_profile = profiles[f'π{i+1}']
            
            # Enforce constraints for minimize criteria
            for critere in self.criteres_minimiser:
                if current_profile[critere] < next_profile[critere]:
                    # Take the average to maintain some of the KNN characteristics
                    avg = (current_profile[critere] + next_profile[critere]) / 2
                    current_profile[critere] = avg
                    next_profile[critere] = avg
            
            # Enforce constraints for maximize criteria
            for critere in self.criteres_maximiser:
                if current_profile[critere] > next_profile[critere]:
                    # Take the average to maintain some of the KNN characteristics
                    avg = (current_profile[critere] + next_profile[critere]) / 2
                    current_profile[critere] = avg
                    next_profile[critere] = avg
    
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
            super_nutri_score = self.categorize_super_nutri_score(
                quantile_class,  # Use quantile-based classification for pessimistic
                product['ecoscore_grade'],  # Assuming 'ecoscore_grade' is in the dataset
                product['is_bio']  # Assuming 'is_bio' is in the dataset
            )
            super_nutri_scores.append(super_nutri_score)
        
        # Add the classes to the DataFrame
        self.data['knn_class'] = knn_classes
        self.data['quantile_class'] = quantile_classes
        self.data['super_nutri_score'] = super_nutri_scores
        # Save the updated DataFrame to a new CSV file
        self.data.to_csv(output_file, index=False)
        print(f"Processed file saved to {output_file}")

        # Save the Super-NutriScore classifications to a text file
        with open("super_nutri_scores.txt", "w", encoding="utf-8") as f:
            for index, row in self.data.iterrows():
                f.write(f"Product {index + 1}: {row['super_nutri_score']}\n")
        print("Super-NutriScore classifications saved to super_nutri_scores.txt")

        # Save the DataFrame to an Excel file
        output_excel_file = "super_nutri_scores.xlsx"
        columns_to_save = ['code', 'ecoscore_grade', 'is_bio', 'knn_class', 'quantile_class', 'super_nutri_score']
        self.data[columns_to_save].to_excel(output_excel_file, index=False)
        print(f"Data saved to {output_excel_file}")



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
file_path = "DB_cereales_Groupe_Allani.csv"
weights = [2, 2, 2, 2, 1, 1, 1, 4, 4]  # Adjust weights as needed
lambda_values = [0.5, 0.6, 0.7]  # Lambda values to test

electre = ElectreTri(file_path, weights, None)
# Generate quantile and KNN profiles
quantile_profiles = electre.calculate_profiles_quantiles()
knn_profiles = electre.calculate_profiles_knn()

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


# List of input files
input_files = [
    "output_with_classes_lambda_optimistic_0.5.csv",
    "output_with_classes_lambda_optimistic_0.6.csv",
    "output_with_classes_lambda_optimistic_0.7.csv",
    "output_with_classes_lambda_pessimestic_0.5.csv",
    "output_with_classes_lambda_pessimestic_0.6.csv",
    "output_with_classes_lambda_pessimestic_0.7.csv"
]

def generate_profile_counts(input_file):
    """
    Generate count statistics for both KNN and quantile methods from an input file
    """
    # Read the input file
    df = pd.read_csv(input_file)
    
    # Get counts for KNN method
    knn_counts = df['knn_class'].value_counts().reset_index()
    knn_counts.columns = ['Profile', 'Count']
    knn_counts = knn_counts.sort_values('Profile')
    
    # Get counts for quantile method
    quantile_counts = df['quantile_class'].value_counts().reset_index()
    quantile_counts.columns = ['Profile', 'Count']
    quantile_counts = quantile_counts.sort_values('Profile')
    
    # Generate output filenames
    base_name = input_file.replace('.csv', '')
    knn_output = f"{base_name}_knn_counts.csv"
    quantile_output = f"{base_name}_quantile_counts.csv"
    
    # Save to CSV files
    knn_counts.to_csv(knn_output, index=False)
    quantile_counts.to_csv(quantile_output, index=False)
    
    print(f"Generated count files for {input_file}:")
    print(f"- {knn_output}")
    print(f"- {quantile_output}")

# Process all input files
for file in input_files:
    generate_profile_counts(file)
    
import pandas as pd
import glob

def combine_count_files():
    # Initialize an empty list to store all dataframes
    all_dfs = []
    
    # Lambda values and methods
    lambda_values = ['0.5', '0.6', '0.7']
    approaches = ['optimistic', 'pessimestic']
    methods = ['knn', 'quantile']
    
    # Process each combination
    for approach in approaches:
        for lambda_val in lambda_values:
            for method in methods:
                # Construct filename pattern
                filename = f"output_with_classes_lambda_{approach}_{lambda_val}_{method}_counts.csv"
                
                try:
                    # Read the CSV file
                    df = pd.read_csv(filename)
                    
                    # Add columns to identify the source
                    df['Approach'] = approach
                    df['Lambda'] = lambda_val
                    df['Method'] = method
                    
                    # Rename columns for clarity
                    df = df.rename(columns={'Profile': 'Class'})
                    
                    # Add to list of dataframes
                    all_dfs.append(df)
                    
                except FileNotFoundError:
                    print(f"Warning: File not found - {filename}")
    
    # Combine all dataframes
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Reorder columns for better readability
        column_order = ['Class', 'Count', 'Approach', 'Lambda', 'Method']
        combined_df = combined_df[column_order]
        
        # Sort the dataframe
        combined_df = combined_df.sort_values(['Approach', 'Lambda', 'Method', 'Class'])
        
        # Save to CSV
        output_filename = 'combined_profile_counts.csv'
        combined_df.to_csv(output_filename, index=False)
        print(f"Combined results saved to {output_filename}")
        
        # Display summary
        print("\nSummary of combined results:")
        print(combined_df.to_string())
    else:
        print("No files were found to combine")

# Run the combination
combine_count_files()