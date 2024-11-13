import pandas as pd
import requests
import json
import time

class OpenFoodFactsClient:
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
        
    def _fetch_page(self, nutriscore=None, ecoscore=None, page=1):
        """Fetch products filtered by nutriscore or ecoscore from the OpenFoodFacts API"""
        params = {
            "categories_tags_en": "Pastas",
            "fields": self.required_fields,
            "page": page,
            "page_size": 50
        }
        if nutriscore:
            params["nutrition_grades_tags"] = nutriscore
        if ecoscore:
            params["ecoscore_tags"] = ecoscore

        try:
            response = requests.get(
                self.base_url,
                params=params,
                headers=self.headers
            )
            response.raise_for_status()
            products = response.json().get('products', [])
            
            # Add additives count and check for Bio label
            for product in products:
                # Count additives
                additives_tags = product.get('additives_tags', [])
                product['additives_count'] = len(additives_tags)
                
                # Check for Bio in labels
                labels = product.get('labels', '').split(',')
                product['is_bio'] = 'Bio' in labels
            
            return products
        except requests.exceptions.RequestException as e:
            print(f"Error fetching products: {e}")
            return []

    def collect_additional_products(self, existing_codes, current_nutriscore_e, current_ecoscore_e, nutriscore_target=60, ecoscore_target=30):
        """Collect additional products to meet the target count for Nutriscore E and Ecoscore E"""
        additional_products = []
        collected_nutriscore_e = current_nutriscore_e
        collected_ecoscore_e = current_ecoscore_e
        page = 1
        
        # Fetch products with Nutriscore E
        while collected_nutriscore_e < nutriscore_target:
            products = self._fetch_page(nutriscore='e', page=page)
            if not products:
                break

            for product in products:
                product_code = product.get('code')
                
                if product_code not in existing_codes:
                    additional_products.append(product)
                    existing_codes.add(product_code)
                    
                    # Count Nutriscore E products
                    if product.get('nutriscore_grade', '').lower() == 'e':
                        collected_nutriscore_e += 1
                        print(f"Collected Nutriscore E: {collected_nutriscore_e}")
                    
                    # Check if Ecoscore is also E for this product
                    if product.get('ecoscore_grade', '').lower() == 'e':
                        collected_ecoscore_e += 1
                        print(f"Collected Ecoscore E: {collected_ecoscore_e}")

                # Break if Nutriscore target is reached
                if collected_nutriscore_e >= nutriscore_target:
                    break
            page += 1
            time.sleep(1)

        # Reset page for Ecoscore E collection
        page = 1

        # Fetch products with Ecoscore E
        while collected_ecoscore_e < ecoscore_target:
            products = self._fetch_page(ecoscore='e', page=page)
            if not products:
                break
            
            for product in products:
                product_code = product.get('code')
                
                if product_code not in existing_codes:
                    additional_products.append(product)
                    existing_codes.add(product_code)
                    
                    # Count only Ecoscore E products
                    if product.get('ecoscore_grade', '').lower() == 'e':
                        collected_ecoscore_e += 1
                        print(f"Collected Ecoscore E: {collected_ecoscore_e}")

                if collected_ecoscore_e >= ecoscore_target:
                    break
            page += 1
            time.sleep(1)

        return additional_products

# Load existing data and track unique product codes
def load_existing_data(csv_file_path):
    df = pd.read_csv(csv_file_path)
    existing_codes = set(df['code'].astype(str))

    # Calculate current Nutriscore E and Ecoscore E counts
    current_nutriscore_e = df['nutriscore_grade'].str.upper().value_counts().get('E', 0)
    current_ecoscore_e = df['ecoscore_grade'].str.upper().value_counts().get('E', 0)

    return existing_codes, current_nutriscore_e, current_ecoscore_e

def save_additional_products_as_json(additional_products, filename="additional_pasta_products.json"):
    """Save additional products to a new JSON file"""
    if additional_products:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(additional_products, f, indent=2, ensure_ascii=False)
        print(f"Additional products saved to {filename}")
    else:
        print("No additional products to save.")

def main():
    existing_csv_path = 'dataPastas.csv'  # Original CSV file path
    additional_json_path = 'additional_pasta_products.json'  # New JSON file for additional products

    # Load existing product codes and current counts
    existing_codes, current_nutriscore_e, current_ecoscore_e = load_existing_data(existing_csv_path)

    # Instantiate OpenFoodFactsClient and collect additional products
    client = OpenFoodFactsClient()
    additional_products = client.collect_additional_products(
        existing_codes, 
        current_nutriscore_e, 
        current_ecoscore_e, 
        nutriscore_target=60, 
        ecoscore_target=30
    )

    # Save additional products to a new JSON file
    save_additional_products_as_json(additional_products, additional_json_path)

if __name__ == "__main__":
    main()
