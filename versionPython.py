##TO DO LIST
#1. Verify that the products are truly pasta products.
#2. Add a column for count the additives.
#3. Add a column for Yuka Score.

import requests
import json
from collections import defaultdict
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
        
    def _fetch_page(self, nutriscore, page=1):
        """Fetch a single page of products with specific nutriscore"""
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
            print(f"Error fetching page {page} for nutriscore {nutriscore}: {e}")
            return []
    def _has_valid_images(self, product):
        """Check if product has at least one valid image"""
        image_fields = ['image_url', 'image_small_url', 'image_front_url', 'image_front_small_url']
        return any(product.get(field) for field in image_fields)

    def _has_valid_ecoscore(self, product):
        """Check if product has a valid ecoscore (A-E)"""
        return product.get('ecoscore_grade', '').lower() in ['a', 'b', 'c', 'd', 'e']

    def _collect_products_by_nutriscore(self, nutriscore, target_count):
        """Collect specified number of products for a given nutriscore"""
        collected_products = []
        page = 1
        
        while len(collected_products) < target_count and page <= 10:  # Limit to 10 pages
            products = self._fetch_page(nutriscore, page)
            if not products:
                break
                
            for product in products:
                if (self._has_valid_images(product) and 
                    self._has_valid_ecoscore(product) and 
                    len(collected_products) < target_count):
                    collected_products.append(product)
            
            page += 1
            time.sleep(1)  # Be nice to the API
            
        return collected_products

    def collect_all_products(self):
        """Collect all required products according to specifications"""
        first_150 = []
        bio_count = 0  # Track total bio products
        target_bio = 75  # Target number of bio products
        
        # First 150 products (30 each of A-E)
        for grade in ['a', 'b', 'c', 'd', 'e']:
            print(f"\nCollecting products with Nutriscore {grade.upper()}...")
            products = self._collect_products_by_nutriscore(grade, 30)
            
            # Count bio products in this batch
            for product in products:
                if product['is_bio']:
                    bio_count += 1
            
            first_150.extend(products)
            print(f"Collected {len(products)} products with Nutriscore {grade.upper()} (Bio products so far: {bio_count})")
        
        # Remaining 150 products (any distribution)
        print("\nCollecting remaining products...")
        remaining_150 = []
        page = 1
        
        while len(remaining_150) < 150 and page <= 20:  # Limit to 20 pages
            for grade in ['a', 'b', 'c', 'd', 'e']:
                products = self._fetch_page(grade, page)
                
                for product in products:
                    if (self._has_valid_images(product) and 
                        self._has_valid_ecoscore(product) and 
                        product not in first_150 and 
                        product not in remaining_150 and 
                        len(remaining_150) < 150):
                        
                        # If we need more bio products, only add bio products
                        if bio_count < target_bio:
                            if product['is_bio']:
                                remaining_150.append(product)
                                bio_count += 1
                        # If we have enough bio products, only add non-bio products
                        elif bio_count == target_bio:
                            if not product['is_bio']:
                                remaining_150.append(product)
                        # If we somehow have too many bio products, only add non-bio products
                        else:
                            if not product['is_bio']:
                                remaining_150.append(product)
                
                time.sleep(1)  # Be nice to the API
            page += 1

        total_products = first_150 + remaining_150
        final_bio_count = sum(1 for p in total_products if p['is_bio'])
        print(f"\nFinal collection complete:")
        print(f"Total products: {len(total_products)}")
        print(f"Total Bio products: {final_bio_count}")
        
        return total_products

    def save_products(self, products, filename="pasta_products.json"):
        """Save collected products to a JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(products, f, indent=2, ensure_ascii=False)
        print(f"\nProducts saved to {filename}")

    def print_statistics(self, products):
        """Print distribution statistics for collected products"""
        def count_distribution(product_list):
            dist = defaultdict(int)
            for product in product_list:
                grade = product.get('nutriscore_grade', '').lower()
                dist[grade] += 1
            return dict(dist)

        first_150_dist = count_distribution(products['first_150'])
        remaining_dist = count_distribution(products['remaining_150'])

        print("\nProduct Distribution:")
        print("\nFirst 150 products:")
        for grade in ['a', 'b', 'c', 'd', 'e']:
            print(f"Nutriscore {grade.upper()}: {first_150_dist.get(grade, 0)}/30")

        print("\nRemaining 150 products:")
        for grade in sorted(remaining_dist.keys()):
            print(f"Nutriscore {grade.upper()}: {remaining_dist.get(grade, 0)}")

def main():
    client = OpenFoodFactsClient()
    print("Starting to collect pasta products...")
    
    products = client.collect_all_products()
    client.save_products(products)
    
    # if len(products['first_150']) == 150 and len(products['remaining_150']) == 150:
    # else:
    #     print("\nFailed to collect required number of products:")
    #     print(f"First 150: {len(products['first_150'])} collected")
    #     print(f"Remaining 150: {len(products['remaining_150'])} collected")

if __name__ == "__main__":
    main()