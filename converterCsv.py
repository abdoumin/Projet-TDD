import json
import csv

def json_to_csv(json_file_path, csv_file_path, encoding='utf-8'):
    """
    Convert JSON to CSV, flattening nested dictionaries and structuring nutriments.
    
    Args:
        json_file_path (str): Path to input JSON file
        csv_file_path (str): Path to output CSV file
        encoding (str): Character encoding of the input file (default: 'utf-8')
    """
    desired_nutriments_order = ['iron_prepared_100g','sodium_100g','sodium_prepared_modifier','vitamin-d_prepared_100g','carbohydrates_serving','sugars_prepared_value','salt_prepared_100g','carbon-footprint-from-known-ingredients_serving','magnesium','carbohydrates_unit','fiber_modifier','energy-kj_prepared_unit','vitamin-b2_label','fiber_prepared_modifier','fr-acides-gras-insatures_label','trans-fat_label','potassium_prepared_100g','sugars_100g','calcium_unit','calcium_prepared_100g','phosphorus_100g','phosphorus_label','fruits-vegetables-nuts-estimate_prepared_unit','sugars_value','energy_value','nutrition-score-fr-producer_value','fr-acides-gras-insatures_unit','energy_prepared_100g','vitamin-b2_serving','trans-fat_prepared','trans-fat_prepared_unit','vitamin-d_prepared_serving','fiber_unit','fat_unit','added-sugars_prepared_100g','added-sugars_label','cholesterol_prepared_serving','fruits-vegetables-nuts-dried_serving','energy-kj_prepared','sugars_prepared_serving','potassium_serving','magnesium_100g','cholesterol_serving','added-sugars_prepared_serving','alcohol','fruits-vegetables-nuts-estimate_100g','saturated-fat_value','zinc_value','fruits-vegetables-nuts_serving','potassium_value','saturated-fat_prepared','saturated-fat_prepared_value','energy_prepared_unit','nutrition-score-fr','sugars_modifier','energy','saturated-fat_unit','proteins_prepared_value','energy-kcal_value','carbohydrates','energy-kcal_prepared_serving','calcium_prepared_serving','vitamin-d_prepared_value','saturated-fat_modifier','fiber_serving','proteins_unit','calcium_100g','fat_prepared_serving','energy_unit','fruits-vegetables-nuts-estimate_prepared_100g','alcohol_value','energy_prepared_value','alcohol_unit','potassium_prepared_serving','fruits-vegetables-nuts-estimate_value','fat_prepared_unit','fruits-vegetables-legumes-estimate-from-ingredients_serving','alcohol_prepared_100g','nutrition-score-fr-producer','energy-kcal_prepared_value','sugars_prepared','polyunsaturated-fat_prepared_100g','salt','salt_prepared_value','energy-kj_prepared_serving','sugars_prepared_unit','proteins_value','trans-fat_prepared_serving','sodium_prepared_unit','iron_prepared_unit','energy-kcal_prepared_100g','sodium','fruits-vegetables-nuts_unit','salt_value','fr-acides-gras-insatures','vitamin-pp_100g','added-sugars_100g','carbon-footprint-from-known-ingredients_100g','energy_serving','energy-kcal_prepared_unit','trans-fat_100g','calcium_prepared','energy_prepared_serving','fruits-vegetables-nuts-estimate_serving','calcium_prepared_value','iron_prepared_value','fruits-vegetables-nuts-estimate-from-ingredients_100g','magnesium_unit','vitamin-b1_100g','vitamin-b1_value','saturated-fat_prepared_modifier','fruits-vegetables-nuts-dried_value','energy-kj_value','added-sugars','energy-kj_unit','vitamin-b1','fiber_prepared_100g','sodium_unit','iron_value','energy-kj_prepared_value','salt_prepared_modifier','sugars_serving','alcohol_prepared_serving','polyunsaturated-fat_label','vitamin-d_serving','sodium_value','magnesium_value','potassium_prepared','proteins_prepared_unit','calcium_value','phosphorus_unit','sodium_prepared_value','vitamin-d_prepared','energy-kcal','fruits-vegetables-nuts_100g','added-sugars_prepared_value','fruits-vegetables-nuts-estimate','fiber_prepared_unit','alcohol_prepared_unit','sugars_prepared_100g','potassium_prepared_unit','iron_prepared','vitamin-b1_label','fruits-vegetables-nuts-estimate_prepared','vitamin-b2_100g','carbon-footprint-from-meat-or-fish_serving','proteins_serving','monounsaturated-fat_prepared','cholesterol','zinc','proteins_prepared','calcium','polyunsaturated-fat_prepared_serving','carbon-footprint-from-meat-or-fish_product','energy-kj_100g','fat_100g','energy_100g','salt_prepared_unit','fat_prepared','sugars','energy_prepared','magnesium_serving','zinc_serving','potassium_unit','vitamin-pp_serving','vitamin-b2','sodium_prepared_100g','monounsaturated-fat_prepared_serving','polyunsaturated-fat_prepared_value','vitamin-d','sodium_prepared','energy-kcal_unit','trans-fat_prepared_value','alcohol_prepared','carbohydrates_prepared_unit','iron_label','alcohol_100g','vitamin-b1_unit','fruits-vegetables-nuts-dried_unit','salt_prepared_serving','trans-fat_value','fruits-vegetables-nuts','fruits-vegetables-nuts-dried_100g','saturated-fat_100g','cholesterol_prepared_unit','salt_serving','added-sugars_prepared_unit','fruits-vegetables-nuts-estimate_prepared_value','alcohol_prepared_value','carbon-footprint-from-meat-or-fish_100g','proteins_prepared_100g','fiber_100g','cholesterol_prepared','sodium_serving','trans-fat_serving','fiber_prepared','energy-kcal_100g','fiber_value','cholesterol_prepared_100g','nutrition-score-fr_100g','fat_serving','added-sugars_value','phosphorus','is_bio','cholesterol_value','energy-kj_prepared_100g','energy-kj','carbohydrates_prepared','salt_100g','calcium_prepared_unit','polyunsaturated-fat_prepared','vitamin-pp_value','fat','iron_serving','trans-fat_prepared_100g','fat_prepared_value','fr-acides-gras-insatures_100g','vitamin-d_prepared_unit','energy-kcal_prepared','fruits-vegetables-nuts-estimate-from-ingredients_serving','vitamin-pp_label','nova-group_serving','sodium_prepared_serving','monounsaturated-fat_prepared_value','monounsaturated-fat_label','saturated-fat_prepared_serving','carbon-footprint-from-known-ingredients_product','iron_unit','fruits-vegetables-nuts-estimate_prepared_serving','vitamin-d_value','saturated-fat_prepared_100g','proteins','energy-kcal_value_computed','fruits-vegetables-nuts_value','carbohydrates_prepared_value','phosphorus_value','fiber_prepared_serving','fr-acides-gras-insatures_value','trans-fat_unit','iron','phosphorus_serving','calcium_label','vitamin-b2_unit','energy-kcal_serving','vitamin-b1_serving','polyunsaturated-fat_prepared_unit','fruits-vegetables-nuts-estimate_label','zinc_label','proteins_100g','cholesterol_prepared_value','zinc_100g','salt_unit','fruits-vegetables-nuts-estimate_unit','nova-group_100g','carbohydrates_100g','nutrition-score-fr-producer_unit','added-sugars_unit','fiber_label','zinc_unit','vitamin-d_100g','energy-kj_serving','vitamin-pp_unit','carbohydrates_prepared_serving','magnesium_label','proteins_prepared_serving','potassium_label','cholesterol_unit','saturated-fat_serving','fruits-vegetables-legumes-estimate-from-ingredients_100g','monounsaturated-fat_prepared_unit','fruits-vegetables-nuts_label','cholesterol_100g','nutrition-score-fr-producer_100g','potassium_prepared_value','trans-fat','fat_prepared_100g','iron_100g','potassium','sodium_modifier','vitamin-b2_value','energy-kj_value_computed','cholesterol_label','vitamin-pp','monounsaturated-fat_prepared_100g','alcohol_serving','salt_modifier','calcium_serving','fruits-vegetables-nuts-dried','salt_prepared','carbohydrates_prepared_100g','added-sugars_prepared','nutrition-score-fr-producer_serving','saturated-fat_prepared_unit','saturated-fat','vitamin-d_label','fiber_prepared_value','alcohol_prepared_modifier','potassium_100g','nova-group','added-sugars_serving','fat_prepared_modifier','iron_prepared_serving','sugars_unit','fiber','vitamin-d_unit','carbohydrates_value','fat_value']
    
    try:
        with open(json_file_path, 'r', encoding=encoding) as json_file:
            data = json.load(json_file)
    except UnicodeDecodeError:
        with open(json_file_path, 'r', encoding='latin-1') as json_file:
            data = json.load(json_file)
    
    if not isinstance(data, list):
        data = [data]
    
    processed_data = []
    for item in data:
        processed_item = {}
        for key, value in item.items():
            if isinstance(value, list):
                processed_item[key] = ','.join(str(v) for v in value)
            else:
                processed_item[key] = value
        
        # Flatten and order nutriments according to desired_nutriments_order
        if 'nutriments' in processed_item:
            nutriments = processed_item.pop('nutriments')
            ordered_nutriments = {key: nutriments.get(key, '') for key in desired_nutriments_order}
            processed_item.update(ordered_nutriments)
        
        processed_data.append(processed_item)
    
    all_keys = set()
    for item in processed_data:
        all_keys.update(item.keys())
    
    # Desired column order with additional risk fields
    desired_order = [
        'code', 'Yuka_score', 'Yuka_class', 'Nbr additifs à risque', 'Nbr additifs à risque modéré',
        'Nbr additifs à risque limité', 'Nbr additifs sans risque','additives_tags', 'additives_count', 'product_name', 'labels', 
        'brands', 'categories_tags_en', 'nutriscore_grade', 'ecoscore_grade', 'ingredients_text', 
         'image_url', 'image_small_url', 'image_front_url', 
        'image_front_small_url'
    ] + desired_nutriments_order  # Include nutriments in the final order
    
    # Fill missing fields in each item
    for item in processed_data:
        for field in ['Yuka_score', 'Yuka_class', 'Nbr additifs à risque', 
                      'Nbr additifs à risque modéré', 'Nbr additifs à risque limité', 
                      'Nbr additifs sans risque']:
            if field not in item:
                item[field] = ''
    
    # Ensure fieldnames cover all keys in data
    fieldnames = desired_order + [key for key in all_keys if key not in desired_order]
    
    if processed_data:
        with open(csv_file_path, 'w', newline='', encoding='utf-8-sig') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(processed_data)

# Example usage
if __name__ == "__main__":        
    json_to_csv('additional_pasta_products.json', 'outputAddtional2.csv')
