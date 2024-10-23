import json
import csv

def json_to_csv(json_file_path, csv_file_path, encoding='utf-8'):
    """
    Convert JSON to CSV, combining arrays into comma-separated strings.
    
    Args:
        json_file_path (str): Path to input JSON file
        csv_file_path (str): Path to output CSV file
        encoding (str): Character encoding of the input file (default: 'utf-8')
    """
    try:
        # Read JSON file with specified encoding
        with open(json_file_path, 'r', encoding=encoding) as json_file:
            data = json.load(json_file)
    except UnicodeDecodeError:
        # If UTF-8 fails, try with another common encoding
        with open(json_file_path, 'r', encoding='latin-1') as json_file:
            data = json.load(json_file)
    
    # Ensure data is a list of dictionaries
    if not isinstance(data, list):
        data = [data]
    
    # Process all values, converting arrays to comma-separated strings
    processed_data = []
    for item in data:
        processed_item = {}
        for key, value in item.items():
            if isinstance(value, list):
                processed_item[key] = ','.join(str(v) for v in value)
            else:
                processed_item[key] = value
        processed_data.append(processed_item)
    
    # Flatten 'nutriments' dictionary into the main item dictionary
    all_keys = set()
    for item in processed_data:
        if 'nutriments' in item:
            nutriments = item.pop('nutriments')
            item.update(nutriments)
        all_keys.update(item.keys())
    # Define the desired order of columns
    desired_order = [
        'code', 'product_name', 'labels', 'brands', 'categories_tags_en', 'nutriscore_grade', 'Yuka_score', 'Yuka_class', 'ecoscore_grade',
        'ingredients_text', 'additives_tags', 'additives_count',
        'image_url', 'image_small_url', 'image_front_url', 'image_front_small_url'
    ]

    # Ensure 'Yuka_score' and 'Yuka_class' are included in the data
    for item in processed_data:
        if 'Yuka_score' not in item:
            item['Yuka_score'] = ''
        if 'Yuka_class' not in item:
            item['Yuka_class'] = ''

    # Ensure all keys are included in the fieldnames
    fieldnames = [key for key in desired_order if key in all_keys]

    # Ensure 'Yuka_score' and 'Yuka_class' are included in the fieldnames
    if 'Yuka_score' not in fieldnames:
        fieldnames.append('Yuka_score')
    if 'Yuka_class' not in fieldnames:
        fieldnames.append('Yuka_class')
    
    # Add all remaining keys to the fieldnames
    remaining_keys = [key for key in all_keys if key not in fieldnames]
    fieldnames.extend(remaining_keys)
    
    # Write to CSV with UTF-8 encoding and BOM for Excel compatibility
    if processed_data:
        with open(csv_file_path, 'w', newline='', encoding='utf-8-sig') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(processed_data)

# Example usage
if __name__ == "__main__":        
    # Convert to CSV
    json_to_csv('pasta_products.json', 'output.csv')