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
    
    # Write to CSV with UTF-8 encoding and BOM for Excel compatibility
    if processed_data:
        fieldnames = processed_data[0].keys()
        with open(csv_file_path, 'w', newline='', encoding='utf-8-sig') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(processed_data)

# Example usage
if __name__ == "__main__":
    # Example JSON data
    example_json = """[
        {
            "name": "John",
            "skills": ["Python", "JavaScript", "SQL"],
            "age": 30
        },
        {
            "name": "Jane",
            "skills": ["Java", "C++"],
            "age": 28
        }
    ]"""
    
    # Save example JSON to a file
    with open('example.json', 'w', encoding='utf-8') as f:
        f.write(example_json)
    
    # Convert to CSV
    json_to_csv('pasta_products.json', 'output.csv')