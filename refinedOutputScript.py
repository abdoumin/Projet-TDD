import csv
import json

def filter_csv_columns(csv_data, columns):
    filtered_data = [{col: row[col] for col in columns if col in row} for row in csv_data]
    return filtered_data

def csv_to_csv(input_csv_file, output_csv_file, columns):
    with open(input_csv_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        csv_data = list(reader)

    filtered_data = filter_csv_columns(csv_data, columns)

    with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        writer.writerows(filtered_data)

if __name__ == "__main__":
    # Example CSV input
    columns_to_keep = ["code","product_name","nutriscore_grade","ecoscore_grade","is_bio","Yuka_score","Yuka_class","fruits-vegetables-nuts-estimate-from-ingredients_serving","sugars_100g","energy_value","energy_unit","energy-kcal_value","saturated-fat_value","sodium_100g","proteins_100g","fiber_100g","fruits-vegetables-nuts-estimate-from-ingredients_serving","Nbr additifs à risque","Nbr additifs à risque modéré","Nbr additifs à risque limité","Nbr additifs sans risque","additives_count"]
    input_csv_file = 'dataPastas-outputAddtional2.csv'
    output_csv_file = 'dataPastas-refined-outputAddtional2.csv'

    csv_to_csv(input_csv_file, output_csv_file, columns_to_keep)
