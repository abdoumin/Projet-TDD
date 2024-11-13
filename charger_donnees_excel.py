import pandas as pd
import numpy as np

criteres = {
        'a_minimiser': [
            'energy', 
            'acides_gras_saturés', 
            'sucres', 
            'sodium', 
            'nombre_additifs'
        ],
        'a_maximiser': [
            'proteines', 
            'fibres', 
            'fruits_legumes'
        ]
    }

def construire_profils_quantiles(df, criteres):
    """Construit les profils limites en utilisant les quantiles"""
    profils = {}
    quantiles = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    for q_index, quantile in enumerate(quantiles, 1):
        profil = {}
        
        # Critères à minimiser
        for critere in criteres['a_minimiser']:
            profil[critere] = df[critere].quantile(quantile)
        
        # Critères à maximiser
        for critere in criteres['a_maximiser']:
            profil[critere] = df[critere].quantile(1 - quantile)
        
        profils[f'π{q_index}'] = profil
    
    return profils

def charger_donnees(chemin_fichier):
    """Charge les données depuis un fichier Excel"""
    try:
        df = pd.read_excel(chemin_fichier)
        return df
    except Exception as e:
        print(f"Erreur de chargement : {e}")
        return None


construire_profils_quantiles(charger_donnees("dataPastas.xlsx"),criteres=criteres)


# data = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
# quantiles = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
# quantile_values = np.quantile(data, quantiles)
# print(quantile_values)
