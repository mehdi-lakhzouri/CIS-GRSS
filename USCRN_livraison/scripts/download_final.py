"""
USCRN - Telechargement des 4 annees pour les 5 stations retenues.

A lancer APRES select_stations.py, une fois les 5 stations choisies.
Remplis la liste STATIONS ci-dessous avec les noms exacts du CSV.

Lancer :  python download_final.py
"""

import os, time, shutil
import requests

BASE = "https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02"
ANNEES = [2022, 2023, 2024, 2025]
DEST = "USCRN_final"
CACHE = "uscrn_cache"   # reutilise les fichiers deja telecharges par select_stations.py

# >>> A REMPLIR avec les 5 stations choisies (noms exacts, sans CRNH0203- ni .txt)
STATIONS = [
    # --- les 5 stations principales, une par climat ---
    # (disponibilite moyenne de SOIL_MOISTURE_5 sur 2022-2025)
    "AZ_Tucson_11_W",      # aride                 97.6 %
    "OK_Goodwell_2_SE",    # plaines semi-arides   98.9 %   <- promue (ex-jumelle)
    "MS_Newton_5_ENE",     # subtropical humide   100.0 %   <- remplace AL_Selma_13_WNW
    "NY_Ithaca_13_E",      # continental froid     97.9 %
    "CA_Fallbrook_5_NE",   # mediterraneen        100.0 %

    # --- jumelle : 3.8 km de Goodwell_2_SE, controle de coherence spatiale ---
    "OK_Goodwell_2_E",     #                       90.2 %  (panne janv-avr 2022)

    # --- ECARTEES apres verification sur les 4 annees ---
    # AL_Selma_13_WNW : panne sept 2023 -> fevr 2024, 85.3 % de moyenne
    # AL_Selma_6_SSE  : aucun rayonnement ni temperature de surface
    # AL_Clanton_2_NE : idem, et 0 % de sol en 2022 et 2023
    # GA_Newton_8_W   : 0 % en 2022
]
def main():
    if not STATIONS:
        print("Remplis d'abord la liste STATIONS en haut du fichier."); return
    os.makedirs(DEST, exist_ok=True)
    total = len(STATIONS) * len(ANNEES)
    n = 0
    for st in STATIONS:
        for an in ANNEES:
            n += 1
            nom = f"CRNH0203-{an}-{st}.txt"
            out = os.path.join(DEST, nom)
            if os.path.exists(out):
                print(f"[{n}/{total}] deja la   : {nom}"); continue
            # deja present dans le cache de la selection ? on copie au lieu de retelecharger
            cache = os.path.join(CACHE, nom)
            if os.path.exists(cache):
                shutil.copy2(cache, out)
                print(f"[{n}/{total}] depuis cache: {nom}"); continue
            try:
                r = requests.get(f"{BASE}/{an}/{nom}", timeout=300)
                r.raise_for_status()
                with open(out, "wb") as f:
                    f.write(r.content)
                print(f"[{n}/{total}] OK {nom}  ({len(r.content)/1e6:.1f} Mo)")
            except Exception as e:
                print(f"[{n}/{total}] ECHEC {nom} : {e}")
            time.sleep(0.3)
    print(f"\nFini. Fichiers dans ./{DEST}/")

if __name__ == "__main__":
    main()
