"""
USCRN - Verification d'une liste de stations sur les 4 annees 2022-2025.

Le test de selection ne portait que sur 2025. Ce script telecharge les 4 annees
de chaque candidate et mesure la disponibilite reelle de SOIL_MOISTURE_5,
annee par annee et par periode du decoupage (train / validation / test).

Lancer :  python verify_4years.py
Duree  :  ~5 min pour une dizaine de candidates
"""

import os, time
import numpy as np
import pandas as pd
import requests

BASE = "https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02"
ANNEES = [2022, 2023, 2024, 2025]
CACHE = "uscrn_verif"
os.makedirs(CACHE, exist_ok=True)

# Remplacants possibles pour le climat subtropical humide.
# Classes par qualite 2025 et par amplitude d'humidite du sol.
CANDIDATES = [
    "AL_Clanton_2_NE",           # 1387 mm, amplitude 0.098-0.583 (la plus large)
    "AL_Gainesville_2_NE",       # 1425 mm, 0.047-0.493
    "MS_Newton_5_ENE",           # 1334 mm, 0.074-0.383
    "LA_Monroe_26_N",            # 1422 mm, 0.033-0.395
    "AR_Batesville_8_WNW",       # 1157 mm, 0.073-0.450
    "NC_Durham_11_W",            # 1214 mm, 0.071-0.437
    "SC_Blackville_3_W",         # 1210 mm, 0.073-0.353
    "AL_Northport_2_S",          # 1128 mm, 0.126-0.470
    "GA_Newton_8_W",             # 879 mm  -- a une jumelle a 13.5 km
    "GA_Newton_11_SW",           # 1031 mm -- jumelle de la precedente
]

COLS = ["WBANNO","UTC_DATE","UTC_TIME","LST_DATE","LST_TIME","CRX_VN","LONGITUDE","LATITUDE",
        "T_CALC","T_HR_AVG","T_MAX","T_MIN","P_CALC","SOLARAD","SOLARAD_FLAG","SOLARAD_MAX",
        "SOLARAD_MAX_FLAG","SOLARAD_MIN","SOLARAD_MIN_FLAG","SUR_TEMP_TYPE","SUR_TEMP",
        "SUR_TEMP_FLAG","SUR_TEMP_MAX","SUR_TEMP_MAX_FLAG","SUR_TEMP_MIN","SUR_TEMP_MIN_FLAG",
        "RH_HR_AVG","RH_HR_AVG_FLAG","SOIL_MOISTURE_5","SOIL_MOISTURE_10","SOIL_MOISTURE_20",
        "SOIL_MOISTURE_50","SOIL_MOISTURE_100","SOIL_TEMP_5","SOIL_TEMP_10","SOIL_TEMP_20",
        "SOIL_TEMP_50","SOIL_TEMP_100"]


def recuperer(station, annee):
    nom = f"CRNH0203-{annee}-{station}.txt"
    for dossier in (CACHE, "uscrn_cache", "USCRN_final"):
        p = os.path.join(dossier, nom)
        if os.path.exists(p):
            return p
    p = os.path.join(CACHE, nom)
    r = requests.get(f"{BASE}/{annee}/{nom}", timeout=300)
    r.raise_for_status()
    with open(p, "wb") as f:
        f.write(r.content)
    time.sleep(0.2)
    return p


def dispo(chemin, colonne="SOIL_MOISTURE_5", code=-99.0):
    df = pd.read_csv(chemin, sep=r"\s+", names=COLS, header=None)
    v = pd.to_numeric(df[colonne], errors="coerce").replace(code, np.nan)
    autres = {c: round(pd.to_numeric(df[c], errors="coerce")
                       .replace({-9999.0: np.nan, -99999.0: np.nan}).notna().mean()*100, 1)
              for c in ("P_CALC", "SOLARAD", "SUR_TEMP")}
    return round(v.notna().mean()*100, 1), autres


def main():
    lignes = []
    for i, st in enumerate(CANDIDATES, 1):
        r = {"station": st}
        ok = True
        for an in ANNEES:
            try:
                p = recuperer(st, an)
                d, autres = dispo(p)
                r[an] = d
                if an == 2025:
                    r.update({f"{k}_25": v for k, v in autres.items()})
            except Exception as e:
                r[an] = np.nan; ok = False
                print(f"  !! {st} {an} : {e}")
        if ok:
            r["TRAIN"] = round((r[2022] + r[2023]) / 2, 1)
            r["moyenne"] = round(np.mean([r[a] for a in ANNEES]), 1)
            etat = "OK " if min(r[a] for a in ANNEES) >= 95 else "-- "
            print(f"  {etat}[{i:2d}/{len(CANDIDATES)}] {st[:30]:30s} "
                  + "  ".join(f"{a}={r[a]:5.1f}" for a in ANNEES)
                  + f"   moy={r['moyenne']:5.1f}")
        lignes.append(r)

    t = pd.DataFrame(lignes)
    cols = ["station"] + ANNEES + ["TRAIN", "moyenne", "P_CALC_25", "SOLARAD_25", "SUR_TEMP_25"]
    t = t[[c for c in cols if c in t.columns]].sort_values("moyenne", ascending=False)
    t.to_csv("verif_4ans.csv", index=False)

    print("\n" + "=" * 92)
    print("CLASSEMENT (toutes les annees comptent, le minimum decide)\n")
    print(t.to_string(index=False))
    print("\nRetenir la premiere station dont AUCUNE annee ne descend sous 95 %.")
    print("Detail complet -> verif_4ans.csv")


if __name__ == "__main__":
    main()
