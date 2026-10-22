"""
USCRN - Selection automatique des 5 stations pour le challenge TSYP14 Track 2.

Ce script :
  1. liste les stations presentes dans les 4 annees 2022-2025
  2. ecarte celles sans capteur d'humidite du sol
  3. les regroupe par climat
  4. mesure la disponibilite reelle de SOIL_MOISTURE_5
  5. sort un classement, 2 meilleures candidates par climat

Lancer :  python select_stations.py
Besoin  :  pip install pandas requests
Duree   :  ~15-25 min (il telecharge une annee par candidate)
"""

import os, re, io, sys, time
import requests
import pandas as pd

BASE = "https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02"
ANNEES = [2022, 2023, 2024, 2025]
ANNEE_TEST = 2025          # annee utilisee pour le pre-test de disponibilite
CACHE = "uscrn_cache"
os.makedirs(CACHE, exist_ok=True)

COLS = ["WBANNO","UTC_DATE","UTC_TIME","LST_DATE","LST_TIME","CRX_VN","LONGITUDE",
        "LATITUDE","T_CALC","T_HR_AVG","T_MAX","T_MIN","P_CALC","SOLARAD",
        "SOLARAD_FLAG","SOLARAD_MAX","SOLARAD_MAX_FLAG","SOLARAD_MIN","SOLARAD_MIN_FLAG",
        "SUR_TEMP_TYPE","SUR_TEMP","SUR_TEMP_FLAG","SUR_TEMP_MAX","SUR_TEMP_MAX_FLAG",
        "SUR_TEMP_MIN","SUR_TEMP_MIN_FLAG","RH_HR_AVG","RH_HR_AVG_FLAG",
        "SOIL_MOISTURE_5","SOIL_MOISTURE_10","SOIL_MOISTURE_20","SOIL_MOISTURE_50",
        "SOIL_MOISTURE_100","SOIL_TEMP_5","SOIL_TEMP_10","SOIL_TEMP_20","SOIL_TEMP_50",
        "SOIL_TEMP_100"]

# Regroupement climatique par code d'etat.
# Objectif : diversite environnementale (cf. doc "Update datasets").
CLIMATS = {
    "1_aride":            ["AZ", "NV", "NM", "UT"],
    "2_plaines_semiaride":["KS", "NE", "CO", "OK", "SD", "WY", "MT", "ND"],
    "3_subtropical_humide":["AL", "GA", "SC", "NC", "MS", "LA", "FL", "AR", "TN", "VA"],
    "4_continental_froid":["MN", "MI", "WI", "ME", "NY", "NH", "VT", "IA", "IL", "IN", "OH", "PA"],
    "5_mediterraneen":    ["CA", "OR", "WA", "ID"],
}
ETAT2CLIMAT = {e: c for c, lst in CLIMATS.items() for e in lst}

# L'Alaska et Hawaii n'ont pas de sondes d'humidite du sol exploitables.
EXCLUS = {"AK", "HI"}


def lister_stations(annee):
    """Noms de stations presentes dans le dossier d'une annee."""
    r = requests.get(f"{BASE}/{annee}/", timeout=120)
    r.raise_for_status()
    noms = re.findall(rf"CRNH0203-{annee}-([A-Za-z0-9_.\-]+)\.txt", r.text)
    return set(noms)


def telecharger(station, annee):
    """Telecharge (et met en cache) un fichier station-annee."""
    nom = f"CRNH0203-{annee}-{station}.txt"
    chemin = os.path.join(CACHE, nom)
    if not os.path.exists(chemin):
        r = requests.get(f"{BASE}/{annee}/{nom}", timeout=300)
        r.raise_for_status()
        with open(chemin, "wb") as f:
            f.write(r.content)
    return chemin


def profiler(chemin):
    """Disponibilite de SOIL_MOISTURE_5 et quelques stats."""
    df = pd.read_csv(chemin, sep=r"\s+", names=COLS, header=None,
                     dtype={"LST_DATE": str, "UTC_DATE": str})
    sm = pd.to_numeric(df["SOIL_MOISTURE_5"], errors="coerce").replace(-99.0, pd.NA)
    pr = pd.to_numeric(df["P_CALC"], errors="coerce").replace(-9999.0, pd.NA)
    t = pd.to_numeric(df["T_HR_AVG"], errors="coerce").replace(-9999.0, pd.NA)
    return {
        "lignes": len(df),
        "sm_dispo_%": round(sm.notna().mean() * 100, 1),
        "sm_min": None if sm.notna().sum() == 0 else round(float(sm.min()), 3),
        "sm_max": None if sm.notna().sum() == 0 else round(float(sm.max()), 3),
        "pluie_an_mm": None if pr.notna().sum() == 0 else round(float(pr.sum()), 0),
        "temp_moy_C": None if t.notna().sum() == 0 else round(float(t.mean()), 1),
        "lat": round(float(df["LATITUDE"].iloc[0]), 2),
        "lon": round(float(df["LONGITUDE"].iloc[0]), 2),
    }


def main():
    print("1/4  Lecture des listings annuels...")
    presentes = None
    for a in ANNEES:
        s = lister_stations(a)
        print(f"     {a} : {len(s)} stations")
        presentes = s if presentes is None else (presentes & s)
    print(f"     -> {len(presentes)} stations presentes les 4 annees\n")

    print("2/4  Filtrage par etat et climat...")
    candidates = []
    for st in sorted(presentes):
        etat = st.split("_")[0]
        if etat in EXCLUS or etat not in ETAT2CLIMAT:
            continue
        candidates.append((ETAT2CLIMAT[etat], st))
    for c in sorted(CLIMATS):
        n = sum(1 for g, _ in candidates if g == c)
        print(f"     {c:24s} : {n} candidates")
    print(f"     -> {len(candidates)} candidates au total\n")

    print(f"3/4  Test de disponibilite sur {ANNEE_TEST} (telechargement)...")
    lignes = []
    for i, (climat, st) in enumerate(candidates, 1):
        try:
            p = telecharger(st, ANNEE_TEST)
            info = profiler(p)
            info.update(climat=climat, station=st)
            lignes.append(info)
            drapeau = "OK " if info["sm_dispo_%"] >= 90 else "-- "
            print(f"  {drapeau}[{i:3d}/{len(candidates)}] {st[:42]:42s} "
                  f"sol={info['sm_dispo_%']:5.1f}%")
        except Exception as e:
            print(f"  !! [{i:3d}/{len(candidates)}] {st[:42]:42s} erreur: {e}")
        time.sleep(0.2)

    res = pd.DataFrame(lignes)
    if res.empty:
        print("Aucune donnee recuperee."); sys.exit(1)

    res = res[["climat","station","sm_dispo_%","sm_min","sm_max",
               "pluie_an_mm","temp_moy_C","lat","lon","lignes"]]
    res = res.sort_values(["climat","sm_dispo_%"], ascending=[True, False])
    res.to_csv("uscrn_candidates.csv", index=False)

    print("\n4/4  RESULTAT — 2 meilleures par climat\n")
    print("=" * 100)
    for c in sorted(CLIMATS):
        sub = res[(res.climat == c) & (res["sm_dispo_%"] >= 90)].head(2)
        print(f"\n### {c}")
        if sub.empty:
            print("   aucune station au-dessus de 90% — baisser le seuil ou elargir les etats")
        else:
            print(sub.drop(columns=["climat"]).to_string(index=False))
    print("\n" + "=" * 100)
    print("\nClassement complet -> uscrn_candidates.csv")
    print("Choisis 1 station par climat, puis telecharge ses 4 annees.")


if __name__ == "__main__":
    main()
