"""
USCRN - Nettoyage et controle qualite.

Entree  : les fichiers CRNH0203-<annee>-<station>.txt
Sortie  : uscrn_clean.parquet  +  uscrn_rapport_qualite.csv

Etapes :
  1. Parsing avec les 38 colonnes officielles
  2. Horodatage UTC unique
  3. Remplacement des codes de valeur manquante par NaN
  4. Application des drapeaux qualite de la NOAA (flag != 0 -> valeur invalidee)
  5. Controles de plausibilite physique
  6. Controle de coherence verticale entre profondeurs de sol
  7. Controle de continuite temporelle
  8. Export

Lancer :  python clean_uscrn.py <dossier_des_txt>
"""

import os, sys, glob
import numpy as np
import pandas as pd

COLS = ["WBANNO","UTC_DATE","UTC_TIME","LST_DATE","LST_TIME","CRX_VN","LONGITUDE","LATITUDE",
        "T_CALC","T_HR_AVG","T_MAX","T_MIN","P_CALC","SOLARAD","SOLARAD_FLAG","SOLARAD_MAX",
        "SOLARAD_MAX_FLAG","SOLARAD_MIN","SOLARAD_MIN_FLAG","SUR_TEMP_TYPE","SUR_TEMP",
        "SUR_TEMP_FLAG","SUR_TEMP_MAX","SUR_TEMP_MAX_FLAG","SUR_TEMP_MIN","SUR_TEMP_MIN_FLAG",
        "RH_HR_AVG","RH_HR_AVG_FLAG","SOIL_MOISTURE_5","SOIL_MOISTURE_10","SOIL_MOISTURE_20",
        "SOIL_MOISTURE_50","SOIL_MOISTURE_100","SOIL_TEMP_5","SOIL_TEMP_10","SOIL_TEMP_20",
        "SOIL_TEMP_50","SOIL_TEMP_100"]

SM   = [f"SOIL_MOISTURE_{d}" for d in (5,10,20,50,100)]
ST   = [f"SOIL_TEMP_{d}"     for d in (5,10,20,50,100)]
TEMP = ["T_CALC","T_HR_AVG","T_MAX","T_MIN","SUR_TEMP","SUR_TEMP_MAX","SUR_TEMP_MIN"]
RAD  = ["SOLARAD","SOLARAD_MAX","SOLARAD_MIN"]

# Codes de valeur manquante, par famille de colonnes
SENTINELLES = {-9999.0: TEMP + ST + ["P_CALC","RH_HR_AVG"],
               -99999.0: RAD,
               -99.0:   SM}

# Drapeau qualite -> colonne(s) qu'il invalide. 0 = bon, tout le reste = suspect.
FLAGS = {"SOLARAD_FLAG":"SOLARAD", "SOLARAD_MAX_FLAG":"SOLARAD_MAX",
         "SOLARAD_MIN_FLAG":"SOLARAD_MIN", "SUR_TEMP_FLAG":"SUR_TEMP",
         "SUR_TEMP_MAX_FLAG":"SUR_TEMP_MAX", "SUR_TEMP_MIN_FLAG":"SUR_TEMP_MIN",
         "RH_HR_AVG_FLAG":"RH_HR_AVG"}

# Bornes physiques. Hors de ces bornes, la valeur est impossible.
BORNES = {**{c:(0.0, 0.70) for c in SM},        # teneur en eau volumique m3/m3
          **{c:(-60.0, 70.0) for c in TEMP+ST}, # degres Celsius
          **{c:(0.0, 1600.0) for c in RAD},     # W/m2
          "P_CALC": (0.0, 200.0),               # mm par heure
          "RH_HR_AVG": (0.0, 100.0)}            # pourcentage


def charger(chemin):
    df = pd.read_csv(chemin, sep=r"\s+", names=COLS, header=None,
                     dtype={"UTC_DATE": str, "UTC_TIME": str,
                            "LST_DATE": str, "LST_TIME": str,
                            "SUR_TEMP_TYPE": str, "CRX_VN": str})
    base = os.path.basename(chemin).replace(".txt", "").split("-")
    df["station"] = base[2]
    df["annee"] = int(base[1])
    return df


def nettoyer(df, journal):
    n = len(df)

    # 2. horodatage UTC
    df["time"] = pd.to_datetime(df.UTC_DATE + df.UTC_TIME.str.zfill(4),
                                format="%Y%m%d%H%M", utc=True).dt.tz_localize(None)

    # 3. codes de valeur manquante
    for code, cols in SENTINELLES.items():
        for c in cols:
            v = pd.to_numeric(df[c], errors="coerce")
            k = int((v == code).sum())
            if k: journal.append((df.station.iloc[0], df.annee.iloc[0], "manquante", c, k))
            df[c] = v.replace(code, np.nan)
    for c in ["LONGITUDE","LATITUDE"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 4. drapeaux qualite NOAA
    for fl, cible in FLAGS.items():
        mauvais = df[fl].astype(str).str.strip() != "0"
        k = int((mauvais & df[cible].notna()).sum())
        if k: journal.append((df.station.iloc[0], df.annee.iloc[0], "flag_qualite", cible, k))
        df.loc[mauvais, cible] = np.nan
    df = df.drop(columns=list(FLAGS.keys()))

    # 5. plausibilite physique
    for c, (lo, hi) in BORNES.items():
        hors = (df[c] < lo) | (df[c] > hi)
        k = int(hors.sum())
        if k: journal.append((df.station.iloc[0], df.annee.iloc[0], "hors_bornes", c, k))
        df.loc[hors, c] = np.nan

    # 6. coherence verticale : un saut a 5 cm sans reponse en dessous ET sans pluie
    #    pour l'expliquer est physiquement impossible. La pluie est le seul moteur
    #    naturel d'une hausse brutale, donc on l'exclut avant de signaler.
    d5  = df["SOIL_MOISTURE_5"].diff().abs()
    d10 = df["SOIL_MOISTURE_10"].diff().abs()
    pluie3h = df["P_CALC"].rolling(3, min_periods=1).sum()
    suspect = (d5 > 0.10) & (d10 < 0.01) & (pluie3h < 0.5)
    df["flag_incoherence_verticale"] = suspect.fillna(False)
    k = int(suspect.sum())
    if k: journal.append((df.station.iloc[0], df.annee.iloc[0], "incoherence_verticale",
                          "SOIL_MOISTURE_5", k))

    # 7. continuite temporelle
    df = df.sort_values("time")
    dup = int(df.time.duplicated().sum())
    if dup: journal.append((df.station.iloc[0], df.annee.iloc[0], "doublon_horaire", "time", dup))
    df = df.drop_duplicates(subset="time")
    attendu = pd.date_range(df.time.min(), df.time.max(), freq="h")
    manque = len(attendu) - len(df)
    if manque: journal.append((df.station.iloc[0], df.annee.iloc[0], "heure_absente", "time", manque))
    df = df.set_index("time").reindex(attendu).rename_axis("time").reset_index()
    df["station"] = df.station.ffill().bfill()
    df["annee"] = df.annee.ffill().bfill().astype(int)
    df["flag_incoherence_verticale"] = df.flag_incoherence_verticale.fillna(False).astype(bool)

    journal.append((df.station.iloc[0], df.annee.iloc[0], "lignes_brutes", "-", n))
    return df


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "."
    fichiers = sorted(glob.glob(os.path.join(src, "CRNH0203-*.txt")))
    if not fichiers:
        print(f"Aucun fichier CRNH0203-*.txt dans {src}"); sys.exit(1)

    journal, morceaux = [], []
    for f in fichiers:
        d = nettoyer(charger(f), journal)
        morceaux.append(d)
        print(f"  {os.path.basename(f)[:46]:46s} {len(d):5d} lignes")

    df = pd.concat(morceaux, ignore_index=True).sort_values(["station","time"])
    garder = (["time","station","annee","LATITUDE","LONGITUDE","T_HR_AVG","T_MAX","T_MIN",
               "P_CALC","SOLARAD","SUR_TEMP","RH_HR_AVG"] + SM + ST
              + ["flag_incoherence_verticale"])
    df = df[garder]
    df.to_parquet("uscrn_clean.parquet", index=False)

    rap = pd.DataFrame(journal, columns=["station","annee","probleme","colonne","n"])
    rap.to_csv("uscrn_rapport_qualite.csv", index=False)

    print(f"\n{len(df):,} lignes -> uscrn_clean.parquet")
    print("\n=== PROBLEMES PAR TYPE ===")
    print(rap[rap.probleme != "lignes_brutes"].groupby("probleme").n.sum().to_string())
    print("\n=== COMPLETUDE DE SOIL_MOISTURE_5 APRES NETTOYAGE ===")
    t = (df.groupby(["station","annee"]).SOIL_MOISTURE_5
           .apply(lambda s: round(s.notna().mean()*100, 1)).unstack())
    print(t.to_string())


if __name__ == "__main__":
    main()
