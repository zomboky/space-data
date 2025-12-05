# %%
import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset

# %%
# # ----------------- CECI EST POUR LE PREMIER FICHIER SEULEMENT -----------------
# # Définition du chemin d'accès au fichier unique
# base_folder = "Donnees_Spatiales"
# year_folder = "2020_NOAA_15"
# file_name = "poes_n15_20200101_proc.nc"
# file_path = os.path.join(base_folder, year_folder, file_name)

# # Liste pour accumuler les DataFrames (seulement un dans ce cas)
# df_grouped_grouped_list = []

# print(f"Traitement du fichier : {file_path}")

# try:
#     # Ouvre le fichier NetCDF
#     ds = Dataset(file_path, "r")
    
#     # Extrait les variables
#     data = {
#         "year": ds.variables["year"][:],
#         "day": ds.variables["day"][:],
#         "msec": ds.variables["msec"][:],
#         # Arrondi L_IGRF à une décimale
#         "L_IGRF": np.round(ds.variables["L_IGRF"][:], 1),
#         "p6": ds.variables["mep_pro_tel90_flux_p6"][:]
#     }
    
#     # Crée le DataFrame et l'ajoute à la liste
#     df_file = pd.DataFrame(data)
#     df_list.append(df_file)
    
#     # Ferme le fichier
#     ds.close()
# except FileNotFoundError:
#     print(f"Erreur : Le fichier {file_path} n'a pas été trouvé. Veuillez vérifier le chemin.")
# except Exception as e:
#     print(f"Erreur lors de la lecture du fichier {file_name}: {e}")

# # Concaténer tous les fichiers
# df = pd.concat(df_list, ignore_index=True)
# print("DataFrame global créé avec", len(df), "lignes.")

# %%
# ----------------- CECI EST POUR TOUTES LES ANNEES -----------------



# # Dossier principal contenant les sous-dossiers par année
base_folder = "Donnees_Spatiales"
subfolders = [os.path.join(base_folder, f"{year}_NOAA_15") for year in range(2020, 2025)]

# # Liste pour accumuler les DataFrames
df_list = []

for folder in subfolders:
    print(f"Traitement du dossier : {folder}")
    
    for file in os.listdir(folder):
        if file.endswith("_proc.nc"):
            file_path = os.path.join(folder, file)
            try:
                ds = Dataset(file_path, "r")
                
                data = {
                    "year": ds.variables["year"][:],
                    "day": ds.variables["day"][:],
                    "msec": ds.variables["msec"][:],
                    "L_IGRF": np.round(ds.variables["L_IGRF"][:], 1),
                    "p6": ds.variables["mep_pro_tel90_flux_p6"][:]
                }
                
                df_file = pd.DataFrame(data)
                df_list.append(df_file)
                
                ds.close()
            except Exception as e:
                print(f"Erreur sur le fichier {file}: {e}")

# Concaténer tous les fichiers
df = pd.concat(df_list, ignore_index=True)
print("DataFrame global créé avec", len(df), "lignes.")



# %%
# Convertir msec en timedelta correctement
time = pd.to_datetime(df['year'].astype(str), format='%Y') \
       + pd.to_timedelta(df['day'] - 1, unit='D') \
       + pd.to_timedelta(df['msec'], unit='ms')               #msec depuis minuit 


# %%
df['datetime'] = time

# %%
df.head(10)
#df.to_csv("Donnees_Spatiales/df_noaa15.csv", index=False)

# %% [markdown]
# On cherche à tracer le flux et L en fonction du temps : 
# 

# %%

import matplotlib.pyplot as plt

# %%
# Graphe pour L_IGRF
plt.figure(figsize=(18,4))
plt.plot(df['datetime'], df['L_IGRF'], color='blue', alpha=0.7)
plt.xlabel("Datetime")
plt.ylabel("L_IGRF")
plt.title("Évolution de L_IGRF dans le temps")
plt.ylim(0, 10)  # Échelle adaptée à L_IGRF
plt.show()

# Graphe pour p6 flux
plt.figure(figsize=(18,4))
plt.plot(df['datetime'], df['p6'], color='red', alpha=0.5)
plt.xlabel("Datetime")
plt.ylabel("p6 flux")
plt.title("Évolution du flux p6 dans le temps")
plt.show()

# %%
# import plotly.express as px

# # Définir la période de 20 jours
# start_time = df['datetime'].min()
# end_time = start_time + pd.Timedelta(days=20)
# df_20days = df[(df['datetime'] >= start_time) & (df['datetime'] <= end_time)]

# # Tracer le graphe interactif
# fig = px.line(df_20days, x='datetime', y='L_IGRF',
#               title=f"Évolution de L_IGRF sur 20 jours ({start_time.date()} → {end_time.date()})",
#               labels={'L_IGRF': 'L_IGRF', 'datetime': 'Date'})
# fig.update_yaxes(range=[0,10])  # Limiter l'échelle Y pour L
# fig.show()


# %%
# import sys
# !{sys.executable} -m pip install nbformat plotly


# %%


# Tracer
plt.figure(figsize=(18,4))
plt.plot(df['datetime'], df['p6'], color='red', alpha=0.5)
plt.xlabel("Datetime")
plt.ylabel("p6 flux")
plt.title("Évolution du flux p6 dans le temps (1 ≤ p6 ≤ 7.5)")
plt.show()

# %%
df.describe()

# %%
# Filtrer les outliers : ne garder que L_IGRF ≤ 10
df_filtered_L = df[df['L_IGRF'] <= 7]

# Tracer
plt.figure(figsize=(18,4))
plt.plot(df_filtered_L['datetime'], df_filtered_L['L_IGRF'], color='blue', alpha=0.7)
plt.xlabel("Datetime")
plt.ylabel("L_IGRF")
plt.title("Évolution de L_IGRF dans le temps (L_IGRF ≤ 7)")
plt.show()

# %% [markdown]
# On prend une valeur de temps toutes les 12H 
# à cette valeur de temps ex 2 janvier à 12h00 on prends le L_IGRF correspondant et on l'arrondi à 0.1 (exemple on a 2.9)
# ensuite on prend tous les p6 qui correspondnent à 2.9 et on en fait la moyenne donc on a une valeur de temps, une valeur de L_IGRF et une valeur de p6 

# %% [markdown]
# 

# %% [markdown]
# Donc je commence le filtrage selon ces instructions 
#  

# %%
# 1. Filtrage de 2.8 à 7
df_filtered = df[(df['L_IGRF'] <= 7) & (df['L_IGRF'] > 2.8)].copy()

# 2. On renomme la colonne
df_filtered["L_IGRF_rounded"] = df_filtered["L_IGRF"].round(1)


# %%
# On prend une valeur toutes les 12 heures

df_filtered['datetime_12H'] = df['datetime'].dt.floor('12h') 
df_filtered["datetime_12H"] = pd.to_datetime(df_filtered["datetime_12H"])

# 3. Regroupement par l'intervalle de 12H et par la valeur arrondie de L_IGRF, 
# puis calcul de la moyenne de p6.
df_grouped = df_filtered.groupby(["datetime_12H", "L_IGRF_rounded"])["p6"].mean().reset_index()


# 4. Renommer la colonne p6 agrégée et L_IGRF arrondie, puis trier
df_grouped = df_grouped.rename(columns={
    "p6": "p6_filtered",
    "L_IGRF_rounded": "L_IGRF"
})
# Trier par 'datetime_12H' puis par 'L_IGRF' dans l'ordre croissant.
df_filtered = df_filtered.sort_values(by=['datetime_12H', 'L_IGRF'])

# Afficher le résultat
print(df_filtered.head(100))

plt.plot(df_filtered['L_IGRF'], '.')



# %%
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

# 1. Tri correct du dataframe
df_grouped = df_grouped.sort_values(["datetime_12H", "L_IGRF"])

# 2. Calcul des quantiles pour la couleur (après filtrage + groupby)
vmin = df_grouped["p6_filtered"].quantile(0.05)
vmax = df_grouped["p6_filtered"].quantile(0.95)

# 3. Plot scatter
plt.figure(figsize=(15,3))

scatter = plt.scatter(
    df_grouped["datetime_12H"],
    df_grouped["L_IGRF"],
    c=df_grouped["p6_filtered"],
    cmap="jet",
    norm=LogNorm(vmin=vmin, vmax=vmax)
)

plt.colorbar(scatter, label="Flux P6 (log scale)")
plt.xlabel("Temps (par pas de 12h)")
plt.ylabel("L (arrondi à 0.1)")
plt.title("Diagramme L – Temps avec Flux P6 (spectre)")
plt.tight_layout()
plt.show()


# %% [markdown]
# il faut un dataframe avec le Bz Vitesse du vent solaire avec omni et -?- 
# 
# df-4-train = df(l=4)
# 
# 
# pour le training on prend en x_train(Vsw, Bz, F)  #le f c'est solar radio flux
#                              y_train(p6)
# modele = linéaire
# modele.fit(xtrain, ytrain)
# 
# y_predict = modele.predict(x_test) et on doit obtenir y_test 
# 
# on obtient une courbe avec un R²
# 
# 
# 
# Pour entrainer la prédiction prendre la valeur du flux y'a 12h
# 

# %%
import matplotlib.dates as mdates

# --- Calcul du seuil (95ème percentile) ---
threshold_95 = df_grouped['p6_filtered'].dropna().quantile(0.95)
print(f"Seuil 95ème percentile pour p6_filtered : {threshold_95:.3f}")

# --- Détection des événements ---
df_events = df_grouped[df_grouped['p6_filtered'] >= threshold_95].copy()
df_events = df_events.sort_values('datetime_12H')
print(f"Nombre d'événements détectés (p6_filtered ≥ 95e perc): {len(df_events)}")

# --- Affichage d'un petit extrait ---
print(df_events[['datetime_12H', 'L_IGRF', 'p6_filtered']].head(10))

# --- Tracé : p6_filtered (fond) + événements (points rouges) + ligne de seuil ---
plt.figure(figsize=(18,4))
# tracé complet du flux p6_filtered (faible alpha pour lisibilité)
plt.plot(df_grouped['datetime_12H'], df_grouped['p6_filtered'], color='lightgray', linewidth=0.6, label='p6 (moyenne 12h)', alpha=0.7)

# événements en points rouges
plt.scatter(df_events['datetime_12H'], df_events['p6_filtered'], color='red', s=16, label=f'Événements (≥ {threshold_95:.3f})')

# ligne horizontale du seuil
plt.axhline(threshold_95, color='k', linestyle='--', linewidth=1, label=f'95e perc = {threshold_95:.3f}')

# mise en forme temporelle de l'axe x
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
plt.xlabel("Datetime")
plt.ylabel("p6 flux (moyenne 12h)")
plt.title("Événements de flux p6 (≥ 95ᵉ percentile) dans le temps")
plt.legend()
plt.tight_layout()
plt.show()

# --- Tracé alternatif : uniquement les événements ---
plt.figure(figsize=(18,2))
plt.scatter(df_events['datetime_12H'], df_events['p6_filtered'], color='red', s=18)
plt.xlabel("Datetime")
plt.ylabel("p6_filtered (événements)")
plt.title("Événements p6 (seulement)")
plt.gca().xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
plt.tight_layout()
plt.show()

# %%


# %% [markdown]
# On remarque qu'il y a beaucoup trop d'évenements. On tente l'établissement d'un Z-score pour les filtrer. On teste plusieurs valeurs et on prendre finalement 1 sigma. 

# %%
from scipy import stats

z_scores = np.abs(stats.zscore(df_grouped['p6_filtered']))
outliers = df_grouped[z_scores > 1]  # seuil Z à 1 sigma
print(f"Nombre d'outliers détectés : {len(outliers)}")


# %% [markdown]
# On trace ces outliers : 

# %%
# --- Tracé complet du flux p6_filtered ---
plt.figure(figsize=(18,4))
plt.plot(df_grouped['datetime_12H'], df_grouped['p6_filtered'], color='lightgray', linewidth=0.6, alpha=0.7, label='p6 (moyenne 12h)')

# --- Tracé des outliers ---
plt.scatter(outliers['datetime_12H'], outliers['p6_filtered'], color='red', s=20, label=f'Outliers (Z>1)')

# --- Mise en forme temporelle de l'axe x ---
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))

# --- Labels et titre ---
plt.xlabel("Datetime")
plt.ylabel("p6 flux (moyenne 12h)")
plt.title("Outliers de flux p6 détectés par Z-score (>1σ)")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ON CHERCHE MAINTENANT A IMPORTER LES DONNES OMNI : BZ VITESSE_SOLAR_WIND ET F10.7 ON UTILISE SUNPY : 
# 

# %%
from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.timeseries import TimeSeries

range = a.Time('2020/01/01', '2024/12/31')
dataset = a.cdaweb.Dataset('OMNI2_H0_MRG1HR')
result = Fido.search(range, dataset)
print(result)


# %%
# Téléchargement des fichiers


dl_files = Fido.fetch(result, path=os.path.join("Donnees_Spatiales/omni", '{file}'))
print("Fichiers téléchargés :")
for f in dl_files:
    print(f)

# %%

# Création du TimeSeries et conversion en DataFrame


TIME = TimeSeries(dl_files, concatenate=True)
df_omni = TIME.to_dataframe()

# %%
df_omni.head()

# %%
df_omni.columns.tolist()

# %%
# On extrait les données nécessaires

df_omni = df_omni[["BZ_GSM", "V", "F10_INDEX"]]
df_omni.head()

# %% [markdown]
# Vérification des NaN : 
# 

# %%
df_omni.isna().any().any()

# %% [markdown]
# On a des Nan, on cherche à les supprimer

# %%
df_omni_clean = df_omni.dropna()
print(f"Ancien nombre de lignes : {len(df_omni)}, nouveau : {len(df_omni_clean)}")

# %% [markdown]
# On cherche des outliers dans le dataframe : 
# 
# 

# %%

# Colonnes à tracer
cols = df_omni.columns.tolist()  # ['BZ_GSM', 'V', 'F10_INDEX']

# Tracé
for col in cols:
    plt.figure(figsize=(18,4))
    plt.plot(df_omni.index, df_omni[col], color='blue', alpha=0.7)
    plt.xlabel("Datetime")
    plt.ylabel(col)
    plt.title(f"Évolution de {col} dans le temps")
    plt.tight_layout()
    plt.show()

# %%
print(df_omni.columns)
print(df_omni.index)


# %%
from scipy import stats

sig = 1.5

z_scores_V = np.abs(stats.zscore(df_omni_clean['V']))
omni_outliers = df_omni_clean[z_scores_V > sig]  
print(f"Nombre d'outliers détectés pour V: {len(omni_outliers)}")

z_scores_Bz = np.abs(stats.zscore(df_omni_clean['BZ_GSM']))
omni_outliers = df_omni_clean[z_scores_Bz > sig]  
print(f"Nombre d'outliers détectés pour Bz: {len(omni_outliers)}")

z_scores_f10 = np.abs(stats.zscore(df_omni_clean['F10_INDEX']))
omni_outliers = df_omni_clean[z_scores_f10 > sig]  
print(f"Nombre d'outliers détectés pour F10.7: {len(omni_outliers)}")

df_omni_filtered = df_omni_clean.copy()

df_omni_filtered = df_omni_filtered[
    (np.abs(stats.zscore(df_omni_filtered['V'], nan_policy='omit')) <= sig) &
    (np.abs(stats.zscore(df_omni_filtered['BZ_GSM'], nan_policy='omit')) <= sig) &
    (np.abs(stats.zscore(df_omni_filtered['F10_INDEX'], nan_policy='omit')) <= sig)
]

# %%
df_omni_filtered.head(10)

# %%

# Colonnes à tracer
cols = df_omni_filtered.columns.tolist()  # ['BZ_GSM', 'V', 'F10_INDEX']

# Tracé
for col in cols:
    plt.figure(figsize=(18,4))
    plt.plot(df_omni.index, df_omni[col], color='blue', alpha=0.7)
    plt.xlabel("Datetime")
    plt.ylabel(col)
    plt.title(f"Évolution de {col} dans le temps")
    plt.tight_layout()
    plt.show()

# %%
# Colonnes à analyser
cols = ["V", "BZ_GSM", "F10_INDEX"]

# Calcul des percentiles
percentiles = {col: df_omni_filtered[col].quantile(0.95) for col in cols}

# Affichage des stats
print("=== 95e percentiles ===")
for col, p95 in percentiles.items():
    print(f"{col} : {p95:.3f}")

# Tracer
for col in cols:
    p95 = percentiles[col]

    plt.figure(figsize=(18,4))
    plt.plot(df_omni_filtered.index, df_omni_filtered[col], label=col, alpha=0.6)
    
    # Ligne horizontale = seuil du 95e percentile
    plt.axhline(p95, color='red', linestyle='--', linewidth=2,
                label=f"95e percentile = {p95:.2f}")
    
    # Points > seuil (évènements)
    mask = df_omni_filtered[col] > p95
    plt.scatter(df_omni_filtered.index[mask],
                df_omni_filtered[col][mask],
                color='orange', s=15, label='Évènements > 95e percentile')
    
    plt.title(f"Évolution de {col} + seuil 95e percentile")
    plt.xlabel("Temps")
    plt.ylabel(col)
    plt.legend()
    plt.tight_layout()
    plt.show()

# %% [markdown]
# On assemble les données Omni et NOAA, on moyenne les données omni sur 12h : 

# %%
df_omni_filtered['time_12h'] = df_omni_filtered.index.floor("12h")

# %% [markdown]
# On compte combien de lignes NOAA avec des L différents existent pour chaque tranche de 12h

# %%
counts_per_timestamp = (
    df_grouped
    .groupby('datetime_12H')
    .size()
    .reset_index(name='count'))

counts_per_timestamp.head(10)

# %% [markdown]
# On merge ces "counts" dans omni : 

# %%
df_omni_with_counts = pd.merge(
    df_omni_filtered,
    counts_per_timestamp,
    left_on='time_12h',
    right_on='datetime_12H',
    how='left'
)


# %% [markdown]
# On duplique les lignes OMNI en fonction du nombre de points L de NOAA

# %%
# --- 1) Duplique OMNI par timestamp ----
df_omni_repeated = (
    df_omni_filtered
    .reindex(df_grouped['datetime_12H'].values, method='pad')
    .reset_index(drop=True)
)

# --- 2) Ajoute L directement depuis NOAA ---
df_omni_repeated['L_IGRF'] = df_grouped['L_IGRF'].values

# --- 3) Fusion parfaite des deux tableaux ---
df_merged = pd.concat(
    [df_grouped.reset_index(drop=True),
     df_omni_repeated.reset_index(drop=True)],
    axis=1
)

print(len(df_merged), "lignes fusionnées")


# %% [markdown]
# On trie les deux dataframe dans le meme ordre avant concaténation, NOAA par ordre de L_IGRF et OMNI par odre de temps
# 

# %%
df_noaa_sorted = df_grouped.sort_values(['datetime_12H', 'L_IGRF']).reset_index(drop=True)


# %%
df_omni_expanded = df_omni_expanded.sort_values('datetime_12H').reset_index(drop=True)

# %%
df_omni_expanded.head()

# %%
# %%
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# --- 1. Préparation des features et target ---

# Features Omni (déjà dupliquées pour correspondre à chaque L de NOAA)
X = df_omni_expanded[['V', 'BZ_GSM', 'F10_INDEX']].fillna(0)

# Target NOAA alignée
y = df_grouped['p6_filtered']

# --- 2. Split chronologique train/val/test 60/20/20 ---
n = len(X)
train_end = int(0.6 * n)
val_end   = int(0.8 * n)

X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
X_val, y_val     = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
X_test, y_test   = X.iloc[val_end:], y.iloc[val_end:]

# --- 3. Création du modèle pipeline et entraînement ---
model = Pipeline([
    ('scaler', StandardScaler()),
    ('reg', LinearRegression())
])
model.fit(X_train, y_train)

# --- 4. Prédictions ---
y_train_pred = model.predict(X_train)
y_val_pred   = model.predict(X_val)
y_test_pred  = model.predict(X_test)

# --- 5. Évaluation ---
print("R² train :", r2_score(y_train, y_train_pred))
print("R² val   :", r2_score(y_val, y_val_pred))
print("R² test  :", r2_score(y_test, y_test_pred))
print("RMSE test:", mean_squared_error(y_test, y_test_pred, squared=False))

# --- 6. Visualisation ---
plt.figure(figsize=(18,4))
plt.plot(y_test.reset_index(drop=True), label="Réel")
plt.plot(y_test_pred, label="Prévu", alpha=0.7)
plt.xlabel("Index")
plt.ylabel("p6_filtered")
plt.title("Prédiction du flux p6 sans lag")
plt.legend()
plt.show()



