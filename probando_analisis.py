import pandas as pd
import numpy as np
from datetime import datetime
from matplotlib.pylab import plt
import seaborn as sns

# import matplotlib.ticker as mticker
# from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
import matplotlib.gridspec as gridspec

# from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib as mpl
from scipy import stats

matplotlib.rcParams["text.usetex"] = True
cmap = plt.colormaps["viridis"]  # plasma
viridis = mpl.colormaps["plasma"]
# matplotlib.rcParams["text.usetex"] = True
variables_contaminacion = {
    "co_concentracion": "CO",
    "direccion_viento": "Dirección del Viento",
    "humedad_relativa": "Humedad Relativa",
    "humedad_relativa_10m": "Humedad Relativa 10 m",
    "humedad_relativa_2m": "Humedad Relativa 2 m",
    "no_concentracion": "NO",
    "no2_concentracion": "NO2",
    "o3_concentracion": "O3",
    "pm10_concentracion": "PM10",
    "pm25_concentracion": "PM2.5",
    "pst_concentracion": "PST",
    "precipitacion_liquida": "Precipitación Líquida",
    "presion_atmosferica": "Presión Atmosférica",
    "radiacion_solar_global": "Radiación Solar Global",
    "radiacion_uvb": "Radiación UVB",
    "so2_concentracion": "SO2",
    "temperatura": "Temperatura",
    "temperatura_10m": "Temperatura a 10 m",
    "temperatura_2m": "Temperatura a 2 m",
    "velocidad_viento": "Velocidad del Viento",
}
id_municipal = [
    5001,
    5308,
    5360,
    5631,
    8001,
    8758,
    11001,
    15491,
    15759,
    47001,
    47189,
    68001,
    76001,
    76520,
    76892,
    81001,
    85001,
]

sizeclas = []
variable_climaticas = []
for us, ot in enumerate(list(variables_contaminacion.keys())):
    # for municipio in id_municipal:
    # variable_climaticas.append(list(variables_contaminacion.values())[us])
    datf = pd.read_csv("datos/" + ot + ".csv")
    # datf = datf.dropna()
    # xz = np.array(list(datf["Concentracion"]))
   
    file_txt =ot+".txt"
    fdata = open("idmunicipal/"+file_txt, "w")
    fdata.write("")

    try:

        fdata.write(str(np.unique(datf["Id_municipal"]))+ "\n")

    finally:



    # xx = list(xz[np.array(list(datf["Id_municipal"])) == municipio])

    # Q1 = np.percentile(xx, 25)
    # Q3 = np.percentile(xx, 75)
    # iqr = Q3 - Q1

    # # Calcular los límites teóricos de los bigotes
    # limite_inferior = Q1 - 1.5 * iqr
    # limite_superior = Q3 + 1.5 * iqr

    # # Ajustar los límites a los valores del conjunto de datos
    # bigote_inferior = np.min([u for u in xx if u >= limite_inferior])
    # bigote_superior = np.max([u for u in xx if u <= limite_superior])

    # dats_Concentracion = list(
    #     np.array(xx)[
    #         (np.array(xx) > bigote_inferior) & (np.array(xx) < bigote_superior)
    #     ]
    # )
    # k = np.histogram_bin_edges(xx, bins="auto")
    # frc, cbin = np.histogram(xx, bins=k)
    # frcnormal = frc / sum(frc)
    # plt.plot(cbin[:-1], frc, "o")

    # sizeclas.append(frcnormal)
    # df = {"x": xx}
    # pm = sns.histplot(x="x", data=df)
    # # plt.xscale("log")
    # # plt.yscale("log")
    # plt.show()
    # if us == 0:
    #     break
