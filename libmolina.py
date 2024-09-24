import numpy as np
from datetime import datetime
import pandas as pd
from scipy import stats
from matplotlib.pylab import plt
from PIL import Image
import matplotlib.gridspec as gridspec
import matplotlib
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
import matplotlib as mpl
from iminuit import Minuit
from probfit import Chi2Regression


def imshow_plots(img_path, figsize=(30, 25)):
    """
    Muestra una imagen en un gráfico con un tamaño específico.

    Parámetros:
    img_path (str): La ruta del archivo de la imagen a mostrar.
    figsize (tuple): El tamaño de la figura en pulgadas (ancho, alto).
    """
    fig = plt.figure(figsize=figsize)
    img = np.asarray(Image.open(img_path))
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def fig_plots(
    x,
    y,
    pplt,
    color="blue",
    nejex="Eje x",
    nejey="Eje y",
    titulo="Titulo",
    labelsize=20,
    fontsize=20,
):
    """
    Muestra una imagen en un gráfico con un tamaño específico.

    Parámetros:
    img_path (str): La ruta del archivo de la imagen a mostrar.
    figsize (tuple): El tamaño de la figura en pulgadas (ancho, alto).
    """
    pplt.plot(x, y, "-", color=color)
    pplt.tick_params(labelsize=labelsize * 0.8)
    pplt.tick_params(labelsize=labelsize * 0.8)
    pplt.xlabel(nejex, labelpad=20, fontsize=labelsize)
    pplt.ylabel(nejey, labelpad=20, fontsize=labelsize)
    pplt.title(titulo, fontsize=fontsize)


def datos_plots(
    o,
    matriz_dataframe,
    nNbins,
    densidad,
    eje_x=["Eje x", "Eje x"],
    eje_y=["Eje y", "Eje y"],
    titulo=["Titulo", "Titulo"],
    labelsize=20,
    fontsize=20,
    figsize=(30, 10),
    wspace=0.3,
):
    """
    Muestra una imagen en un gráfico con un tamaño específico.

    Parámetros:
    img_path (str): La ruta del archivo de la imagen a mostrar.
    figsize (tuple): El tamaño de la figura en pulgadas (ancho, alto).
    """
    matriz_data_frame = [u for u in matriz_dataframe if u]
    nbins = [u for u in nNbins if u]
    figura, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    for u, lista_dataframe in enumerate(matriz_data_frame):
        for df in lista_dataframe:
            df[densidad].plot(style=".", ax=ax[u])
            ax[u].tick_params(labelsize=labelsize * 0.9)
            ax[u].tick_params(labelsize=labelsize * 0.9)
            ax[u].set_xlabel(eje_x[u], labelpad=20, fontsize=labelsize)
            ax[u].set_ylabel(eje_y[u], labelpad=20, fontsize=labelsize)
            ax[u].set_title(titulo[u], pad=25, fontsize=fontsize)

    figura.tight_layout()
    plt.subplots_adjust(wspace=wspace)
    figura.savefig(f"plots/dispers{o}.png", bbox_inches="tight")


def figure_plots(
    o,
    matriz_dataframe,
    nNbins,
    densidad,
    eje_x=["Eje x", "Eje x"],
    eje_y=["Eje y", "Eje y"],
    titulo=["Titulo", "Titulo"],
    labelsize=20,
    fontsize=20,
    figsize=(30, 10),
    wspace=0.3,
):
    """
    Muestra una imagen en un gráfico con un tamaño específico.

    Parámetros:
    img_path (str): La ruta del archivo de la imagen a mostrar.
    figsize (tuple): El tamaño de la figura en pulgadas (ancho, alto).
    """
    matriz_data_frame = [u for u in matriz_dataframe if u]
    nbins = [u for u in nNbins if u]
    figura, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    for u, lista_dataframe in enumerate(matriz_data_frame):
        for df in lista_dataframe:
            df[densidad].hist(
                bins=nbins[u],
                ax=ax[u],
                density=True,
                grid=False,
                histtype="step",
                linewidth=3,
            )
            # frm = r'${:.0f}$'
            # xlist = np.linspace(np.min(df[densidad]), np.max(df[densidad]), 6)
            # ax[u].xaxis.set_major_locator(mticker.FixedLocator(xlist))
            # ax[u].set_xticklabels([frm.format(x) for x in xlist],fontsize=14)
            ax[u].tick_params(labelsize=labelsize * 0.9)
            ax[u].tick_params(labelsize=labelsize * 0.9)
            ax[u].set_xlabel(eje_x[u], labelpad=20, fontsize=labelsize)
            ax[u].set_ylabel(eje_y[u], labelpad=20, fontsize=labelsize)
            ax[u].set_title(titulo[u], pad=25, fontsize=fontsize)
            # ax[u].yaxis.set_data_interval(0, 0.065 ,True)

    figura.tight_layout()
    plt.subplots_adjust(wspace=wspace)
    figura.savefig(f"plots/Graf{o}.png", bbox_inches="tight")


def bulk_data_grouping(conjunto_de_datos, agrupando_chunk, directorio_save):
    """
    Esta función agrupa y guarda  conjunto masivos  de datos

    conjunto_de_datos: conjuntos de datos pd.read_csv(...)

    agrupando_chunk: variable que contiene el nombre de una columna dataframe
      con la que quieres agrupar el conjunto masivo de datos.

    directorio_save: esta varialbe debe contener la direccion
    donde vas a gurdar los datos agrupados.

    """
    grupos_acumulados = {}
    for chunk in conjunto_de_datos:
        grouped = chunk.groupby(agrupando_chunk)
        for nombre_grupo, grupo in grouped:
            if nombre_grupo in grupos_acumulados:
                grupos_acumulados[nombre_grupo] = pd.concat(
                    [grupos_acumulados[nombre_grupo], grupo]
                )
            else:
                grupos_acumulados[nombre_grupo] = grupo

    # Guardar cada grupo acumulado en archivos CSV separados
    for nombre_grupo, grupo in grupos_acumulados.items():
        # se puede modificar el nombre del grupo con las dos lineas de codigo comentadas
        # y realizando un par de modificaciones
        # i = np.where(np.array(list(variables_contaminacion.values())) == nombre_grupo)[0]
        # name_archivo = np.array(list(variables_contaminacion.keys()))[i]
        nombre_archivo = f"{nombre_grupo}.csv"
        grupo.to_csv(directorio_save + nombre_archivo, index=False)
        print(f"Guardado: {nombre_archivo}")


def umbral_iqr(dataframe, variable):
    """
    Esta función filtra los datos hubicado dentro de los bigotes del boxplot, eliminando
    los valores atípicos que se encuentran fuera del rango delimitado
    por los bigotes inferior y superior

    dataframe: este parametro recibe un dataframe.
    variable: columna especifica de datos del dataframe

    """

    Q1 = dataframe[variable].quantile(0.25)
    Q3 = dataframe[variable].quantile(0.75)
    iqr = Q3 - Q1
    lower_limit = Q1 - 1.5 * iqr
    upper_limit = Q3 + 1.5 * iqr

    return lower_limit, upper_limit


def covertir_formato_24_horas(fecha_12_horas):
    """
    esta función convierte los datos de fecha que están en
    formato 12 horas  a fechas en formato 24 horas

    fecha_12_horas: este parámetro recibe una lista de fechas
    en formato 12 horas
    """
    # Convertir la fecha y hora de 12 horas a formato de 24 horas
    fecha_en_formato_24_horas = []
    for formatear_fecha in fecha_12_horas:
        fecha_24_horas = datetime.strptime(formatear_fecha, "%d/%m/%Y %I:%M:%S %p")
        fecha_en_formato_24_horas.append(fecha_24_horas)
    fecha_hora_con_formato = [
        darformato.strftime("%d/%m/%Y %H:%M:%S")
        for darformato in fecha_en_formato_24_horas
    ]

    return fecha_hora_con_formato


def fechas_en_horas(lista_fechas_horas, fecha_referencia=datetime(1970, 1, 1, 0, 0, 0)):
    """
    Esta funcion recibe datos de fechas en formato  "%d/%m/%Y %H:%M:%S" y las convierte
    a horas.

    lista_fechas_horas: parametro que recibe una lista de fechas en formato "%d/%m/%Y %H:%M:%S"
    fecha_referencia: parametro que recibe una fecha de referencia, por defecto es
    1970, 1, 1, 0, 0, 0

    """

    horas = []
    for fecha_hora in lista_fechas_horas:
        # Convertir la fecha y hora a datetime
        fecha_hora_dt = datetime.strptime(fecha_hora, "%d/%m/%Y %H:%M:%S")
        # Calcular la diferencia en horas respecto a la fecha de referencia
        diferencia = (fecha_hora_dt - fecha_referencia).total_seconds() / 3600
        horas.append(diferencia)
    return horas


def corr_pearsonr0(sizeclas):
    array_corr_perason = []
    array_corr_pevalue = []
    for xi in range(len(sizeclas[:, 0])):
        corr_perason = []
        pvalue_perason = []
        for yi in range(len(sizeclas[:, 0])):
            varcorrpearson = stats.pearsonr(sizeclas[xi, :], sizeclas[yi, :])
            corr_perason.append(varcorrpearson[0])
            pvalue_perason.append(varcorrpearson[1])
            # Jensen_Shannon_divergence(sizeclas[xi,:], sizeclas[yi,:]))  #
        array_corr_perason.append(corr_perason)
        array_corr_pevalue.append(pvalue_perason)
    array_corr_perason = np.array(array_corr_perason)
    array_corr_pevalue = np.array(array_corr_pevalue)
    return array_corr_perason, array_corr_pevalue


def plot_corr_pearsonr(
    harvest,
    titulos,
    farmers,
    figsize=(30, 25),
    labelsize=10,
    fontsize=10,
    fontz=10,
    tight_layout=True,
    pad=20,
):
    fig = plt.figure(figsize=figsize, tight_layout=tight_layout)
    gs = gridspec.GridSpec(1, 2)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax = [ax0, ax1]
    Color = ["black", "white"]
    for ut, ax in enumerate(ax):
        pc = ax.imshow(harvest[ut])

        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(len(farmers[0])), labels=farmers[0])
        ax.set_yticks(np.arange(len(farmers[1])), labels=farmers[1])

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(farmers[0])):
            for j in range(len(farmers[1])):
                text = ax.text(
                    j,
                    i,
                    round(harvest[ut][i, j], 3),
                    ha="center",
                    va="center",
                    fontsize=fontz,
                    color=Color[ut],
                )

        ax.set_title(titulos[ut], fontsize=fontsize, pad=pad)
        ax.tick_params(labelsize=labelsize)
        ax.tick_params(labelsize=labelsize)
    fig.tight_layout()
    fig.savefig("cvclima.pdf")
    return plt.show()


def corr_pearsonr(sizeclas):
    array_corr_perason = []
    array_corr_pevalue = []
    for xi in range(len(sizeclas[:, 0])):
        corr_perason = []
        pvalue_perason = []
        for yi in range(len(sizeclas[:, 0])):
            varcorrpearson = stats.pearsonr(sizeclas[xi, :], sizeclas[yi, :])
            corr_perason.append(varcorrpearson[0])
            pvalue_perason.append(varcorrpearson[1])
            # Jensen_Shannon_divergence(sizeclas[xi,:], sizeclas[yi,:]))  #
        array_corr_perason.append(corr_perason)
        array_corr_pevalue.append(pvalue_perason)
    array_corr_perason = np.array(array_corr_perason)
    array_corr_pevalue = np.array(array_corr_pevalue)
    return array_corr_perason, array_corr_pevalue


def plt_corr_pears(
    harvest,
    titulos,
    farmers,
    figsize=(30, 25),
    labelsize=10,
    fontsize=10,
    tight_layout=True,
    padd=20,
    formato=r"${:.2f}$",
    descript=["Descripción", "Descripción"],
    nNums=10,
    size="4%",
    padr="15%",
    descrip_bar=20,
    var_clima="varclima",
    wspace=1.5,
):
    fig = plt.figure(figsize=figsize, tight_layout=tight_layout)
    xmicol1 = "viridis"
    xcmap = mpl.colormaps[xmicol1].resampled(250)
    xnewcolors = xcmap(np.linspace(0, 1, 250))
    cmap = ListedColormap(xnewcolors)
    gs = gridspec.GridSpec(1, 2)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax = [ax0, ax1]
    for ut, ax in enumerate(ax):
        pc = ax.imshow(harvest[ut])

        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(len(farmers[0])), labels=farmers[0])
        ax.set_yticks(np.arange(len(farmers[1])), labels=farmers[1])

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.set_title(titulos[ut], fontsize=fontsize, pad=padd)
        ax.tick_params(labelsize=labelsize)
        ax.tick_params(labelsize=labelsize)

        ticks_loc = np.linspace(np.min(harvest[ut]), np.max(harvest[ut]), nNums)

        norm = matplotlib.colors.Normalize(
            vmin=np.min(harvest[ut]), vmax=np.max(harvest[ut])
        )
        sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
        ax2_divider = make_axes_locatable(ax)
        cax0 = ax2_divider.append_axes("right", size=size, pad=padr)
        cbar0 = fig.colorbar(pc, extend="both", cax=cax0, orientation="vertical")
        cbar0.ax.set_ylabel(descript[ut], fontsize=descrip_bar)
        cbar0.ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        cbar0.ax.set_yticklabels(
            [formato.format(x) for x in ticks_loc], fontsize=descrip_bar * 0.8
        )
    fig.tight_layout()
    plt.subplots_adjust(wspace=wspace)
    fig.savefig(f"plots/{var_clima}_corr.png", bbox_inches="tight")
    return plt.show()


def model_fit(x, m, b):
    f = b + x * m
    return f


def plots_lineal_model(
    z,
    mx,
    n_var=["i", "j"],
    titX=["x", "x"],
    titY=["y", "y"],
    limX=[1, 1],
    limY=[1, 1],
    ngr=0,
):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
    for u, array_freq in enumerate(mx):
        enx = []
        eny = []
        nN = len(np.array(array_freq)[:, 0])
        for i in range(nN):
            for j in range(i + 1, nN):
                ai = list(np.array(array_freq)[i])
                aj = list(np.array(array_freq)[j])
                corr, _ = pearsonr(ai, aj)
                if corr > 0.7:
                    enx += list(ai)
                    eny += list(aj)
                    xi = np.array(ai, dtype=float)
                    yi = np.array(aj, dtype=float)
                    ax[u].plot(xi[xi < z], yi[xi < z], "o", label="Datos")

        x = np.array(enx, dtype=float)
        y = np.array(eny, dtype=float)
        xx = x[x < z]
        yy = y[x < z]
        chi2 = Chi2Regression(model_fit, xx, yy)
        p = Minuit(chi2, m=1, b=1)
        p.migrad()

        m_fit = p.values["m"]
        b_fit = p.values["b"]
        print(f"Parámetros ajustados: a={m_fit:.2f}, b={b_fit:.2f}")
        ax[u].plot(
            xx,
            model_fit(xx, m_fit, b_fit),
            color="black",
            label=f"Ajuste: a={m_fit:.2f}, b={b_fit:.2f}",
        )

        ax[u].set_xlabel(f"{titX[u]}", labelpad=20, fontsize=30)
        ax[u].set_ylabel(f"{titY[u]}", labelpad=20, fontsize=30)
        ax[u].set_title(f"{n_var[u]}", pad=25, fontsize=30)
        ax[u].tick_params(labelsize=30 * 0.9)
        ax[u].tick_params(labelsize=30 * 0.9)
        ax[u].set_xlim(-0.005, limX[u])
        ax[u].set_ylim(-0.005, limY[u])
        # ax[u].legend(fontsize=15)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    fig.savefig(f"plots/gxGraf{ngr}.png", bbox_inches="tight")
    plt.show()


def param_lineal_model(z, array_freq, xmunic):
    enx = []
    eny = []
    valcorr = []
    mun_val = []
    nN = len(np.array(array_freq)[:, 0])
    for i in range(nN):
        for j in range(i + 1, nN):
            ai = list(np.array(array_freq)[i])
            aj = list(np.array(array_freq)[j])
            corr, _ = pearsonr(ai, aj)
            if corr > 0.7:
                valcorr.append(corr)
                mun_val.append([xmunic[i], xmunic[j]])
                enx += list(ai)
                eny += list(aj)
                xi = np.array(ai, dtype=float)
                yi = np.array(aj, dtype=float)

    x = np.array(enx, dtype=float)
    y = np.array(eny, dtype=float)
    xx = x[x < z]
    yy = y[x < z]
    chi2 = Chi2Regression(model_fit, xx, yy)
    p = Minuit(chi2, m=1, b=1)
    p.migrad()

    m_fit = round(p.values["m"], 3)
    b_fit = round(p.values["b"], 3)
    return m_fit, b_fit, mun_val, valcorr
