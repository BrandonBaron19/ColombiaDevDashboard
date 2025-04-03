"""
Módulo de Visualizaciones para Datos del Banco Mundial

Este módulo genera visualizaciones avanzadas a partir de los datos procesados del Banco Mundial.
Las visualizaciones incluyen series temporales, gráficos comparativos, análisis multivariable,
mapas geoespaciales y dashboards integrados.

Categorías de visualizaciones:
1. Series temporales avanzadas
2. Gráficos de comparación y distribución
3. Análisis multivariable y correlaciones
4. Mapas y visualizaciones geoespaciales
5. Dashboards y visualizaciones compuestas
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap
import seaborn as sns
from datetime import datetime
import json
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constantes y configuración
DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
DIRECTORIO_DATOS = os.path.join(DIRECTORIO_ACTUAL, "datos_procesados")
DIRECTORIO_VISUALIZACIONES = os.path.join(DIRECTORIO_ACTUAL, "visualizaciones")

# Definición de archivos a cargar
ARCHIVOS_DATOS = {
    'maestro': 'datos_maestro.csv',
    'economia': 'datos_economia.csv',
    'social': 'datos_social.csv',
    'educacion': 'datos_educacion.csv',
    'ambiente': 'datos_ambiente.csv',
    'gobernanza': 'datos_gobernanza.csv'
}

# Indicadores clave por tema
INDICADORES = {
    'economia': {
        'PIB': 'NY.GDP.MKTP.CD',
        'PIB_PER_CAPITA': 'NY.GDP.PCAP.CD',
        'CRECIMIENTO_PIB': 'NY.GDP.MKTP.KD.ZG',
        'INFLACION': 'FP.CPI.TOTL.ZG',
        'EXPORTACIONES': 'NE.EXP.GNFS.ZS',
        'IMPORTACIONES': 'NE.IMP.GNFS.ZS'
    },
    'social': {
        'POBREZA': 'SI.POV.NAHC',
        'GINI': 'SI.POV.GINI',
        'DESEMPLEO': 'SL.UEM.TOTL.ZS',
        'ESPERANZA_VIDA': 'SP.DYN.LE00.IN',
        'POBLACION': 'SP.POP.TOTL',
        'POBLACION_URBANA': 'SP.URB.TOTL.IN.ZS'
    },
    'educacion': {
        'ALFABETIZACION': 'SE.ADT.LITR.ZS',
        'FINALIZACION_PRIMARIA': 'SE.PRM.CMPT.ZS',
        'FINALIZACION_SECUNDARIA': 'SE.SEC.CMPT.LO.ZS',
        'INSCRIPCION_TERCIARIA': 'SE.TER.ENRR',
        'GASTO_EDUCACION': 'SE.XPD.TOTL.GD.ZS'
    },
    'ambiente': {
        'ENERGIA_RENOVABLE': 'EG.FEC.RNEW.ZS',
        'EMISIONES_CO2': 'EN.ATM.CO2E.PC',
        'AREA_FORESTAL': 'AG.LND.FRST.ZS',
        'AGUA_DULCE': 'ER.H2O.FWTL.ZS',
        'EXPOSICION_PM25': 'EN.ATM.PM25.MC.M3'
    },
    'gobernanza': {
        'HOMICIDIOS': 'VC.IHR.PSRC.P5',
        'VOZ_RENDICION': 'VA.EST',
        'ESTADO_DERECHO': 'RL.EST',
        'CONTROL_CORRUPCION': 'CC.EST',
        'EFECTIVIDAD_GOBIERNO': 'GE.EST'
    }
}

# Períodos predefinidos para análisis temporales
PERIODOS_PREDEFINIDOS = {
    '2000-2005': (2000, 2005),
    '2006-2010': (2006, 2010),
    '2011-2015': (2011, 2015),
    '2016-2020': (2016, 2020),
    '2021-2023': (2021, 2023)
}

def cargar_datos_procesados():
    """
    Carga todos los archivos de datos procesados.
    
    Returns:
        dict: Diccionario con DataFrames de datos procesados
    """
    logger.info("Cargando datos procesados...")
    
    # Diccionario para almacenar los DataFrames
    dfs = {}
    
    # Cargar cada archivo
    for nombre, archivo in ARCHIVOS_DATOS.items():
        ruta_completa = os.path.join(DIRECTORIO_DATOS, archivo)
        if os.path.exists(ruta_completa):
            df = pd.read_csv(ruta_completa, encoding='utf-8')
            dfs[nombre] = df
            logger.info(f"  Cargado: {archivo} ({df.shape[0]} filas, {df.shape[1]} columnas)")
        else:
            logger.warning(f"  ¡Advertencia! No se encontró el archivo: {ruta_completa}")
    
    return dfs

# 1. VISUALIZACIONES DE SERIES TEMPORALES AVANZADAS

def crear_serie_temporal_multilinea(df, codigo_indicadores, titulo="Series Temporales", ancho=900, alto=500):
    """
    Crea un gráfico de líneas múltiples para varios indicadores a lo largo del tiempo.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos (formato largo)
        codigo_indicadores (list): Lista de códigos de indicadores a incluir
        titulo (str): Título del gráfico
        ancho (int): Ancho del gráfico en píxeles
        alto (int): Alto del gráfico en píxeles
        
    Returns:
        go.Figure: Figura de Plotly
    """
    # Filtrar datos para los indicadores seleccionados
    datos_filtrados = df[df['Indicator Code'].isin(codigo_indicadores)].copy()
    
    # Crear el gráfico
    fig = px.line(
        datos_filtrados, 
        x="Año", 
        y="Valor",
        color="Indicator Name",
        title=titulo,
        labels={"Valor": "Valor del indicador", "Año": "Año", "Indicator Name": "Indicador"},
        width=ancho,
        height=alto,
    )
    
    # Mejorar el diseño
    fig.update_layout(
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        xaxis=dict(
            tickmode='linear',
            tick0=datos_filtrados['Año'].min(),
            dtick=5
        )
    )
    
    # Añadir líneas de tendencia
    for indicador in datos_filtrados['Indicator Name'].unique():
        datos_ind = datos_filtrados[datos_filtrados['Indicator Name'] == indicador]
        x_vals = datos_ind['Año']
        y_vals = datos_ind['Valor']
        
        # Solo añadir línea de tendencia si hay suficientes puntos
        if len(x_vals) > 5:
            # Crear polinomio de grado 2 para la tendencia
            coefs = np.polyfit(x_vals, y_vals, 2)
            y_trend = np.polyval(coefs, x_vals)
            
            # Añadir línea de tendencia como línea punteada
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_trend,
                mode='lines',
                line=dict(dash='dash', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    return fig

def crear_serie_temporal_area(df, codigo_indicador, titulo="Evolución Temporal", usar_promedio_movil=True, ancho=800, alto=450):
    """
    Crea un gráfico de área para un indicador con promedio móvil.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos (formato largo)
        codigo_indicador (str): Código del indicador a visualizar
        titulo (str): Título del gráfico
        usar_promedio_movil (bool): Si es True, muestra el promedio móvil de 3 años
        ancho (int): Ancho del gráfico en píxeles
        alto (int): Alto del gráfico en píxeles
        
    Returns:
        go.Figure: Figura de Plotly
    """
    # Filtrar datos para el indicador seleccionado
    datos_filtrados = df[df['Indicator Code'] == codigo_indicador].copy()
    
    if datos_filtrados.empty:
        logger.warning(f"No se encontraron datos para el indicador {codigo_indicador}")
        return None
    
    # Ordenar por año
    datos_filtrados = datos_filtrados.sort_values('Año')
    
    # Calcular promedio móvil si no existe en los datos
    if usar_promedio_movil and 'Promedio_Movil_3' not in datos_filtrados.columns:
        datos_filtrados['Promedio_Movil_3'] = datos_filtrados['Valor'].rolling(window=3, min_periods=1).mean()
    
    # Crear el gráfico
    fig = go.Figure()
    
    # Añadir área para valores originales
    fig.add_trace(go.Scatter(
        x=datos_filtrados['Año'],
        y=datos_filtrados['Valor'],
        fill='tozeroy',
        mode='lines',
        name=datos_filtrados['Indicator Name'].iloc[0],
        line=dict(width=0.5, color='rgb(73, 102, 141)'),
        fillcolor='rgba(73, 102, 141, 0.3)'
    ))
    
    # Añadir línea para promedio móvil si se solicita
    if usar_promedio_movil and 'Promedio_Movil_3' in datos_filtrados.columns:
        fig.add_trace(go.Scatter(
            x=datos_filtrados['Año'],
            y=datos_filtrados['Promedio_Movil_3'],
            mode='lines',
            name='Promedio móvil (3 años)',
            line=dict(width=2, color='rgb(214, 39, 40)')
        ))
    
    # Configurar el diseño
    fig.update_layout(
        title=titulo,
        template="plotly_white",
        width=ancho,
        height=alto,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            tickmode='linear',
            tick0=datos_filtrados['Año'].min(),
            dtick=5
        )
    )
    
    return fig

def crear_heatmap_temporal(df, codigo_indicador, titulo="Evolución por Año", ancho=800, alto=400):
    """
    Crea un mapa de calor temporal para un indicador específico.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos (formato largo)
        codigo_indicador (str): Código del indicador a visualizar
        titulo (str): Título del gráfico
        ancho (int): Ancho del gráfico en píxeles
        alto (int): Alto del gráfico en píxeles
        
    Returns:
        go.Figure: Figura de Plotly
    """
    # Filtrar datos para el indicador seleccionado
    datos_filtrados = df[df['Indicator Code'] == codigo_indicador].copy()
    
    if datos_filtrados.empty:
        logger.warning(f"No se encontraron datos para el indicador {codigo_indicador}")
        return None
    
    # Obtener años y valores
    años = datos_filtrados['Año'].values
    valores = datos_filtrados['Valor'].values
    
    # Ordenar por año
    indices_ordenados = np.argsort(años)
    años = años[indices_ordenados]
    valores = valores[indices_ordenados]
    
    # Crear el mapa de calor
    fig = go.Figure(data=go.Heatmap(
        z=[valores],
        x=años,
        y=[datos_filtrados['Indicator Name'].iloc[0]],
        colorscale='Viridis',
        colorbar=dict(title="Valor")
    ))
    
    # Configurar el diseño
    fig.update_layout(
        title=titulo,
        template="plotly_white",
        width=ancho,
        height=alto,
        yaxis=dict(autorange="reversed")
    )
    
    return fig

# 2. GRÁFICOS DE COMPARACIÓN Y DISTRIBUCIÓN

def crear_boxplot_por_periodos(df, codigo_indicador, periodos=None, titulo="Distribución por Períodos", ancho=800, alto=500):
    """
    Crea un boxplot que muestra la distribución de un indicador por períodos de tiempo.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos (formato largo)
        codigo_indicador (str): Código del indicador a visualizar
        periodos (dict, optional): Diccionario con nombres de períodos y rangos de años
        titulo (str): Título del gráfico
        ancho (int): Ancho del gráfico en píxeles
        alto (int): Alto del gráfico en píxeles
        
    Returns:
        go.Figure: Figura de Plotly
    """
    # Definir períodos si no se proporcionan
    if periodos is None:
        periodos = PERIODOS_PREDEFINIDOS
    
    # Filtrar datos para el indicador
    datos_filtrados = df[df['Indicator Code'] == codigo_indicador].copy()
    
    if datos_filtrados.empty:
        logger.warning(f"No se encontraron datos para el indicador {codigo_indicador}")
        return None
    
    # Añadir columna de período
    def asignar_periodo(año):
        for nombre, (inicio, fin) in periodos.items():
            if inicio <= año <= fin:
                return nombre
        return 'Otro'
    
    datos_filtrados['Periodo'] = datos_filtrados['Año'].apply(asignar_periodo)
    
    # Crear el boxplot
    fig = px.box(
        datos_filtrados,
        x='Periodo',
        y='Valor',
        title=f"{titulo}: {datos_filtrados['Indicator Name'].iloc[0]}",
        labels={"Valor": "Valor del indicador", "Periodo": "Período"},
        width=ancho,
        height=alto,
        color='Periodo',
        category_orders={"Periodo": list(periodos.keys())}
    )
    
    # Mejorar el diseño
    fig.update_layout(
        template="plotly_white",
        boxmode='group',
        xaxis=dict(categoryorder='array', categoryarray=list(periodos.keys()))
    )
    
    return fig

def crear_barras_comparativas(df, codigo_indicadores, año_inicio, año_fin, titulo="Comparación de Indicadores", ancho=800, alto=600):
    """
    Crea un gráfico de barras que compara múltiples indicadores entre dos años.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos (formato largo)
        codigo_indicadores (list): Lista de códigos de indicadores a comparar
        año_inicio (int): Año de inicio para comparación
        año_fin (int): Año de fin para comparación
        titulo (str): Título del gráfico
        ancho (int): Ancho del gráfico en píxeles
        alto (int): Alto del gráfico en píxeles
        
    Returns:
        go.Figure: Figura de Plotly
    """
    # Filtrar datos
    datos_filtrados = df[df['Indicator Code'].isin(codigo_indicadores) & 
                         df['Año'].isin([año_inicio, año_fin])].copy()
    
    if datos_filtrados.empty:
        logger.warning(f"No se encontraron datos para los indicadores y años especificados")
        return None
    
    # Pivotar datos para tener indicadores como índice y años como columnas
    tabla_pivot = datos_filtrados.pivot_table(
        values='Valor',
        index='Indicator Name',
        columns='Año',
        aggfunc='first'
    ).reset_index()
    
    # Calcular cambio porcentual
    if año_inicio in tabla_pivot.columns and año_fin in tabla_pivot.columns:
        tabla_pivot['Cambio %'] = ((tabla_pivot[año_fin] / tabla_pivot[año_inicio]) - 1) * 100
    
    # Crear gráfico de barras agrupadas
    fig = go.Figure()
    
    # Barra para el año inicial
    fig.add_trace(go.Bar(
        x=tabla_pivot['Indicator Name'],
        y=tabla_pivot[año_inicio] if año_inicio in tabla_pivot.columns else [],
        name=f'Año {año_inicio}',
        marker_color='rgba(55, 83, 109, 0.7)'
    ))
    
    # Barra para el año final
    fig.add_trace(go.Bar(
        x=tabla_pivot['Indicator Name'],
        y=tabla_pivot[año_fin] if año_fin in tabla_pivot.columns else [],
        name=f'Año {año_fin}',
        marker_color='rgba(26, 118, 255, 0.7)'
    ))
    
    # Configurar el diseño
    fig.update_layout(
        title=titulo,
        template="plotly_white",
        width=ancho,
        height=alto,
        barmode='group',
        xaxis=dict(tickangle=-45, title=""),
        yaxis=dict(title="Valor del indicador"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Añadir etiquetas con cambio porcentual
    if 'Cambio %' in tabla_pivot.columns:
        for i, indicador in enumerate(tabla_pivot['Indicator Name']):
            cambio = tabla_pivot.loc[i, 'Cambio %']
            color = 'green' if cambio >= 0 else 'red'
            fig.add_annotation(
                x=indicador,
                y=max(tabla_pivot.loc[i, año_inicio], tabla_pivot.loc[i, año_fin]) * 1.1,
                text=f"{cambio:.1f}%",
                showarrow=False,
                font=dict(size=10, color=color)
            )
    
    return fig

def crear_variacion_respecto_base(df, codigo_indicador, año_base=2000, titulo="Variación respecto al año base", ancho=800, alto=450):
    """
    Crea un gráfico que muestra la variación porcentual respecto a un año base.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos (formato largo)
        codigo_indicador (str): Código del indicador a visualizar
        año_base (int): Año base para calcular variaciones
        titulo (str): Título del gráfico
        ancho (int): Ancho del gráfico en píxeles
        alto (int): Alto del gráfico en píxeles
        
    Returns:
        go.Figure: Figura de Plotly
    """
    # Filtrar datos para el indicador
    datos_filtrados = df[df['Indicator Code'] == codigo_indicador].copy()
    
    if datos_filtrados.empty:
        logger.warning(f"No se encontraron datos para el indicador {codigo_indicador}")
        return None
    
    # Ordenar por año
    datos_filtrados = datos_filtrados.sort_values('Año')
    
    # Obtener el valor para el año base
    valor_base = datos_filtrados[datos_filtrados['Año'] == año_base]['Valor'].values
    
    if len(valor_base) == 0:
        logger.warning(f"No se encontró valor para el año base {año_base}")
        # Usar el primer año disponible como base alternativa
        año_base = datos_filtrados['Año'].min()
        valor_base = datos_filtrados[datos_filtrados['Año'] == año_base]['Valor'].values
        
        if len(valor_base) == 0:
            return None
    
    valor_base = valor_base[0]
    
    # Calcular variación porcentual
    datos_filtrados['Variación %'] = ((datos_filtrados['Valor'] / valor_base) - 1) * 100
    
    # Crear el gráfico
    fig = go.Figure()
    
    # Añadir barra para cada año
    fig.add_trace(go.Bar(
        x=datos_filtrados['Año'],
        y=datos_filtrados['Variación %'],
        marker_color='rgba(58, 71, 80, 0.6)',
        marker_line_color='rgba(8, 48, 107, 1)',
        marker_line_width=1.5,
        name=datos_filtrados['Indicator Name'].iloc[0]
    ))
    
    # Añadir línea de cero
    fig.add_shape(
        type="line",
        x0=datos_filtrados['Año'].min(),
        y0=0,
        x1=datos_filtrados['Año'].max(),
        y1=0,
        line=dict(color="red", width=2, dash="dot")
    )
    
    # Configurar el diseño
    fig.update_layout(
        title=f"{titulo}: {datos_filtrados['Indicator Name'].iloc[0]} (Base: {año_base})",
        template="plotly_white",
        width=ancho,
        height=alto,
        xaxis=dict(
            tickmode='linear',
            tick0=datos_filtrados['Año'].min(),
            dtick=5,
            title="Año"
        ),
        yaxis=dict(title="Variación porcentual (%)"),
        showlegend=False
    )
    
    return fig

# 3. ANÁLISIS MULTIVARIABLE Y CORRELACIONES

def crear_matriz_correlacion(df, codigo_indicadores, años_recientes=5, titulo="Matriz de Correlación", ancho=800, alto=700):
    """
    Crea una matriz de correlación entre múltiples indicadores.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos (formato largo)
        codigo_indicadores (list): Lista de códigos de indicadores para analizar
        años_recientes (int): Número de años recientes a considerar
        titulo (str): Título del gráfico
        ancho (int): Ancho del gráfico en píxeles
        alto (int): Alto del gráfico en píxeles
        
    Returns:
        go.Figure: Figura de Plotly
    """
    # Filtrar datos para los indicadores seleccionados
    datos_filtrados = df[df['Indicator Code'].isin(codigo_indicadores)].copy()
    
    if datos_filtrados.empty:
        logger.warning("No se encontraron datos para los indicadores especificados")
        return None
    
    # Obtener años más recientes
    años_max = datos_filtrados['Año'].max()
    años_a_incluir = list(range(años_max - años_recientes + 1, años_max + 1))
    datos_filtrados = datos_filtrados[datos_filtrados['Año'].isin(años_a_incluir)]
    
    # Pivotar datos para tener indicadores como columnas
    tabla_pivot = datos_filtrados.pivot_table(
        values='Valor',
        index='Año',
        columns='Indicator Name',
        aggfunc='first'
    )
    
    # Calcular matriz de correlación
    correlacion = tabla_pivot.corr()
    
    # Crear el mapa de calor
    fig = px.imshow(
        correlacion,
        x=correlacion.columns,
        y=correlacion.columns,
        color_continuous_scale='RdBu_r',
        range_color=[-1, 1],
        title=f"{titulo} (últimos {años_recientes} años)",
        width=ancho,
        height=alto,
        labels=dict(color="Correlación")
    )
    
    # Añadir texto con valores de correlación
    for i, row in enumerate(correlacion.index):
        for j, col in enumerate(correlacion.columns):
            fig.add_annotation(
                x=col,
                y=row,
                text=f"{correlacion.iloc[i, j]:.2f}",
                showarrow=False,
                font=dict(color="black" if abs(correlacion.iloc[i, j]) < 0.5 else "white")
            )
    
    # Configurar el diseño
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(tickangle=-45)
    )
    
    return fig

def crear_scatter_matrix(df, codigo_indicadores, años_recientes=10, titulo="Matriz de Dispersión", ancho=900, alto=800):
    """
    Crea una matriz de gráficos de dispersión entre múltiples indicadores.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos (formato largo)
        codigo_indicadores (list): Lista de códigos de indicadores para analizar
        años_recientes (int): Número de años recientes a considerar
        titulo (str): Título del gráfico
        ancho (int): Ancho del gráfico en píxeles
        alto (int): Alto del gráfico en píxeles
        
    Returns:
        go.Figure: Figura de Plotly
    """
    # Filtrar datos para los indicadores seleccionados (máximo 4 indicadores para legibilidad)
    codigo_indicadores = codigo_indicadores[:4] if len(codigo_indicadores) > 4 else codigo_indicadores
    datos_filtrados = df[df['Indicator Code'].isin(codigo_indicadores)].copy()
    
    if datos_filtrados.empty:
        logger.warning("No se encontraron datos para los indicadores especificados")
        return None
    
    # Obtener años más recientes
    años_max = datos_filtrados['Año'].max()
    años_a_incluir = list(range(años_max - años_recientes + 1, años_max + 1))
    datos_filtrados = datos_filtrados[datos_filtrados['Año'].isin(años_a_incluir)]
    
    # Pivotar datos para tener indicadores como columnas
    tabla_pivot = datos_filtrados.pivot_table(
        values='Valor',
        index='Año',
        columns='Indicator Name',
        aggfunc='first'
    ).reset_index()
    
    # Verificar que hay suficientes datos
    if tabla_pivot.shape[0] < 3:
        logger.warning("No hay suficientes datos para crear la matriz de dispersión")
        return None
    
    # Crear la matriz de dispersión
    fig = px.scatter_matrix(
        tabla_pivot,
        dimensions=tabla_pivot.columns[1:],  # Excluir la columna 'Año'
        color='Año',
        title=f"{titulo} (últimos {años_recientes} años)",
        width=ancho,
        height=alto,
        labels=tabla_pivot.columns,
        color_continuous_scale='Viridis'
    )
    
    # Mejorar el diseño
    fig.update_layout(
        template="plotly_white",
        coloraxis_showscale=True
    )
    
    # Ajustar tamaño de texto en los ejes
    fig.update_traces(diagonal_visible=False)
    
    return fig

def crear_grafico_burbujas(df, codigo_x, codigo_y, codigo_tamaño, años_seleccionados=None, titulo="Gráfico de Burbujas", ancho=800, alto=600):
    """
    Crea un gráfico de burbujas que muestra la relación entre tres indicadores.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos (formato largo)
        codigo_x (str): Código del indicador para el eje X
        codigo_y (str): Código del indicador para el eje Y
        codigo_tamaño (str): Código del indicador para el tamaño de las burbujas
        años_seleccionados (list): Lista de años a incluir
        titulo (str): Título del gráfico
        ancho (int): Ancho del gráfico en píxeles
        alto (int): Alto del gráfico en píxeles
        
    Returns:
        go.Figure: Figura de Plotly
    """
    # Si no se proporcionan años, usar los últimos 10 años disponibles
    if años_seleccionados is None:
        años_max = df['Año'].max()
        años_seleccionados = list(range(años_max - 9, años_max + 1))
    
    # Crear DataFrames para cada indicador
    df_x = df[df['Indicator Code'] == codigo_x].copy()
    df_y = df[df['Indicator Code'] == codigo_y].copy()
    df_tamaño = df[df['Indicator Code'] == codigo_tamaño].copy()
    
    # Verificar si hay datos
    if df_x.empty or df_y.empty or df_tamaño.empty:
        logger.warning("No se encontraron datos para uno o más indicadores")
        return None
    
    # Filtrar por años seleccionados
    df_x = df_x[df_x['Año'].isin(años_seleccionados)]
    df_y = df_y[df_y['Año'].isin(años_seleccionados)]
    df_tamaño = df_tamaño[df_tamaño['Año'].isin(años_seleccionados)]
    
    # Crear un DataFrame combinado
    df_combinado = pd.DataFrame({
        'Año': df_x['Año'],
        'X': df_x['Valor'],
        'Y': df_y['Valor'],
        'Tamaño': df_tamaño['Valor']
    })
    
    # Eliminar filas con valores faltantes
    df_combinado = df_combinado.dropna()
    
    if df_combinado.empty:
        logger.warning("No hay datos completos para los años e indicadores seleccionados")
        return None
    
    # Nombres de los indicadores para las etiquetas
    nombre_x = df_x['Indicator Name'].iloc[0] if not df_x.empty else codigo_x
    nombre_y = df_y['Indicator Name'].iloc[0] if not df_y.empty else codigo_y
    nombre_tamaño = df_tamaño['Indicator Name'].iloc[0] if not df_tamaño.empty else codigo_tamaño
    
    # Crear el gráfico de burbujas
    fig = px.scatter(
        df_combinado,
        x='X',
        y='Y',
        size='Tamaño',
        color='Año',
        hover_name='Año',
        title=titulo,
        labels={
            'X': nombre_x,
            'Y': nombre_y,
            'Tamaño': nombre_tamaño,
            'Año': 'Año'
        },
        width=ancho,
        height=alto,
        color_continuous_scale='Viridis',
        size_max=50
    )
    
    # Mejorar el diseño
    fig.update_layout(
        template="plotly_white",
        coloraxis_colorbar=dict(title="Año")
    )
    
    # Añadir etiquetas con años
    for i, row in df_combinado.iterrows():
        fig.add_annotation(
            x=row['X'],
            y=row['Y'],
            text=str(int(row['Año'])),
            showarrow=False,
            font=dict(size=9)
        )
    
    return fig

# 4. MAPAS Y VISUALIZACIONES GEOESPACIALES

def crear_mapa_colombia_folium(directorio_actual, titulo="Mapa de Colombia"):
    """
    Crea un mapa básico de Colombia usando Folium.
    
    Args:
        directorio_actual (str): Directorio actual para buscar archivos GeoJSON
        titulo (str): Título del mapa
        
    Returns:
        folium.Map: Mapa de Folium
    """
    # Coordenadas centrales de Colombia
    lat_centro = 4.5709
    lon_centro = -74.2973
    
    # Crear mapa base
    mapa = folium.Map(
        location=[lat_centro, lon_centro],
        zoom_start=6,
        tiles='CartoDB positron'
    )
    
    # Intentar cargar archivo GeoJSON de Colombia si existe
    archivo_geojson = os.path.join(directorio_actual, 'datos_geograficos', 'colombia.geojson')
    
    if os.path.exists(archivo_geojson):
        # Leer el archivo GeoJSON
        with open(archivo_geojson, 'r', encoding='utf-8') as f:
            datos_geo = json.load(f)
        
        # Añadir capa GeoJSON
        folium.GeoJson(
            datos_geo,
            name="Colombia",
            style_function=lambda x: {
                'fillColor': '#3186cc',
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.3
            }
        ).add_to(mapa)
        
        # Añadir control de capas
        folium.LayerControl().add_to(mapa)
    else:
        # Si no hay archivo GeoJSON, añadir un marcador simple en Bogotá
        folium.Marker(
            [4.7110, -74.0721],
            popup="Bogotá, Colombia",
            tooltip="Capital de Colombia"
        ).add_to(mapa)
    
    # Añadir título al mapa
    titulo_html = f'''
         <h3 align="center" style="font-size:16px"><b>{titulo}</b></h3>
         '''
    mapa.get_root().html.add_child(folium.Element(titulo_html))
    
    return mapa

def crear_heatmap_indicador_tiempo(df, codigo_indicador, titulo="Evolución de Indicador por Año", ancho=800, alto=500):
    """
    Crea un heatmap que muestra la evolución de un indicador a lo largo del tiempo.
    No es un mapa geográfico sino una visualización de intensidad.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos (formato largo)
        codigo_indicador (str): Código del indicador a visualizar
        titulo (str): Título del gráfico
        ancho (int): Ancho del gráfico en píxeles
        alto (int): Alto del gráfico en píxeles
        
    Returns:
        go.Figure: Figura de Plotly
    """
    # Filtrar datos para el indicador
    datos_filtrados = df[df['Indicator Code'] == codigo_indicador].copy()
    
    if datos_filtrados.empty:
        logger.warning(f"No se encontraron datos para el indicador {codigo_indicador}")
        return None
    
    # Ordenar por año y eliminar años sin datos
    datos_filtrados = datos_filtrados.sort_values('Año')
    datos_filtrados = datos_filtrados.dropna(subset=['Valor'])
    
    # Crear el heatmap
    fig = go.Figure(data=go.Heatmap(
        z=[datos_filtrados['Valor']],
        x=datos_filtrados['Año'],
        y=[datos_filtrados['Indicator Name'].iloc[0]],
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="Valor")
    ))
    
    # Configurar el diseño
    fig.update_layout(
        title=titulo,
        template="plotly_white",
        width=ancho,
        height=alto,
        xaxis=dict(
            tickmode='linear',
            tick0=datos_filtrados['Año'].min(),
            dtick=5,
            title="Año"
        ),
        yaxis=dict(
            title="",
            showticklabels=True
        )
    )
    
    return fig

# 5. DASHBOARDS Y VISUALIZACIONES COMPUESTAS

def crear_dashboard_economico(dfs, ancho=1000, alto=800):
    """
    Crea un dashboard económico compuesto con múltiples visualizaciones.
    
    Args:
        dfs (dict): Diccionario con DataFrames de datos procesados
        ancho (int): Ancho del dashboard en píxeles
        alto (int): Alto del dashboard en píxeles
        
    Returns:
        go.Figure: Figura de Plotly
    """
    # Verificar si tenemos los datos necesarios
    if 'economia' not in dfs:
        logger.warning("No se encontraron datos económicos")
        return None
    
    df_economia = dfs['economia']
    
    # Definir indicadores a mostrar
    indicador_pib = INDICADORES['economia']['PIB']
    indicador_pib_crecimiento = INDICADORES['economia']['CRECIMIENTO_PIB']
    indicador_exportaciones = INDICADORES['economia']['EXPORTACIONES']
    indicador_importaciones = INDICADORES['economia']['IMPORTACIONES']
    
    # Crear figura con subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "PIB (US$ a precios actuales)",
            "Crecimiento del PIB (% anual)",
            "Exportaciones e Importaciones (% del PIB)",
            "PIB: Variación respecto al 2000"
        ),
        specs=[
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "bar"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    # 1. Gráfico de PIB
    df_pib = df_economia[df_economia['Indicator Code'] == indicador_pib].copy()
    if not df_pib.empty:
        # Ordenar por año
        df_pib = df_pib.sort_values('Año')
        
        # Añadir línea de PIB
        fig.add_trace(
            go.Scatter(
                x=df_pib['Año'],
                y=df_pib['Valor'] / 1e9,  # Convertir a miles de millones
                mode='lines+markers',
                name='PIB',
                line=dict(color='rgba(55, 83, 109, 1)', width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # Añadir línea de tendencia
        x_vals = df_pib['Año']
        y_vals = df_pib['Valor'] / 1e9
        coefs = np.polyfit(x_vals, y_vals, 2)
        y_trend = np.polyval(coefs, x_vals)
        
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_trend,
                mode='lines',
                name='Tendencia',
                line=dict(color='rgba(255, 0, 0, 0.6)', width=2, dash='dash')
            ),
            row=1, col=1
        )
    
    # 2. Gráfico de crecimiento del PIB
    df_crecimiento = df_economia[df_economia['Indicator Code'] == indicador_pib_crecimiento].copy()
    if not df_crecimiento.empty:
        # Ordenar por año
        df_crecimiento = df_crecimiento.sort_values('Año')
        
        # Añadir barras de crecimiento
        fig.add_trace(
            go.Bar(
                x=df_crecimiento['Año'],
                y=df_crecimiento['Valor'],
                name='Crecimiento del PIB',
                marker_color='rgba(26, 118, 255, 0.7)'
            ),
            row=1, col=2
        )
        
        # Añadir línea de cero
        fig.add_shape(
            type="line",
            x0=df_crecimiento['Año'].min(),
            y0=0,
            x1=df_crecimiento['Año'].max(),
            y1=0,
            line=dict(color="red", width=1.5, dash="dot"),
            row=1, col=2
        )
    
    # 3. Gráfico de exportaciones e importaciones
    df_exp = df_economia[df_economia['Indicator Code'] == indicador_exportaciones].copy()
    df_imp = df_economia[df_economia['Indicator Code'] == indicador_importaciones].copy()
    
    if not df_exp.empty and not df_imp.empty:
        # Ordenar por año
        df_exp = df_exp.sort_values('Año')
        df_imp = df_imp.sort_values('Año')
        
        # Añadir línea de exportaciones
        fig.add_trace(
            go.Scatter(
                x=df_exp['Año'],
                y=df_exp['Valor'],
                mode='lines',
                name='Exportaciones',
                line=dict(color='rgba(0, 128, 0, 0.8)', width=2)
            ),
            row=2, col=1
        )
        
        # Añadir línea de importaciones
        fig.add_trace(
            go.Scatter(
                x=df_imp['Año'],
                y=df_imp['Valor'],
                mode='lines',
                name='Importaciones',
                line=dict(color='rgba(220, 20, 60, 0.8)', width=2)
            ),
            row=2, col=1
        )
    
    # 4. Variación del PIB respecto al 2000
    if not df_pib.empty:
        # Obtener valor del año 2000
        valor_2000 = df_pib[df_pib['Año'] == 2000]['Valor'].values
        
        if len(valor_2000) > 0:
            valor_2000 = valor_2000[0]
            
            # Calcular variación porcentual
            df_pib['Var_2000'] = ((df_pib['Valor'] / valor_2000) - 1) * 100
            
            # Añadir barras de variación
            fig.add_trace(
                go.Bar(
                    x=df_pib['Año'],
                    y=df_pib['Var_2000'],
                    name='Variación vs 2000',
                    marker_color='rgba(50, 171, 96, 0.7)'
                ),
                row=2, col=2
            )
    
    # Configurar diseño
    fig.update_layout(
        title="Dashboard Económico de Colombia",
        template="plotly_white",
        width=ancho,
        height=alto,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Actualizar ejes
    fig.update_yaxes(title_text="Miles de millones de US$", row=1, col=1)
    fig.update_yaxes(title_text="% anual", row=1, col=2)
    fig.update_yaxes(title_text="% del PIB", row=2, col=1)
    fig.update_yaxes(title_text="% variación", row=2, col=2)
    
    fig.update_xaxes(dtick=5, row=1, col=1)
    fig.update_xaxes(dtick=5, row=1, col=2)
    fig.update_xaxes(dtick=5, row=2, col=1)
    fig.update_xaxes(dtick=5, row=2, col=2)
    
    return fig

def crear_dashboard_social(dfs, ancho=1000, alto=800):
    """
    Crea un dashboard social compuesto con múltiples visualizaciones.
    
    Args:
        dfs (dict): Diccionario con DataFrames de datos procesados
        ancho (int): Ancho del dashboard en píxeles
        alto (int): Alto del dashboard en píxeles
        
    Returns:
        go.Figure: Figura de Plotly
    """
    # Verificar si tenemos los datos necesarios
    if 'social' not in dfs:
        logger.warning("No se encontraron datos sociales")
        return None
    
    df_social = dfs['social']
    
    # Definir indicadores a mostrar
    indicador_pobreza = INDICADORES['social']['POBREZA']
    indicador_gini = INDICADORES['social']['GINI']
    indicador_desempleo = INDICADORES['social']['DESEMPLEO']
    indicador_esperanza = INDICADORES['social']['ESPERANZA_VIDA']
    
    # Crear figura con subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Pobreza (% de la población)",
            "Desigualdad (Índice de Gini)",
            "Desempleo (% de la población activa)",
            "Esperanza de vida al nacer (años)"
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    # 1. Gráfico de pobreza
    df_pobreza = df_social[df_social['Indicator Code'] == indicador_pobreza].copy()
    if not df_pobreza.empty:
        # Ordenar por año
        df_pobreza = df_pobreza.sort_values('Año')
        
        # Añadir línea de pobreza
        fig.add_trace(
            go.Scatter(
                x=df_pobreza['Año'],
                y=df_pobreza['Valor'],
                mode='lines+markers',
                name='Pobreza',
                line=dict(color='rgba(214, 39, 40, 0.8)', width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
    
    # 2. Gráfico de Gini
    df_gini = df_social[df_social['Indicator Code'] == indicador_gini].copy()
    if not df_gini.empty:
        # Ordenar por año
        df_gini = df_gini.sort_values('Año')
        
        # Añadir línea de Gini
        fig.add_trace(
            go.Scatter(
                x=df_gini['Año'],
                y=df_gini['Valor'],
                mode='lines+markers',
                name='Gini',
                line=dict(color='rgba(44, 160, 44, 0.8)', width=2),
                marker=dict(size=6)
            ),
            row=1, col=2
        )
    
    # 3. Gráfico de desempleo
    df_desempleo = df_social[df_social['Indicator Code'] == indicador_desempleo].copy()
    if not df_desempleo.empty:
        # Ordenar por año
        df_desempleo = df_desempleo.sort_values('Año')
        
        # Añadir línea de desempleo
        fig.add_trace(
            go.Scatter(
                x=df_desempleo['Año'],
                y=df_desempleo['Valor'],
                mode='lines+markers',
                name='Desempleo',
                line=dict(color='rgba(31, 119, 180, 0.8)', width=2),
                marker=dict(size=6)
            ),
            row=2, col=1
        )
        
        # Añadir área sombreada
        fig.add_trace(
            go.Scatter(
                x=df_desempleo['Año'],
                y=df_desempleo['Valor'],
                fill='tozeroy',
                fillcolor='rgba(31, 119, 180, 0.2)',
                line=dict(width=0),
                showlegend=False
            ),
            row=2, col=1
        )
    
    # 4. Gráfico de esperanza de vida
    df_esperanza = df_social[df_social['Indicator Code'] == indicador_esperanza].copy()
    if not df_esperanza.empty:
        # Ordenar por año
        df_esperanza = df_esperanza.sort_values('Año')
        
        # Añadir línea de esperanza de vida
        fig.add_trace(
            go.Scatter(
                x=df_esperanza['Año'],
                y=df_esperanza['Valor'],
                mode='lines+markers',
                name='Esperanza de vida',
                line=dict(color='rgba(148, 103, 189, 0.8)', width=2),
                marker=dict(size=6)
            ),
            row=2, col=2
        )
    
    # Configurar diseño
    fig.update_layout(
        title="Indicadores de Desarrollo Social de Colombia",
        template="plotly_white",
        width=ancho,
        height=alto,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Actualizar ejes
    fig.update_yaxes(title_text="% de la población", row=1, col=1)
    fig.update_yaxes(title_text="Índice (0-100)", row=1, col=2)
    fig.update_yaxes(title_text="% de la población activa", row=2, col=1)
    fig.update_yaxes(title_text="Años", row=2, col=2)
    
    fig.update_xaxes(dtick=5, row=1, col=1)
    fig.update_xaxes(dtick=5, row=1, col=2)
    fig.update_xaxes(dtick=5, row=2, col=1)
    fig.update_xaxes(dtick=5, row=2, col=2)
    
    return fig

def generar_visualizaciones_demo(dfs):
    """
    Genera visualizaciones de demostración a partir de los datos procesados.
    
    Args:
        dfs (dict): Diccionario con DataFrames de datos procesados
    """
    # Crear directorio para visualizaciones si no existe
    if not os.path.exists(DIRECTORIO_VISUALIZACIONES):
        os.makedirs(DIRECTORIO_VISUALIZACIONES)
    
    logger.info("\nGenerando visualizaciones de demostración...")
    
    # 1. Series temporales
    logger.info("\n1. Visualizaciones de series temporales:")
    
    # 1.1 Gráfico multilinea de PIB y crecimiento
    if 'economia' in dfs:
        fig_pib_crecimiento = crear_serie_temporal_multilinea(
            dfs['economia'],
            [INDICADORES['economia']['PIB_PER_CAPITA'], INDICADORES['economia']['CRECIMIENTO_PIB']],
            "Evolución del PIB per cápita y Crecimiento del PIB"
        )
        
        if fig_pib_crecimiento:
            fig_pib_crecimiento.write_html(os.path.join(DIRECTORIO_VISUALIZACIONES, "pib_crecimiento.html"))
            logger.info("  - Creado: pib_crecimiento.html")
    
    # 1.2 Gráfico de área para esperanza de vida
    if 'social' in dfs:
        fig_esperanza = crear_serie_temporal_area(
            dfs['social'],
            INDICADORES['social']['ESPERANZA_VIDA'],
            "Evolución de la Esperanza de Vida en Colombia"
        )
        
        if fig_esperanza:
            fig_esperanza.write_html(os.path.join(DIRECTORIO_VISUALIZACIONES, "esperanza_vida.html"))
            logger.info("  - Creado: esperanza_vida.html")
    
    # 1.3 Heatmap temporal para tasa de homicidios
    if 'gobernanza' in dfs:
        fig_homicidios = crear_heatmap_temporal(
            dfs['gobernanza'],
            INDICADORES['gobernanza']['HOMICIDIOS'],
            "Evolución de la Tasa de Homicidios (por 100.000 habitantes)"
        )
        
        if fig_homicidios:
            fig_homicidios.write_html(os.path.join(DIRECTORIO_VISUALIZACIONES, "homicidios_heatmap.html"))
            logger.info("  - Creado: homicidios_heatmap.html")
    
    # 2. Gráficos de comparación
    logger.info("\n2. Visualizaciones comparativas:")
    
    # 2.1 Boxplot para desempleo por períodos
    if 'social' in dfs:
        fig_desempleo_box = crear_boxplot_por_periodos(
            dfs['social'],
            INDICADORES['social']['DESEMPLEO'],
            titulo="Distribución del Desempleo por Períodos"
        )
        
        if fig_desempleo_box:
            fig_desempleo_box.write_html(os.path.join(DIRECTORIO_VISUALIZACIONES, "desempleo_boxplot.html"))
            logger.info("  - Creado: desempleo_boxplot.html")
    
    # 2.2 Barras comparativas para indicadores educativos
    if 'educacion' in dfs:
        fig_educacion_comp = crear_barras_comparativas(
            dfs['educacion'],
            [INDICADORES['educacion']['FINALIZACION_PRIMARIA'], 
             INDICADORES['educacion']['FINALIZACION_SECUNDARIA'], 
             INDICADORES['educacion']['INSCRIPCION_TERCIARIA']],
            2000, 2020,
            "Comparación de Indicadores Educativos: 2000 vs 2020"
        )
        
        if fig_educacion_comp:
            fig_educacion_comp.write_html(os.path.join(DIRECTORIO_VISUALIZACIONES, "educacion_comparativa.html"))
            logger.info("  - Creado: educacion_comparativa.html")
    
    # 2.3 Variación respecto al año base para exportaciones
    if 'economia' in dfs:
        fig_exp_var = crear_variacion_respecto_base(
            dfs['economia'],
            INDICADORES['economia']['EXPORTACIONES'],
            titulo="Variación de Exportaciones respecto al año 2000"
        )
        
        if fig_exp_var:
            fig_exp_var.write_html(os.path.join(DIRECTORIO_VISUALIZACIONES, "exportaciones_variacion.html"))
            logger.info("  - Creado: exportaciones_variacion.html")
    
    # 3. Análisis multivariable
    logger.info("\n3. Análisis multivariable:")
    
    # 3.1 Matriz de correlación para indicadores económicos
    if 'economia' in dfs:
        fig_corr_eco = crear_matriz_correlacion(
            dfs['economia'],
            [INDICADORES['economia']['PIB_PER_CAPITA'], 
             INDICADORES['economia']['CRECIMIENTO_PIB'], 
             INDICADORES['economia']['EXPORTACIONES'], 
             INDICADORES['economia']['IMPORTACIONES']],
            titulo="Correlación entre Indicadores Económicos"
        )
        
        if fig_corr_eco:
            fig_corr_eco.write_html(os.path.join(DIRECTORIO_VISUALIZACIONES, "correlacion_economia.html"))
            logger.info("  - Creado: correlacion_economia.html")
    
    # 3.2 Scatter matrix para indicadores sociales
    if 'social' in dfs:
        fig_scatter_social = crear_scatter_matrix(
            dfs['social'],
            [INDICADORES['social']['ESPERANZA_VIDA'], 
             INDICADORES['social']['POBLACION_URBANA'], 
             INDICADORES['social']['GINI'], 
             INDICADORES['social']['DESEMPLEO']],
            titulo="Relaciones entre Indicadores Sociales"
        )
        
        if fig_scatter_social:
            fig_scatter_social.write_html(os.path.join(DIRECTORIO_VISUALIZACIONES, "scatter_social.html"))
            logger.info("  - Creado: scatter_social.html")
    
    # 3.3 Gráfico de burbujas (PIB per cápita, esperanza de vida, población)
    if 'economia' in dfs and 'social' in dfs:
        # Combinar DataFrames para este análisis específico
        df_combinado = pd.concat([dfs['economia'], dfs['social']])
        
        fig_burbujas = crear_grafico_burbujas(
            df_combinado,
            INDICADORES['economia']['PIB_PER_CAPITA'],
            INDICADORES['social']['ESPERANZA_VIDA'],
            INDICADORES['social']['POBLACION'],
            años_seleccionados=[2000, 2005, 2010, 2015, 2020],
            titulo="Relación entre PIB per cápita, Esperanza de Vida y Población"
        )
        
        if fig_burbujas:
            fig_burbujas.write_html(os.path.join(DIRECTORIO_VISUALIZACIONES, "burbujas_desarrollo.html"))
            logger.info("  - Creado: burbujas_desarrollo.html")
    
    # 4. Mapas y visualizaciones geoespaciales
    logger.info("\n4. Visualizaciones geoespaciales:")
    
    # 4.1 Mapa de Colombia con Folium
    try:
        mapa_colombia = crear_mapa_colombia_folium(DIRECTORIO_ACTUAL, "Mapa de Colombia")
        mapa_colombia.save(os.path.join(DIRECTORIO_VISUALIZACIONES, "mapa_colombia.html"))
        logger.info("  - Creado: mapa_colombia.html")
    except Exception as e:
        logger.error(f"  - Error al crear mapa de Colombia: {e}")
    
    # 4.2 Heatmap para energía renovable
    if 'ambiente' in dfs:
        fig_energia = crear_heatmap_indicador_tiempo(
            dfs['ambiente'],
            INDICADORES['ambiente']['ENERGIA_RENOVABLE'],
            "Evolución del Consumo de Energía Renovable (%)"
        )
        
        if fig_energia:
            fig_energia.write_html(os.path.join(DIRECTORIO_VISUALIZACIONES, "energia_renovable_heatmap.html"))
            logger.info("  - Creado: energia_renovable_heatmap.html")
    
    # 5. Dashboards compuestos
    logger.info("\n5. Dashboards compuestos:")
    
    # 5.1 Dashboard económico
    fig_dash_eco = crear_dashboard_economico(dfs)
    if fig_dash_eco:
        fig_dash_eco.write_html(os.path.join(DIRECTORIO_VISUALIZACIONES, "dashboard_economico.html"))
        logger.info("  - Creado: dashboard_economico.html")
    
    # 5.2 Dashboard social
    fig_dash_social = crear_dashboard_social(dfs)
    if fig_dash_social:
        fig_dash_social.write_html(os.path.join(DIRECTORIO_VISUALIZACIONES, "dashboard_social.html"))
        logger.info("  - Creado: dashboard_social.html")
    
    logger.info(f"\nVisualizaciones guardadas en: {DIRECTORIO_VISUALIZACIONES}")
    logger.info("Abra estos archivos HTML en un navegador para visualizarlos interactivamente.")

def main():
    """Función principal que ejecuta todo el proceso de visualización"""
    logger.info("Iniciando generación de visualizaciones avanzadas...")
    
    # 1. Cargar datos procesados
    dfs = cargar_datos_procesados()
    
    if not dfs:
        logger.error("No se pudieron cargar los datos procesados. Ejecute primero el script de preprocesamiento.")
        return
    
    # 2. Generar visualizaciones de demostración
    generar_visualizaciones_demo(dfs)
    
    logger.info("\n¡Proceso de generación de visualizaciones completado con éxito!")

if __name__ == "__main__":
    main()