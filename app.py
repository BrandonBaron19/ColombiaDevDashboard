"""
Dashboard Colombia - Indicadores de Desarrollo

Esta aplicación Streamlit muestra un análisis interactivo de los indicadores de desarrollo
de Colombia en diferentes dimensiones: económica, social, educativa, ambiental y de gobernanza.

Los datos provienen de indicadores del Banco Mundial procesados previamente.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import datetime
import json

# Configuraciones de página
st.set_page_config(
    page_title="Dashboard Colombia - Indicadores de Desarrollo",
    page_icon="🇨🇴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constantes
DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
DIRECTORIO_DATOS = os.path.join(DIRECTORIO_ACTUAL, "datos_procesados")

# Definición de indicadores clave por categoría
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

# Categorización de indicadores para análisis
CATEGORIAS_ANALISIS = {
    'Economía': ['NY.GDP.PCAP.CD', 'NY.GDP.MKTP.KD.ZG', 'FP.CPI.TOTL.ZG', 'NE.EXP.GNFS.ZS', 'NE.IMP.GNFS.ZS'],
    'Social': ['SI.POV.NAHC', 'SI.POV.GINI', 'SL.UEM.TOTL.ZS', 'SP.DYN.LE00.IN', 'SP.URB.TOTL.IN.ZS'],
    'Educación': ['SE.PRM.CMPT.ZS', 'SE.SEC.CMPT.LO.ZS', 'SE.TER.ENRR', 'SE.XPD.TOTL.GD.ZS'],
    'Medio Ambiente': ['EG.FEC.RNEW.ZS', 'EN.ATM.CO2E.PC', 'AG.LND.FRST.ZS', 'ER.H2O.FWTL.ZS'],
    'Seguridad y Gobernanza': ['VC.IHR.PSRC.P5', 'VA.EST', 'RL.EST', 'CC.EST', 'GE.EST']
}

# Definición de alternativas para indicadores clave
INDICADORES_ALTERNATIVOS = {
    'SI.POV.NAHC': ['SI.POV.DDAY', 'SI.POV.LMIC', 'SI.POV.UMIC', 'SI.POV.GINI'],  # Alternativas para pobreza
    'NY.GDP.PCAP.CD': ['NY.GDP.MKTP.CD', 'NY.GDP.MKTP.KD'],  # Alternativas para PIB
    'SL.UEM.TOTL.ZS': [],  # Sin alternativas para desempleo
    'SP.DYN.LE00.IN': ['SP.DYN.LE00.MA.IN', 'SP.DYN.LE00.FE.IN'],  # Alternativas para esperanza de vida
    'VC.IHR.PSRC.P5': []  # Sin alternativas para homicidios
}

# Indicadores para panorama general
INDICADORES_PANORAMA = {
    'PIB per cápita (US$)': 'NY.GDP.PCAP.CD',
    'Desigualdad (Índice de Gini)': 'SI.POV.GINI',
    'Desempleo (%)': 'SL.UEM.TOTL.ZS',
    'Esperanza de vida (años)': 'SP.DYN.LE00.IN',
    'Tasa de homicidios (por 100.000)': 'VC.IHR.PSRC.P5'
}

# Eventos históricos para visualizaciones de seguridad
EVENTOS_HISTORICOS = {
    2002: "Inicio gobierno Uribe",
    2010: "Inicio gobierno Santos",
    2016: "Acuerdo de paz con FARC"
}

# Definición de archivos a cargar
ARCHIVOS_DATOS = {
    'maestro': 'datos_maestro.csv',
    'economia': 'datos_economia.csv',
    'social': 'datos_social.csv',
    'educacion': 'datos_educacion.csv',
    'ambiente': 'datos_ambiente.csv',
    'gobernanza': 'datos_gobernanza.csv'
}

# Función para cargar los datos
@st.cache_data
def cargar_datos():
    """
    Carga los datos procesados desde los archivos CSV.
    
    Returns:
        dict: Diccionario con DataFrames
    """
    # Diccionario para almacenar los DataFrames
    dfs = {}
    
    # Cargar cada archivo
    for nombre, archivo in ARCHIVOS_DATOS.items():
        ruta_completa = os.path.join(DIRECTORIO_DATOS, archivo)
        if os.path.exists(ruta_completa):
            try:
                df = pd.read_csv(ruta_completa, encoding='utf-8')
                dfs[nombre] = df
            except Exception as e:
                st.error(f"Error al cargar {archivo}: {e}")
        else:
            st.warning(f"Archivo no encontrado: {ruta_completa}")
    
    return dfs

# Funciones para crear visualizaciones
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
        return None
    
    # Ordenar por año
    datos_filtrados = datos_filtrados.sort_values('Año')
    
    # Obtener el valor para el año base
    valor_base = datos_filtrados[datos_filtrados['Año'] == año_base]['Valor'].values
    
    if len(valor_base) == 0:
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

def crear_mapa_colombia():
    """
    Crea un mapa básico de Colombia usando Folium.
    
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
    
    # Añadir marcador para Bogotá
    folium.Marker(
        [4.7110, -74.0721],
        popup="Bogotá, Colombia",
        tooltip="Capital de Colombia"
    ).add_to(mapa)
    
    return mapa

def crear_dashboard_economico(df_economia):
    """
    Crea un dashboard económico compuesto con múltiples visualizaciones.
    
    Args:
        df_economia (pd.DataFrame): DataFrame con datos económicos
        
    Returns:
        go.Figure: Figura de Plotly
    """
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
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=700
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
        periodos = {
            '2000-2005': (2000, 2005),
            '2006-2010': (2006, 2010),
            '2011-2015': (2011, 2015),
            '2016-2020': (2016, 2020),
            '2021-2023': (2021, 2023)
        }
    
    # Filtrar datos para el indicador
    datos_filtrados = df[df['Indicator Code'] == codigo_indicador].copy()
    
    if datos_filtrados.empty:
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

# Función para mostrar panorama general
def mostrar_panorama_general(dfs, year_start, year_end):
    """
    Muestra un panorama general de Colombia con indicadores clave.
    
    Args:
        dfs (dict): Diccionario con DataFrames de datos
        year_start (int): Año de inicio para filtrar
        year_end (int): Año de fin para filtrar
    """
    st.header("Panorama General de Colombia")
    
    # Introducción
    st.markdown("""
    Esta sección presenta un resumen de los principales indicadores de desarrollo de Colombia
    que permiten comprender la evolución del país en diferentes dimensiones. Seleccione el rango
    de años en el panel lateral para ajustar los datos mostrados.
    """)
    
    # Filtrar los datos por rango de años
    df_filtered = {}
    for key, df in dfs.items():
        if key != 'maestro':  # No filtrar df_maestro
            df_filtered[key] = df[(df['Año'] >= year_start) & (df['Año'] <= year_end)]
        else:
            df_filtered[key] = df
    
    # Métricas principales en tarjetas
    st.subheader("Indicadores Clave (últimos datos disponibles)")
    
    # Crear fila de métricas
    cols = st.columns(len(INDICADORES_PANORAMA))
    
    # Para cada indicador clave
    for i, (nombre, codigo) in enumerate(INDICADORES_PANORAMA.items()):
        df_temp = None
        datos_encontrados = False
        codigo_usado = codigo
        
        # Primero buscar el indicador principal
        for df_name, df in df_filtered.items():
            if df_name != 'maestro' and codigo in df['Indicator Code'].values:
                df_temp = df
                datos_encontrados = True
                break
        
        # Si no encuentra el indicador principal, probar con alternativas
        if not datos_encontrados and codigo in INDICADORES_ALTERNATIVOS:
            for codigo_alt in INDICADORES_ALTERNATIVOS[codigo]:
                for df_name, df in df_filtered.items():
                    if df_name != 'maestro' and codigo_alt in df['Indicator Code'].values:
                        df_temp = df
                        datos_encontrados = True
                        codigo_usado = codigo_alt
                        # Actualizar el nombre para indicar que es un indicador alternativo
                        if codigo == 'SI.POV.NAHC' and codigo_alt == 'SI.POV.DDAY':
                            nombre = 'Pobreza ($3.65/día) (%)'
                        elif codigo == 'SI.POV.NAHC':
                            nombre = 'Desigualdad (alt.)'
                        break
                if datos_encontrados:
                    break
        
        # Mostrar el resultado en la columna correspondiente
        with cols[i]:
            if datos_encontrados:
                # Buscar el valor más reciente disponible
                ultimo_dato = df_temp[df_temp['Indicator Code'] == codigo_usado].sort_values('Año', ascending=False)
                
                if not ultimo_dato.empty and not pd.isna(ultimo_dato.iloc[0]['Valor']):
                    valor = ultimo_dato.iloc[0]['Valor']
                    año = int(ultimo_dato.iloc[0]['Año'])
                    
                    # Si es el indicador de pobreza alternativo, añadir un asterisco
                    if codigo == 'SI.POV.NAHC' and codigo_usado != codigo:
                        st.metric(
                            label=nombre + "*",
                            value=f"{valor:.1f}" if isinstance(valor, (int, float)) else valor,
                            delta=f"Dato de {año}"
                        )
                        st.caption("*Indicador alternativo")
                    else:
                        st.metric(
                            label=nombre,
                            value=f"{valor:.1f}" if isinstance(valor, (int, float)) else valor,
                            delta=f"Dato de {año}"
                        )
                else:
                    st.metric(label=nombre, value="No disponible", delta="Sin datos válidos")
            else:
                st.metric(label=nombre, value="No disponible", delta="Indicador no encontrado")
    
    # Gráfico de evolución del PIB y crecimiento
    st.subheader("Evolución Económica")
    
    if 'economia' in df_filtered:
        fig_pib = crear_serie_temporal_multilinea(
            df_filtered['economia'],
            [INDICADORES['economia']['PIB_PER_CAPITA'], INDICADORES['economia']['CRECIMIENTO_PIB']],
            "Evolución del PIB per cápita y Crecimiento del PIB"
        )
        
        if fig_pib:
            st.plotly_chart(fig_pib, use_container_width=True)
        else:
            st.warning("No hay datos suficientes para visualizar la evolución económica.")
    else:
        st.warning("No se encuentran datos económicos para visualizar.")
    
    # Indicadores sociales
    st.subheader("Desarrollo Social")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'social' in df_filtered:
            # Intentar primero con SI.POV.NAHC
            indicador_pobreza_principal = INDICADORES['social']['POBREZA']
            fig_pobreza = None
            
            # Verificar si el indicador principal existe y tiene datos
            if indicador_pobreza_principal in df_filtered['social']['Indicator Code'].values:
                datos_temp = df_filtered['social'][df_filtered['social']['Indicator Code'] == indicador_pobreza_principal]
                if not datos_temp.empty and not datos_temp['Valor'].isna().all():
                    fig_pobreza = crear_variacion_respecto_base(
                        df_filtered['social'],
                        indicador_pobreza_principal,
                        año_base=year_start,
                        titulo=f"Variación de la Pobreza (Base: {year_start})"
                    )
            
            # Si no hay gráfico con el indicador principal, probar con alternativas
            if fig_pobreza is None:
                for alt_code in INDICADORES_ALTERNATIVOS['SI.POV.NAHC']:
                    if alt_code in df_filtered['social']['Indicator Code'].values:
                        fig_pobreza = crear_variacion_respecto_base(
                            df_filtered['social'],
                            alt_code,
                            año_base=year_start,
                            titulo=f"Variación de Pobreza (indicador alternativo, Base: {year_start})"
                        )
                        if fig_pobreza:
                            st.info(f"Se está utilizando un indicador alternativo porque el indicador principal de pobreza no tiene datos suficientes.")
                            break
            
            if fig_pobreza:
                st.plotly_chart(fig_pobreza, use_container_width=True)
            else:
                st.warning("No hay datos suficientes sobre pobreza para crear la visualización.")
        else:
            st.warning("No se encuentran datos sociales para visualizar.")
    
    with col2:
        if 'social' in df_filtered:
            fig_desempleo = crear_serie_temporal_multilinea(
                df_filtered['social'],
                [INDICADORES['social']['DESEMPLEO']],
                "Evolución de la Tasa de Desempleo"
            )
            
            if fig_desempleo:
                st.plotly_chart(fig_desempleo, use_container_width=True)
            else:
                st.warning("No hay datos suficientes sobre desempleo.")
        else:
            st.warning("No se encuentran datos sociales para visualizar.")
    
    # Mapa de Colombia
    st.subheader("Mapa de Colombia")
    mapa = crear_mapa_colombia()
    folium_static(mapa, width=800, height=500)

# Función para mostrar desarrollo económico
def mostrar_desarrollo_economico(dfs, chart_type, year_start, year_end):
    """
    Muestra visualizaciones relacionadas con el desarrollo económico.
    
    Args:
        dfs (dict): Diccionario con DataFrames de datos
        chart_type (str): Tipo de visualización a mostrar
        year_start (int): Año de inicio para filtrar
        year_end (int): Año de fin para filtrar
    """
    st.header("Desarrollo Económico de Colombia")
    
    # Filtrar por rango de años
    if 'economia' in dfs:
        df_eco = dfs['economia']
        df_eco_filtered = df_eco[(df_eco['Año'] >= year_start) & (df_eco['Año'] <= year_end)]
        
        if chart_type == "Dashboard Económico":
            # Mostrar dashboard económico completo
            fig = crear_dashboard_economico(df_eco_filtered)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hay suficientes datos para crear el dashboard económico.")
        
        elif chart_type == "Evolución del PIB":
            st.subheader("Evolución del PIB de Colombia")
            
            # Selector de indicador PIB
            pib_indicators = {
                'PIB (US$ a precios actuales)': INDICADORES['economia']['PIB'],
                'PIB per cápita (US$ a precios actuales)': INDICADORES['economia']['PIB_PER_CAPITA'],
                'Crecimiento del PIB (% anual)': INDICADORES['economia']['CRECIMIENTO_PIB'],
                'PIB per cápita (US$ a precios constantes de 2015)': 'NY.GDP.PCAP.KD'
            }
            
            selected_pib_indicator = st.selectbox(
                "Seleccione el indicador del PIB:",
                list(pib_indicators.keys())
            )
            
            if selected_pib_indicator:
                codigo_indicador = pib_indicators[selected_pib_indicator]
                
                # Crear gráfico según el tipo de indicador
                if 'KD.ZG' in codigo_indicador:  # Es un indicador de crecimiento
                    fig = px.bar(
                        df_eco_filtered[df_eco_filtered['Indicator Code'] == codigo_indicador],
                        x='Año',
                        y='Valor',
                        title=f"Evolución de {selected_pib_indicator}",
                        labels={'Valor': '% anual', 'Año': 'Año'},
                        color_discrete_sequence=['#1f77b4']
                    )
                    
                    # Añadir línea de cero
                    fig.add_shape(
                        type="line",
                        x0=year_start,
                        y0=0,
                        x1=year_end,
                        y1=0,
                        line=dict(color="red", width=1.5, dash="dot")
                    )
                else:  # Es un indicador de nivel
                    fig = px.area(
                        df_eco_filtered[df_eco_filtered['Indicator Code'] == codigo_indicador],
                        x='Año',
                        y='Valor',
                        title=f"Evolución de {selected_pib_indicator}",
                        labels={'Valor': 'Valor (US$)', 'Año': 'Año'},
                        color_discrete_sequence=['#7fb3d5']
                    )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Añadir análisis de tendencia
                if len(df_eco_filtered[df_eco_filtered['Indicator Code'] == codigo_indicador]) > 2:
                    data_for_trend = df_eco_filtered[df_eco_filtered['Indicator Code'] == codigo_indicador].sort_values('Año')
                    
                    if not data_for_trend.empty:
                        x = data_for_trend['Año'].values
                        y = data_for_trend['Valor'].values
                        
                        if len(x) > 2 and len(y) > 2:
                            coefs = np.polyfit(x, y, 2)
                            trend_eq = f"Tendencia: {coefs[0]:.6f}x² + {coefs[1]:.2f}x + {coefs[2]:.2f}"
                            
                            st.info(f"""
                            **Análisis de Tendencia:**
                            
                            La evolución del {selected_pib_indicator} muestra una tendencia polinómica de segundo grado.
                            
                            {trend_eq}
                            
                            El valor más reciente ({int(data_for_trend['Año'].max())}) es {data_for_trend.iloc[-1]['Valor']:.2f}.
                            """)
            else:
                st.warning("Seleccione un indicador para visualizar.")
        
        elif chart_type == "Comercio Internacional":
            st.subheader("Comercio Internacional")
            
            # Indicadores de comercio internacional
            comercio_indicators = {
                'Exportaciones de bienes y servicios (% del PIB)': INDICADORES['economia']['EXPORTACIONES'],
                'Importaciones de bienes y servicios (% del PIB)': INDICADORES['economia']['IMPORTACIONES'],
                'Exportaciones de productos de alta tecnología (US$ a precios actuales)': 'TX.VAL.TECH.CD',
                'Balanza comercial (% del PIB)': 'NE.RSB.GNFS.ZS'
            }
            
            # Permitir selección múltiple para comparar
            selected_comercio_indicators = st.multiselect(
                "Seleccione indicadores de comercio para comparar:",
                list(comercio_indicators.keys()),
                default=list(comercio_indicators.keys())[:2]
            )
            
            if selected_comercio_indicators:
                # Obtener códigos de los indicadores seleccionados
                codigos_seleccionados = [comercio_indicators[ind] for ind in selected_comercio_indicators]
                
                # Crear gráfico de líneas múltiples
                fig = crear_serie_temporal_multilinea(
                    df_eco_filtered,
                    codigos_seleccionados,
                    "Evolución de Indicadores de Comercio Internacional"
                )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No hay datos suficientes para los indicadores seleccionados.")
                
                # Comparación de períodos
                if len(selected_comercio_indicators) > 0:
                    st.subheader("Comparación entre períodos")
                    
                    # Definir períodos
                    periodos = {
                        f'{year_start}-{year_start+4}': (year_start, year_start+4),
                        f'{year_end-4}-{year_end}': (year_end-4, year_end)
                    }
                    
                    # Calcular promedios por período
                    datos_periodos = {}
                    
                    for nombre, (inicio, fin) in periodos.items():
                        datos_periodo = df_eco_filtered[
                            (df_eco_filtered['Año'] >= inicio) & 
                            (df_eco_filtered['Año'] <= fin) &
                            (df_eco_filtered['Indicator Code'].isin(codigos_seleccionados))
                        ]
                        
                        # Agrupar por indicador y calcular promedio
                        if not datos_periodo.empty:
                            promedios = datos_periodo.groupby('Indicator Name')['Valor'].mean()
                            datos_periodos[nombre] = promedios
                    
                    # Crear DataFrame para la tabla
                    if datos_periodos and len(datos_periodos) == 2:
                        df_comparacion = pd.DataFrame(datos_periodos)
                        df_comparacion['Variación (%)'] = ((df_comparacion[list(periodos.keys())[1]] / 
                                                         df_comparacion[list(periodos.keys())[0]]) - 1) * 100
                        
                        # Mostrar tabla
                        st.dataframe(df_comparacion.style.format({
                            list(periodos.keys())[0]: "{:.2f}",
                            list(periodos.keys())[1]: "{:.2f}",
                            'Variación (%)': "{:.2f}%"
                        }))
            else:
                st.warning("Seleccione al menos un indicador para visualizar.")
        
        elif chart_type == "Inflación y Estabilidad":
            st.subheader("Inflación y Estabilidad Económica")
            
            # Indicadores de inflación y estabilidad
            inflacion_indicators = {
                'Inflación, precios al consumidor (% anual)': INDICADORES['economia']['INFLACION'],
                'Inflación, precios al productor (% anual)': 'FP.WPI.TOTL',
                'Tipo de cambio (UMN por US$, promedio del período)': 'PA.NUS.FCRF'
            }
            
            # Permitir selección de indicador
            selected_inflacion_indicator = st.selectbox(
                "Seleccione un indicador de inflación:",
                list(inflacion_indicators.keys())
            )
            
            if selected_inflacion_indicator:
                codigo_indicador = inflacion_indicators[selected_inflacion_indicator]
                
                # Crear gráfico
                datos_inflacion = df_eco_filtered[df_eco_filtered['Indicator Code'] == codigo_indicador]
                
                if not datos_inflacion.empty:
                    fig = px.line(
                        datos_inflacion,
                        x='Año',
                        y='Valor',
                        title=f"Evolución de {selected_inflacion_indicator}",
                        labels={'Valor': 'Valor', 'Año': 'Año'},
                        markers=True
                    )
                    
                    # Añadir área sombreada
                    fig.add_trace(
                        go.Scatter(
                            x=datos_inflacion['Año'],
                            y=datos_inflacion['Valor'],
                            fill='tozeroy',
                            fillcolor='rgba(31, 119, 180, 0.2)',
                            line=dict(width=0),
                            showlegend=False
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Análisis de inflación
                    if 'inflación' in selected_inflacion_indicator.lower():
                        max_inflation = datos_inflacion['Valor'].max()
                        max_year = datos_inflacion.loc[datos_inflacion['Valor'].idxmax(), 'Año']
                        avg_inflation = datos_inflacion['Valor'].mean()
                        
                        st.info(f"""
                        **Análisis de Inflación:**
                        
                        - Inflación máxima: {max_inflation:.2f}% (año {int(max_year)})
                        - Inflación promedio en el período: {avg_inflation:.2f}%
                        - La inflación muestra una {'tendencia a la baja' if datos_inflacion.iloc[-1]['Valor'] < avg_inflation else 'tendencia al alza'} en los últimos años.
                        """)
                else:
                    st.warning(f"No hay datos disponibles para {selected_inflacion_indicator}.")
            else:
                st.warning("Seleccione un indicador para visualizar.")
        
        elif chart_type == "Variación temporal":
            st.subheader("Análisis de Variación Temporal")
            
            # Indicadores disponibles
            indicadores_disponibles = df_eco_filtered['Indicator Name'].unique()
            
            # Permitir selección de indicador
            selected_indicator = st.selectbox(
                "Seleccione un indicador para analizar su variación:",
                indicadores_disponibles
            )
            
            if selected_indicator:
                # Obtener código del indicador
                codigo_indicador = df_eco_filtered[df_eco_filtered['Indicator Name'] == selected_indicator]['Indicator Code'].iloc[0]
                
                # Permitir selección de año base
                años_disponibles = sorted(df_eco_filtered[df_eco_filtered['Indicator Code'] == codigo_indicador]['Año'].unique())
                
                if años_disponibles:
                    año_base = st.selectbox(
                        "Seleccione año base para el cálculo de variación:",
                        años_disponibles,
                        index=0
                    )
                    
                    # Crear gráfico de variación
                    fig = crear_variacion_respecto_base(
                        df_eco_filtered,
                        codigo_indicador,
                        año_base=año_base,
                        titulo=f"Variación de {selected_indicator} respecto al año {año_base}"
                    )
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"No hay datos suficientes para calcular la variación desde {año_base}.")
                else:
                    st.warning("No hay años disponibles para el indicador seleccionado.")
            else:
                st.warning("Seleccione un indicador para visualizar.")
    else:
        st.error("No se encuentran datos económicos para visualizar.")

# Función para mostrar desarrollo social
def mostrar_desarrollo_social(dfs, chart_type, year_start, year_end):
    """
    Muestra visualizaciones relacionadas con el desarrollo social.
    
    Args:
        dfs (dict): Diccionario con DataFrames de datos
        chart_type (str): Tipo de visualización a mostrar
        year_start (int): Año de inicio para filtrar
        year_end (int): Año de fin para filtrar
    """
    st.header("Desarrollo Social de Colombia")
    
    # Filtrar por rango de años
    if 'social' in dfs:
        df_social = dfs['social']
        df_social_filtered = df_social[(df_social['Año'] >= year_start) & (df_social['Año'] <= year_end)]
        
        if chart_type == "Pobreza y Desigualdad":
            st.subheader("Pobreza y Desigualdad")
            
            # Indicadores de pobreza y desigualdad
            indicadores = {
                'Tasa de pobreza (línea nacional)': INDICADORES['social']['POBREZA'],
                'Índice de Gini': INDICADORES['social']['GINI']
            }
            
            # Seleccionar indicador
            selected_indicator = st.selectbox(
                "Seleccione un indicador:",
                list(indicadores.keys())
            )
            
            if selected_indicator:
                codigo_indicador = indicadores[selected_indicator]
                
                # Crear gráfico de evolución
                datos_ind = df_social_filtered[df_social_filtered['Indicator Code'] == codigo_indicador]
                
                if not datos_ind.empty:
                    fig = px.line(
                        datos_ind,
                        x='Año',
                        y='Valor',
                        title=f"Evolución de {selected_indicator}",
                        labels={'Valor': 'Valor', 'Año': 'Año'},
                        markers=True
                    )
                    
                    # Mejorar el diseño
                    fig.update_layout(
                        template="plotly_white",
                        hovermode="x"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Análisis de tendencia
                    if len(datos_ind) > 2:
                        datos_ind = datos_ind.sort_values('Año')
                        first_value = datos_ind.iloc[0]['Valor']
                        last_value = datos_ind.iloc[-1]['Valor']
                        change = ((last_value / first_value) - 1) * 100
                        
                        st.info(f"""
                        **Análisis de tendencia:**
                        
                        - Valor en {int(datos_ind.iloc[0]['Año'])}: {first_value:.2f}
                        - Valor en {int(datos_ind.iloc[-1]['Año'])}: {last_value:.2f}
                        - Cambio porcentual: {change:.1f}% {'de aumento' if change > 0 else 'de reducción'}
                        
                        {selected_indicator} ha {'aumentado' if change > 0 else 'disminuido'} en el período analizado.
                        """)
                else:
                    st.warning(f"No hay datos disponibles para {selected_indicator}.")
            else:
                st.warning("Seleccione un indicador para visualizar.")
        
        elif chart_type == "Demografía":
            st.subheader("Indicadores Demográficos")
            
            # Indicadores demográficos
            indicadores = {
                'Población total': INDICADORES['social']['POBLACION'],
                'Población urbana (% del total)': INDICADORES['social']['POBLACION_URBANA'],
                'Tasa de fertilidad (nacimientos por mujer)': 'SP.DYN.TFRT.IN',
                'Esperanza de vida al nacer (años)': INDICADORES['social']['ESPERANZA_VIDA']
            }
            
            # Selección múltiple de indicadores
            selected_indicators = st.multiselect(
                "Seleccione indicadores para visualizar:",
                list(indicadores.keys()),
                default=[list(indicadores.keys())[0]]
            )
            
            if selected_indicators:
                codigos = [indicadores[ind] for ind in selected_indicators]
                
                # Crear gráfico multilinea
                fig = crear_serie_temporal_multilinea(
                    df_social_filtered,
                    codigos,
                    "Evolución de Indicadores Demográficos"
                )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No hay datos suficientes para los indicadores seleccionados.")
                
                # Visualización especial para población total
                if 'Población total' in selected_indicators:
                    st.subheader("Crecimiento Poblacional")
                    
                    datos_pob = df_social_filtered[df_social_filtered['Indicator Code'] == INDICADORES['social']['POBLACION']]
                    
                    if not datos_pob.empty:
                        # Calcular tasa de crecimiento anual
                        datos_pob = datos_pob.sort_values('Año')
                        datos_pob['Tasa_Crecimiento'] = datos_pob['Valor'].pct_change() * 100
                        
                        # Eliminar primer registro que tendrá NaN
                        datos_pob = datos_pob.dropna(subset=['Tasa_Crecimiento'])
                        
                        if not datos_pob.empty:
                            fig = px.bar(
                                datos_pob,
                                x='Año',
                                y='Tasa_Crecimiento',
                                title="Tasa de crecimiento poblacional anual (%)",
                                labels={'Tasa_Crecimiento': '% anual', 'Año': 'Año'},
                                color_discrete_sequence=['#2471A3']
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Seleccione al menos un indicador para visualizar.")
        
        elif chart_type == "Empleo":
            st.subheader("Empleo y Mercado Laboral")
            
            # Indicadores de empleo
            indicador_desempleo = INDICADORES['social']['DESEMPLEO']  # Código para desempleo
            
            # Crear gráfico de desempleo
            datos_desempleo = df_social_filtered[df_social_filtered['Indicator Code'] == indicador_desempleo]
            
            if not datos_desempleo.empty:
                # Gráfico de evolución
                fig = px.line(
                    datos_desempleo,
                    x='Año',
                    y='Valor',
                    title="Evolución de la Tasa de Desempleo",
                    labels={'Valor': '% de la población activa', 'Año': 'Año'},
                    markers=True
                )
                
                # Añadir área sombreada
                fig.add_trace(
                    go.Scatter(
                        x=datos_desempleo['Año'],
                        y=datos_desempleo['Valor'],
                        fill='tozeroy',
                        fillcolor='rgba(214, 39, 40, 0.2)',
                        line=dict(width=0),
                        showlegend=False
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Análisis por períodos
                st.subheader("Análisis del Desempleo por Períodos")
                
                fig_boxplot = crear_boxplot_por_periodos(
                    df_social_filtered,
                    indicador_desempleo,
                    titulo="Distribución del Desempleo por Períodos"
                )
                
                if fig_boxplot:
                    st.plotly_chart(fig_boxplot, use_container_width=True)
                else:
                    st.warning("No hay datos suficientes para crear el boxplot de desempleo.")
            else:
                st.warning("No hay datos disponibles sobre desempleo.")
        
        elif chart_type == "Indicadores Sociales Combinados":
            st.subheader("Análisis Combinado de Indicadores Sociales")
            
            # Crear gráfico de burbujas relacionando diversos indicadores
            st.write("Este gráfico muestra la relación entre esperanza de vida, PIB per cápita y población total a lo largo del tiempo.")
            
            # Combinar datos sociales y económicos si están disponibles
            if 'economia' in dfs:
                df_combinado = pd.concat([df_social_filtered, dfs['economia'][(dfs['economia']['Año'] >= year_start) & (dfs['economia']['Año'] <= year_end)]])
                
                # Seleccionar años para el gráfico de burbujas
                step = max(1, (year_end - year_start) // 5)  # Mostrar máximo 5 puntos
                años_seleccionados = list(range(year_start, year_end + 1, step))
                
                fig_burbujas = crear_grafico_burbujas(
                    df_combinado,
                    INDICADORES['economia']['PIB_PER_CAPITA'],
                    INDICADORES['social']['ESPERANZA_VIDA'],
                    INDICADORES['social']['POBLACION'],
                    años_seleccionados=años_seleccionados,
                    titulo="Relación entre PIB per cápita, Esperanza de Vida y Población"
                )
                
                if fig_burbujas:
                    st.plotly_chart(fig_burbujas, use_container_width=True)
                    
                    # Añadir explicación
                    st.markdown("""
                    **Interpretación del gráfico:**
                    
                    Este gráfico de burbujas muestra tres dimensiones:
                    - **Eje X**: PIB per cápita en US$ a precios actuales
                    - **Eje Y**: Esperanza de vida al nacer en años
                    - **Tamaño de la burbuja**: Población total
                    - **Color**: Año
                    
                    Cada burbuja representa Colombia en un año específico. La progresión del color muestra la evolución a través del tiempo.
                    """)
                else:
                    st.warning("No hay datos suficientes para crear el gráfico de burbujas.")
            else:
                st.warning("Faltan datos económicos para realizar el análisis combinado.")
    else:
        st.error("No se encuentran datos sociales para visualizar.")

# Función para mostrar educación
def mostrar_educacion(dfs, chart_type, year_start, year_end):
    """
    Muestra visualizaciones relacionadas con educación.
    
    Args:
        dfs (dict): Diccionario con DataFrames de datos
        chart_type (str): Tipo de visualización a mostrar
        year_start (int): Año de inicio para filtrar
        year_end (int): Año de fin para filtrar
    """
    st.header("Indicadores de Educación de Colombia")
    
    # Filtrar por rango de años
    if 'educacion' in dfs:
        df_edu = dfs['educacion']
        df_edu_filtered = df_edu[(df_edu['Año'] >= year_start) & (df_edu['Año'] <= year_end)]
        
        if chart_type == "Tasas de Finalización":
            st.subheader("Tasas de Finalización por Nivel Educativo")
            
            # Indicadores de tasas de finalización
            indicadores = {
                'Tasa de finalización de primaria': INDICADORES['educacion']['FINALIZACION_PRIMARIA'],
                'Tasa de finalización de secundaria inferior': INDICADORES['educacion']['FINALIZACION_SECUNDARIA']
            }
            
            # Crear gráfico de líneas para ambos indicadores
            codigos = list(indicadores.values())
            
            fig = crear_serie_temporal_multilinea(
                df_edu_filtered,
                codigos,
                "Evolución de Tasas de Finalización por Nivel"
            )
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hay datos suficientes para los indicadores seleccionados.")
            
            # Análisis comparativo entre períodos
            st.subheader("Comparación de Períodos")
            
            # Definir tres períodos para comparación
            periodo_inicial = (year_start, min(year_start + 5, year_end))
            periodo_medio = (year_start + (year_end - year_start) // 3, year_start + 2 * (year_end - year_start) // 3)
            periodo_final = (max(year_end - 5, year_start), year_end)
            
            periodos = {
                f'{periodo_inicial[0]}-{periodo_inicial[1]}': periodo_inicial,
                f'{periodo_medio[0]}-{periodo_medio[1]}': periodo_medio,
                f'{periodo_final[0]}-{periodo_final[1]}': periodo_final
            }
            
            # Calcular promedios por período para cada indicador
            datos_comparacion = []
            
            for nombre_ind, codigo_ind in indicadores.items():
                promedios_periodos = {}
                
                for nombre_periodo, (inicio, fin) in periodos.items():
                    datos_periodo = df_edu_filtered[
                        (df_edu_filtered['Año'] >= inicio) & 
                        (df_edu_filtered['Año'] <= fin) &
                        (df_edu_filtered['Indicator Code'] == codigo_ind)
                    ]
                    
                    if not datos_periodo.empty:
                        promedio = datos_periodo['Valor'].mean()
                        promedios_periodos[nombre_periodo] = promedio
                    else:
                        promedios_periodos[nombre_periodo] = None
                
                # Añadir a la lista de comparación
                if promedios_periodos:
                    fila = {'Indicador': nombre_ind}
                    fila.update(promedios_periodos)
                    datos_comparacion.append(fila)
            
            # Crear DataFrame y mostrar
            if datos_comparacion:
                df_comparacion = pd.DataFrame(datos_comparacion)
                
                # Formatear tabla
                st.dataframe(df_comparacion.style.format({
                    col: "{:.2f}%" if col != 'Indicador' else "{}" 
                    for col in df_comparacion.columns
                }))
                
                # Calcular y mostrar cambio porcentual
                if len(periodos) >= 2:
                    primer_periodo = list(periodos.keys())[0]
                    ultimo_periodo = list(periodos.keys())[-1]
                    
                    df_comparacion['Cambio %'] = ((df_comparacion[ultimo_periodo] / df_comparacion[primer_periodo]) - 1) * 100
                    
                    st.write(f"**Cambio porcentual entre {primer_periodo} y {ultimo_periodo}:**")
                    
                    for i, row in df_comparacion.iterrows():
                        cambio = row['Cambio %']
                        if pd.notna(cambio):
                            st.write(f"- {row['Indicador']}: {cambio:.1f}% {'de aumento' if cambio > 0 else 'de disminución'}")
            else:
                st.warning("No hay datos suficientes para realizar comparaciones entre períodos.")
        
        elif chart_type == "Inscripción por Nivel":
            st.subheader("Tasas de Inscripción por Nivel Educativo")
            
            # Indicadores de inscripción
            indicadores = {
                'Inscripción en educación primaria (% bruto)': 'SE.PRM.ENRR',
                'Inscripción en educación secundaria (% bruto)': 'SE.SEC.ENRR',
                'Inscripción en educación terciaria (% bruto)': INDICADORES['educacion']['INSCRIPCION_TERCIARIA']
            }
            
            # Selección de indicadores
            selected_indicators = st.multiselect(
                "Seleccione niveles educativos para comparar:",
                list(indicadores.keys()),
                default=list(indicadores.keys())
            )
            
            if selected_indicators:
                codigos = [indicadores[ind] for ind in selected_indicators]
                
                # Crear gráfico de líneas
                fig = crear_serie_temporal_multilinea(
                    df_edu_filtered,
                    codigos,
                    "Evolución de las Tasas de Inscripción por Nivel"
                )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                    # Análisis de brechas entre niveles
                    st.subheader("Análisis de Brechas entre Niveles Educativos")
                    
                    # Calcular la diferencia entre niveles para el último año disponible
                    ultimo_año = df_edu_filtered['Año'].max()
                    valores_ultimo_año = {}
                    
                    for nombre, codigo in indicadores.items():
                        datos = df_edu_filtered[
                            (df_edu_filtered['Año'] == ultimo_año) &
                            (df_edu_filtered['Indicator Code'] == codigo)
                        ]
                        
                        if not datos.empty:
                            valores_ultimo_año[nombre] = datos['Valor'].iloc[0]
                    
                    # Mostrar análisis si hay al menos dos niveles
                    if len(valores_ultimo_año) >= 2:
                        niveles = list(valores_ultimo_año.keys())
                        valores = list(valores_ultimo_año.values())
                        
                        # Ordenar niveles por valor (ascendente)
                        niveles_ordenados = [x for _, x in sorted(zip(valores, niveles))]
                        
                        # Calcular brechas
                        brechas = []
                        for i in range(len(niveles_ordenados) - 1):
                            nivel1 = niveles_ordenados[i]
                            nivel2 = niveles_ordenados[i + 1]
                            brecha = valores_ultimo_año[nivel2] - valores_ultimo_año[nivel1]
                            brechas.append((nivel1, nivel2, brecha))
                        
                        # Mostrar brechas
                        st.write(f"**Brechas entre niveles educativos en {int(ultimo_año)}:**")
                        
                        for nivel1, nivel2, brecha in brechas:
                            st.write(f"- Brecha entre {nivel1.split('(')[0]} y {nivel2.split('(')[0]}: {brecha:.1f} puntos porcentuales")
                else:
                    st.warning("No hay datos suficientes para los indicadores seleccionados.")
            else:
                st.warning("Seleccione al menos un nivel educativo para visualizar.")
        
        elif chart_type == "Gasto en Educación":
            st.subheader("Gasto en Educación")
            
            # Indicadores de gasto
            indicador_gasto = INDICADORES['educacion']['GASTO_EDUCACION']  # Gasto público en educación (% del PIB)
            
            # Crear gráfico de área
            datos_gasto = df_edu_filtered[df_edu_filtered['Indicator Code'] == indicador_gasto]
            
            if not datos_gasto.empty:
                fig = px.area(
                    datos_gasto,
                    x='Año',
                    y='Valor',
                    title="Gasto Público en Educación (% del PIB)",
                    labels={'Valor': '% del PIB', 'Año': 'Año'},
                    color_discrete_sequence=['#3498DB']
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Análisis del gasto
                promedio = datos_gasto['Valor'].mean()
                maximo = datos_gasto['Valor'].max()
                año_maximo = datos_gasto.loc[datos_gasto['Valor'].idxmax(), 'Año']
                ultimo_valor = datos_gasto.sort_values('Año').iloc[-1]['Valor']
                ultimo_año = datos_gasto.sort_values('Año').iloc[-1]['Año']
                
                st.info(f"""
                **Análisis del Gasto en Educación:**
                
                - Gasto promedio durante el período: {promedio:.2f}% del PIB
                - Gasto máximo: {maximo:.2f}% del PIB en {int(año_maximo)}
                - Último dato disponible: {ultimo_valor:.2f}% del PIB en {int(ultimo_año)}
                
                El gasto en educación es un indicador clave del compromiso gubernamental con el desarrollo educativo.
                """)
            else:
                st.warning("No hay datos disponibles sobre gasto en educación.")
        
        elif chart_type == "Comparativa entre Niveles":
            st.subheader("Comparativa entre Niveles Educativos")
            
            # Años para comparar
            años_comparacion = [year_start, year_end] if year_end > year_start else [year_start]
            
            # Indicadores para comparar
            indicadores = {
                'Tasa de finalización de primaria': INDICADORES['educacion']['FINALIZACION_PRIMARIA'],
                'Tasa de finalización de secundaria inferior': INDICADORES['educacion']['FINALIZACION_SECUNDARIA'],
                'Inscripción en educación terciaria (% bruto)': INDICADORES['educacion']['INSCRIPCION_TERCIARIA']
            }
            
            # Crear gráfico de barras comparativas
            fig = crear_barras_comparativas(
                df_edu_filtered,
                list(indicadores.values()),
                min(años_comparacion),
                max(años_comparacion),
                "Comparación de Indicadores Educativos"
            )
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hay datos suficientes para crear la comparación.")
            
            # Añadir explicación
            st.markdown("""
            **Interpretación del gráfico:**
            
            Este gráfico compara diferentes indicadores educativos entre el año inicial y final del período seleccionado.
            La comparación permite observar cómo ha evolucionado el acceso y la permanencia en los diferentes niveles
            del sistema educativo.
            
            Las tasas de finalización muestran el porcentaje de estudiantes que completan un nivel educativo,
            mientras que las tasas de inscripción muestran el porcentaje de la población en edad escolar que está
            matriculada en cada nivel.
            """)
    else:
        st.error("No se encuentran datos de educación para visualizar.")

# Función para mostrar medio ambiente
def mostrar_medio_ambiente(dfs, chart_type, year_start, year_end):
    """
    Muestra visualizaciones relacionadas con medio ambiente.
    
    Args:
        dfs (dict): Diccionario con DataFrames de datos
        chart_type (str): Tipo de visualización a mostrar
        year_start (int): Año de inicio para filtrar
        year_end (int): Año de fin para filtrar
    """
    st.header("Indicadores Ambientales de Colombia")
    
    # Filtrar por rango de años
    if 'ambiente' in dfs:
        df_amb = dfs['ambiente']
        df_amb_filtered = df_amb[(df_amb['Año'] >= year_start) & (df_amb['Año'] <= year_end)]
        
        if chart_type == "Energía Renovable":
            st.subheader("Consumo de Energía Renovable")
            
            # Indicador de energía renovable
            indicador_renovable = INDICADORES['ambiente']['ENERGIA_RENOVABLE']  # Consumo de energía renovable (% del total)
            
            # Crear gráfico
            datos_renovable = df_amb_filtered[df_amb_filtered['Indicator Code'] == indicador_renovable]
            
            if not datos_renovable.empty:
                # Gráfico de área
                fig = px.area(
                    datos_renovable,
                    x='Año',
                    y='Valor',
                    title="Consumo de Energía Renovable (% del consumo total de energía final)",
                    labels={'Valor': '% del total', 'Año': 'Año'},
                    color_discrete_sequence=['#27AE60']
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Análisis de tendencia
                if len(datos_renovable) > 2:
                    datos_renovable = datos_renovable.sort_values('Año')
                    first_value = datos_renovable.iloc[0]['Valor']
                    last_value = datos_renovable.iloc[-1]['Valor']
                    change = ((last_value / first_value) - 1) * 100
                    
                    st.info(f"""
                    **Análisis de tendencia en energía renovable:**
                    
                    - Porcentaje en {int(datos_renovable.iloc[0]['Año'])}: {first_value:.1f}%
                    - Porcentaje en {int(datos_renovable.iloc[-1]['Año'])}: {last_value:.1f}%
                    - Cambio: {change:.1f}% {'de aumento' if change > 0 else 'de disminución'}
                    
                    El consumo de energía renovable como porcentaje del total ha {'aumentado' if change > 0 else 'disminuido'} en el período analizado.
                    """)
            else:
                st.warning("No hay datos disponibles sobre consumo de energía renovable.")
        
        elif chart_type == "Emisiones de CO2":
            st.subheader("Emisiones de CO2")
            
            # Indicador de emisiones de CO2
            indicador_co2 = INDICADORES['ambiente']['EMISIONES_CO2']  # Emisiones de CO2 (toneladas métricas per cápita)
            
            # Crear gráfico
            datos_co2 = df_amb_filtered[df_amb_filtered['Indicator Code'] == indicador_co2]
            
            if not datos_co2.empty:
                # Gráfico de barras
                fig = px.bar(
                    datos_co2,
                    x='Año',
                    y='Valor',
                    title="Emisiones de CO2 (toneladas métricas per cápita)",
                    labels={'Valor': 'Toneladas métricas per cápita', 'Año': 'Año'},
                    color_discrete_sequence=['#E74C3C']
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Análisis de emisiones
                if len(datos_co2) > 2:
                    datos_co2 = datos_co2.sort_values('Año')
                    avg_emissions = datos_co2['Valor'].mean()
                    max_emissions = datos_co2['Valor'].max()
                    year_max = datos_co2.loc[datos_co2['Valor'].idxmax(), 'Año']
                    
                    # Comparar con promedio mundial si está disponible
                    st.info(f"""
                    **Análisis de emisiones de CO2:**
                    
                    - Emisiones promedio durante el período: {avg_emissions:.2f} toneladas per cápita
                    - Emisiones máximas: {max_emissions:.2f} toneladas per cápita en {int(year_max)}
                    - Tendencia: {'Creciente' if datos_co2.iloc[-1]['Valor'] > datos_co2.iloc[0]['Valor'] else 'Decreciente'}
                    
                    Las emisiones de CO2 son un indicador clave del impacto ambiental y la contribución al cambio climático.
                    """)
            else:
                st.warning("No hay datos disponibles sobre emisiones de CO2.")
        
        elif chart_type == "Áreas Forestales":
            st.subheader("Áreas Forestales")
            
            # Indicador de áreas forestales
            indicador_forestal = INDICADORES['ambiente']['AREA_FORESTAL']  # Área selvática (% del área de tierra)
            
            # Crear gráfico
            datos_forestal = df_amb_filtered[df_amb_filtered['Indicator Code'] == indicador_forestal]
            
            if not datos_forestal.empty:
                # Gráfico de línea con área
                fig = px.area(
                    datos_forestal,
                    x='Año',
                    y='Valor',
                    title="Área Forestal (% del área total de tierra)",
                    labels={'Valor': '% del área total', 'Año': 'Año'},
                    color_discrete_sequence=['#196F3D']
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Análisis de pérdida de bosques
                if len(datos_forestal) > 2:
                    datos_forestal = datos_forestal.sort_values('Año')
                    first_value = datos_forestal.iloc[0]['Valor']
                    last_value = datos_forestal.iloc[-1]['Valor']
                    change_percent = ((last_value / first_value) - 1) * 100
                    
                    # Estimación de pérdida en hectáreas (asumiendo área total de Colombia: 1,141,748 km²)
                    area_total_km2 = 1141748
                    area_tierra_km2 = area_total_km2 * 0.96  # Aproximadamente 96% es tierra
                    
                    perdida_porcentaje = first_value - last_value
                    perdida_km2 = (perdida_porcentaje / 100) * area_tierra_km2
                    
                    if perdida_porcentaje > 0:
                        st.warning(f"""
                        **Análisis de deforestación:**
                        
                        - Cobertura forestal en {int(datos_forestal.iloc[0]['Año'])}: {first_value:.2f}% del territorio
                        - Cobertura forestal en {int(datos_forestal.iloc[-1]['Año'])}: {last_value:.2f}% del territorio
                        - Cambio porcentual: {change_percent:.2f}%
                        - Pérdida estimada: aproximadamente {perdida_km2:.0f} km² (equivalente a {perdida_km2/100:.0f} hectáreas)
                        
                        La deforestación es un problema crítico para la biodiversidad y la mitigación del cambio climático.
                        """)
                    else:
                        st.success(f"""
                        **Análisis de cobertura forestal:**
                        
                        - Cobertura forestal en {int(datos_forestal.iloc[0]['Año'])}: {first_value:.2f}% del territorio
                        - Cobertura forestal en {int(datos_forestal.iloc[-1]['Año'])}: {last_value:.2f}% del territorio
                        - Cambio porcentual: +{-change_percent:.2f}%
                        - Ganancia estimada: aproximadamente {-perdida_km2:.0f} km² (equivalente a {-perdida_km2/100:.0f} hectáreas)
                        
                        El aumento de la cobertura forestal es un indicador positivo para la conservación ambiental.
                        """)
            else:
                st.warning("No hay datos disponibles sobre áreas forestales.")
        
        elif chart_type == "Recursos Hídricos":
            st.subheader("Recursos Hídricos")
            
            # Indicador de recursos hídricos
            indicador_agua = INDICADORES['ambiente']['AGUA_DULCE']  # Recursos intern

def mostrar_medio_ambiente(dfs, chart_type, year_start, year_end):
    """
    Muestra visualizaciones relacionadas con medio ambiente.
    
    Args:
        dfs (dict): Diccionario con DataFrames de datos
        chart_type (str): Tipo de visualización a mostrar
        year_start (int): Año de inicio para filtrar
        year_end (int): Año de fin para filtrar
    """
    st.header("Indicadores Ambientales de Colombia")
    
    # Filtrar por rango de años
    if 'ambiente' in dfs:
        df_amb = dfs['ambiente']
        df_amb_filtered = df_amb[(df_amb['Año'] >= year_start) & (df_amb['Año'] <= year_end)]
        
        if chart_type == "Energía Renovable":
            st.subheader("Consumo de Energía Renovable")
            
            # Indicador de energía renovable
            indicador_renovable = INDICADORES['ambiente']['ENERGIA_RENOVABLE']  # Consumo de energía renovable (% del total)
            
            # Crear gráfico
            datos_renovable = df_amb_filtered[df_amb_filtered['Indicator Code'] == indicador_renovable]
            
            if not datos_renovable.empty:
                # Gráfico de área
                fig = px.area(
                    datos_renovable,
                    x='Año',
                    y='Valor',
                    title="Consumo de Energía Renovable (% del consumo total de energía final)",
                    labels={'Valor': '% del total', 'Año': 'Año'},
                    color_discrete_sequence=['#27AE60']
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Análisis de tendencia
                if len(datos_renovable) > 2:
                    datos_renovable = datos_renovable.sort_values('Año')
                    first_value = datos_renovable.iloc[0]['Valor']
                    last_value = datos_renovable.iloc[-1]['Valor']
                    change = ((last_value / first_value) - 1) * 100
                    
                    st.info(f"""
                    **Análisis de tendencia en energía renovable:**
                    
                    - Porcentaje en {int(datos_renovable.iloc[0]['Año'])}: {first_value:.1f}%
                    - Porcentaje en {int(datos_renovable.iloc[-1]['Año'])}: {last_value:.1f}%
                    - Cambio: {change:.1f}% {'de aumento' if change > 0 else 'de disminución'}
                    
                    El consumo de energía renovable como porcentaje del total ha {'aumentado' if change > 0 else 'disminuido'} en el período analizado.
                    """)
            else:
                st.warning("No hay datos disponibles sobre consumo de energía renovable.")
        
        elif chart_type == "Emisiones de CO2":
            st.subheader("Emisiones de CO2")
            
            # Indicador de emisiones de CO2
            indicador_co2 = INDICADORES['ambiente']['EMISIONES_CO2']  # Emisiones de CO2 (toneladas métricas per cápita)
            
            # Crear gráfico
            datos_co2 = df_amb_filtered[df_amb_filtered['Indicator Code'] == indicador_co2]
            
            if not datos_co2.empty:
                # Gráfico de barras
                fig = px.bar(
                    datos_co2,
                    x='Año',
                    y='Valor',
                    title="Emisiones de CO2 (toneladas métricas per cápita)",
                    labels={'Valor': 'Toneladas métricas per cápita', 'Año': 'Año'},
                    color_discrete_sequence=['#E74C3C']
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Análisis de emisiones
                if len(datos_co2) > 2:
                    datos_co2 = datos_co2.sort_values('Año')
                    avg_emissions = datos_co2['Valor'].mean()
                    max_emissions = datos_co2['Valor'].max()
                    year_max = datos_co2.loc[datos_co2['Valor'].idxmax(), 'Año']
                    
                    # Comparar con promedio mundial si está disponible
                    st.info(f"""
                    **Análisis de emisiones de CO2:**
                    
                    - Emisiones promedio durante el período: {avg_emissions:.2f} toneladas per cápita
                    - Emisiones máximas: {max_emissions:.2f} toneladas per cápita en {int(year_max)}
                    - Tendencia: {'Creciente' if datos_co2.iloc[-1]['Valor'] > datos_co2.iloc[0]['Valor'] else 'Decreciente'}
                    
                    Las emisiones de CO2 son un indicador clave del impacto ambiental y la contribución al cambio climático.
                    """)
            else:
                st.warning("No hay datos disponibles sobre emisiones de CO2.")
        
        elif chart_type == "Áreas Forestales":
            st.subheader("Áreas Forestales")
            
            # Indicador de áreas forestales
            indicador_forestal = INDICADORES['ambiente']['AREA_FORESTAL']  # Área selvática (% del área de tierra)
            
            # Crear gráfico
            datos_forestal = df_amb_filtered[df_amb_filtered['Indicator Code'] == indicador_forestal]
            
            if not datos_forestal.empty:
                # Gráfico de línea con área
                fig = px.area(
                    datos_forestal,
                    x='Año',
                    y='Valor',
                    title="Área Forestal (% del área total de tierra)",
                    labels={'Valor': '% del área total', 'Año': 'Año'},
                    color_discrete_sequence=['#196F3D']
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Análisis de pérdida de bosques
                if len(datos_forestal) > 2:
                    datos_forestal = datos_forestal.sort_values('Año')
                    first_value = datos_forestal.iloc[0]['Valor']
                    last_value = datos_forestal.iloc[-1]['Valor']
                    change_percent = ((last_value / first_value) - 1) * 100
                    
                    # Estimación de pérdida en hectáreas (asumiendo área total de Colombia: 1,141,748 km²)
                    area_total_km2 = 1141748
                    area_tierra_km2 = area_total_km2 * 0.96  # Aproximadamente 96% es tierra
                    
                    perdida_porcentaje = first_value - last_value
                    perdida_km2 = (perdida_porcentaje / 100) * area_tierra_km2
                    
                    if perdida_porcentaje > 0:
                        st.warning(f"""
                        **Análisis de deforestación:**
                        
                        - Cobertura forestal en {int(datos_forestal.iloc[0]['Año'])}: {first_value:.2f}% del territorio
                        - Cobertura forestal en {int(datos_forestal.iloc[-1]['Año'])}: {last_value:.2f}% del territorio
                        - Cambio porcentual: {change_percent:.2f}%
                        - Pérdida estimada: aproximadamente {perdida_km2:.0f} km² (equivalente a {perdida_km2/100:.0f} hectáreas)
                        
                        La deforestación es un problema crítico para la biodiversidad y la mitigación del cambio climático.
                        """)
                    else:
                        st.success(f"""
                        **Análisis de cobertura forestal:**
                        
                        - Cobertura forestal en {int(datos_forestal.iloc[0]['Año'])}: {first_value:.2f}% del territorio
                        - Cobertura forestal en {int(datos_forestal.iloc[-1]['Año'])}: {last_value:.2f}% del territorio
                        - Cambio porcentual: +{-change_percent:.2f}%
                        - Ganancia estimada: aproximadamente {-perdida_km2:.0f} km² (equivalente a {-perdida_km2/100:.0f} hectáreas)
                        
                        El aumento de la cobertura forestal es un indicador positivo para la conservación ambiental.
                        """)
            else:
                st.warning("No hay datos disponibles sobre áreas forestales.")
        
        elif chart_type == "Recursos Hídricos":
            st.subheader("Recursos Hídricos")
            
            # Indicador de recursos hídricos
            indicador_agua = INDICADORES['ambiente']['AGUA_DULCE']  # Recursos internos renovables de agua dulce
            
            # Crear gráfico
            datos_agua = df_amb_filtered[df_amb_filtered['Indicator Code'] == indicador_agua]
            
            if not datos_agua.empty:
                # Gráfico de línea
                fig = px.line(
                    datos_agua,
                    x='Año',
                    y='Valor',
                    title="Recursos Renovables de Agua Dulce per cápita",
                    labels={'Valor': 'Metros cúbicos per cápita', 'Año': 'Año'},
                    markers=True,
                    color_discrete_sequence=['#3498DB']
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Análisis de recursos hídricos
                if len(datos_agua) > 2:
                    datos_agua = datos_agua.sort_values('Año')
                    first_value = datos_agua.iloc[0]['Valor']
                    last_value = datos_agua.iloc[-1]['Valor']
                    change_percent = ((last_value / first_value) - 1) * 100
                    
                    st.info(f"""
                    **Análisis de recursos hídricos:**
                    
                    - Recursos de agua dulce per cápita en {int(datos_agua.iloc[0]['Año'])}: {first_value:.0f} metros cúbicos
                    - Recursos de agua dulce per cápita en {int(datos_agua.iloc[-1]['Año'])}: {last_value:.0f} metros cúbicos
                    - Cambio porcentual: {change_percent:.1f}%
                    
                    {"La disminución de recursos hídricos per cápita puede deberse tanto al crecimiento poblacional como al agotamiento de fuentes de agua." if change_percent < 0 else "El aumento de recursos hídricos per cápita sugiere una mejora en la gestión del agua o un cambio demográfico favorable."}
                    """)
            else:
                st.warning("No hay datos disponibles sobre recursos hídricos.")
    else:
        st.error("No se encuentran datos ambientales para visualizar.")

# Función para mostrar seguridad y gobernanza
def mostrar_seguridad_gobernanza(dfs, chart_type, year_start, year_end):
    """
    Muestra visualizaciones relacionadas con seguridad y gobernanza.
    
    Args:
        dfs (dict): Diccionario con DataFrames de datos
        chart_type (str): Tipo de visualización a mostrar
        year_start (int): Año de inicio para filtrar
        year_end (int): Año de fin para filtrar
    """
    st.header("Seguridad y Gobernanza en Colombia")
    
    # Filtrar por rango de años
    if 'gobernanza' in dfs:
        df_gob = dfs['gobernanza']
        df_gob_filtered = df_gob[(df_gob['Año'] >= year_start) & (df_gob['Año'] <= year_end)]
        
        if chart_type == "Tasa de Homicidios":
            st.subheader("Evolución de la Tasa de Homicidios")
            
            # Indicador de homicidios
            indicador_homicidios = INDICADORES['gobernanza']['HOMICIDIOS']  # Homicidios intencionales (por cada 100.000 habitantes)
            
            # Crear gráfico
            datos_homicidios = df_gob_filtered[df_gob_filtered['Indicator Code'] == indicador_homicidios]
            
            if not datos_homicidios.empty:
                # Gráfico de línea con área
                fig = px.area(
                    datos_homicidios,
                    x='Año',
                    y='Valor',
                    title="Tasa de Homicidios (por cada 100.000 habitantes)",
                    labels={'Valor': 'Tasa por 100.000 habitantes', 'Año': 'Año'},
                    color_discrete_sequence=['#C0392B']
                )
                
                # Línea de tendencia
                datos_homicidios = datos_homicidios.sort_values('Año')
                x = datos_homicidios['Año']
                y = datos_homicidios['Valor']
                
                if len(x) > 5:
                    # Calcular tendencia polinómica
                    z = np.polyfit(x, y, 3)
                    p = np.poly1d(z)
                    
                    # Añadir línea de tendencia
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=p(x),
                            mode='lines',
                            name='Tendencia',
                            line=dict(color='rgba(0, 0, 0, 0.7)', width=2, dash='dash')
                        )
                    )
                
                # Añadir eventos históricos significativos
                for año, evento in EVENTOS_HISTORICOS.items():
                    if year_start <= año <= year_end:
                        fig.add_vline(
                            x=año,
                            line_width=1,
                            line_dash="dash",
                            line_color="gray"
                        )
                        
                        # Encontrar el valor de la tasa para ese año
                        valor_año = datos_homicidios[datos_homicidios['Año'] == año]['Valor'].values
                        y_pos = valor_año[0] if len(valor_año) > 0 else max(y) * 0.8
                        
                        fig.add_annotation(
                            x=año,
                            y=y_pos,
                            text=evento,
                            showarrow=True,
                            arrowhead=1,
                            ax=0,
                            ay=-40
                        )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Análisis de tendencia
                if len(datos_homicidios) > 5:
                    max_rate = datos_homicidios['Valor'].max()
                    max_year = datos_homicidios.loc[datos_homicidios['Valor'].idxmax(), 'Año']
                    
                    min_rate = datos_homicidios['Valor'].min()
                    min_year = datos_homicidios.loc[datos_homicidios['Valor'].idxmin(), 'Año']
                    
                    recent_rate = datos_homicidios.iloc[-1]['Valor']
                    recent_year = datos_homicidios.iloc[-1]['Año']
                    
                    change_since_max = ((recent_rate / max_rate) - 1) * 100
                    
                    st.markdown(f"""
                    **Análisis de la Tasa de Homicidios:**
                    
                    - Tasa máxima: {max_rate:.1f} por 100.000 habitantes en {int(max_year)}
                    - Tasa mínima: {min_rate:.1f} por 100.000 habitantes en {int(min_year)}
                    - Tasa reciente ({int(recent_year)}): {recent_rate:.1f} por 100.000 habitantes
                    - Cambio desde el máximo: {change_since_max:.1f}%
                    
                    La tasa de homicidios es un indicador clave de seguridad ciudadana y ha mostrado una tendencia general 
                    {'a la baja' if change_since_max < 0 else 'al alza'} en el período analizado.
                    """)
            else:
                st.warning("No hay datos disponibles sobre tasas de homicidios.")
        
        elif chart_type == "Indicadores de Gobernanza":
            st.subheader("Indicadores de Gobernanza del Banco Mundial")
            
            # Indicadores de gobernanza
            indicadores = {
                'Control de la Corrupción': INDICADORES['gobernanza']['CONTROL_CORRUPCION'],
                'Efectividad Gubernamental': INDICADORES['gobernanza']['EFECTIVIDAD_GOBIERNO'],
                'Estado de Derecho': INDICADORES['gobernanza']['ESTADO_DERECHO'],
                'Voz y Rendición de Cuentas': INDICADORES['gobernanza']['VOZ_RENDICION']
            }
            
            # Selección de indicadores
            selected_indicators = st.multiselect(
                "Seleccione indicadores de gobernanza para visualizar:",
                list(indicadores.keys()),
                default=list(indicadores.keys())
            )
            
            if selected_indicators:
                codigos = [indicadores[ind] for ind in selected_indicators]
                
                # Crear gráfico de líneas múltiples
                fig = crear_serie_temporal_multilinea(
                    df_gob_filtered,
                    codigos,
                    "Evolución de Indicadores de Gobernanza"
                )
                
                if fig:
                    # Actualizar rango Y para mejor comparación (los índices van de -2.5 a 2.5)
                    fig.update_layout(
                        yaxis=dict(
                            range=[-2.5, 2.5],
                            tickvals=[-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5],
                            title="Estimación (-2.5 débil; 2.5 fuerte)"
                        )
                    )
                    
                    # Añadir línea de cero
                    fig.add_shape(
                        type="line",
                        x0=year_start,
                        y0=0,
                        x1=year_end,
                        y1=0,
                        line=dict(color="black", width=1, dash="dot")
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Explicación
                    st.markdown("""
                    **Interpretación de los indicadores:**
                    
                    Los indicadores de gobernanza del Banco Mundial varían entre aproximadamente -2.5 (débil) y 2.5 (fuerte).
                    
                    - **Control de la Corrupción**: Percepción sobre control de la corrupción y captura del estado por élites e intereses privados.
                    - **Efectividad Gubernamental**: Calidad de los servicios públicos, la administración y su independencia de presiones políticas.
                    - **Estado de Derecho**: Confianza en las reglas de la sociedad, calidad de ejecución de contratos, derechos de propiedad, policía y tribunales.
                    - **Voz y Rendición de Cuentas**: Libertad de expresión, asociación y medios, y capacidad de los ciudadanos de participar en la selección de su gobierno.
                    """)
                else:
                    st.warning("No hay datos suficientes para los indicadores seleccionados.")
            else:
                st.warning("Seleccione al menos un indicador para visualizar.")
        
        elif chart_type == "Mapa de Colombia":
            st.subheader("Mapa de Colombia")
            
            # Crear mapa básico
            mapa = crear_mapa_colombia()
            folium_static(mapa, width=800, height=600)
            
            st.markdown("""
            **Contexto geográfico:**
            
            Colombia tiene una superficie aproximada de 1.141.748 km² y está dividida en 32 departamentos.
            Su ubicación estratégica le permite tener costas tanto en el océano Pacífico como en el mar Caribe,
            y comparte fronteras con Panamá, Venezuela, Brasil, Perú y Ecuador.
            
            La diversidad geográfica de Colombia, que incluye la región Andina, Caribe, Pacífica, Orinoquía y Amazonía,
            influye significativamente en los patrones de desarrollo económico, social y ambiental del país.
            """)
        
        elif chart_type == "Evolución de Seguridad":
            st.subheader("Análisis Temporal de Seguridad")
            
            # Comparación de períodos para tasa de homicidios
            indicador_homicidios = INDICADORES['gobernanza']['HOMICIDIOS']  # Homicidios intencionales
            
            datos_homicidios = df_gob_filtered[df_gob_filtered['Indicator Code'] == indicador_homicidios]
            
            if not datos_homicidios.empty:
                # Análisis por períodos
                st.write("**Análisis de la Tasa de Homicidios por Períodos**")
                
                fig_boxplot = crear_boxplot_por_periodos(
                    df_gob_filtered,
                    indicador_homicidios,
                    titulo="Distribución de la Tasa de Homicidios por Períodos"
                )
                
                if fig_boxplot:
                    st.plotly_chart(fig_boxplot, use_container_width=True)
                    
                    # Contextualización histórica
                    st.markdown("""
                    **Contexto histórico:**
                    
                    La evolución de la tasa de homicidios en Colombia refleja las diferentes etapas del conflicto
                    armado interno y las políticas de seguridad implementadas por los diferentes gobiernos. Se pueden identificar varias etapas importantes:
                    
                    - **1990-2002**: Período de alta violencia asociada al narcotráfico y la expansión de grupos armados.
                    - **2002-2010**: Implementación de la política de seguridad democrática, con reducción significativa de los índices de violencia.
                    - **2010-2016**: Proceso de paz con las FARC y continuación de la tendencia decreciente.
                    - **2016-presente**: Período post-acuerdo de paz, con desafíos en la implementación y surgimiento de nuevas dinámicas de violencia.
                    
                    Los cambios en la tasa de homicidios responden tanto a factores internos (políticas públicas, acuerdos de paz)
                    como a factores externos (dinámicas del narcotráfico internacional, situación en países vecinos).
                    """)
                else:
                    st.warning("No hay datos suficientes para crear el análisis por períodos.")
                
                # Variación temporal
                st.write("**Variación Temporal de la Tasa de Homicidios**")
                
                fig_var = crear_variacion_respecto_base(
                    df_gob_filtered,
                    indicador_homicidios,
                    año_base=year_start,
                    titulo=f"Variación de la Tasa de Homicidios (Base: {year_start})"
                )
                
                if fig_var:
                    st.plotly_chart(fig_var, use_container_width=True)
                else:
                    st.warning(f"No hay datos suficientes para calcular la variación desde {year_start}.")
            else:
                st.warning("No hay datos disponibles sobre tasas de homicidios.")
    else:
        st.error("No se encuentran datos de gobernanza para visualizar.")

# Función para mostrar análisis multivariable
def mostrar_analisis_multivariable(dfs, year_start, year_end):
    """
    Muestra análisis multivariable que relaciona diferentes indicadores.
    
    Args:
        dfs (dict): Diccionario con DataFrames de datos
        year_start (int): Año de inicio para filtrar
        year_end (int): Año de fin para filtrar
    """
    st.header("Análisis Multivariable de Indicadores")
    
    st.markdown("""
    Esta sección permite explorar las relaciones entre diferentes indicadores de desarrollo,
    identificando correlaciones y patrones que pueden revelar dinámicas complejas del
    desarrollo de Colombia.
    """)
    
    # Filtrar datos por rango de años
    dfs_filtered = {}
    for key, df in dfs.items():
        if key != 'maestro':  # No filtrar df_maestro
            dfs_filtered[key] = df[(df['Año'] >= year_start) & (df['Año'] <= year_end)]
        else:
            dfs_filtered[key] = df
    
    # Combinar todos los DataFrames en uno solo para análisis
    df_combined = pd.concat([df for key, df in dfs_filtered.items() if key != 'maestro'])
    
    # 1. Matriz de correlación
    st.subheader("Matriz de Correlación entre Indicadores")
    
    # Selección de categorías de indicadores
    # Permitir selección de categorías
    selected_categories = st.multiselect(
        "Seleccione categorías de indicadores para analizar correlaciones:",
        list(CATEGORIAS_ANALISIS.keys()),
        default=list(CATEGORIAS_ANALISIS.keys())[:2]  # Por defecto, seleccionar las primeras dos categorías
    )
    
    if selected_categories:
        # Obtener indicadores de las categorías seleccionadas
        indicadores_seleccionados = []
        for cat in selected_categories:
            indicadores_seleccionados.extend(CATEGORIAS_ANALISIS[cat])
        
        # Número de años recientes a considerar
        años_recientes = st.slider(
            "Número de años recientes a considerar para la correlación:",
            min_value=3,
            max_value=20,
            value=10
        )
        
        # Crear matriz de correlación
        fig_corr = crear_matriz_correlacion(
            df_combined,
            indicadores_seleccionados,
            años_recientes=años_recientes,
            titulo=f"Correlación entre Indicadores de Desarrollo"
        )
        
        if fig_corr:
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Explicación
            st.markdown("""
            **Interpretación de la matriz de correlación:**
            
            La matriz muestra la correlación entre pares de indicadores, con valores que van de -1 a 1:
            
            - **Correlación positiva (azul)**: Cuando un indicador aumenta, el otro también tiende a aumentar.
            - **Correlación negativa (rojo)**: Cuando un indicador aumenta, el otro tiende a disminuir.
            - **Sin correlación (blanco/claro)**: No hay una relación clara entre los indicadores.
            
            La intensidad del color representa la fuerza de la correlación: colores más intensos indican correlaciones más fuertes.
            """)
        else:
            st.warning("No hay datos suficientes para crear la matriz de correlación con los indicadores seleccionados.")
    else:
        st.warning("Seleccione al menos una categoría de indicadores para analizar correlaciones.")
    
    # 2. Gráfico de burbujas para análisis multidimensional
    st.subheader("Análisis Multidimensional: Gráfico de Burbujas")
    
    # Permitir selección de indicadores
    st.write("Seleccione tres indicadores para visualizar su relación a lo largo del tiempo:")
    
    # Obtener lista de todos los indicadores disponibles
    todos_indicadores = {}
    for cat, ind_list in CATEGORIAS_ANALISIS.items():
        for ind_code in ind_list:
            # Buscar nombre del indicador
            ind_name = None
            for _, df in dfs_filtered.items():
                if ind_code in df['Indicator Code'].values:
                    ind_name = df[df['Indicator Code'] == ind_code]['Indicator Name'].iloc[0]
                    break
            
            if ind_name:
                todos_indicadores[f"{ind_name} ({ind_code})"] = ind_code
    
    # Columnas para selección de indicadores
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ind_x = st.selectbox(
            "Indicador para eje X:",
            list(todos_indicadores.keys()),
            index=0 if todos_indicadores else None
        )
    
    with col2:
        ind_y = st.selectbox(
            "Indicador para eje Y:",
            list(todos_indicadores.keys()),
            index=min(1, len(todos_indicadores)-1) if todos_indicadores else None
        )
    
    with col3:
        ind_size = st.selectbox(
            "Indicador para tamaño:",
            list(todos_indicadores.keys()),
            index=min(2, len(todos_indicadores)-1) if todos_indicadores else None
        )
    
    if ind_x and ind_y and ind_size:
        codigo_x = todos_indicadores[ind_x]
        codigo_y = todos_indicadores[ind_y]
        codigo_size = todos_indicadores[ind_size]
        
        # Número de años para mostrar
        step = max(1, (year_end - year_start) // 8)  # Mostrar máximo 8 puntos
        años_seleccionados = list(range(year_start, year_end + 1, step))
        
        # Crear gráfico de burbujas
        fig_burbujas = crear_grafico_burbujas(
            df_combined,
            codigo_x,
            codigo_y,
            codigo_size,
            años_seleccionados=años_seleccionados,
            titulo=f"Relación entre indicadores seleccionados a lo largo del tiempo"
        )
        
        if fig_burbujas:
            st.plotly_chart(fig_burbujas, use_container_width=True)
            
            # Explicación
            st.markdown(f"""
            **Interpretación del gráfico de burbujas:**
            
            Este gráfico muestra la evolución simultánea de tres indicadores de desarrollo de Colombia:
            
            - **Eje X**: {ind_x.split(' (')[0]}
            - **Eje Y**: {ind_y.split(' (')[0]}
            - **Tamaño de las burbujas**: {ind_size.split(' (')[0]}
            - **Color**: Representa el año
            
            Cada burbuja corresponde a un año específico. La progresión del color y las etiquetas de año muestran cómo han evolucionado estos indicadores en conjunto a lo largo del tiempo.
            
            Observe patrones de movimiento y cambios de tamaño para identificar relaciones entre estos indicadores y posibles puntos de inflexión en el desarrollo de Colombia.
            """)
        else:
            st.warning("""
            No se pudo crear el gráfico de burbujas con los indicadores seleccionados. 
            Esto puede deberse a datos incompletos o falta de coincidencia en los años disponibles para estos indicadores.
            Pruebe con otra combinación de indicadores o un rango de años diferente.
            """)
    else:
        st.warning("Seleccione los tres indicadores para crear el gráfico de burbujas.")
    
    # 3. Scatter matrix para visualizar múltiples relaciones
    st.subheader("Matriz de Dispersión para Análisis Multivariable")
    
    # Permitir selección de indicadores (máximo 4 para legibilidad)
    selected_indicators_scatter = st.multiselect(
        "Seleccione indicadores para la matriz de dispersión (máximo 4):",
        list(todos_indicadores.keys()),
        default=list(todos_indicadores.keys())[:min(3, len(todos_indicadores))]
    )
    
    if len(selected_indicators_scatter) > 4:
        st.warning("Ha seleccionado más de 4 indicadores. Solo se mostrarán los primeros 4 para mantener la legibilidad.")
        selected_indicators_scatter = selected_indicators_scatter[:4]
    
    if selected_indicators_scatter and len(selected_indicators_scatter) >= 2:
        # Obtener códigos de indicadores
        codigos_scatter = [todos_indicadores[ind] for ind in selected_indicators_scatter]
        
        # Crear scatter matrix
        fig_scatter = crear_scatter_matrix(
            df_combined,
            codigos_scatter,
            años_recientes=min(year_end - year_start + 1, 15),  # Limitar a máximo 15 años para claridad
            titulo="Matriz de Relaciones entre Indicadores Seleccionados"
        )
        
        if fig_scatter:
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Explicación
            st.markdown("""
            **Interpretación de la matriz de dispersión:**
            
            Esta matriz muestra la relación entre cada par de indicadores seleccionados. Cada celda representa un diagrama de dispersión entre dos indicadores, con puntos coloreados según el año.
            
            - Un patrón claro y definido indica una relación fuerte entre los indicadores.
            - Una nube dispersa de puntos sugiere que no hay una relación clara.
            - La progresión de colores muestra cómo ha cambiado la relación con el tiempo.
            
            Observe diferentes celdas para identificar relaciones bivariadas interesantes entre los indicadores seleccionados.
            """)
        else:
            st.warning("""
            No se pudo crear la matriz de dispersión con los indicadores seleccionados.
            Esto puede deberse a datos incompletos o incompatibles. Intente con otra selección de indicadores.
            """)
    elif len(selected_indicators_scatter) == 1:
        st.warning("Seleccione al menos dos indicadores para crear la matriz de dispersión.")
    else:
        st.warning("Seleccione indicadores para crear la matriz de dispersión.")

# Función principal para la aplicación
def main():
    """Función principal que ejecuta la aplicación Streamlit."""
    # DIAGNÓSTICO: Si queremos mostrar el diagnóstico de pobreza, descomentar esta línea
    
    # Título y descripción
    st.title("🇨🇴 Colombia en Perspectiva: Indicadores de Desarrollo")
    st.markdown("""
    Este dashboard interactivo permite explorar la evolución de Colombia a través de diversos indicadores
    de desarrollo económico, social, educativo, ambiental y de gobernanza durante las últimas décadas.
    """)
    
    # Cargar datos
    with st.spinner("Cargando datos..."):
        dfs = cargar_datos()
    
    if not dfs:
        st.error("No se pudieron cargar los datos. Verifique que los archivos existan en el directorio 'datos_procesados'.")
        return
    
    # Crear menú lateral
    st.sidebar.title("Navegación")
    seccion = st.sidebar.radio(
        "Seleccione una sección:",
        ["Panorama General", "Desarrollo Económico", "Desarrollo Social", 
         "Educación", "Medio Ambiente", "Seguridad y Gobernanza", 
         "Análisis Multivariable"]
    )
    
    # Filtros generales
    st.sidebar.title("Filtros")
    
    # Rango de años (aplicable a todas las secciones)
    min_year = 2000
    max_year = 2023
    
    years_range = st.sidebar.slider(
        "Rango de años",
        min_value=min_year,
        max_value=max_year,
        value=(2000, 2023)
    )
    
    year_start, year_end = years_range
    
    # Selector de visualización por tema
    if seccion != "Panorama General" and seccion != "Análisis Multivariable":
        st.sidebar.title("Opciones de Visualización")
        
        # Opciones específicas para cada sección
        if seccion == "Desarrollo Económico":
            chart_type = st.sidebar.selectbox(
                "Tipo de visualización",
                ["Dashboard Económico", "Evolución del PIB", "Comercio Internacional", 
                 "Inflación y Estabilidad", "Variación temporal"]
            )
        
        elif seccion == "Desarrollo Social":
            chart_type = st.sidebar.selectbox(
                "Tipo de visualización",
                ["Pobreza y Desigualdad", "Demografía", "Empleo", "Indicadores Sociales Combinados"]
            )
        
        elif seccion == "Educación":
            chart_type = st.sidebar.selectbox(
                "Tipo de visualización",
                ["Tasas de Finalización", "Inscripción por Nivel", "Gasto en Educación", 
                 "Comparativa entre Niveles"]
            )
        
        elif seccion == "Medio Ambiente":
            chart_type = st.sidebar.selectbox(
                "Tipo de visualización",
                ["Energía Renovable", "Emisiones de CO2", "Áreas Forestales", 
                 "Recursos Hídricos"]
            )
        
        elif seccion == "Seguridad y Gobernanza":
            chart_type = st.sidebar.selectbox(
                "Tipo de visualización",
                ["Tasa de Homicidios", "Indicadores de Gobernanza", "Mapa de Colombia", 
                 "Evolución de Seguridad"]
            )
    
    # Mostrar información según la sección seleccionada
    if seccion == "Panorama General":
        mostrar_panorama_general(dfs, year_start, year_end)
    
    elif seccion == "Desarrollo Económico":
        mostrar_desarrollo_economico(dfs, chart_type, year_start, year_end)
    
    elif seccion == "Desarrollo Social":
        mostrar_desarrollo_social(dfs, chart_type, year_start, year_end)
    
    elif seccion == "Educación":
        mostrar_educacion(dfs, chart_type, year_start, year_end)
    
    elif seccion == "Medio Ambiente":
        mostrar_medio_ambiente(dfs, chart_type, year_start, year_end)
    
    elif seccion == "Seguridad y Gobernanza":
        mostrar_seguridad_gobernanza(dfs, chart_type, year_start, year_end)
    
    elif seccion == "Análisis Multivariable":
        mostrar_analisis_multivariable(dfs, year_start, year_end)
    
    # Pie de página
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: gray; font-size: 0.8em;">
        Dashboard creado con Streamlit y Plotly | Datos: Banco Mundial | Última actualización: Marzo 2025
        </div>
        """,
        unsafe_allow_html=True
    )

# Ejecutar aplicación principal
if __name__ == "__main__":
    main()