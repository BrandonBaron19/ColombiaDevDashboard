"""
Módulo de Preprocesamiento de Datos del Banco Mundial

Este módulo realiza el preprocesamiento y análisis de archivos de datos del Banco Mundial.
Carga, limpia, transforma y analiza datos del conjunto de indicadores de desarrollo del Banco Mundial,
creando subconjuntos temáticos e indicadores derivados.

El flujo de trabajo principal:
1. Cargar archivos de datos crudos y metadatos
2. Limpiar y transformar datos a un formato estructurado
3. Calcular indicadores derivados (tasas de crecimiento, promedios móviles)
4. Crear un dataframe maestro con metadatos integrados
5. Crear subconjuntos temáticos para diferentes dominios de análisis
6. Verificar que los indicadores críticos estén presentes
7. Realizar análisis exploratorio
8. Guardar datos procesados en archivos CSV
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constantes
INDICADORES_CRITICOS = {
    'POBREZA': 'SI.POV.NAHC',
    'GINI': 'SI.POV.GINI',
    'PIB_PER_CAPITA': 'NY.GDP.PCAP.CD',
    'CRECIMIENTO_PIB': 'NY.GDP.MKTP.KD.ZG',
    'TASA_HOMICIDIOS': 'VC.IHR.PSRC.P5'
}

CATEGORIAS_INDICADORES = {
    'NY': 'Economía',
    'SI': 'Pobreza y Desigualdad',
    'SL': 'Trabajo y Empleo',
    'SE': 'Educación',
    'SH': 'Salud',
    'SP': 'Población y Demografía',
    'EN': 'Medio Ambiente',
    'EG': 'Energía',
    'VC': 'Crimen y Seguridad',
    'VA': 'Gobernanza',
    'TX': 'Comercio y Exportaciones',
    'TM': 'Comercio e Importaciones'
}

# Definir indicadores temáticos
INDICADORES_TEMATICOS = {
    'economia': [
        'NY.GDP.MKTP.CD',     # PIB (US$ a precios actuales)
        'NY.GDP.PCAP.CD',     # PIB per cápita (US$ a precios actuales)
        'NY.GDP.MKTP.KD.ZG',  # Crecimiento del PIB (% anual)
        'FP.CPI.TOTL.ZG',     # Inflación, precios al consumidor (% anual)
        'NE.EXP.GNFS.ZS',     # Exportaciones de bienes y servicios (% del PIB)
        'NE.IMP.GNFS.ZS'      # Importaciones de bienes y servicios (% del PIB)
    ],
    'social': [
        'SI.POV.NAHC',        # Tasa de incidencia de la pobreza, línea de pobreza nacional
        'SI.POV.GINI',        # Índice de Gini
        'SL.UEM.TOTL.ZS',     # Desempleo, total (% de la población activa total)
        'SP.DYN.LE00.IN',     # Esperanza de vida al nacer, total (años)
        'SP.POP.TOTL',        # Población, total
        'SP.URB.TOTL.IN.ZS'   # Población urbana (% del total)
    ],
    'educacion': [
        'SE.ADT.LITR.ZS',     # Tasa de alfabetización, total de adultos
        'SE.PRM.CMPT.ZS',     # Tasa de finalización de la educación de nivel primario
        'SE.SEC.CMPT.LO.ZS',  # Tasa de finalización de la educación de nivel secundario inferior
        'SE.TER.ENRR',        # Inscripción escolar, nivel terciario (% bruto)
        'SE.XPD.TOTL.GD.ZS'   # Gasto público en educación, total (% del PIB)
    ],
    'ambiente': [
        'EG.FEC.RNEW.ZS',     # Consumo de energía renovable (% del total)
        'EN.ATM.CO2E.PC',     # Emisiones de CO2 (toneladas métricas per cápita)
        'AG.LND.FRST.ZS',     # Área selvática (% del área de tierra)
        'ER.H2O.FWTL.ZS',     # Recursos internos renovables de agua dulce per cápita
        'EN.ATM.PM25.MC.M3'   # Exposición a PM2.5, media anual
    ],
    'gobernanza': [
        'VC.IHR.PSRC.P5',     # Homicidios intencionales (por cada 100.000 habitantes)
        'VA.EST',             # Voice and Accountability: Estimate
        'RL.EST',             # Rule of Law: Estimate
        'CC.EST',             # Control of Corruption: Estimate
        'GE.EST'              # Government Effectiveness: Estimate
    ]
}

def obtener_rutas_archivos():
    """Define las rutas de archivos para datos y directorios de salida."""
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    archivo_datos = os.path.join(directorio_actual, "API_COL_DS2_es_csv_v2_17895.csv")
    archivo_metadatos_pais = os.path.join(directorio_actual, "Metadata_Country_API_COL_DS2_es_csv_v2_17895.csv")
    archivo_metadatos_indicadores = os.path.join(directorio_actual, "Metadata_Indicator_API_COL_DS2_es_csv_v2_17895.csv")
    directorio_salida = os.path.join(directorio_actual, "datos_procesados")
    
    return {
        'datos': archivo_datos,
        'metadatos_pais': archivo_metadatos_pais,
        'metadatos_indicadores': archivo_metadatos_indicadores,
        'directorio_salida': directorio_salida
    }

def cargar_datos_crudos(rutas_archivos):
    """
    Carga los archivos de datos del Banco Mundial incluyendo datos principales y metadatos.
    
    Args:
        rutas_archivos (dict): Diccionario con rutas a los archivos de datos
    
    Returns:
        tuple: Tres DataFrames con los datos cargados
    """
    logger.info("Cargando datos crudos...")
    
    # Cargar datos principales (saltando las primeras 4 filas de metadatos)
    try:
        datos = pd.read_csv(rutas_archivos['datos'], skiprows=4, encoding='utf-8')
        metadatos_pais = pd.read_csv(rutas_archivos['metadatos_pais'], encoding='utf-8')
        metadatos_indicadores = pd.read_csv(rutas_archivos['metadatos_indicadores'], encoding='utf-8')
        
        logger.info(f"Datos cargados: {datos.shape[0]} filas y {datos.shape[1]} columnas")
        return datos, metadatos_pais, metadatos_indicadores
    except Exception as e:
        logger.error(f"Error al cargar datos: {e}")
        raise

def limpiar_y_transformar_datos(datos, metadatos_indicadores):
    """
    Limpia y transforma los datos:
    1. Selecciona columnas necesarias
    2. Transforma de formato ancho a formato largo
    3. Convierte tipos de datos
    4. Maneja valores faltantes preservando indicadores clave
    5. Añade información de categoría de indicador
    
    Args:
        datos (pd.DataFrame): DataFrame de datos principales
        metadatos_indicadores (pd.DataFrame): DataFrame de metadatos de indicadores
        
    Returns:
        pd.DataFrame: DataFrame transformado y limpio
    """
    logger.info("Limpiando y transformando datos...")
    
    # Seleccionar solo columnas necesarias
    columnas_años = [col for col in datos.columns if col.isdigit() and len(col) == 4]
    columnas_seleccionadas = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'] + columnas_años
    datos_limpios = datos[columnas_seleccionadas].copy()
    
    # Transformar de formato ancho a formato largo
    logger.info("Transformando de formato ancho a formato largo...")
    datos_largo = datos_limpios.melt(
        id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'],
        var_name='Año',
        value_name='Valor'
    )
    
    # Convertir tipos de datos
    datos_largo['Año'] = pd.to_numeric(datos_largo['Año'])
    datos_largo['Valor'] = pd.to_numeric(datos_largo['Valor'], errors='coerce')
    
    # Manejar valores faltantes preservando indicadores clave
    # Verificar si el indicador de pobreza existe en los datos originales
    indicador_pobreza = INDICADORES_CRITICOS['POBREZA']
    logger.info(f"Verificando si el indicador de pobreza {indicador_pobreza} existe en los datos...")
    tiene_pobreza = indicador_pobreza in datos_largo['Indicator Code'].values
    logger.info(f"El indicador de pobreza {indicador_pobreza} existe: {tiene_pobreza}")
    
    if tiene_pobreza:
        # Preservar datos de pobreza incluso con valores faltantes
        datos_pobreza = datos_largo[datos_largo['Indicator Code'] == indicador_pobreza].copy()
        otros_datos = datos_largo[datos_largo['Indicator Code'] != indicador_pobreza].copy()
        
        # Eliminar valores faltantes para otros datos
        otros_datos_filtrados = otros_datos.dropna(subset=['Valor'])
        
        # Recombinar con datos de pobreza (incluidos valores nulos)
        datos_filtrados = pd.concat([otros_datos_filtrados, datos_pobreza], ignore_index=True)
        
        logger.info(f"Registros para {indicador_pobreza}: {len(datos_pobreza)}")
        logger.info(f"Registros para {indicador_pobreza} con valores no nulos: {len(datos_pobreza.dropna(subset=['Valor']))}")
    else:
        # Si el indicador de pobreza no existe, proceder normalmente
        datos_filtrados = datos_largo.dropna(subset=['Valor'])
    
    # Añadir información de categoría de indicador
    if 'INDICATOR_CODE' in metadatos_indicadores.columns:
        # Función para asignar categoría basada en el prefijo del código
        def asignar_categoria(codigo):
            if pd.isna(codigo):
                return 'Otros'
            prefijo = codigo.split('.')[0]
            return CATEGORIAS_INDICADORES.get(prefijo, 'Otros')
        
        # Crear diccionario de mapeo
        mapa_categorias = {
            codigo: asignar_categoria(codigo) 
            for codigo in metadatos_indicadores['INDICATOR_CODE']
        }
        
        # Añadir columna de categoría
        datos_filtrados['Categoría'] = datos_filtrados['Indicator Code'].map(mapa_categorias)
    else:
        datos_filtrados['Categoría'] = 'Sin categoría'
    
    # Verificación final de la presencia del indicador de pobreza
    logger.info(f"Verificando si {indicador_pobreza} sigue presente después de la transformación...")
    tiene_pobreza_final = indicador_pobreza in datos_filtrados['Indicator Code'].values
    logger.info(f"Indicador de pobreza en datos transformados: {tiene_pobreza_final}")
    
    logger.info(f"Datos transformados: {datos_filtrados.shape[0]} filas y {datos_filtrados.shape[1]} columnas")
    return datos_filtrados

def calcular_indicadores_derivados(datos):
    """
    Calcula indicadores derivados:
    - Tasas de crecimiento interanuales
    - Promedios móviles
    - Diferencias respecto a años base
    
    Args:
        datos (pd.DataFrame): Datos limpios en formato largo
        
    Returns:
        pd.DataFrame: DataFrame con indicadores derivados
    """
    logger.info("Calculando indicadores derivados...")
    
    datos_derivados = datos.copy()
    indicadores_criticos = list(INDICADORES_CRITICOS.values())
    
    # Asegurar que los datos estén ordenados por indicador y año
    datos_derivados = datos_derivados.sort_values(['Indicator Code', 'Año'])
    
    # Lista para almacenar DataFrames con indicadores derivados
    dfs_derivados = []
    
    # Procesar cada indicador único
    for indicador, grupo in datos_derivados.groupby('Indicator Code'):
        # Procesar si es crítico o tiene suficientes datos
        es_critico = indicador in indicadores_criticos
        
        if es_critico or len(grupo) > 5:
            if es_critico:
                logger.info(f"Procesando indicador crítico: {indicador} ({len(grupo)} registros)")
            
            # Crear DataFrame temporal para este indicador
            df_temp = grupo.copy()
            
            # Calcular tasa de crecimiento interanual (solo para valores positivos)
            if (df_temp['Valor'] > 0).all():
                # Configurar índice para operación shift
                df_temp = df_temp.sort_values('Año').set_index('Año')
                
                # Calcular cambio porcentual
                df_temp['Crecimiento_Anual'] = df_temp['Valor'].pct_change() * 100
                
                # Restablecer índice
                df_temp = df_temp.reset_index()
            
            # Calcular promedio móvil de 3 años
            df_temp = df_temp.sort_values('Año')
            df_temp['Promedio_Movil_3'] = df_temp['Valor'].rolling(window=3, min_periods=1).mean()
            
            # Calcular promedio móvil de 5 años
            df_temp['Promedio_Movil_5'] = df_temp['Valor'].rolling(window=5, min_periods=1).mean()
            
            # Calcular cambio desde el año 2000 (o primer año disponible)
            año_base = 2000
            valores_base = df_temp[df_temp['Año'] == año_base]['Valor'].values
            
            if len(valores_base) > 0:
                valor_base = valores_base[0]
                df_temp['Cambio_Desde_2000_Abs'] = df_temp['Valor'] - valor_base
                # Cambio porcentual solo si el valor base no es cero
                if valor_base != 0:
                    df_temp['Cambio_Desde_2000_Pct'] = (df_temp['Valor'] / valor_base - 1) * 100
            
            # Añadir a la lista de DataFrames
            dfs_derivados.append(df_temp)
    
    # Combinar todos los DataFrames derivados
    if dfs_derivados:
        datos_derivados_combinados = pd.concat(dfs_derivados, ignore_index=True)
        
        # Verificar indicadores críticos
        for indicador in indicadores_criticos:
            tiene_indicador = indicador in datos_derivados_combinados['Indicator Code'].values
            logger.info(f"Indicador crítico {indicador} en datos derivados: {tiene_indicador}")
            if tiene_indicador:
                logger.info(f"  Registros: {len(datos_derivados_combinados[datos_derivados_combinados['Indicator Code'] == indicador])}")
        
        logger.info(f"Indicadores derivados calculados: {datos_derivados_combinados.shape[0]} filas")
        return datos_derivados_combinados
    else:
        logger.warning("No se pudieron calcular indicadores derivados")
        return datos_derivados

def crear_dataframe_maestro(datos_transformados, metadatos_pais, metadatos_indicadores):
    """
    Crea un DataFrame maestro que integra toda la información necesaria,
    incluyendo datos transformados y metadatos relevantes.
    
    Args:
        datos_transformados (pd.DataFrame): Datos transformados
        metadatos_pais (pd.DataFrame): Metadatos del país
        metadatos_indicadores (pd.DataFrame): Metadatos de indicadores
        
    Returns:
        pd.DataFrame: DataFrame maestro unificado
    """
    logger.info("Creando DataFrame maestro...")
    
    # Comenzar con los datos transformados
    df_maestro = datos_transformados.copy()
    
    # Añadir información relevante de metadatos de indicadores
    if 'INDICATOR_CODE' in metadatos_indicadores.columns and 'SOURCE_NOTE' in metadatos_indicadores.columns:
        # Crear diccionario de mapeo
        mapa_descripciones = dict(zip(metadatos_indicadores['INDICATOR_CODE'], 
                                      metadatos_indicadores['SOURCE_NOTE']))
        
        # Añadir columna de descripción
        df_maestro['Descripcion_Indicador'] = df_maestro['Indicator Code'].map(mapa_descripciones)
    
    # Añadir metadatos del país (región, grupo de ingresos)
    if 'Country Code' in metadatos_pais.columns and 'Region' in metadatos_pais.columns:
        # Crear diccionarios de mapeo
        mapa_region = dict(zip(metadatos_pais['Country Code'], metadatos_pais['Region']))
        
        # Añadir columnas
        df_maestro['Region'] = df_maestro['Country Code'].map(mapa_region)
        
        # Añadir grupo de ingresos si está disponible
        if 'Income_Group' in metadatos_pais.columns:
            mapa_ingresos = dict(zip(metadatos_pais['Country Code'], metadatos_pais['Income_Group']))
            df_maestro['Grupo_Ingresos'] = df_maestro['Country Code'].map(mapa_ingresos)
    
    # Añadir timestamp de procesamiento
    df_maestro['Fecha_Procesamiento'] = datetime.now().strftime('%Y-%m-%d')
    
    # Filtrar años recientes para análisis más relevante (2000 en adelante)
    df_maestro_filtrado = df_maestro[df_maestro['Año'] >= 2000].copy()
    
    # Verificar indicador de pobreza
    indicador_pobreza = INDICADORES_CRITICOS['POBREZA']
    tiene_pobreza = indicador_pobreza in df_maestro_filtrado['Indicator Code'].values
    logger.info(f"Indicador de pobreza {indicador_pobreza} en DataFrame maestro: {tiene_pobreza}")
    
    logger.info(f"DataFrame maestro creado: {df_maestro_filtrado.shape[0]} filas y {df_maestro_filtrado.shape[1]} columnas")
    return df_maestro_filtrado

def crear_subconjuntos_tematicos(df_maestro):
    """
    Crea subconjuntos de datos temáticos para diferentes áreas de análisis.
    
    Args:
        df_maestro (pd.DataFrame): DataFrame maestro unificado
    
    Returns:
        dict: Diccionario con DataFrames temáticos
    """
    logger.info("Creando subconjuntos temáticos...")
    
    # Diccionario para almacenar subconjuntos
    subconjuntos = {}
    
    # Crear cada subconjunto temático
    for tema, codigos in INDICADORES_TEMATICOS.items():
        # Filtrar por códigos de indicador
        subconjunto = df_maestro[df_maestro['Indicator Code'].isin(codigos)].copy()
        
        # Verificar registros para cada indicador
        for codigo in codigos:
            cantidad_registros = len(subconjunto[subconjunto['Indicator Code'] == codigo])
            logger.info(f"  Indicador {codigo} en tema '{tema}': {cantidad_registros} registros")
        
        # Verificar si se encontraron datos
        if not subconjunto.empty:
            subconjuntos[tema] = subconjunto
            logger.info(f"  Subconjunto '{tema}' creado: {subconjunto.shape[0]} filas")
        else:
            logger.warning(f"  No se encontraron datos para el tema '{tema}'")
    
    return subconjuntos

def verificar_indicadores_criticos(subconjuntos, datos_originales):
    """
    Verifica si los indicadores críticos están presentes en los subconjuntos.
    Si faltan, intenta añadirlos manualmente desde los datos originales.
    
    Args:
        subconjuntos (dict): Diccionario con DataFrames temáticos
        datos_originales (pd.DataFrame): DataFrame de datos originales
    
    Returns:
        dict: Diccionario actualizado con DataFrames temáticos
    """
    mapeo_criticos = {
        'social': [INDICADORES_CRITICOS['POBREZA'], INDICADORES_CRITICOS['GINI']],
        'economia': [INDICADORES_CRITICOS['PIB_PER_CAPITA'], INDICADORES_CRITICOS['CRECIMIENTO_PIB']],
        'gobernanza': [INDICADORES_CRITICOS['TASA_HOMICIDIOS']]
    }
    
    logger.info("Verificando indicadores críticos:")
    for tema, indicadores in mapeo_criticos.items():
        if tema in subconjuntos:
            df = subconjuntos[tema]
            for indicador in indicadores:
                existe = indicador in df['Indicator Code'].values
                logger.info(f"  {tema} - {indicador}: {'✓ Presente' if existe else '✗ Ausente'}")
                
                # Si el indicador crítico está ausente, intentar añadirlo manualmente
                if not existe and indicador == INDICADORES_CRITICOS['POBREZA']:
                    logger.warning(f"    ¡Alerta! Indicador de pobreza {indicador} ausente. Añadiendo manualmente...")
                    
                    # Buscar en los datos originales
                    datos_indicador = datos_originales[datos_originales['Indicator Code'] == indicador]
                    
                    if not datos_indicador.empty:
                        # Transformar a formato largo
                        columnas_años = [col for col in datos_indicador.columns if col.isdigit() and len(col) == 4]
                        datos_indicador_largo = datos_indicador.melt(
                            id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'],
                            value_vars=columnas_años,
                            var_name='Año',
                            value_name='Valor'
                        )
                        
                        # Convertir tipos
                        datos_indicador_largo['Año'] = pd.to_numeric(datos_indicador_largo['Año'])
                        datos_indicador_largo['Valor'] = pd.to_numeric(datos_indicador_largo['Valor'], errors='coerce')
                        
                        # Añadir columnas necesarias
                        datos_indicador_largo['Categoría'] = 'Pobreza y Desigualdad'
                        
                        # Añadir columnas adicionales para que coincida con el formato
                        for col in df.columns:
                            if col not in datos_indicador_largo.columns:
                                datos_indicador_largo[col] = None
                        
                        # Combinar con el subconjunto existente
                        subconjuntos[tema] = pd.concat([df, datos_indicador_largo], ignore_index=True)
                        
                        logger.info(f"    ✓ Indicador {indicador} añadido manualmente. Registros añadidos: {len(datos_indicador_largo)}")
                    else:
                        logger.error(f"    ✗ Indicador {indicador} no encontrado en los datos originales.")
    
    return subconjuntos

def guardar_datos_procesados(df_maestro, subconjuntos, directorio_salida):
    """
    Guarda los datos procesados en archivos CSV.
    
    Args:
        df_maestro (pd.DataFrame): DataFrame maestro
        subconjuntos (dict): Diccionario con DataFrames temáticos
        directorio_salida (str): Ruta del directorio de salida
    """
    logger.info("Guardando datos procesados...")
    
    # Crear directorio para datos procesados si no existe
    if not os.path.exists(directorio_salida):
        os.makedirs(directorio_salida)
    
    # Guardar DataFrame maestro
    archivo_maestro = os.path.join(directorio_salida, "datos_maestro.csv")
    df_maestro.to_csv(archivo_maestro, index=False, encoding='utf-8')
    logger.info(f"  DataFrame maestro guardado en: {archivo_maestro}")
    
    # Guardar subconjuntos temáticos
    for tema, df in subconjuntos.items():
        archivo_tema = os.path.join(directorio_salida, f"datos_{tema}.csv")
        
        # Verificar presencia de indicador de pobreza para el tema social
        if tema == 'social':
            tiene_pobreza = INDICADORES_CRITICOS['POBREZA'] in df['Indicator Code'].values
            logger.info(f"  Indicador de pobreza en subconjunto 'social' antes de guardar: {tiene_pobreza}")
        
        df.to_csv(archivo_tema, index=False, encoding='utf-8')
        logger.info(f"  Subconjunto '{tema}' guardado en: {archivo_tema}")

def analisis_exploratorio(df_maestro, subconjuntos):
    """
    Realiza un análisis exploratorio básico de los datos procesados.
    
    Args:
        df_maestro (pd.DataFrame): DataFrame maestro
        subconjuntos (dict): Diccionario con DataFrames temáticos
    """
    logger.info("Análisis Exploratorio de Datos:")
    
    # Información general del DataFrame maestro
    logger.info("Resumen del DataFrame maestro:")
    logger.info(f"  Total de indicadores: {df_maestro['Indicator Code'].nunique()}")
    logger.info(f"  Rango de años: {df_maestro['Año'].min()} - {df_maestro['Año'].max()}")
    logger.info(f"  Categorías de indicadores: {df_maestro['Categoría'].nunique()}")
    
    # Completitud de datos por año
    completitud_por_año = df_maestro.groupby('Año').count()['Valor']
    logger.info("Completitud de datos por año (muestra):")
    logger.info(completitud_por_año.tail(10))
    
    # Análisis de valores faltantes
    valores_faltantes = df_maestro.isna().sum()
    valores_faltantes = valores_faltantes[valores_faltantes > 0]
    if not valores_faltantes.empty:
        logger.info("Valores faltantes por columna:")
        for col, count in valores_faltantes.items():
            logger.info(f"  {col}: {count}")
    
    # Análisis de valores extremos (outliers) para indicadores clave
    logger.info("Estadísticas descriptivas para indicadores seleccionados:")
    for tema, df in subconjuntos.items():
        logger.info(f"  Tema: {tema}")
        for indicador, grupo in df.groupby('Indicator Name'):
            if not grupo['Valor'].empty and not grupo['Valor'].isna().all():
                logger.info(f"    {indicador}:")
                logger.info(f"      Mín: {grupo['Valor'].min():.2f}, Máx: {grupo['Valor'].max():.2f}, Media: {grupo['Valor'].mean():.2f}")
            else:
                logger.info(f"    {indicador}: No hay datos disponibles o son todos nulos")

def verificar_salida_final(directorio_salida):
    """
    Realiza una verificación final de los archivos de salida.
    
    Args:
        directorio_salida (str): Ruta del directorio de salida
    """
    archivo_social = os.path.join(directorio_salida, "datos_social.csv")
    if os.path.exists(archivo_social):
        try:
            df_social = pd.read_csv(archivo_social)
            tiene_pobreza = INDICADORES_CRITICOS['POBREZA'] in df_social['Indicator Code'].values
            logger.info(f"Verificación final - Indicador de pobreza en datos_social.csv: {tiene_pobreza}")
            if tiene_pobreza:
                cantidad_registros = len(df_social[df_social['Indicator Code'] == INDICADORES_CRITICOS['POBREZA']])
                logger.info(f"Número de registros: {cantidad_registros}")
        except Exception as e:
            logger.error(f"Error al verificar el archivo final: {e}")

def main():
    """Función principal que ejecuta todo el proceso de preprocesamiento"""
    logger.info("Iniciando preprocesamiento de datos...")
    
    try:
        # Obtener rutas de archivos
        rutas_archivos = obtener_rutas_archivos()
        
        # 1. Cargar datos crudos
        datos, metadatos_pais, metadatos_indicadores = cargar_datos_crudos(rutas_archivos)
        
        # 2. Limpiar y transformar datos
        datos_transformados = limpiar_y_transformar_datos(datos, metadatos_indicadores)
        
        # 3. Calcular indicadores derivados
        datos_con_derivados = calcular_indicadores_derivados(datos_transformados)
        
        # 4. Crear DataFrame maestro
        df_maestro = crear_dataframe_maestro(datos_con_derivados, metadatos_pais, metadatos_indicadores)
        
        # 5. Crear subconjuntos temáticos
        subconjuntos = crear_subconjuntos_tematicos(df_maestro)
        
        # 5b. Verificar indicadores críticos y añadirlos si faltan
        subconjuntos = verificar_indicadores_criticos(subconjuntos, datos)
        
        # 6. Realizar análisis exploratorio
        analisis_exploratorio(df_maestro, subconjuntos)
        
        # 7. Guardar datos procesados
        guardar_datos_procesados(df_maestro, subconjuntos, rutas_archivos['directorio_salida'])
        
        # Verificación final
        verificar_salida_final(rutas_archivos['directorio_salida'])
        
        logger.info("¡Proceso de preprocesamiento completado con éxito!")
        
    except Exception as e:
        logger.error(f"Error en el flujo de preprocesamiento: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()