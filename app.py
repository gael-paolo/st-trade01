import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import timedelta
import datetime
import io
import numpy as np # Necesario para float('nan') y cálculos

# --- Funciones de Estrategia e Indicadores ---

@st.cache_data
def Media_Movil_Simple(df: pd.DataFrame, longitud: int, columna: str = "Close") -> pd.Series:
    """Calcula la Media Móvil Simple (SMA)."""
    return df[columna].rolling(window=longitud, min_periods=longitud).mean()

@st.cache_data
def Media_Movil_Exponencial(df: pd.DataFrame, longitud: int, columna: str = "Close") -> pd.Series:
    """Calcula la Media Móvil Exponencial (EMA)."""
    return df[columna].ewm(span=longitud, min_periods=longitud, adjust=False).mean()

# Generador de Señales SMA Crossover
def generar_señales_cruce_sma(df: pd.DataFrame, sma_rapida: int, sma_lenta: int):
    """Genera señales de compra/venta basadas en el cruce de dos SMAs."""
    df['Media_R'] = Media_Movil_Simple(df, longitud=sma_rapida)
    df['Media_L'] = Media_Movil_Simple(df, longitud=sma_lenta)
    df['Tipo_Media'] = 'SMA'
    
    df['Cruce_Arriba'] = ((df['Media_R'].shift(1) < df['Media_L'].shift(1)) & (df['Media_R'] > df['Media_L'])).astype(int)
    df['Cruce_Abajo'] = ((df['Media_R'].shift(1) > df['Media_L'].shift(1)) & (df['Media_R'] < df['Media_L'])).astype(int)
    
    df['Señal'] = 0
    df.loc[df['Cruce_Arriba'] == 1, 'Señal'] = 1  # Compra
    df.loc[df['Cruce_Abajo'] == 1, 'Señal'] = -1 # Venta
    
    return df

# Generador de Señales EMA Crossover
def generar_señales_cruce_ema(df: pd.DataFrame, ema_rapida: int, ema_lenta: int):
    """Genera señales de compra/venta basadas en el cruce de dos EMAs."""
    df['Media_R'] = Media_Movil_Exponencial(df, longitud=ema_rapida)
    df['Media_L'] = Media_Movil_Exponencial(df, longitud=ema_lenta)
    df['Tipo_Media'] = 'EMA'
    
    df['Cruce_Arriba'] = ((df['Media_R'].shift(1) < df['Media_L'].shift(1)) & (df['Media_R'] > df['Media_L'])).astype(int)
    df['Cruce_Abajo'] = ((df['Media_R'].shift(1) > df['Media_L'].shift(1)) & (df['Media_R'] < df['Media_L'])).astype(int)
    
    df['Señal'] = 0
    df.loc[df['Cruce_Arriba'] == 1, 'Señal'] = 1  # Compra
    df.loc[df['Cruce_Abajo'] == 1, 'Señal'] = -1 # Venta
    
    return df

# Función placeholder para otras estrategias futuras
def generar_señales_otras(df: pd.DataFrame, estrategia_nombre: str):
    df['Media_R'] = np.nan
    df['Media_L'] = np.nan
    df['Tipo_Media'] = estrategia_nombre
    df['Señal'] = 0
    return df


# --- Función de Backtesting (MODIFICADA para SL/TP) ---

def ejecutar_backtest(df, capital_inicial=10000, sl_pct=0.005, tp_pct=0.01, comision=0.0):
    """Simula operaciones Long/Short, cerrando por SL/TP o señal opuesta."""
    
    # 1. Preparación de Posición y Precios
    df['Position_Temp'] = df['Señal'].replace({1: 1, -1: -1}) 
    df['Position_Temp'].ffill(inplace=True)
    df['Posicion'] = df['Position_Temp'].shift(1).fillna(0)
    
    # Precios de la Vela Siguiente (necesario para simular SL/TP)
    df['Open_Next'] = df['Open'].shift(-1)
    df['High_Next'] = df['High'].shift(-1)
    df['Low_Next'] = df['Low'].shift(-1)
    df['Close_Next'] = df['Close'].shift(-1)
    
    # Inicializar la columna de Capital
    df['Capital'] = float(capital_inicial)
    df['Retorno_Estrategia'] = 0.0
    
    # Estado de la posición y precio de entrada para simulación
    posicion_abierta = 0 # 1:Largo, -1:Corto, 0:Cerrada
    precio_entrada = 0.0

    # 2. Simulación Fila por Fila para Gestión de Riesgo (Iteración)
    for i in range(1, len(df)):
        
        # Precio de cierre de la vela anterior (que es el precio de entrada de la actual)
        price_close = df.loc[df.index[i-1], 'Close']
        price_open_next = df.loc[df.index[i], 'Open']
        
        # --- A. Cierre de Posición Abierta ---
        if posicion_abierta != 0:
            
            # Definición de niveles SL/TP
            if posicion_abierta == 1: # LARGO
                sl_price = precio_entrada * (1 - sl_pct)
                tp_price = precio_entrada * (1 + tp_pct)
                
                # SL/TP en la vela actual
                sl_hit = (df.loc[df.index[i], 'Low'] <= sl_price)
                tp_hit = (df.loc[df.index[i], 'High'] >= tp_price)
                
                # Orden de prioridad: SL > TP > Señal Opuesta
                if sl_hit:
                    retorno = -sl_pct
                    df.loc[df.index[i], 'Retorno_Estrategia'] = retorno # Aplicar pérdida
                    posicion_abierta = 0 # Cerrar posición
                elif tp_hit:
                    retorno = tp_pct
                    df.loc[df.index[i], 'Retorno_Estrategia'] = retorno # Aplicar ganancia
                    posicion_abierta = 0 # Cerrar posición
                elif df.loc[df.index[i], 'Señal'] == -1:
                    # Cerrado por Señal Inversa
                    retorno = (df.loc[df.index[i], 'Close'] / precio_entrada) - 1
                    df.loc[df.index[i], 'Retorno_Estrategia'] = retorno 
                    posicion_abierta = 0
            
            elif posicion_abierta == -1: # CORTO
                sl_price = precio_entrada * (1 + sl_pct)
                tp_price = precio_entrada * (1 - tp_pct)
                
                sl_hit = (df.loc[df.index[i], 'High'] >= sl_price)
                tp_hit = (df.loc[df.index[i], 'Low'] <= tp_price)
                
                if sl_hit:
                    retorno = -sl_pct
                    df.loc[df.index[i], 'Retorno_Estrategia'] = retorno
                    posicion_abierta = 0
                elif tp_hit:
                    retorno = tp_pct
                    df.loc[df.index[i], 'Retorno_Estrategia'] = retorno
                    posicion_abierta = 0
                elif df.loc[df.index[i], 'Señal'] == 1:
                    # Cerrado por Señal Inversa
                    retorno = 1 - (df.loc[df.index[i], 'Close'] / precio_entrada)
                    df.loc[df.index[i], 'Retorno_Estrategia'] = retorno
                    posicion_abierta = 0

        # --- B. Apertura de Nueva Posición (Si no hay posición abierta) ---
        if posicion_abierta == 0:
            if df.loc[df.index[i], 'Señal'] == 1:
                posicion_abierta = 1
                precio_entrada = price_open_next # Entrar en el OPEN de la vela actual
            elif df.loc[df.index[i], 'Señal'] == -1:
                posicion_abierta = -1
                precio_entrada = price_open_next # Entrar en el OPEN de la vela actual
        
        # --- C. Actualización de Capital ---
        # Si la posición está abierta pero no se cerró por SL/TP, el retorno en la vela es 0.
        # Si se cerró (SL/TP/Señal), el retorno ya se aplicó.
        
        # El capital se actualiza con el retorno de la estrategia para esa vela
        capital_anterior = df.loc[df.index[i-1], 'Capital']
        retorno_aplicado = df.loc[df.index[i], 'Retorno_Estrategia']
        df.loc[df.index[i], 'Capital'] = capital_anterior * (1 + retorno_aplicado)
        
        # Actualizar la Posición en el DataFrame final (para visualización)
        df.loc[df.index[i], 'Posicion'] = posicion_abierta

    # 3. Métricas Finales
    # Limpiar columnas temporales de precios
    df.drop(columns=['Open_Next', 'High_Next', 'Low_Next', 'Close_Next'], inplace=True) 

    # Recalcular el retorno total de las ganancias aplicadas
    retorno_total = (df['Capital'].iloc[-1] / capital_inicial) - 1
    
    dias_operados = (df.index[-1] - df.index[0]).days if not df.empty else 1
    retorno_anualizado = ((1 + retorno_total) ** (252 / dias_operados)) - 1 if dias_operados > 0 else 0
    
    # Drawdown
    df['Max_Capital'] = df['Capital'].cummax()
    df['Drawdown'] = (df['Max_Capital'] - df['Capital']) / df['Max_Capital']
    max_drawdown = df['Drawdown'].max() if not df.empty else 0
    
    # Ratio de Sharpe (usando el retorno aplicado)
    std_dev = df['Retorno_Estrategia'].std()
    sharpe_ratio = df['Retorno_Estrategia'].mean() / std_dev * (252**0.5) if std_dev > 0 else 0
    
    metricas = {
        "Capital Inicial": f"${capital_inicial:,.2f}",
        "Capital Final": f"${df['Capital'].iloc[-1]:,.2f}",
        "Retorno Total (%)": f"{(retorno_total * 100):.2f}%",
        "Retorno Anualizado (%)": f"{(retorno_anualizado * 100):.2f}%",
        "Máximo Drawdown (%)": f"{(max_drawdown * 100):.2f}%",
        "Ratio de Sharpe (252d)": f"{sharpe_ratio:.2f}"
    }
    
    return df, metricas, dias_operados

# --- Función de Exportación a Excel ---

def to_excel(df: pd.DataFrame) -> bytes:
    """Convierte el DataFrame a un objeto de bytes de Excel."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer: 
        df.to_excel(writer, index=True, sheet_name='Resultados Backtest')
    processed_data = output.getvalue()
    return processed_data


# --- Configuración de Streamlit ---

st.set_page_config(layout="wide", page_title="Backtester - EMA Scalping")

st.title("📈 Backtester: Cruce de EMA (Scalping Long/Short con SL/TP)")
st.markdown("Herramienta de prueba para estrategias de trading de alta frecuencia.")

HOY = datetime.date.today() 

# --- Inicializar Session State para las fechas ---
if 'fecha_inicio' not in st.session_state:
    st.session_state.fecha_inicio = HOY - timedelta(days=60)
if 'fecha_fin' not in st.session_state:
    st.session_state.fecha_fin = HOY

# --- Barra Lateral (Inputs) ---

st.sidebar.header("⚙️ Configuración de la Estrategia")

# 1. Selección de Estrategia
st.sidebar.subheader("Selección de Estrategia")
estrategia_seleccionada = st.sidebar.selectbox(
    "Estrategia a Evaluar",
    [
        "Cruce de Medias Exponenciales (EMA)", 
        "Cruce de Medias Móviles (SMA)", 
        "RSI Crossover (Próximamente)", 
    ],
    index=0
)

# 2. Parámetros de la Estrategia (Condicional)
st.sidebar.subheader("Parámetros de la Estrategia")

if estrategia_seleccionada == "Cruce de Medias Exponenciales (EMA)":
    st.sidebar.markdown("##### Media Móvil Exponencial (EMA)")
    ema_rapida = st.sidebar.slider("EMA Rápida (Períodos)", min_value=2, max_value=50, value=12, key="ema_r_slider")
    ema_lenta = st.sidebar.slider("EMA Lenta (Períodos)", min_value=2, max_value=200, value=26, key="ema_l_slider")
    
    if ema_rapida >= ema_lenta:
        st.sidebar.error("La EMA Rápida debe ser menor que la EMA Lenta.")
        st.stop()
    
    estrategia_params = {'ema_rapida': ema_rapida, 'ema_lenta': ema_lenta}

elif estrategia_seleccionada == "Cruce de Medias Móviles (SMA)":
    st.sidebar.markdown("##### Media Móvil Simple (SMA)")
    sma_rapida = st.sidebar.slider("SMA Rápida (Períodos)", min_value=2, max_value=50, value=5, key="sma_r_slider")
    sma_lenta = st.sidebar.slider("SMA Lenta (Períodos)", min_value=2, max_value=200, value=10, key="sma_l_slider")
    
    if sma_rapida >= sma_lenta:
        st.sidebar.error("La SMA Rápida debe ser menor que la SMA Lenta.")
        st.stop()
    
    estrategia_params = {'sma_rapida': sma_rapida, 'sma_lenta': sma_lenta}

else:
    estrategia_params = {'estrategia_nombre': estrategia_seleccionada}


# --- Secciones de Datos y Fechas ---

# 3. Entrada de Datos
st.sidebar.subheader("Datos del Ticker")
top_tickers = ["BTC-USD", "ETH-USD", "MSFT", "AAPL", "NVDA", "TSLA"]
ticker = st.sidebar.selectbox(
    "Símbolo de Ticker", 
    options=top_tickers,
    index=0,
)
usar_ticker_personalizado = st.sidebar.checkbox("Usar ticker personalizado")
if usar_ticker_personalizado:
    ticker = st.sidebar.text_input("Ticker personalizado (Ej: NFLX)", "NFLX")

intervalo = st.sidebar.selectbox(
    "Intervalo", 
    ["1m", "2m", "5m", "15m", "30m", "60m", "1h", "1d"],
    index=2,
    help="1m, 5m, 15m son comunes para Scalping."
)

LIMITES_YFINANCE = {
    "1m": {"max_dias": 7, "sugerido_dias": 5, "nombre": "1 minuto"},
    "2m": {"max_dias": 60, "sugerido_dias": 30, "nombre": "2 minutos"},
    "5m": {"max_dias": 60, "sugerido_dias": 30, "nombre": "5 minutos"},
    "15m": {"max_dias": 60, "sugerido_dias": 30, "nombre": "15 minutos"},
    "30m": {"max_dias": 60, "sugerido_dias": 30, "nombre": "30 minutos"},
    "60m": {"max_dias": 730, "sugerido_dias": 90, "nombre": "1 hora"},
    "1h": {"max_dias": 730, "sugerido_dias": 90, "nombre": "1 hora"},
    "1d": {"max_dias": None, "sugerido_dias": 365, "nombre": "1 día"},
}

limite_info = LIMITES_YFINANCE[intervalo]
fecha_inicio_sugerida = HOY - timedelta(days=limite_info["sugerido_dias"])

if limite_info["max_dias"]:
    fecha_minima_permitida = HOY - timedelta(days=limite_info["max_dias"])
else:
    fecha_minima_permitida = datetime.date(2000, 1, 1)

if 'fecha_inicio' not in st.session_state or st.session_state.get('ultimo_intervalo') != intervalo:
    st.session_state.fecha_inicio = fecha_inicio_sugerida
    st.session_state.ultimo_intervalo = intervalo

st.sidebar.markdown("### 📅 Rango de Fechas")
col_start, col_end = st.sidebar.columns(2)

fecha_inicio = col_start.date_input(
    "Fecha de Inicio", 
    value=st.session_state.fecha_inicio,
    min_value=fecha_minima_permitida,
    max_value=HOY,
    key="input_fecha_inicio"
)

fecha_fin = col_end.date_input(
    "Fecha de Fin", 
    value=st.session_state.fecha_fin, 
    max_value=HOY,
    key="input_fecha_fin"
)

st.session_state.fecha_inicio = fecha_inicio
st.session_state.fecha_fin = fecha_fin

if limite_info["max_dias"]:
    dias_solicitados = (fecha_fin - fecha_inicio).days
    if dias_solicitados > limite_info["max_dias"]:
        st.sidebar.error(f"⚠️ Límite de {limite_info['max_dias']} días excedido.")
        st.stop() 

# 4. Parámetros del Backtest
st.sidebar.subheader("Configuración de la Simulación")
capital_inicial = st.sidebar.number_input("Capital Inicial", min_value=0.01, value=10000.00, step=0.01, format="%.2f")

# --- Parámetros SL/TP ---
st.sidebar.markdown("### 🛑 Stop Loss / Take Profit (%)")
# Los valores se ingresan como porcentaje (ej: 0.5%) y se dividen por 100.0 para usarse como fracción.
stop_loss_pct = st.sidebar.number_input("Stop Loss (SL) %", min_value=0.01, value=0.50, step=0.01) / 100.0
take_profit_pct = st.sidebar.number_input("Take Profit (TP) %", min_value=0.01, value=1.00, step=0.01) / 100.0


# --- Proceso Principal ---

if st.sidebar.button("🚀 Ejecutar Backtest"):
    try:
        # Descargar datos
        st.subheader(f"⬇️ Descargando Datos para **{ticker.upper()}**")
        if fecha_inicio >= fecha_fin:
            st.error("La Fecha de Inicio debe ser anterior a la Fecha de Fin.")
            st.stop()
            
        df = yf.download(ticker, start=fecha_inicio, end=fecha_fin + timedelta(days=1), interval=intervalo, multi_level_index=False)
        
        if df.empty:
            st.error(f"No se pudieron descargar datos para {ticker} en el rango y/o intervalo especificado.")
            st.stop()
        
        # Manejo de Zona Horaria: Convertir a EST (New York) si es necesario
        if df.index.tz is None:
            df.index = df.index.tz_localize('America/New_York')
        else:
             df.index = df.index.tz_convert('America/New_York')

        st.success(f"Datos descargados: {df.index.min().date()} a {df.index.max().date()} ({len(df)} velas) - Horario: EST (New York)")

        # Aplicar Estrategia
        if estrategia_seleccionada == "Cruce de Medias Móviles (SMA)":
            df_estrategia = generar_señales_cruce_sma(df.copy(), **estrategia_params)
            media_r_nombre = f"SMA {estrategia_params['sma_rapida']}"
            media_l_nombre = f"SMA {estrategia_params['sma_lenta']}"
            
        elif estrategia_seleccionada == "Cruce de Medias Exponenciales (EMA)":
            df_estrategia = generar_señales_cruce_ema(df.copy(), **estrategia_params)
            media_r_nombre = f"EMA {estrategia_params['ema_rapida']}"
            media_l_nombre = f"EMA {estrategia_params['ema_lenta']}"
            
        else: # Estrategias futuras
             st.warning("Estrategia en desarrollo. No se generaron señales de trading.")
             df_estrategia = generar_señales_otras(df.copy(), estrategia_seleccionada)
             media_r_nombre = 'N/A'
             media_l_nombre = 'N/A'

        # Limpieza y Backtest
        cols_to_check = ['Close']
        if 'Media_R' in df_estrategia.columns and not df_estrategia['Media_R'].isnull().all():
            cols_to_check.extend(['Media_R', 'Media_L'])

        df_estrategia_limpio = df_estrategia.dropna(subset=cols_to_check).copy()

        if df_estrategia_limpio.empty:
            st.warning("⚠️ **Datos Insuficientes:** Intenta ampliar el rango de fechas.")
            st.stop()

        # Ejecutar Backtest (Capturando los 3 valores)
        df_resultados, metricas, dias_operados = ejecutar_backtest(
            df_estrategia_limpio, 
            capital_inicial=capital_inicial,
            sl_pct=stop_loss_pct,
            tp_pct=take_profit_pct
        )

        # --- Resultados ---
        
        st.header(f"📊 Resultados del Backtest: **{estrategia_seleccionada}**")
        
        # 1. Métricas
        st.subheader("Métricas de Rendimiento")
        col1, col2, col3 = st.columns(3)
        col1.metric("Retorno Total", metricas['Retorno Total (%)'], delta=metricas['Retorno Anualizado (%)'])
        col2.metric("Capital Final", metricas['Capital Final'])
        col3.metric("Máximo Drawdown", metricas['Máximo Drawdown (%)'])
        st.markdown(f"**Ratio de Sharpe:** {metricas['Ratio de Sharpe (252d)']}")
        st.info("ℹ️ Un Ratio de Sharpe mayor a 1.0 es generalmente bueno.")
        st.markdown("---")
        
        # 2. Gráfico de Evolución del Capital
        st.subheader("Evolución del Capital")
        
        capital_acumulado = df_resultados['Capital']
        
        fig_cap, ax_cap = plt.subplots(figsize=(10, 3))
        ax_cap.plot(capital_acumulado.index, capital_acumulado, label="Capital Acumulado", color="blue", linewidth=1.5)
        ax_cap.fill_between(
            x=capital_acumulado.index,
            y1=capital_acumulado,
            y2=df_resultados['Max_Capital'],
            where=(capital_acumulado < df_resultados['Max_Capital']),
            color="red",
            alpha=0.3,
            label="Drawdown"
        )
        ax_cap.set_title("Evolución del Capital", size=12)
        ax_cap.legend(loc='best', fontsize=9)
        ax_cap.grid(True, alpha=0.3)
        ax_cap.tick_params(labelsize=9)
        
        st.pyplot(fig_cap)
        st.markdown("---")

        # 3. Gráfico de Velas con Medias y Cruces
        st.subheader("🕯️ Visualización de la Estrategia")
        
        # 3.1 Crear series de plotting con np.nan para evitar errores de mplfinance
        df_resultados['Señal_Compra_Plot'] = df_resultados.apply(
            lambda row: row['Low'] if row['Señal'] == 1 else np.nan, axis=1
        )
        df_resultados['Señal_Venta_Plot'] = df_resultados.apply(
            lambda row: row['High'] if row['Señal'] == -1 else np.nan, axis=1
        )
        
        # 3.2 Definir los addplots
        addplots = []
        if estrategia_seleccionada.startswith("Cruce de Medias"):
            addplots.extend([
                mpf.make_addplot(df_resultados['Media_R'], label=media_r_nombre, color="lime", width=1.0, panel=0),
                mpf.make_addplot(df_resultados['Media_L'], label=media_l_nombre, color="blue", width=1.0, panel=0),
            ])
            
        addplots.extend([
            mpf.make_addplot(df_resultados['Señal_Compra_Plot'], type='scatter', markersize=100, marker='^', color='green', panel=0, alpha=1.0),
            mpf.make_addplot(df_resultados['Señal_Venta_Plot'], type='scatter', markersize=100, marker='v', color='red', panel=0, alpha=1.0),
        ])
            
        # 3.3 Crear la figura de mplfinance
        fig_static, axlist = mpf.plot(
            df_resultados, 
            type="candle", 
            style="binance",
            volume=True,      
            addplot=addplots,
            title=f"{estrategia_seleccionada} en {ticker.upper()} ({intervalo})",
            returnfig=True,
            figsize=(14, 7),
            panel_ratios=(3, 1)
        )
        
        ax = axlist[0]
        ax.legend(loc='upper left', fontsize=10)
        
        st.pyplot(fig_static)
        st.markdown("---")

        # 4. Datos de la Estrategia y Exportación
        st.info(f"📊 **Total de velas:** {len(df_resultados)} | **Señales de Compra/Venta:** {(df_resultados['Señal'] != 0).sum()} | **Días simulados:** {dias_operados}")
        st.subheader("Datos de la Estrategia (Últimas 10 Filas)")
        
        cols_to_display = ['Open', 'High', 'Low', 'Close', 'Volume']
        if estrategia_seleccionada.startswith("Cruce de Medias"):
            cols_to_display.extend(['Media_R', 'Media_L'])
            
        cols_to_display.extend(['Señal', 'Posicion', 'Retorno_Estrategia', 'Capital'])
        
        st.dataframe(df_resultados[cols_to_display].tail(10))

        # --- Sección de Descarga a Excel (con corrección de Timezone) ---
        st.markdown("---")
        st.subheader("📥 Descargar Datos Completos del Backtest")

        # 1. Preparar el DataFrame para Excel
        df_excel = df_resultados.copy()
        
        # 2. SOLUCIÓN DEL ERROR: Eliminar la información de la zona horaria del índice
        if df_excel.index.tz is not None:
            df_excel.index = df_excel.index.tz_localize(None) 
        
        # 3. Redondeo de columnas numéricas
        for col in ['Capital', 'Max_Capital', 'Drawdown', 'Close', 'Open', 'High', 'Low', 'Media_R', 'Media_L', 'Señal', 'Posicion', 'Retorno_Estrategia']:
            if col in df_excel.columns:
                df_excel[col] = df_excel[col].round(4)
            
        # Generar el archivo Excel
        excel_data = to_excel(df_excel)

        # Crear el nombre del archivo dinámicamente
        ticker_upper = ticker.upper()
        fecha_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        file_name = f"Backtest_{ticker_upper}_{estrategia_seleccionada.replace(' ', '_').replace('(', '').replace(')', '')}_{fecha_str}.xlsx"
        
        # Botón de descarga
        st.download_button(
            label="💾 Descargar Resultados a Excel",
            data=excel_data,
            file_name=file_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Descarga el DataFrame completo con precios, indicadores, señales, posiciones y capital."
        )


    except Exception as e:
        st.error(f"Ocurrió un error durante el backtest: {e}")
        st.exception(e)

else:
    st.info("Presiona **'🚀 Ejecutar Backtest'** para comenzar la simulación de scalping.")