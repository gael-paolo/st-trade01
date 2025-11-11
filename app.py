import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go 
import mplfinance as mpf
from datetime import timedelta
import datetime 

# --- Funciones de Estrategia e Indicadores ---

@st.cache_data
def Media_Movil_Simple(df: pd.DataFrame, longitud: int, columna: str = "Close") -> pd.Series:
    """Calcula la Media Móvil Simple (SMA)."""
    return df[columna].rolling(window=longitud, min_periods=longitud).mean()

def generar_señales_cruce_sma(df: pd.DataFrame, sma_rapida: int, sma_lenta: int):
    """Genera señales de compra/venta basadas en el cruce de dos SMAs."""
    df['SMA_R'] = Media_Movil_Simple(df, longitud=sma_rapida)
    df['SMA_L'] = Media_Movil_Simple(df, longitud=sma_lenta)
    
    df['Cruce_Arriba'] = ((df['SMA_R'].shift(1) < df['SMA_L'].shift(1)) & (df['SMA_R'] > df['SMA_L'])).astype(int)
    df['Cruce_Abajo'] = ((df['SMA_R'].shift(1) > df['SMA_L'].shift(1)) & (df['SMA_R'] < df['SMA_L'])).astype(int)
    
    df['Señal'] = 0
    df.loc[df['Cruce_Arriba'] == 1, 'Señal'] = 1
    df.loc[df['Cruce_Abajo'] == 1, 'Señal'] = -1
    
    return df

# --- Función de Backtesting ---

def ejecutar_backtest_sma(df, capital_inicial=10000, comision=0.0):
    """Simula las operaciones y calcula métricas de backtesting."""
    
    df['Position_Temp'] = df['Señal'].replace({1: 1, -1: 0})
    df['Position_Temp'].ffill(inplace=True)
    df['Posicion'] = df['Position_Temp'].shift(1).fillna(0)
    
    df['Retorno_Activo'] = df['Close'].pct_change()
    df['Retorno_Estrategia'] = df['Retorno_Activo'] * df['Posicion']
    
    df['Capital'] = capital_inicial * (1 + df['Retorno_Estrategia']).cumprod()
    
    capital_final = df['Capital'].iloc[-1] if not df['Capital'].empty else capital_inicial
    retorno_total = (capital_final / capital_inicial) - 1
    
    dias_operados = (df.index[-1] - df.index[0]).days if not df.empty else 1
    retorno_anualizado = ((1 + retorno_total) ** (252 / dias_operados)) - 1 if dias_operados > 0 else 0
    
    df['Max_Capital'] = df['Capital'].cummax()
    df['Drawdown'] = (df['Max_Capital'] - df['Capital']) / df['Max_Capital']
    max_drawdown = df['Drawdown'].max() if not df.empty else 0
    
    std_dev = df['Retorno_Estrategia'].std()
    sharpe_ratio = df['Retorno_Estrategia'].mean() / std_dev * (252**0.5) if std_dev > 0 else 0
    
    metricas = {
        "Capital Inicial": f"${capital_inicial:,.2f}",
        "Capital Final": f"${capital_final:,.2f}",
        "Retorno Total (%)": f"{(retorno_total * 100):.2f}%",
        "Retorno Anualizado (%)": f"{(retorno_anualizado * 100):.2f}%",
        "Máximo Drawdown (%)": f"{(max_drawdown * 100):.2f}%",
        "Ratio de Sharpe (252d)": f"{sharpe_ratio:.2f}"
    }
    
    return df, metricas

# --- Configuración de Streamlit ---

st.set_page_config(layout="wide", page_title="Scalping Backtester - SMA Crossover")

st.title("📈 Scalping Backtester con Cruce de Medias Móviles")
st.markdown("Herramienta de prueba para estrategias de trading de alta frecuencia.")

HOY = datetime.date.today() 

# --- Inicializar Session State para las fechas ---
if 'fecha_inicio' not in st.session_state:
    st.session_state.fecha_inicio = HOY - timedelta(days=150)
if 'fecha_fin' not in st.session_state:
    st.session_state.fecha_fin = HOY

# --- Barra Lateral (Inputs) ---

st.sidebar.header("⚙️ Configuración de la Estrategia")

# 1. Entrada de Datos
st.sidebar.subheader("Datos")

# Top 10 tickers más importantes
top_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "BRK.B", "V", "JPM"]
ticker = st.sidebar.selectbox(
    "Símbolo de Ticker", 
    options=top_tickers,
    index=0,
    help="Selecciona uno de los 10 tickers más importantes del mercado"
)

# Permitir entrada personalizada
usar_ticker_personalizado = st.sidebar.checkbox("Usar ticker personalizado")
if usar_ticker_personalizado:
    ticker = st.sidebar.text_input("Ticker personalizado (Ej: NFLX)", "NFLX")

# Intervalo primero (ANTES de las fechas)
intervalo = st.sidebar.selectbox(
    "Intervalo", 
    ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"],
    index=4,  # 30m por defecto
    help="Selecciona el intervalo de tiempo para cada vela"
)

# Definir rangos sugeridos y límites según el intervalo
LIMITES_YFINANCE = {
    "1m": {"max_dias": 7, "sugerido_dias": 5, "nombre": "1 minuto"},
    "2m": {"max_dias": 60, "sugerido_dias": 30, "nombre": "2 minutos"},
    "5m": {"max_dias": 60, "sugerido_dias": 30, "nombre": "5 minutos"},
    "15m": {"max_dias": 60, "sugerido_dias": 30, "nombre": "15 minutos"},
    "30m": {"max_dias": 60, "sugerido_dias": 30, "nombre": "30 minutos"},
    "60m": {"max_dias": 730, "sugerido_dias": 90, "nombre": "1 hora"},
    "90m": {"max_dias": 60, "sugerido_dias": 30, "nombre": "90 minutos"},
    "1h": {"max_dias": 730, "sugerido_dias": 90, "nombre": "1 hora"},
    "1d": {"max_dias": None, "sugerido_dias": 365, "nombre": "1 día"},
    "5d": {"max_dias": None, "sugerido_dias": 365, "nombre": "5 días"},
    "1wk": {"max_dias": None, "sugerido_dias": 730, "nombre": "1 semana"},
    "1mo": {"max_dias": None, "sugerido_dias": 1825, "nombre": "1 mes"},
    "3mo": {"max_dias": None, "sugerido_dias": 1825, "nombre": "3 meses"},
}

limite_info = LIMITES_YFINANCE[intervalo]

# Calcular fecha de inicio sugerida
fecha_inicio_sugerida = HOY - timedelta(days=limite_info["sugerido_dias"])

# Si hay límite máximo, ajustar
if limite_info["max_dias"]:
    fecha_minima_permitida = HOY - timedelta(days=limite_info["max_dias"])
else:
    fecha_minima_permitida = datetime.date(2000, 1, 1)  # Sin límite práctico

# Mostrar información del intervalo seleccionado
if limite_info["max_dias"]:
    st.sidebar.info(f"⏱️ **Intervalo {limite_info['nombre']}**\n\n"
                   f"📊 Rango sugerido: **{limite_info['sugerido_dias']} días**\n\n"
                   f"⚠️ Límite máximo: **{limite_info['max_dias']} días**")
else:
    st.sidebar.info(f"⏱️ **Intervalo {limite_info['nombre']}**\n\n"
                   f"📊 Rango sugerido: **{limite_info['sugerido_dias']} días**\n\n"
                   f"✅ Sin límite de histórico")

st.sidebar.markdown("### 📅 Rango de Fechas")

# Inicializar con fecha sugerida si no existe en session_state
if 'fecha_inicio' not in st.session_state or st.session_state.get('ultimo_intervalo') != intervalo:
    st.session_state.fecha_inicio = fecha_inicio_sugerida
    st.session_state.ultimo_intervalo = intervalo

col_start, col_end = st.sidebar.columns(2)

# Fecha de Inicio con validación
fecha_inicio = col_start.date_input(
    "Fecha de Inicio", 
    value=st.session_state.fecha_inicio,
    min_value=fecha_minima_permitida,
    max_value=HOY,
    key="input_fecha_inicio"
)

# Fecha de Fin
fecha_fin = col_end.date_input(
    "Fecha de Fin", 
    value=st.session_state.fecha_fin, 
    max_value=HOY,
    key="input_fecha_fin"
)

# Actualizar session_state
st.session_state.fecha_inicio = fecha_inicio
st.session_state.fecha_fin = fecha_fin

# Validación del rango según el límite
if limite_info["max_dias"]:
    dias_solicitados = (fecha_fin - fecha_inicio).days
    if dias_solicitados > limite_info["max_dias"]:
        st.sidebar.error(f"⚠️ El intervalo **{intervalo}** tiene un límite de **{limite_info['max_dias']} días**. "
                        f"Has seleccionado **{dias_solicitados} días**.")
        st.sidebar.warning(f"💡 Ajusta el rango a máximo {limite_info['max_dias']} días o elige un intervalo mayor (1h, 1d, etc.)")
        st.stop() 

# 2. Parámetros de la Estrategia
st.sidebar.subheader("Parámetros de SMA")
sma_rapida = st.sidebar.slider("SMA Rápida (Períodos)", min_value=2, max_value=50, value=5)
sma_lenta = st.sidebar.slider("SMA Lenta (Períodos)", min_value=2, max_value=200, value=10)

# 3. Parámetros del Backtest
st.sidebar.subheader("Backtesting")
capital_inicial = st.sidebar.number_input("Capital Inicial", min_value=0.01, value=10000.00, step=0.01, format="%.2f")

if sma_rapida >= sma_lenta:
    st.sidebar.error("La SMA Rápida debe ser menor que la SMA Lenta.")
    st.stop()

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
        
        # Convertir a horario del mercado (EST - Eastern Time)
        if df.index.tz is not None:
            df.index = df.index.tz_convert('America/New_York')
        
        st.success(f"Datos descargados: {df.index.min().date()} a {df.index.max().date()} ({len(df)} velas) - Horario: EST (New York)")

        # Aplicar Estrategia
        df_estrategia = generar_señales_cruce_sma(df.copy(), sma_rapida, sma_lenta)
        df_estrategia_limpio = df_estrategia.dropna().copy()
        
        if df_estrategia_limpio.empty:
            st.warning(f"""
                ⚠️ **Datos Insuficientes:**
                El DataFrame queda vacío después de calcular las SMAs {sma_rapida} y {sma_lenta} debido al rango corto o SMAs largas.
                **Intenta ampliar el rango de fechas.**
            """)
            st.stop()

        # Ejecutar Backtest
        df_resultados, metricas = ejecutar_backtest_sma(df_estrategia_limpio, capital_inicial=capital_inicial)

        # --- Resultados ---
        
        st.header("📊 Resultados del Backtest")
        
        # 1. Métricas
        st.subheader("Métricas de Rendimiento")
        col1, col2, col3 = st.columns(3)
        col1.metric("Retorno Total", metricas['Retorno Total (%)'], delta=metricas['Retorno Anualizado (%)'])
        col2.metric("Capital Final", metricas['Capital Final'])
        col3.metric("Máximo Drawdown", metricas['Máximo Drawdown (%)'], help="La mayor caída de un pico a un valle antes de alcanzar un nuevo pico.")
        
        st.markdown(f"**Ratio de Sharpe:** {metricas['Ratio de Sharpe (252d)']}")
        st.info("""
            ℹ️ **Ratio de Sharpe (Explicación):**
            Mide el rendimiento de la inversión ajustado por el riesgo. Un número **mayor a 1.0** es generalmente bueno,
            indicando que la estrategia ofrece un exceso de rendimiento (por encima de la tasa libre de riesgo)
            por cada unidad de volatilidad (riesgo) que asume.
        """)
        
        st.markdown("---")
        
        # 2. Gráfico de Evolución del Capital y Drawdown (MÁS PEQUEÑO)
        st.subheader("Evolución del Capital y Máxima Pérdida (Drawdown)")
        
        capital_acumulado = df_resultados['Capital']
        
        # Reducir el tamaño del gráfico de 10,5 a 10,3
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

        # 3. Gráfico de Velas con SMAs y Cruces - ESTÁTICO (mplfinance)
        st.subheader("📊 Visualización de la Estrategia y Señales (Estático)")
        
        # Crear series de plotting con NaN donde no hay señal
        df_resultados['Señal_Compra_Plot'] = df_resultados.apply(
            lambda row: row['Close'] if row['Señal'] == 1 else None, axis=1
        )
        df_resultados['Señal_Venta_Plot'] = df_resultados.apply(
            lambda row: row['Close'] if row['Señal'] == -1 else None, axis=1
        )
        
        # Definir los addplots
        media_mov_plots = [
            mpf.make_addplot(df_resultados['SMA_R'], label=f"SMA {sma_rapida}", color="green", type="line"),
            mpf.make_addplot(df_resultados['SMA_L'], label=f"SMA {sma_lenta}", color="blue", type="line"),
            mpf.make_addplot(df_resultados['Señal_Compra_Plot'], type='scatter', markersize=150, marker='^', color='lime'),
            mpf.make_addplot(df_resultados['Señal_Venta_Plot'], type='scatter', markersize=150, marker='v', color='red'),
        ]

        # Crear la figura de mplfinance
        fig_static, axlist = mpf.plot(
            df_resultados, 
            type="candle", 
            style="yahoo", 
            volume=False,
            addplot=media_mov_plots,
            title=f"Estrategia de Cruce de SMA ({sma_rapida} vs {sma_lenta}) en {ticker.upper()} ({intervalo})",
            returnfig=True,
            figsize=(12, 6)
        )
        
        # Mostrar el gráfico estático en Streamlit
        st.pyplot(fig_static)
        
        st.markdown("---")

        # 4. Gráfico de Velas con SMAs y Cruces - INTERACTIVO (Plotly)
        st.subheader("🕯️ Visualización Interactiva de la Estrategia y Señales")
        
        # Crear Series de Señales para Plotly (reutilizando las ya creadas)
        # Ya están creadas arriba: Señal_Compra_Plot y Señal_Venta_Plot

        # Calcular el rango Y para incluir todos los datos
        min_price = min(df_resultados['Low'].min(), df_resultados['SMA_R'].min(), df_resultados['SMA_L'].min())
        max_price = max(df_resultados['High'].max(), df_resultados['SMA_R'].max(), df_resultados['SMA_L'].max())
        padding = (max_price - min_price) * 0.05 
        y_range = [min_price - padding, max_price + padding]

        # Inicializar la Figura de Plotly
        fig = go.Figure()
        
        # Añadir Gráfico de Velas (Candlestick)
        fig.add_trace(go.Candlestick(
            x=df_resultados.index,
            open=df_resultados['Open'],
            high=df_resultados['High'],
            low=df_resultados['Low'],
            close=df_resultados['Close'],
            name='Velas',
            increasing_line_color='green', 
            decreasing_line_color='red'
        ))

        # Añadir SMAs (Líneas)
        fig.add_trace(go.Scatter(
            x=df_resultados.index,
            y=df_resultados['SMA_R'],
            line=dict(color='lime', width=1.5),
            name=f'SMA Rápida ({sma_rapida})'
        ))
        fig.add_trace(go.Scatter(
            x=df_resultados.index,
            y=df_resultados['SMA_L'],
            line=dict(color='blue', width=1.5),
            name=f'SMA Lenta ({sma_lenta})'
        ))

        # Añadir Señales de Compra
        fig.add_trace(go.Scatter(
            x=df_resultados.index,
            y=df_resultados['Señal_Compra_Plot'],
            mode='markers',
            marker=dict(symbol='triangle-up', size=12, color='green', line=dict(width=1, color='darkgreen')),
            name='Compra'
        ))

        # Añadir Señales de Venta
        fig.add_trace(go.Scatter(
            x=df_resultados.index,
            y=df_resultados['Señal_Venta_Plot'],
            mode='markers',
            marker=dict(symbol='triangle-down', size=12, color='red', line=dict(width=1, color='darkred')),
            name='Venta'
        ))

        # Configurar el formato del eje X según el intervalo
        if intervalo in ['1m', '3m', '5m', '15m', '30m']:
            # Para intervalos de minutos, mostrar hora:minuto
            dtick_config = dict(
                dtick=60000 * int(intervalo.replace('m', '')),  # tick cada intervalo
                tickformat='%H:%M'
            )
        elif intervalo == '1h':
            # Para intervalos de 1 hora
            dtick_config = dict(
                dtick=3600000,  # tick cada hora
                tickformat='%d/%m %H:%M'
            )
        else:
            # Para intervalos diarios
            dtick_config = dict(
                dtick=86400000,  # tick cada día
                tickformat='%d/%m/%Y'
            )
        
        # Configurar Layout
        fig.update_layout(
            title=f"Estrategia de Cruce de SMA ({sma_rapida} vs {sma_lenta}) en {ticker.upper()} ({intervalo})",
            xaxis_title="Fecha/Hora",
            yaxis_title="Precio",
            yaxis=dict(range=y_range, fixedrange=False),
            xaxis=dict(
                rangeslider_visible=False,
                type='date',
                **dtick_config
            ),
            height=650,
            hovermode="x unified",
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        # Mostrar el gráfico en Streamlit
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"📊 **Total de velas mostradas:** {len(df_resultados)} | **Señales de compra:** {(df_resultados['Señal'] == 1).sum()} | **Señales de venta:** {(df_resultados['Señal'] == -1).sum()}")
                        
        st.subheader("Datos de la Estrategia (Últimas 10 Filas)")
        cols_to_display = ['Open', 'High', 'Low', 'Close', 'SMA_R', 'SMA_L', 'Señal', 'Posicion', 'Capital']
        st.dataframe(df_resultados[cols_to_display].tail(10))

    except Exception as e:
        st.error(f"Ocurrió un error durante el backtest: {e}")
        st.exception(e)

else:
    st.info("Presiona **'🚀 Ejecutar Backtest'** para comenzar. Sugerencia: Usa rangos de fechas de meses para intervalos de minutos.")