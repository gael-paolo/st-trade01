import streamlit as st
import pandas as pd
import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt
from datetime import timedelta

# --- Funciones de Estrategia e Indicadores (Sin Cambios) ---

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
    df.loc[df['Cruce_Arriba'] == 1, 'Señal'] = 1  # Señal de Compra (Largo)
    df.loc[df['Cruce_Abajo'] == 1, 'Señal'] = -1 # Señal de Venta/Cierre
    
    return df

# --- Función de Backtesting (Corrección de Posición) ---

def ejecutar_backtest_sma(df, capital_inicial=10000, comision=0.0):
    """Simula las operaciones y calcula métricas de backtesting."""
    
    # Lógica de Posición: Mantiene la posición abierta (1) hasta la señal de cierre (-1 o 0)
    df['Position_Temp'] = df['Señal'].replace({1: 1, -1: 0})
    df['Position_Temp'].ffill(inplace=True)
    df['Posicion'] = df['Position_Temp'].shift(1).fillna(0)
    
    # Calcular retornos
    df['Retorno_Activo'] = df['Close'].pct_change()
    df['Retorno_Estrategia'] = df['Retorno_Activo'] * df['Posicion']
    
    # Simular capital
    df['Capital'] = capital_inicial * (1 + df['Retorno_Estrategia']).cumprod()
    
    # Métricas clave
    capital_final = df['Capital'].iloc[-1] if not df['Capital'].empty else capital_inicial
    retorno_total = (capital_final / capital_inicial) - 1
    
    # Retorno Anualizado (Estimación simple)
    dias_operados = (df.index[-1] - df.index[0]).days if not df.empty else 1
    retorno_anualizado = ((1 + retorno_total) ** (252 / dias_operados)) - 1 if dias_operados > 0 else 0
    
    # Máximo Drawdown
    df['Max_Capital'] = df['Capital'].cummax()
    df['Drawdown'] = (df['Max_Capital'] - df['Capital']) / df['Max_Capital']
    max_drawdown = df['Drawdown'].max() if not df.empty else 0
    
    # Ratio de Sharpe (Simplificado)
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

# --- Barra Lateral (Inputs) ---

st.sidebar.header("⚙️ Configuración de la Estrategia")

# 1. Entrada de Datos
st.sidebar.subheader("Datos")
ticker = st.sidebar.text_input("Símbolo de Ticker (Ej: NFLX)", "NFLX")
fecha_inicio = st.sidebar.date_input("Fecha de Inicio", pd.to_datetime("2024-01-01"))
fecha_fin = st.sidebar.date_input("Fecha de Fin", pd.to_datetime("2024-06-01"))
# Modificación 1: Agregar 1m y 3m al selector de intervalo
intervalo = st.sidebar.selectbox("Intervalo", ["1d", "1h", "30m", "15m", "5m", "2m", "1m"], index=4) # 5m por defecto

# 2. Parámetros de la Estrategia
st.sidebar.subheader("Parámetros de SMA")
sma_rapida = st.sidebar.slider("SMA Rápida (Períodos)", min_value=2, max_value=50, value=5)
sma_lenta = st.sidebar.slider("SMA Lenta (Períodos)", min_value=2, max_value=200, value=10)

# 3. Parámetros del Backtest
st.sidebar.subheader("Backtesting")
# Modificación 2: Permitir centavos en el Capital Inicial
capital_inicial = st.sidebar.number_input("Capital Inicial", min_value=0.01, value=10000.00, step=0.01, format="%.2f")
# comision = st.sidebar.number_input("Comisión por Operación (%)", min_value=0.0, value=0.0, step=0.01)

# Validar SMA
if sma_rapida >= sma_lenta:
    st.sidebar.error("La SMA Rápida debe ser menor que la SMA Lenta.")
    st.stop()

# --- Proceso Principal ---

if st.sidebar.button("🚀 Ejecutar Backtest"):
    try:
        # Descargar datos
        st.subheader(f"⬇️ Descargando Datos para **{ticker.upper()}**")
        df = yf.download(ticker, start=fecha_inicio, end=fecha_fin, interval=intervalo, multi_level_index=False)
        
        if df.empty:
            st.error(f"No se pudieron descargar datos para {ticker} en el rango y/o intervalo especificado.")
            st.stop()
            
        st.success(f"Datos descargados: {df.index.min().date()} a {df.index.max().date()}")

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
        # Modificación 4: Alerta/Mensaje de explicación para el Ratio de Sharpe
        st.info("""
            ℹ️ **Ratio de Sharpe (Explicación):**
            Mide el rendimiento de la inversión ajustado por el riesgo. Un número **mayor a 1.0** es generalmente bueno,
            indicando que la estrategia ofrece un exceso de rendimiento (por encima de la tasa libre de riesgo)
            por cada unidad de volatilidad (riesgo) que asume.
        """)
        
        st.markdown("---")
        
        # 2. Gráfico de Evolución del Capital (Modificación 5)
        st.subheader("Evolución del Capital y Máxima Pérdida (Drawdown)")
        
        # Preparar datos para la visualización de Drawdown
        capital_acumulado = df_resultados['Capital']
        drawdown_data = df_resultados['Drawdown']
        
        # Encontrar el inicio y fin del máximo drawdown para la anotación
        max_drawdown_value = drawdown_data.max()
        if max_drawdown_value > 0:
            end_drawdown_date = drawdown_data[drawdown_data == max_drawdown_value].index[-1]
            peak_date = df_resultados['Max_Capital'].idxmax()
        else:
            end_drawdown_date = capital_acumulado.index[-1]
            peak_date = capital_acumulado.index[0]

        # Crear figura de Matplotlib para el gráfico de Capital/Drawdown
        fig_cap, ax_cap = plt.subplots(figsize=(10, 5)) # Modificación 5: Gráfico más pequeño (figsize=(10, 5))

        ax_cap.plot(capital_acumulado.index, capital_acumulado, label="Capital Acumulado", color="blue")
        
        # Graficar el Drawdown (la diferencia entre Capital Máximo y Capital Actual)
        ax_cap.fill_between(
            x=capital_acumulado.index,
            y1=capital_acumulado,
            y2=df_resultados['Max_Capital'],
            where=(capital_acumulado < df_resultados['Max_Capital']),
            color="red",
            alpha=0.3,
            label="Drawdown"
        )
        
        # Añadir etiquetas y título
        ax_cap.set_title("Evolución del Capital", size=15)
        ax_cap.legend()
        ax_cap.grid(True)
        
        # Mostrar el gráfico en Streamlit
        st.pyplot(fig_cap)
        
        st.markdown("---")

        # 3. Gráfico de Velas con SMAs y Cruces (Modificación 3)
        st.subheader("Visualización de la Estrategia y Señales")
        
        # --- SOLUCIÓN AL ERROR: Asegurar que las series de PLOT tienen el mismo índice ---
        
        # 1. Crear series de plotting con NaN donde no hay señal
        # La señal de compra (1) usa el precio de cierre en ese punto. El resto es NaN.
        df_resultados['Señal_Compra_Plot'] = df_resultados.apply(
            lambda row: row['Close'] if row['Señal'] == 1 else None, axis=1
        )
        
        # La señal de venta (-1) usa el precio de cierre en ese punto. El resto es NaN.
        df_resultados['Señal_Venta_Plot'] = df_resultados.apply(
            lambda row: row['Close'] if row['Señal'] == -1 else None, axis=1
        )
        
        # 2. Definir los addplots (Ahora las series tienen la misma longitud garantizada)
        media_mov_plots = [
            # SMAs (tomadas directamente de las columnas del df_resultados limpio)
            mpf.make_addplot(df_resultados[f'SMA_R'], label=f"SMA {sma_rapida}", color="green", type="line"),
            mpf.make_addplot(df_resultados[f'SMA_L'], label=f"SMA {sma_lenta}", color="blue", type="line"),
            
            # Señales de Compra (usando la nueva columna con NaN)
            mpf.make_addplot(df_resultados['Señal_Compra_Plot'], type='scatter', markersize=150, marker='^', color='lime'),
            
            # Señales de Venta (usando la nueva columna con NaN)
            mpf.make_addplot(df_resultados['Señal_Venta_Plot'], type='scatter', markersize=150, marker='v', color='red'),
        ]

        # 3. Crear la figura de mplfinance
        fig, axlist = mpf.plot(
            df_resultados, 
            type="candle", 
            style="yahoo", 
            volume=False,
            addplot=media_mov_plots,
            title=f"Estrategia de Cruce de SMA ({sma_rapida} vs {sma_lenta}) en {ticker.upper()} ({intervalo})",
            returnfig=True
        )
        
        # Mostrar el gráfico en Streamlit
        st.pyplot(fig)
        
        st.subheader("Datos de la Estrategia (Últimas 10 Filas)")
        # Asegurarse de que las nuevas columnas creadas para el plot no molesten en la tabla de datos
        cols_to_display = ['Open', 'High', 'Low', 'Close', 'SMA_R', 'SMA_L', 'Señal', 'Posicion', 'Capital']
        st.dataframe(df_resultados[cols_to_display].tail(10))
    except Exception as e:
        st.error(f"Ocurrió un error durante el backtest: {e}")

else:
    st.info("Presiona **'🚀 Ejecutar Backtest'** para comenzar. Sugerencia: Usa rangos de fechas de meses para intervalos de minutos.")