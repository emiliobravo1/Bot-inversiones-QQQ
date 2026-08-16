import yfinance as yf
import pandas as pd
import requests
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # <--- Nueva herramienta NLP

load_dotenv()

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

RSI_BAJO = 30
RSI_ALTO = 70
SENTIMIENTO_NEGATIVO = -0.15
TIEMPO_ESPERA_HTTP = 15
CAPITAL_INICIAL_BACKTEST = 10000.0
COMISION_POR_OPERACION = 0.0005
SLIPPAGE_POR_OPERACION = 0.0005


def crear_sesion_http():
    """Crea una sesion HTTP con reintentos para reducir fallos temporales de red."""
    sesion = requests.Session()
    estrategia = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
    )
    adapter = HTTPAdapter(max_retries=estrategia)
    sesion.mount("https://", adapter)
    sesion.mount("http://", adapter)
    return sesion


def validar_configuracion():
    faltantes = []
    if not TOKEN:
        faltantes.append("TELEGRAM_TOKEN")
    if not CHAT_ID:
        faltantes.append("TELEGRAM_CHAT_ID")

    if faltantes:
        raise ValueError(
            "Faltan variables de entorno requeridas: " + ", ".join(faltantes)
        )


def descargar_historial(ticker="QQQ", period="1y"):
    etf = yf.Ticker(ticker)
    try:
        hist = etf.history(period=period)
    except Exception as exc:
        raise RuntimeError(f"Error al descargar datos historicos para {ticker}: {exc}") from exc

    if hist.empty:
        raise RuntimeError("Yahoo Finance devolvio datos vacios")

    return hist

def calcular_rsi(datos, ventana=14):
    delta = datos['Close'].diff()
    ganancias = delta.where(delta > 0, 0)
    perdidas = -delta.where(delta < 0, 0)
    media_ganancias = ganancias.ewm(alpha=1/ventana, adjust=False).mean()
    media_perdidas = perdidas.ewm(alpha=1/ventana, adjust=False).mean()
    
    rs = media_ganancias / media_perdidas
    rsi = 100 - (100 / (1 + rs))
    return rsi

def analizar_sentimiento_noticias(ticker="QQQ"):
    """Descarga titulares recientes y evalúa el nivel de pánico o euforia."""
    try:
        etf = yf.Ticker(ticker)
        noticias = etf.news # Descarga las noticias directamente de Yahoo Finance
    except Exception as exc:
        print(f"No se pudieron leer noticias para {ticker}: {exc}")
        return "Neutral", 0.0
    
    if not noticias:
        return "Neutral", 0.0
        
    analyzer = SentimentIntensityAnalyzer()
    puntaje_total = 0
    
    # Analizamos el título de cada noticia
    for noticia in noticias:
        titulo = noticia.get('title', '')
        # VADER entrega un 'compound' entre -1 (pánico total) y 1 (euforia total)
        puntaje = analyzer.polarity_scores(titulo)['compound']
        puntaje_total += puntaje
        
    promedio = puntaje_total / len(noticias)
    
    # Clasificamos el sentimiento general
    if promedio <= -0.15:
        estado_noticias = "Pesimista / Panico"
    elif promedio >= 0.15:
        estado_noticias = "Optimista / Euforia"
    else:
        estado_noticias = "Neutral"
        
    return estado_noticias, promedio

def enviar_mensaje_telegram(mensaje):
    print(f"Enviando mensaje a Chat ID: {CHAT_ID}")
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": mensaje}
    
    # Capturamos la respuesta de Telegram
    sesion = crear_sesion_http()
    respuesta = sesion.post(url, json=payload, timeout=TIEMPO_ESPERA_HTTP)
    
    # Imprimimos lo que Telegram nos dice
    print("Código de respuesta de Telegram:", respuesta.status_code)
    print("Detalle:", respuesta.text)
    
    # Forzamos un error si Telegram rechaza el mensaje
    respuesta.raise_for_status()


def max_drawdown(serie_equity):
    if not serie_equity:
        return 0.0

    pico = serie_equity[0]
    max_dd = 0.0
    for valor in serie_equity:
        if valor > pico:
            pico = valor
        dd = (valor / pico) - 1.0
        if dd < max_dd:
            max_dd = dd
    return max_dd


def backtest_rapido(ticker="QQQ", period="10y"):
    hist = descargar_historial(ticker=ticker, period=period)
    hist = hist.copy()
    hist["SMA_200"] = hist["Close"].rolling(window=200).mean()
    hist["RSI"] = calcular_rsi(hist)
    hist = hist.dropna(subset=["SMA_200", "RSI"]).copy()

    if hist.empty:
        raise RuntimeError("No hay suficientes datos para ejecutar backtest")

    capital = CAPITAL_INICIAL_BACKTEST
    acciones = 0.0
    en_posicion = False
    operaciones = 0
    operaciones_ganadoras = 0
    comisiones_pagadas = 0.0
    costo_slippage = 0.0
    equity = []
    retorno_diario = []
    equity_previo = None
    capital_entrada = 0.0

    for fila in hist.itertuples():
        precio = float(fila.Close)
        sma_200 = float(fila.SMA_200)
        rsi = float(fila.RSI)

        senal_compra = (rsi < RSI_BAJO) and (precio < sma_200)
        senal_salida = (rsi > 55) or (precio > sma_200)

        if (not en_posicion) and senal_compra:
            comision_compra = capital * COMISION_POR_OPERACION
            capital_neto = capital - comision_compra
            precio_entrada = precio * (1 + SLIPPAGE_POR_OPERACION)
            acciones = capital_neto / precio_entrada
            comisiones_pagadas += comision_compra
            costo_slippage += acciones * (precio_entrada - precio)
            capital_entrada = capital
            capital = 0.0
            en_posicion = True
            operaciones += 1
        elif en_posicion and senal_salida:
            precio_salida = precio * (1 - SLIPPAGE_POR_OPERACION)
            bruto_salida = acciones * precio_salida
            comision_venta = bruto_salida * COMISION_POR_OPERACION
            capital = bruto_salida - comision_venta
            comisiones_pagadas += comision_venta
            costo_slippage += acciones * (precio - precio_salida)
            if capital > capital_entrada:
                operaciones_ganadoras += 1
            acciones = 0.0
            en_posicion = False

        equity_actual = capital if not en_posicion else acciones * precio
        equity.append(equity_actual)

        if equity_previo and equity_previo > 0:
            retorno_diario.append((equity_actual / equity_previo) - 1.0)
        equity_previo = equity_actual

    if en_posicion:
        precio_final = float(hist["Close"].iloc[-1])
        precio_salida = precio_final * (1 - SLIPPAGE_POR_OPERACION)
        bruto_salida = acciones * precio_salida
        comision_venta = bruto_salida * COMISION_POR_OPERACION
        capital = bruto_salida - comision_venta
        comisiones_pagadas += comision_venta
        costo_slippage += acciones * (precio_final - precio_salida)
        if capital > capital_entrada:
            operaciones_ganadoras += 1
        acciones = 0.0
        en_posicion = False
        equity[-1] = capital

    valor_final = capital
    primer_precio = float(hist["Close"].iloc[0])
    ultimo_precio = float(hist["Close"].iloc[-1])
    buy_hold_final = CAPITAL_INICIAL_BACKTEST * (ultimo_precio / primer_precio)

    dias = len(hist)
    anos = dias / 252 if dias > 0 else 0
    cagr = ((valor_final / CAPITAL_INICIAL_BACKTEST) ** (1 / anos) - 1) if anos > 0 else 0.0

    if retorno_diario:
        media = pd.Series(retorno_diario).mean()
        std = pd.Series(retorno_diario).std()
        sharpe = (media / std) * (252 ** 0.5) if std and std > 0 else 0.0
    else:
        sharpe = 0.0

    dd_max = max_drawdown(equity)
    tasa_acierto = (operaciones_ganadoras / operaciones) if operaciones > 0 else 0.0

    return {
        "valor_final": valor_final,
        "buy_hold_final": buy_hold_final,
        "retorno_pct": (valor_final / CAPITAL_INICIAL_BACKTEST - 1.0) * 100,
        "buy_hold_retorno_pct": (buy_hold_final / CAPITAL_INICIAL_BACKTEST - 1.0) * 100,
        "cagr_pct": cagr * 100,
        "sharpe": sharpe,
        "max_drawdown_pct": dd_max * 100,
        "operaciones": operaciones,
        "tasa_acierto_pct": tasa_acierto * 100,
        "comisiones_pagadas": comisiones_pagadas,
        "costo_slippage": costo_slippage,
        "costo_total_friccion": comisiones_pagadas + costo_slippage,
    }


def resumen_backtest_texto(metricas):
    return (
        "Backtest rapido (10y):\n"
        f"- Supuestos: comision {COMISION_POR_OPERACION * 100:.2f}% y slippage {SLIPPAGE_POR_OPERACION * 100:.2f}% por operacion\n"
        f"- Estrategia: {metricas['retorno_pct']:.2f}%\n"
        f"- Buy and Hold: {metricas['buy_hold_retorno_pct']:.2f}%\n"
        f"- CAGR: {metricas['cagr_pct']:.2f}%\n"
        f"- Sharpe: {metricas['sharpe']:.2f}\n"
        f"- Max Drawdown: {metricas['max_drawdown_pct']:.2f}%\n"
        f"- Operaciones: {metricas['operaciones']}\n"
        f"- Tasa de acierto: {metricas['tasa_acierto_pct']:.2f}%\n"
        f"- Comisiones pagadas: ${metricas['comisiones_pagadas']:.2f}\n"
        f"- Costo por slippage: ${metricas['costo_slippage']:.2f}\n"
        f"- Costo total por friccion: ${metricas['costo_total_friccion']:.2f}"
    )

def analizar_etf(ticker="QQQ"):
    print(f"Evaluando {ticker} y leyendo noticias...")
    try:
        hist = descargar_historial(ticker=ticker, period="1y")
    except Exception as exc:
        print(exc)
        return

    if hist.empty:
        print("Error: No se pudieron descargar los datos de Yahoo Finance. Reintentar más tarde.")
        return
    
    hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
    hist['RSI'] = calcular_rsi(hist)
    
    precio_actual = hist['Close'].iloc[-1]
    sma_actual = hist['SMA_200'].iloc[-1]
    rsi_actual = hist['RSI'].iloc[-1]

    if pd.isna(sma_actual) or pd.isna(rsi_actual):
        print("Datos insuficientes para calcular SMA 200 o RSI.")
        return
    
    # --- NUEVA LÓGICA DE SENTIMIENTO ---
    sentimiento_texto, sentimiento_valor = analizar_sentimiento_noticias(ticker)
    
    # Ahora la oportunidad de compra es aún más fuerte si hay pánico en las noticias
    if rsi_actual < RSI_BAJO:
        # Entramos aquí solo con que el RSI sea bajo, garantizando que avise.
        if precio_actual < sma_actual and sentimiento_valor <= SENTIMIENTO_NEGATIVO:
            # Se alinearon los astros: RSI bajo, precio bajo la tendencia y pánico en noticias
            estado = "Oportunidad de Oro (Caida + Panico)"
        else:
            # Caída fuerte a corto plazo, pero no necesariamente bajo la SMA o con pánico
            estado = "Barato (Oportunidad de compra)"

    elif rsi_actual > RSI_ALTO:
        estado = "Inflado (Sobrecomprado)"
    else:
        estado = "Normal"
        
    mensaje_final = (
        f"Resumen de Inversion: {ticker}\n"
        f"Precio Actual: ${precio_actual:.2f}\n"
        f"SMA 200 días: ${sma_actual:.2f}\n"
        f"RSI (14 días): {rsi_actual:.2f}\n"
        f"Sentimiento en Noticias: {sentimiento_texto}\n\n"
        f"Estado del mercado: {estado}"
    )

    try:
        metricas = backtest_rapido(ticker=ticker, period="10y")
        resumen_bt = resumen_backtest_texto(metricas)
        mensaje_final = f"{mensaje_final}\n\n{resumen_bt}"
        print(resumen_bt)
    except Exception as exc:
        print(f"Backtest no disponible: {exc}")
    
    enviar_mensaje_telegram(mensaje_final)

if __name__ == "__main__":
    try:
        validar_configuracion()
        analizar_etf("QQQ")
    except Exception as exc:
        print(f"Error de ejecucion: {exc}")