import yfinance as yf
import pandas as pd
import requests
import os
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # <--- Nueva herramienta NLP

load_dotenv()

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def calcular_rsi(datos, ventana=14):
    delta = datos['Close'].diff()
    ganancias = delta.where(delta > 0, 0)
    perdidas = -delta.where(delta < 0, 0)
    media_ganancias = ganancias.rolling(window=ventana, min_periods=1).mean()
    media_perdidas = perdidas.rolling(window=ventana, min_periods=1).mean()
    rs = media_ganancias / media_perdidas
    rsi = 100 - (100 / (1 + rs))
    return rsi

def analizar_sentimiento_noticias(ticker="QQQ"):
    """Descarga titulares recientes y evalúa el nivel de pánico o euforia."""
    etf = yf.Ticker(ticker)
    noticias = etf.news # Descarga las noticias directamente de Yahoo Finance
    
    if not noticias:
        return "Neutral 😶", 0.0
        
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
        estado_noticias = "Pesimista / Pánico 😨"
    elif promedio >= 0.15:
        estado_noticias = "Optimista / Euforia 🚀"
    else:
        estado_noticias = "Neutral 😶"
        
    return estado_noticias, promedio

def enviar_mensaje_telegram(mensaje):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": mensaje, "parse_mode": "Markdown"}
    requests.post(url, json=payload)

def analizar_etf(ticker="QQQ"):
    print(f"Evaluando {ticker} y leyendo noticias...")
    etf = yf.Ticker(ticker)
    hist = etf.history(period="1y")
    
    hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
    hist['RSI'] = calcular_rsi(hist)
    
    precio_actual = hist['Close'].iloc[-1]
    sma_actual = hist['SMA_200'].iloc[-1]
    rsi_actual = hist['RSI'].iloc[-1]
    
    # --- NUEVA LÓGICA DE SENTIMIENTO ---
    sentimiento_texto, sentimiento_valor = analizar_sentimiento_noticias(ticker)
    
    estado = "Normal 🟡"
    # Ahora la oportunidad de compra es aún más fuerte si hay pánico en las noticias
    if rsi_actual < 30 and precio_actual < sma_actual:
        if sentimiento_valor <= -0.15:
            estado = "Oportunidad de Oro (Caída + Pánico) 🟢🟢"
        else:
            estado = "Barato (Oportunidad de compra) 🟢"
    elif rsi_actual > 70:
        estado = "Inflado (Sobrecomprado) 🔴"
        
    mensaje_final = (
        f"📊 *Resumen de Inversión: {ticker}*\n"
        f"Precio Actual: ${precio_actual:.2f}\n"
        f"SMA 200 días: ${sma_actual:.2f}\n"
        f"RSI (14 días): {rsi_actual:.2f}\n"
        f"📰 Sentimiento en Noticias: {sentimiento_texto}\n\n"
        f"💡 *Estado del mercado:* {estado}"
    )
    
    enviar_mensaje_telegram(mensaje_final)

if __name__ == "__main__":
    analizar_etf("QQQ")