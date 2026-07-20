import yfinance as yf
import pandas as pd
import requests
import os # <--- Nueva librería estándar de Python
from dotenv import load_dotenv # <--- La librería que acabamos de instalar

# Cargar las variables del archivo .env al sistema temporalmente
load_dotenv()

# Leer las credenciales de forma segura
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

def enviar_mensaje_telegram(mensaje):
    """Función que usa la API de Telegram para enviar texto."""
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": mensaje,
        "parse_mode": "Markdown" # Permite enviar texto en negrita (*)
    }
    respuesta = requests.post(url, json=payload)
    if respuesta.status_code == 200:
        print("Mensaje enviado a Telegram con éxito.")
    else:
        print(f"Error al enviar: {respuesta.text}")

def analizar_etf(ticker="QQQ"):
    print(f"Obteniendo datos de {ticker}...")
    etf = yf.Ticker(ticker)
    hist = etf.history(period="1y")
    
    hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
    hist['RSI'] = calcular_rsi(hist)
    
    precio_actual = hist['Close'].iloc[-1]
    sma_actual = hist['SMA_200'].iloc[-1]
    rsi_actual = hist['RSI'].iloc[-1]
    
    estado = "Normal 🟡"
    if rsi_actual < 30 and precio_actual < sma_actual:
        estado = "Barato (Oportunidad de compra) 🟢"
    elif rsi_actual > 70:
        estado = "Inflado (Sobrecomprado) 🔴"
        
    # Construimos el mensaje en texto formateado
    mensaje_final = (
        f"📊 *Resumen de Inversión: {ticker}*\n"
        f"Precio Actual: ${precio_actual:.2f}\n"
        f"SMA 200 días: ${sma_actual:.2f}\n"
        f"RSI (14 días): {rsi_actual:.2f}\n\n"
        f"💡 *Estado del mercado:* {estado}"
    )
    
    # Enviar el mensaje a Telegram
    enviar_mensaje_telegram(mensaje_final)

# Ejecutar la función
analizar_etf("QQQ")