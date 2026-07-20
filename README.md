# 📈 QQQ Market Tracker & Sentiment Bot

Un bot de inversiones automatizado desarrollado en Python que monitorea el rendimiento del ETF QQQ (Nasdaq 100). Combina análisis técnico y análisis de sentimiento de noticias financieras para enviar reportes diarios directamente a Telegram.

El sistema está diseñado para ejecutarse de forma autónoma en la nube utilizando GitHub Actions, facilitando la toma de decisiones de inversión durante las caídas del mercado.

## 🚀 Características Principales

* **Análisis Técnico Avanzado:** Calcula indicadores clave como el RSI (Relative Strength Index) y la SMA 200 (Media Móvil Simple de 200 días) para evaluar la tendencia a largo plazo.
* **Análisis de Sentimiento:** Utiliza `vaderSentiment` para procesar los titulares de noticias financieras y determinar si el sentimiento general del mercado es positivo, negativo o neutral.
* **Notificaciones en Tiempo Real:** Envía un resumen estructurado con los datos del mercado y el sentimiento diario directamente a un chat de Telegram.
* **Automatización en la Nube:** Configurado mediante GitHub Actions (`bot_diario.yml`) para ejecutarse diariamente de forma automática sin necesidad de servidores locales.

## 🛠️ Tecnologías Utilizadas

* **Lenguaje:** Python 3
* **Librerías principales:** `requests`, `pandas`, `vaderSentiment`, `python-dotenv`
* **Automatización:** GitHub Actions
* **Notificaciones:** Telegram Bot API

## ⚙️ Requisitos Previos

Si deseas clonar y ejecutar este bot en tu entorno local, necesitarás:

1. Python 3.8 o superior.
2. Un token de la API de Telegram (obtenido a través de [@BotFather](https://t.me/botfather)).
3. Tu ID de chat de Telegram (obtenido a través de bots como [@userinfobot](https://t.me/userinfobot)).

