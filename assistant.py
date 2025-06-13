import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import os
import logging
from datetime import datetime
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Configurar logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class TelegramAssistant:
    def __init__(self, data_file='data.txt'):
        """Inicializar el asistente virtual para Telegram"""
        self.model = None
        self.data_file = data_file
        self.initialize_model()
        
    def initialize_model(self):
        """Inicializar y entrenar el modelo"""
        data, labels = self.load_data()
        self.train_model(data, labels)

    def load_data(self):
        """Cargar datos de entrenamiento con validación"""
        if not os.path.exists(self.data_file):
            return self.get_default_data()
        
        data = []
        labels = []
        
        try:
            with open(self.data_file, 'r', encoding='utf-8') as file:
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split(',')
                    if len(parts) >= 2:
                        command = parts[0].strip().lower()
                        category = parts[1].strip().lower()
                        if command and category:
                            data.append(command)
                            labels.append(category)
            
            if not data:
                return self.get_default_data()
                
            return data, labels
            
        except Exception as e:
            return self.get_default_data()

    def get_default_data(self):
        """Datos de entrenamiento por defecto"""
        data = [
            # Saludos (10 ejemplos)
            "hola", "buenos días", "buenas tardes", "hey", "saludos", 
            "buenas noches", "qué tal", "hola asistente", "buenos días asistente", "hi",
            
            # Despedidas (10 ejemplos)
            "adiós", "hasta luego", "nos vemos", "chau", "bye", 
            "salir", "terminar", "cerrar", "hasta pronto", "me voy",
            
            # Comandos del hogar (16 ejemplos)
            "encender luz", "enciende las luces", "prender luces", "luz on",
            "encender la luz", "enciende luz", "prender la luz", "iluminar",
            "apagar luz", "apaga las luces", "apagar luces", "luz off",
            "apagar la luz", "apaga luz", "apagar todas las luces", "oscurecer",
            
            # Consultas sobre estado (8 ejemplos)
            "cómo estás", "qué tal", "cómo te encuentras", "todo bien",
            "como estas", "que tal", "cómo va todo", "todo ok",
            
            # Agradecimientos (8 ejemplos)
            "gracias", "muchas gracias", "te agradezco", "thank you",
            "muy bien", "perfecto", "excelente", "genial",
            
            # Información (8 ejemplos)
            "qué hora es", "dime la hora", "hora actual", "qué día es",
            "ayuda", "help", "qué puedes hacer", "comandos disponibles",
            
            # Clima (6 ejemplos)
            "cómo está el clima", "qué tiempo hace", "va a llover",
            "temperatura", "clima hoy", "pronóstico del tiempo"
        ]
        
        labels = [
            # Saludos (10 etiquetas)
            "saludar", "saludar", "saludar", "saludar", "saludar",
            "saludar", "saludar", "saludar", "saludar", "saludar",
            
            # Despedidas (10 etiquetas)
            "despedir", "despedir", "despedir", "despedir", "despedir",
            "despedir", "despedir", "despedir", "despedir", "despedir",
            
            # Comandos del hogar (16 etiquetas)
            "hogar", "hogar", "hogar", "hogar", "hogar", "hogar", "hogar", "hogar",
            "hogar", "hogar", "hogar", "hogar", "hogar", "hogar", "hogar", "hogar",
            
            # Consultas sobre estado (8 etiquetas)
            "consulta", "consulta", "consulta", "consulta",
            "consulta", "consulta", "consulta", "consulta",
            
            # Agradecimientos (8 etiquetas)
            "agradecimiento", "agradecimiento", "agradecimiento", "agradecimiento",
            "agradecimiento", "agradecimiento", "agradecimiento", "agradecimiento",
            
            # Información (8 etiquetas)
            "informacion", "informacion", "informacion", "informacion",
            "informacion", "informacion", "informacion", "informacion",
            
            # Clima (6 etiquetas)
            "clima", "clima", "clima", "clima", "clima", "clima"
        ]
        
        return data, labels

    def train_model(self, data, labels):
        """Entrenar modelo con validación"""
        try:
            if not data or not labels or len(data) != len(labels) or len(set(labels)) < 2:
                return False
            
            self.model = make_pipeline(
                CountVectorizer(lowercase=True, stop_words=None, ngram_range=(1, 2)),
                MultinomialNB(alpha=1.0)
            )
            
            self.model.fit(data, labels)
            return True
            
        except Exception as e:
            logger.error(f"Error entrenando modelo: {e}")
            return False

    def predict_command(self, command):
        """Predecir categoría del comando"""
        if not self.model:
            return "desconocido"
        
        try:
            prediction = self.model.predict([command.lower()])[0]
            probabilities = self.model.predict_proba([command.lower()])[0]
            max_prob = max(probabilities)
            
            if max_prob < 0.3:
                return "desconocido"
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            return "desconocido"

    def generate_response(self, command, category):
        """Generar respuesta según la categoría predicha"""
        try:
            if category == 'hogar':
                return self.handle_home_commands(command)
            elif category == 'saludar':
                responses = [
                    "¡Hola! 🤖 Soy tu asistente virtual, ¿cómo puedo ayudarte?",
                    "¡Buenos días! ☀️ Estoy aquí para asistirte.",
                    "¡Hola! 👋 ¿En qué puedo ayudarte?",
                    "¡Saludos! 🙋‍♂️ ¿Qué necesitas?",
                    "¡Hola! 😊 ¿Cómo puedo ayudarte hoy?"
                ]
                return np.random.choice(responses)
            elif category == 'despedir':
                responses = [
                    "¡Adiós! 👋 Hasta luego.",
                    "¡Nos vemos! 😊 Que tengas un buen día.",
                    "¡Hasta pronto! 🙂 Fue un placer ayudarte.",
                    "¡Chau! 👋 Vuelve cuando necesites ayuda.",
                    "¡Hasta la vista! 😎 Que todo vaya bien."
                ]
                return np.random.choice(responses)
            elif category == 'consulta':
                responses = [
                    "Estoy muy bien, gracias por preguntar. 😊 ¿Y tú cómo estás?",
                    "Todo perfecto por aquí. ✅ ¿Cómo puedo ayudarte?",
                    "Muy bien, listo para ayudarte. 🚀 ¿Qué necesitas?",
                    "Excelente, funcionando al 100%. 💯 ¿En qué te puedo asistir?",
                    "¡Genial! 🎉 Aquí estoy para lo que necesites."
                ]
                return np.random.choice(responses)
            elif category == 'agradecimiento':
                responses = [
                    "¡De nada! 😊 Siempre es un placer ayudar.",
                    "¡No hay de qué! 👍 Para eso estoy aquí.",
                    "¡Con gusto! 🙂 ¿Necesitas algo más?",
                    "¡Un placer! 😄 Estoy aquí cuando me necesites.",
                    "¡Perfecto! 🎯 Me alegra poder ayudarte."
                ]
                return np.random.choice(responses)
            elif category == 'informacion':
                return self.handle_info_commands(command)
            elif category == 'clima':
                responses = [
                    "Lo siento, no tengo acceso a datos meteorológicos en tiempo real. 🌤️ Te recomiendo consultar una aplicación del clima.",
                    "Para información del clima actualizada, te sugiero revisar el pronóstico en tu aplicación favorita. 📱",
                    "No puedo acceder a datos del clima en este momento. ⛅ ¿Hay algo más en lo que pueda ayudarte?"
                ]
                return np.random.choice(responses)
            else:
                suggestions = [
                    "No entiendo esa orden. 🤔 Prueba con comandos como: 'hola', 'encender luces', 'ayuda', 'adiós'.",
                    "Comando no reconocido. ❓ Algunos ejemplos: 'cómo estás', 'apagar luz', 'qué puedes hacer'.",
                    "No comprendo. 😅 Intenta con: 'buenos días', 'encender tv', 'gracias', 'salir'."
                ]
                return np.random.choice(suggestions)
                
        except Exception as e:
            logger.error(f"Error generando respuesta: {e}")
            return "Disculpa, hubo un error procesando tu solicitud. 😔"

    def handle_home_commands(self, command):
        """Manejar comandos del hogar"""
        if any(word in command for word in ["encender", "enciende", "prender", "on"]):
            if any(word in command for word in ["luz", "luces", "iluminación"]):
                return "💡 Encendiendo las luces del hogar."
            elif any(word in command for word in ["tv", "televisión", "televisor"]):
                return "📺 Encendiendo la televisión."
            else:
                return "💡 Encendiendo dispositivo..."
                
        elif any(word in command for word in ["apagar", "apaga", "off"]):
            if any(word in command for word in ["luz", "luces", "iluminación"]):
                return "💡 Apagando las luces del hogar."
            elif any(word in command for word in ["tv", "televisión", "televisor"]):
                return "📺 Apagando la televisión."
            else:
                return "💡 Apagando dispositivo..."
        else:
            return "Comando de hogar no reconocido. 🏠 Intenta con 'encender/apagar luces' o 'encender/apagar tv'."

    def handle_info_commands(self, command):
        """Manejar comandos de información"""
        if any(word in command for word in ["hora", "tiempo"]):
            now = datetime.now()
            time_str = now.strftime("%H:%M:%S")
            date_str = now.strftime("%d/%m/%Y")
            return f"🕐 Son las {time_str} del {date_str}"
        elif any(word in command for word in ["ayuda", "help", "comandos", "qué puedes hacer"]):
            help_text = """📋 *Comandos disponibles:*

🙋‍♂️ *Saludos:* 'hola', 'buenos días', 'buenas tardes'
👋 *Despedidas:* 'adiós', 'hasta luego', 'salir'
🏠 *Hogar:* 'encender/apagar luces', 'encender/apagar tv'
💬 *Consultas:* 'cómo estás', 'qué tal'
ℹ️ *Información:* 'qué hora es', 'ayuda'
🙏 *Agradecimientos:* 'gracias', 'perfecto'

¡Prueba cualquiera de estos comandos! 🚀"""
            return help_text
        else:
            return "ℹ️ Información no disponible. Prueba con 'ayuda' para ver comandos disponibles."

# Instancia global del asistente
assistant = TelegramAssistant()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Comando /start"""
    welcome_msg = """🤖 *¡Bienvenido al Asistente Virtual!*
═══════════════════════════════════════

Soy tu asistente inteligente en Telegram. Puedes escribirme comandos y yo los procesaré para ayudarte.

💡 *Ejemplos:* 'hola', 'encender luces', 'cómo estás', 'ayuda', 'adiós'

¡Empecemos a chatear! 🚀

Escribe /help para ver todos los comandos disponibles."""
    
    await update.message.reply_text(welcome_msg, parse_mode='Markdown')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Comando /help"""
    help_text = """🆘 *Ayuda del Asistente Virtual*

📋 *Comandos disponibles:*

🙋‍♂️ *Saludos:* 'hola', 'buenos días', 'buenas tardes'
👋 *Despedidas:* 'adiós', 'hasta luego', 'salir'  
🏠 *Hogar:* 'encender/apagar luces', 'encender/apagar tv'
💬 *Consultas:* 'cómo estás', 'qué tal'
ℹ️ *Información:* 'qué hora es', 'ayuda'
🙏 *Agradecimientos:* 'gracias', 'perfecto'
🌤️ *Clima:* 'cómo está el clima', 'qué tiempo hace'

*Comandos especiales:*
/start - Mensaje de bienvenida
/help - Esta ayuda
/status - Estado del bot

¡Solo escribe tu mensaje y yo te entenderé! 😊"""
    
    await update.message.reply_text(help_text, parse_mode='Markdown')

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Comando /status"""
    status_msg = """🔧 *Estado del Sistema*

✅ Bot activo y funcionando
🤖 Modelo de IA: Entrenado y listo
📊 Sistema de predicción: Operativo
💾 Base de datos: Cargada

🎯 Listo para procesar tus comandos!"""
    
    await update.message.reply_text(status_msg, parse_mode='Markdown')

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Manejar mensajes de texto"""
    try:
        user_message = update.message.text
        user_name = update.effective_user.first_name
        
        logger.info(f"Mensaje de {user_name}: {user_message}")
        
        # Predecir categoría y generar respuesta
        category = assistant.predict_command(user_message)
        response = assistant.generate_response(user_message, category)
        
        # Enviar respuesta
        await update.message.reply_text(response)
        
    except Exception as e:
        logger.error(f"Error manejando mensaje: {e}")
        await update.message.reply_text("😔 Lo siento, hubo un error procesando tu mensaje. Intenta de nuevo.")

def main():
    """Función principal para ejecutar el bot"""
    # Token del bot de Telegram
    TOKEN = "8026091378:AAFhu6sdIXJ7UHI0xoF1mAdCpZXq_aSU0t0"
    
    # Crear aplicación
    application = Application.builder().token(TOKEN).build()
    
    # Agregar manejadores
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Iniciar bot
    print("🚀 Iniciando bot de Telegram...")
    print("✅ Bot activo. Presiona Ctrl+C para detener.")
    
    # Ejecutar bot
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()