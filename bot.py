import requests
import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from deep_translator import GoogleTranslator
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
import logging

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)

logger = logging.getLogger(__name__)

# Initialize the model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"
model = GPTNeoForCausalLM.from_pretrained(model_name)  # Load the model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set the pad token to the end-of-sequence token

# Device settings (use GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Initialize translators
translator_to_english = GoogleTranslator(source='auto', target='en')
translator_to_turkish = GoogleTranslator(source='en', target='tr')

# Function to get cryptocurrency price from Binance
def get_crypto_price(symbol):
    API_URL = 'https://api.binance.com/api/v3/ticker/price'
    params = {
        'symbol': symbol.upper() + 'USDT'  # Prepare the symbol for API request
    }
    response = requests.get(API_URL, params=params)
    
    if response.status_code == 200:
        try:
            data = response.json()
            price = float(data['price'])
            return f"{symbol.upper()}: ${price:.2f}"  # Return the price formatted
        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"Error parsing price data: {e}")
            return f"An error occurred: {str(e)}"
    else:
        logger.error(f"Failed to fetch data from Binance API. Status code: {response.status_code}")
        return "Unable to fetch data."

# Function for /price command
async def price(update: Update, context) -> None:
    if context.args:
        crypto = context.args[0].upper()  # Get the cryptocurrency symbol
        coin_data = get_crypto_price(crypto)  # Fetch price data
        await update.message.reply_text(f'Current price for {crypto}:\n{coin_data}')
    else:
        await update.message.reply_text('Please enter a cryptocurrency symbol. Example: /price BTC')

# Function for /start command
async def start(update: Update, context) -> None:
    await update.message.reply_text('Welcome to the Crypto Bot! Use /price <cryptocurrency> to get the current price.')

# Function for /help command
async def help_command(update: Update, context) -> None:
    help_text = (
        "With this bot, you can get cryptocurrency prices.\n\n"
        "Usage:\n"
        "/start - Starts the bot.\n"
        "/price <cryptocurrency> - Shows the current price of the specified cryptocurrency. Example: /price BTC\n"
        "/help - Displays this help message."
    )
    await update.message.reply_text(help_text)

# Function that responds to any message
async def chat(update: Update, context) -> None:
    if update.message and update.message.text:
        user_message = update.message.text  # Get user message

        # Translate user message to English
        translated_message = translator_to_english.translate(user_message)

        # Generate a response using GPT-Neo
        input_ids = tokenizer.encode(translated_message, return_tensors='pt').to(device)
        attention_mask = torch.ones_like(input_ids)  # Create an attention mask

        with torch.no_grad():
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=100,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
        bot_response = tokenizer.decode(output[0], skip_special_tokens=True)

        # Translate bot response back to Turkish
        translated_response = translator_to_turkish.translate(bot_response)

        await update.message.reply_text(translated_response)  # Send response back to the user

def main():
    logger.info("Bot is starting...")
    app = ApplicationBuilder().token("TELEGRAM_API_KEY").build()  # Replace with your Telegram bot API key

    app.add_handler(CommandHandler("start", start))  # Handle /start command
    app.add_handler(CommandHandler("price", price))  # Handle /price command
    app.add_handler(CommandHandler("help", help_command))  # Handle /help command
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))  # Handle text messages

    logger.info("Bot is running...")
    app.run_polling()  # Start polling for updates

if __name__ == '__main__':
    main()
