import requests
import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from deep_translator import GoogleTranslator
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# Initialize the text generation model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"
model = GPTNeoForCausalLM.from_pretrained(model_name)  # Load with half precision
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Device settings (use GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Initialize translators
translator_to_english = GoogleTranslator(source='auto', target='en')
translator_to_turkish = GoogleTranslator(source='en', target='tr')

# Function to get the cryptocurrency price from Binance
def get_crypto_price(symbol):
    API_URL = f'https://api.binance.com/api/v3/ticker/price?symbol={symbol.upper()}USDT'
    response = requests.get(API_URL)
    
    if response.status_code == 200:
        try:
            data = response.json()
            price = float(data['price'])
            return f"{symbol.upper()}: ${price:.2f}"
        except (ValueError, KeyError, TypeError) as e:
            return f"An error occurred: {str(e)}"
    else:
        return "Unable to fetch data."

# Function for the /price command
async def price(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.args:
        crypto = context.args[0].upper()
        coin_data = get_crypto_price(crypto)
        await update.message.reply_text(f'Current price for {crypto}:\n{coin_data}')
    else:
        await update.message.reply_text('Please enter a cryptocurrency symbol. Example: /price BTC')

# Function for the /start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Welcome to the Crypto Bot! Use /price <cryptocurrency> to get the current price.')

# Function for the /help command
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = (
        "With this bot, you can get cryptocurrency prices.\n\n"
        "Usage:\n"
        "/start - Starts the bot.\n"
        "/price <cryptocurrency> - Shows the current price of the specified cryptocurrency. Example: /price BTC\n"
        "/help - Displays this help message."
    )
    await update.message.reply_text(help_text)

# Function to respond to any message
async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        user_message = update.message.text
        
        # Translate the user message to English
        translated_message = translator_to_english.translate(user_message)

        # Generate a response using GPT-Neo
        input_ids = tokenizer.encode(translated_message, return_tensors='pt').to(device)
        attention_mask = input_ids.ne(tokenizer.pad_token_id).to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids, 
                attention_mask=attention_mask,  # Use attention mask
                max_length=100, 
                num_return_sequences=1, 
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,  # Reduces repetitive responses
                top_p=0.9,  # Controls diversity via nucleus sampling
                temperature=0.7,  # Lowers the probability of repetitive or extreme outputs
                do_sample=True  # Enables sampling for diverse outputs
            )
        bot_response = tokenizer.decode(output[0], skip_special_tokens=True)

        # Translate the bot response back to Turkish
        translated_response = translator_to_turkish.translate(bot_response)

        await update.message.reply_text(translated_response)
    else:
        print("Error: update.message is None")

def main():
    print("Bot is starting...")
    app = ApplicationBuilder().token("Telegram_API_Key").build()  # Replace with your actual Telegram bot token.

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("price", price))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))

    print("Bot is running...")
    app.run_polling()

if __name__ == '__main__':
    main()

