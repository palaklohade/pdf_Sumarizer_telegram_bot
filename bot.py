import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from llm_api import process_pdf_with_llm, answer_query_from_pdf
load_dotenv()
user_sessions = {}
TOKEN = os.environ.get('TELEGRAM_BOT')

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Welcome! Please upload a PDF file to start.")

    

async def handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file = await context.bot.get_file(update.message.document.file_id)
    file_path = f"downloads/{update.message.document.file_name}"
    os.makedirs('downloads', exist_ok=True)  
    await file.download_to_drive(file_path)
    
    file_size = os.path.getsize(file_path)
    print(f"[DEBUG] File saved to: {file_path}, size: {file_size} bytes")
    await update.message.reply_text("PDF received. Analyzing now...")


    db = process_pdf_with_llm(file_path)
    user_sessions[update.message.chat_id] = db 
    try:
        db = process_pdf_with_llm(file_path)
        if db is None:
            await update.message.reply_text("⚠️ Failed to analyze the PDF. Please check the document format.")
            return
        user_sessions[update.message.chat_id] = db 
        await update.message.reply_text("✅ Analysis done. Please ask your question about the document.")
    except Exception as e:
        print(f"[ERROR] LLM Processing Failed: {e}")
        await update.message.reply_text("❌ An error occurred while analyzing the PDF. Please try again.") 


async def handle_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.chat_id
    query = update.message.text

    if user_id not in user_sessions:
        await update.message.reply_text("Please upload a PDF first before asking questions.")
        return
    if query in ["no", "nah", "n"]:
        await update.message.reply_text("Thank you! I'll always be here if you need to ask any further query. 😊")
        return
    elif query in ["yes", "y", "yeah"]:
        await update.message.reply_text("Sure! Please type your next query.")
        return

    db = user_sessions[user_id]
    answer = answer_query_from_pdf(db, query)
    await update.message.reply_text(f"{answer}")

    await update.message.reply_text("Do you have any other question? (yes/no)")

def main():
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.Document.PDF, handle_pdf))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_query))

    print("Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
