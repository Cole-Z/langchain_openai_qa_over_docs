import os
import discord
from discord.ext import commands
import openai

openai.api_key = os.environ.get("OPENAI_API_KEY")

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI, VectorDBQA

def chat_gpt_response(conversation, model_name="gpt-3.5-turbo", max_response_tokens=250):
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=conversation,
        temperature=0.7,
        max_tokens=max_response_tokens,
    )
    return response['choices'][0]['message']['content'].strip()

def ask_question(conversation, question):
    conversation.append({"role": "user", "content": question})
    response = chat_gpt_response(conversation, model_name="gpt-3.5-turbo")
    conversation.append({"role": "assistant", "content": response})
    return response

with open('kb.txt', 'r', encoding='utf-8') as file:
    kb_info = file.read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(kb_info)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(texts, embeddings)

qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=vectorstore)

conversation = [
    {
        "role": "system",
        "content": "You are a helpful assistant with expertise in Linux and databases."
    }
]

max_conversation_length = 5

intents = discord.Intents.default()
intents.typing = False
intents.presences = False
intents.message_content = True
intents.members = True

bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"{bot.user} has connected to Discord!")

async def process_message(conversation, user_input):
    if len(conversation) > max_conversation_length * 2 - 1:  # Multiply by 2 to account for user and assistant messages
        conversation.pop(0)
        conversation.pop(0)

    response = ask_question(conversation, user_input)
    return response

@bot.command()
async def chat(ctx, *, user_input):
    response = await process_message(conversation, user_input)
    await ctx.send(response)

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if message.guild is None:
        response = await process_message(conversation, message.content)
        await message.channel.send(response)
    else:
        await bot.process_commands(message)

TOKEN = os.getenv("DISCORD_TOKEN")
bot.run(TOKEN)

