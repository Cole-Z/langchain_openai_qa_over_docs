import os
import openai

openai.api_key = os.environ.get("OPENAI_API_KEY")

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI, VectorDBQA

# this just ensures the response being generated is completed. 
def truncate_to_last_sentence(text):
    end_punctuations = [".", "!", "?"]
    for i in range(len(text) - 1, -1, -1):
        if text[i] in end_punctuations:
            return text[:i+1]
    return text

# Refine answer function
def refine_answer(prompt, model_name="text-davinci-002", temperature=0.5, max_tokens=100):
    response = openai.Completion.create(
        engine=model_name,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
        stop=['Human:', 'AI:'],
        echo=False,
    )

    full_response = response.choices[0].text.strip()
    truncated_response = truncate_to_last_sentence(full_response)
    return truncated_response

# Ask question function
def ask_question(qa_instance, question, refine=False):
    initial_answer = qa_instance.run(question)
    if refine:
        prompt = f"You are a knowledgeable assistant. Please provide a more detailed and a refined answer to the question: {question}\n\nInitial answer: {initial_answer}\n\nRefined answer:"
        refined_answer = refine_answer(prompt)
        return refined_answer

    return initial_answer

# Read the contents of the 'kb.txt' file into the zlg_info variable with UTF-8 encoding
with open('kb.txt', 'r', encoding='utf-8') as file:
    kb_info = file.read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(kb_info)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(texts, embeddings)

qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=vectorstore)

# Bot loop
while True:
    user_question = input("Ask a question or type 'exit' to quit: ")
    if user_question.lower() == "exit":
        break

    refined_answer = ask_question(qa, user_question, refine=True)
    print("Bot:", refined_answer)
