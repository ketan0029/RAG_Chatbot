from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import faiss
import numpy as np
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from deep_translator import GoogleTranslator  


genai.configure(api_key="AIzaSyCL02k05rgXKMuAwaVvUL9iPb5XgdFwTwc")  

def load_diet_data(file_path="diet_data.txt"):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def create_vector_store(diet_text):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts([diet_text], embedding_model)
    vector_store.save_local("vector_store")
    return vector_store

if os.path.exists("vector_store/index.faiss"):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local("vector_store", embedding_model, allow_dangerous_deserialization=True)
else:
    diet_text = load_diet_data()
    vector_store = create_vector_store(diet_text)

def query_gemini(prompt):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

class GeminiRetrievalQA:
    def __init__(self, retriever):
        self.retriever = retriever

    def run(self, query):
        retrieved_docs = self.retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        prompt = f"{context}\n\nQuery: {query}"      
        response = query_gemini(prompt)
        if "This question cannot be answered" in response or "not related" in response.lower():
            return "I don't know."
        return response

qa_chain = GeminiRetrievalQA(retriever=vector_store.as_retriever())

app = Flask(__name__)
CORS(app)

def translate_text(text, src_lang, dest_lang):
    """Uses Deep Translator to translate text reliably."""
    try:
        return GoogleTranslator(source=src_lang, target=dest_lang).translate(text)
    except Exception as e:
        return f"Translation Error: {str(e)}"

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        user_input = data.get("message", "")
        selected_language = data.get("language", "en")

        if not user_input:
            return jsonify({"error": "No input message provided"})

        # Translate user query to English
        translated_input = translate_text(user_input, selected_language, "en")

        # Get chatbot response
        response = qa_chain.run(translated_input)

        # Translate response back to selected language
        translated_response = translate_text(response, "en", selected_language)

        return jsonify({"response": translated_response})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(port=5000, debug=True)
