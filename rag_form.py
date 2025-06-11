import os
import json
import faiss
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=API_KEY)

# Models
model = genai.GenerativeModel("gemini-1.5-flash")
embedding_model = genai.GenerativeModel("embedding-001")

# Load product database
def load_rag_data():
    with open("data/products.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        print("✅ Products loaded:", [entry.get("Product", "Unnamed") for entry in data])
        return data

# Generate embedding using Gemini embedding model
def embed_text(text):
    try:
        resp = embedding_model.embed_content(text)
        return np.array(resp["embedding"], dtype=np.float32)
    except Exception as e:
        print(f"❌ Embedding error for '{text}':", e)
        return np.zeros((768,), dtype=np.float32)

# Product Search using FAISS with full semantic context
class ProductSearch:
    def __init__(self, rag_data):
        self.entries = []
        self.embeddings = []

        for item in rag_data:
            context_str = f"{item.get('Product', '')}. Category: {item.get('Category', '')}. Features: {item.get('Features', '')}. Benefits: {item.get('Benefits', '')}"
            emb = embed_text(context_str)
            self.embeddings.append(emb)
            self.entries.append({
                "Category": item.get("Category", ""),
                "Features": item.get("Features", ""),
                "Benefits": item.get("Benefits", "")
            })

        self.embeddings = np.vstack(self.embeddings)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

    def search(self, query, threshold=0.4):
        query_emb = embed_text(query).reshape(1, -1)
        distances, indices = self.index.search(query_emb, k=1)
        best_dist = distances[0][0]
        best_idx = indices[0][0]

        if best_dist < threshold:
            return self.entries[best_idx]
        return None

# Prompt when semantic match found
def construire_prompt_with_context(product, audience, tone, context):
    return (
        f"You are a highly creative and experienced advertising copywriter and storyteller.\n"
        f"Write a **full-length advertising story** (not limited in length) for the following product:\n"
        f"**Product Name**: {product}\n"
        f"**Target Audience**: {audience}\n\n"
        f"**Context:**\n"
        f"- Category: {context.get('Category', 'N/A')}\n"
        f"- Key Features: {context.get('Features', 'N/A')}\n"
        f"- Main Benefits: {context.get('Benefits', 'N/A')}\n\n"
        f"**Instructions:**\n"
        f"- Create an engaging, vivid, and emotional storytelling advertisement.\n"
        f"- Include a scenario or short narrative where a character experiences the product.\n"
        f"- Use sensory language to evoke feelings, visuals, and benefits.\n"
        f"- Make sure the tone is {tone.lower()} and compelling.\n"
        f"- Focus on how the product transforms or improves the life of the consumer.\n"
        f"- This should read like a script for a complete ad campaign or commercial.\n"
    )

# Fallback prompt with no context
def construire_prompt_simple(product, audience, tone):
    return (
        f"You are a highly creative and experienced advertising copywriter and storyteller.\n"
        f"Write a **full-length advertising story** (not limited in length) for the product named '{product}'.\n"
        f"Target Audience: {audience}\n"
        f"Tone: {tone}\n\n"
        f"Instructions:\n"
        f"- Create an engaging, vivid, and emotional storytelling advertisement.\n"
        f"- You can include a scenario or short narrative where a character experiences the product.\n"
        f"- Use sensory language to evoke feelings, visuals, and benefits.\n"
        f"- Focus on how the product transforms or improves the life of the consumer.\n"
        f"- The final result should be captivating, memorable, and persuasive.\n"
    )

# Main generation logic
def generer_story(product_input, audience_input, tone_input):
    rag_data = load_rag_data()
    searcher = ProductSearch(rag_data)
    matched_context = searcher.search(product_input)

    if matched_context:
        prompt = construire_prompt_with_context(
            product=product_input,
            audience=audience_input,
            tone=tone_input,
            context=matched_context
        )
    else:
        print(f"⚠️ No relevant product match found for: '{product_input}', using fallback prompt.")
        prompt = construire_prompt_simple(product_input, audience_input, tone_input)

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Error generating story: {str(e)}"
