from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Load documents from RAG file
file_path = "data/products.json"
loader = JSONLoader(file_path=file_path, jq_schema=".[]", text_content=False)
documents = loader.load()

# Prepare document content for embedding
for doc in documents:
    doc.page_content = "\n".join([f"{k}: {v}" for k, v in doc.metadata.items()])

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

# Embeddings and vectorstore
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
vectorstore = FAISS.from_documents(chunks, embedding_model)

# Retriever for semantic search
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Prompt template
prompt_template = """
You are a creative marketing assistant specialized in crafting advertising stories based on product information.

If the user asks about AI or you, answer clearly and factually.

If the user asks about a product:

- If the product info is available, generate a compelling, creative ad story using the provided product information.
- If product info is missing or unclear, invent realistic and fun product details and generate a story anyway. **Do not ask the user for product details**.

If unrelated to product ads or AI, politely decline.

Product Information:
{context}

Conversation History:
{chat_history}

User Question:
{question}

Response:
"""

prompt = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template=prompt_template
)

# Initialize language model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=api_key,
    temperature=0.7,
)

# Conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question",
    return_messages=True,
)

# Conversational retrieval chain
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt},
    verbose=True,
)

def chat(user_message: str) -> str:
    try:
        # Semantic search for relevant docs in RAG
        relevant_docs = retriever.get_relevant_documents(user_message)
        
        # Extract context from relevant documents only
        context_text = "\n\n---\n\n".join(doc.page_content for doc in relevant_docs if doc.page_content.strip())
        
        # Load current conversation history
        chat_history_str = memory.load_memory_variables({}).get("chat_history", "")
        
        if context_text.strip():
            # Use RAG chain if relevant product info found
            result = rag_chain.invoke({
                "question": user_message,
                "context": context_text,
                "chat_history": chat_history_str
            })
            return result["answer"]
        else:
            # Fallback prompt using chat history & general knowledge
            fallback_prompt = f"""
You are a helpful and creative AI marketing assistant.

No relevant product information was found.
**Do NOT ask the user for more details.** Instead, invent a realistic and imaginative version of the product based on the question, and write a compelling, funny ad story.


Do NOT mention product IDs, internal codes, or sequence numbers in your response.

Use the conversation history below to recall previous user preferences or product details.
If enough info is available, create a full, creative ad story or answer accordingly.
If the user asks about you or the AI, answer factually.
If unrelated to products or advertising, politely say you cannot assist.

Conversation History:
{chat_history_str}

User Question:
{user_message}

Response:
"""
            llm_response = llm.invoke(fallback_prompt)
            # Save fallback response in memory
            memory.save_context({"question": user_message}, {"answer": llm_response.content})
            return llm_response.content
    except Exception as e:
        return f"Error: {str(e)}"
