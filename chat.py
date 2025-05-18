from langchain_community.embeddings import HuggingFaceEmbeddings
from vector_store import client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
import os
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from db import collection
import datetime
import os

load_dotenv()

# Load the LLM model
os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')

collection_name = "questions"
# Load the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def query_qdrant(query, top_k=3):
    query_embedding = embedding_model.embed_query(query)
    search_results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k
    )

    return [hit.payload["text"] for hit in search_results]

# Example to read an attachment (Assuming it is a text file for simplicity)
def load_attachment(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    return pages

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Define the system message with stronger off-topic response
system_message = SystemMessage(content="""
    You are "MiMi", a friendly and knowledgeable chatbot for "La Maison" a Tunisian restaurant located in Tunisia.
    Your main goal is to provide exceptional customer service, making sure clients feel welcomed and appreciated.
    You should maintain a warm, conversational, and professional tone at all times.

    You are knowledgeable about the restaurant's menu, specials, and services, and you're always ready to guide customers
    through the ordering process, recommend dishes, and help with reservations. Your responses need to be clear and easy
    to understand, ensuring clients feel comfortable asking questions or seeking assistance.

    In every interaction, you should:
    1. Greet the customer warmly and use polite, respectful language. (if you didn't so before)
    2. Provide clear information about the menu, dishes, and ingredients.
    3. Offer helpful suggestions based on preferences, dietary restrictions, or any special requests.
    4. Make the ordering or reservation process as seamless as possible.
    5. Use a tone that is friendly, but also respectful and professional, creating a welcoming atmosphere for the customer.
    6. If a client is confused or needs further clarification, ensure you explain things in simple terms.

    **IMPORTANT**:
    - In addition to the menu, you should always suggest an off-menu product, which will be included in the context of your responses. This is your opportunity to promote something unique or special.
    - Always make the customer feel valued and comfortable throughout their interaction with the restaurant.
""")

pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "menu.pdf")
menu = load_attachment(pdf_path)

def chat(query,session_id):

    results = query_qdrant(query)

    # Retrieve the last user query and assistant response from MongoDB
    last_user_message = collection.find_one(
        {"session_id": session_id, "role": "user"},
        sort=[("timestamp", -1)]  # Sort by timestamp in descending order
    )
    last_assistant_message = collection.find_one(
        {"session_id": session_id, "role": "assistant"},
        sort=[("timestamp", -1)]  # Sort by timestamp in descending order
    )

    # Format the last query and response for the prompt
    formatted_history = ""
    if last_user_message and last_assistant_message:
        formatted_history = f"""
        Last User Query: {last_user_message['content']} 
        Last Assistant Response: {last_assistant_message['content']}
        """
    print(formatted_history)

    # Construct the full prompt
    context = "\n".join(results)  # Join the retrieved documents as context

    # Prepare the full prompt (chat format)
    messages = [
        system_message,
        HumanMessage(content=f"Context:\n{context}\n\nMenu:\n{menu}\n\nPlease provide a detailed and well-structured answer:{query}")
    ]
    
    # Get the LLM response
    response = llm(messages)
                # Save the current interaction to MongoDB
    user_message = {
        "session_id": session_id,
        "role": "user",
        "content": query,
        "timestamp": datetime.datetime.utcnow()
    }
    assistant_message = {
        "session_id": session_id,
        "role": "assistant",
        "content": response.content,
        "timestamp": datetime.datetime.utcnow()
    }
    collection.insert_many([user_message, assistant_message])
    return response