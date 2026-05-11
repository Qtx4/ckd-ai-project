import os
from dotenv import load_dotenv

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")

print("API KEY FOUND:", groq_key[:15] if groq_key else "NO KEY")

MODEL_NAME = "llama-3.1-8b-instant"

print("MODEL USED:", MODEL_NAME)

from langchain_groq import ChatGroq

llm = ChatGroq(
    groq_api_key=groq_key,
    model_name=MODEL_NAME,
    temperature=0.3
)

response = llm.invoke("What is CKD?")

print("\nLLM RESPONSE:\n")

if hasattr(response, "content"):
    print(response.content)
else:
    print(response)