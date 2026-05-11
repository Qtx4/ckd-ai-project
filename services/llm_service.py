import os
import re
from dotenv import load_dotenv

load_dotenv()

USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower()

# -----------------------------
# LLM SELECTOR
# -----------------------------

if USE_OLLAMA == "true":

    from langchain_community.llms import Ollama

    llm = Ollama(
        model="phi",
        temperature=0.2
    )

else:

    from langchain_groq import ChatGroq

    groq_key = os.getenv("GROQ_API_KEY")

    if not groq_key:
        raise ValueError("GROQ_API_KEY missing")

    llm = ChatGroq(
        groq_api_key=groq_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.2
    )

# -----------------------------
# ALLOWED MEDICAL KEYWORDS
# -----------------------------

MEDICAL_KEYWORDS = [

    "ckd",
    "kidney",
    "renal",
    "creatinine",
    "egfr",
    "dialysis",
    "nephrologist",
    "proteinuria",
    "albumin",
    "blood urea",
    "kidney failure",
    "transplant",
    "anemia",
    "urine",
    "hypertension",
    "serum creatinine",
    "gfr",
    "blood pressure",
    "diabetes",
    "swelling",
    "urination",
    "kidney stone",
    "electrolyte"
]

# -----------------------------
# LANGUAGE DETECTOR
# -----------------------------

def is_hindi_query(text):

    hindi_words = [
        "kya",
        "kaise",
        "kyu",
        "mujhe",
        "mera",
        "problem",
        "batao",
        "hai",
        "kidney",
        "dialysis"
    ]

    text = text.lower()

    return any(word in text for word in hindi_words)

# -----------------------------
# CKD FILTER
# -----------------------------

def is_medical_query(query):

    query = query.lower()

    # remove symbols
    query = re.sub(r'[^a-zA-Z0-9\s]', '', query)

    for keyword in MEDICAL_KEYWORDS:

        if keyword in query:
            return True

    return False

# -----------------------------
# MAIN FUNCTION
# -----------------------------

def get_llm_response(query, docs):

    try:

        # -----------------------------
        # STRICT FILTER
        # -----------------------------

        if not is_medical_query(query):

            return (
                "I am a CKD medical assistant and can only answer "
                "kidney-related medical questions."
            )

        # -----------------------------
        # LANGUAGE MODE
        # -----------------------------

        if is_hindi_query(query):

            language_instruction = """
- Reply in simple Hindi/Hinglish.
- Use professional and respectful language.
"""

        else:

            language_instruction = """
- Reply in clear professional English.
- Do NOT use Hindi.
"""

        # -----------------------------
        # CONTEXT
        # -----------------------------

        context = "\n".join(docs) if docs else "No medical context available."

        # -----------------------------
        # PROMPT
        # -----------------------------

        prompt = f"""

You are a highly professional AI medical assistant specialized ONLY in Chronic Kidney Disease (CKD).

STRICT RULES:
- ONLY answer kidney-related medical questions.
- Never answer unrelated topics.
- Never answer math, celebrities, coding, movies, cricket, politics, hacking, or jokes.
- Never use slang, memes, vulgar words, or casual language.
- Keep answers medically safe.
- Keep answers concise and patient-friendly.
- For severe conditions advise consulting a nephrologist.

{language_instruction}

MEDICAL CONTEXT:
{context}

USER QUESTION:
{query}

FINAL ANSWER:
"""

        # -----------------------------
        # LLM CALL
        # -----------------------------

        res = llm.invoke(prompt)

        # -----------------------------
        # OUTPUT
        # -----------------------------

        if hasattr(res, "content"):

            answer = res.content.strip()

        else:

            answer = str(res).strip()

        if not answer:

            return "Unable to generate response."

        return answer

    except Exception as e:

        return f"LLM Error: {str(e)}"