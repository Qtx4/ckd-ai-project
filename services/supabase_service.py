from supabase import create_client
from dotenv import load_dotenv
import os
import datetime
import logging

# -----------------------------
# LOAD ENV
# -----------------------------
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# -----------------------------
# CLIENT INIT
# -----------------------------
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------------
# LOGGING SETUP
# -----------------------------
logging.basicConfig(level=logging.INFO)


def save_prediction(data):
    """
    Save patient prediction data to Supabase safely.
    Uses UPSERT to avoid duplicate entries based on email.
    """

    try:
        # -----------------------------
        # ADD TIMESTAMP (IMPORTANT)
        # -----------------------------
        data["created_at"] = datetime.datetime.now().isoformat()

        # -----------------------------
        # UPSERT (NO DUPLICATES)
        # -----------------------------
        response = supabase.table("patient_reports").upsert(
            data,
            on_conflict="email"
        ).execute()

        logging.info("Data saved successfully")

        return {
            "success": True,
            "data": response.data
        }

    except Exception as e:
        logging.error(f"Supabase Error: {str(e)}")

        return {
            "success": False,
            "error": str(e)
        }