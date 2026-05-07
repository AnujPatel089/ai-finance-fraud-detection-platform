import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Load Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")


def generate_fraud_explanation(transaction_data):

    prompt = f"""
    You are an AI fraud analyst.

    Analyze this financial transaction.

    Transaction:
    {transaction_data}

    Explain:
    1. Fraud risk
    2. Suspicious indicators
    3. Why it may be fraud
    4. Recommended action

    Keep explanation simple and professional.
    """

    response = model.generate_content(prompt)

    return response.text