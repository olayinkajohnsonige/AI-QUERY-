from flask import Flask, render_template, request, jsonify
import string
import os
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError

# Load environment variables from the .env file
load_dotenv() 

app = Flask(__name__)

# ⚠️ Change the variable name to GEMINI_API_KEY
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
MODEL_NAME = "gemini-2.5-flash"

def preprocess_question(question):
    # Same preprocessing logic
    processed_q = question.lower()
    processed_q = processed_q.translate(str.maketrans('', '', string.punctuation))
    return processed_q

def query_llm_api(prompt):
    """
    Handles API request to Gemini.
    """
    if not GEMINI_API_KEY:
        return {"error": "Server Configuration Error: API Key not set."}

    try:
        # Initialize the client with the API key
        client = genai.Client(api_key=GEMINI_API_KEY)

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt],
        )
        
        # Check if the response contains text
        if response.text:
            return {"answer": response.text.strip()}
        else:
            return {"error": "Gemini returned an empty response."}

    except APIError as e:
        return {"error": f"Gemini API Error: {e}"}
    except Exception as e:
        return {"error": f"Request failed: {e}"}

# --- Routes remain the same ---

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask-ai", methods=["POST"])
def ask_ai():
    try:
        data = request.get_json()
        question = data.get("question")
        
        if not question:
            return jsonify({"reply": "No question provided.", "processed_q": ""}), 400

        processed_q = preprocess_question(question)
        llm_response = query_llm_api(processed_q)
        
        if "error" in llm_response:
            return jsonify({
                "processed_q": processed_q,
                "reply": f"LLM API Failed: {llm_response['error']}"
            }), 500
        else:
            return jsonify({
                "processed_q": processed_q,
                "reply": llm_response['answer']
            })

    except Exception as e:
        return jsonify({"reply": f"Server processing error: {e}", "processed_q": "Error"}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=os.environ.get('PORT', 5000))