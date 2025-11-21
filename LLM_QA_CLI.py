import os
import sys
import string
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError

# Load environment variables from the .env file
load_dotenv() 

# ⚠️ Change the variable name to GEMINI_API_KEY
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 

# The Model ID
MODEL_NAME = "gemini-2.5-flash"

def preprocess_question(question):
    # Same preprocessing logic
    processed_q = question.lower()
    processed_q = processed_q.translate(str.maketrans('', '', string.punctuation))
    return processed_q

def query_llm_api(prompt):
    """
    Sends the processed prompt to the Google Gemini API.
    """
    if not GEMINI_API_KEY:
        return "Error: GEMINI_API_KEY environment variable not set."

    try:
        # Initialize the client with the API key
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        print("Waiting for Gemini response...")
        
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt],
        )

        # The generated text is directly available in the response object
        return response.text.strip()

    except APIError as e:
        return f"Gemini API Error: {e}"
    except Exception as e:
        return f"An unknown error occurred: {e}"


def main():
    """Main function to run the CLI application."""
    print("\n--- 🤖 LLM Q&A CLI Application (Part A) ---")
    print("Ask your question, or type 'exit' to quit.")
    
    while True:
        try:
            question = input("\n[❓ YOUR QUESTION]: ")
            
            if question.lower() == 'exit':
                print("Exiting application. Goodbye!")
                sys.exit(0)
            
            if not question.strip():
                continue
            
            processed_q = preprocess_question(question)
            
            print(f"[✅ PROCESSED QUESTION]: {processed_q}")
            
            answer = query_llm_api(processed_q)
            
            print(f"[💡 FINAL ANSWER]: {answer}")
            
        except KeyboardInterrupt:
            print("\nExiting application. Goodbye!")
            sys.exit(0)

if __name__ == "__main__":
    main()