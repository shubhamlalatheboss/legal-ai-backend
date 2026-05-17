from fastapi import FastAPI, UploadFile, File
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
import pdfplumber
import os
import io
import json

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

@app.get("/")
def home():
    return {
        "message": "Legal AI Backend Running"
    }

@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):

    try:

        # Validate uploaded file type
        if file.content_type != "application/pdf":
            return {
                "success": False,
                "error": "Only PDF files are supported."
            }

        # Read uploaded file
        contents = await file.read()

        # Extract text from PDF
        extracted_text = ""

        with pdfplumber.open(io.BytesIO(contents)) as pdf:

            if len(pdf.pages) == 0:
                return {
                    "success": False,
                    "error": "Uploaded PDF has no pages."
                }

            for page in pdf.pages:

                text = page.extract_text()

                if text:
                    extracted_text += text + "\n"

        # Check extracted text
        if not extracted_text.strip():
            return {
                "success": False,
                "error": "No readable text found in PDF. Scanned PDFs are not supported yet."
            }

        # Limit text size
        extracted_text = extracted_text[:8000]

        print("===== EXTRACTED TEXT PREVIEW =====")
        print(extracted_text[:1000])

        # Send extracted text to Groq
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """
You are a legal AI assistant.

Analyze the uploaded legal document.

You MUST return ONLY valid JSON.

Use this exact structure:

{
  "summary": "string",
  "key_risks": [
    "risk 1",
    "risk 2"
  ],
  "important_obligations": [
    "obligation 1",
    "obligation 2"
  ],
  "important_clauses": [
    "clause 1",
    "clause 2"
  ],
  "termination_conditions": [
    "condition 1",
    "condition 2"
  ]
}

Do not include markdown.
Do not include explanations outside JSON.
"""
                },
                {
                    "role": "user",
                    "content": extracted_text
                }
            ],
            temperature=0.2
        )

        # Get AI response
        ai_response = response.choices[0].message.content

        print("===== RAW AI RESPONSE =====")
        print(ai_response)

        # Parse JSON response
        parsed_response = json.loads(ai_response)

        # Return structured response
        return {
            "success": True,
            "filename": file.filename,
            "analysis": parsed_response
        }

    except json.JSONDecodeError:

        return {
            "success": False,
            "error": "AI returned invalid JSON format."
        }

    except Exception as e:

        print("===== ERROR =====")
        print(str(e))

        return {
            "success": False,
            "error": str(e)
        }