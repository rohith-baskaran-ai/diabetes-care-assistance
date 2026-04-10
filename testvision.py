import os
import base64
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# put any image path here — screenshot, photo, anything
image_path = r"C:\Users\Rohith\OneDrive\AI Developer\diabetes-care-assistant\WhatsApp Image 2026-04-10 at 12.49.18.jpeg"

base64_image = encode_image(image_path)

response = client.chat.completions.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """You are a medical prescription reader.
Extract all medicines from this prescription.
Return JSON only:
{
  "medicines": [
    {
      "name": "medicine name",
      "dose": "dosage",
      "frequency": "how often",
      "duration": "how long",
      "instructions": "special instructions"
    }
  ],
  "doctor_name": "doctor name if visible",
  "date": "date if visible"
}"""
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ],
    max_tokens=500
)

print(response.choices[0].message.content)