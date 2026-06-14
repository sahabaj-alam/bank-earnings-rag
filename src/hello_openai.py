"""Smoke-test the OpenAI API. Run: python src/hello_openai.py"""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role":"system","content":"You are a concise assistant."},
        {"role":"user","content":"What is JPMorgan Chase? ANswer in one sentence"}
    ]
)
print(response.choices[0].message.content);