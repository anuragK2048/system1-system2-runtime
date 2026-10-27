import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


client = OpenAI(
    base_url="https://inference.baseten.co/v1",
    api_key=os.getenv("BASETEN_API_KEY"),
)

response = client.chat.completions.create(
    model="meta-models/muse-spark-open-1026",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)