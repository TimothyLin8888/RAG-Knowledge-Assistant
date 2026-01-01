from openai import OpenAI
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key = openai_api_key)

async def generate_answer(query: str, context: str) -> str:
    """Generating an answer to the question the user says with the context from Wikipedia
    
    query: The question that the user asks to be found.
    context: The context in relation to the user's question to help.
    """

    prompt = f"Answer the following question based on the context:\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"

    response = client.responses.create(
        model="gpt-5-nano",
        input=prompt
    )

    output = response.output[1].content[0].text

    return output