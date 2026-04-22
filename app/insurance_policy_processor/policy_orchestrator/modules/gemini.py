from uuid import uuid4

from google import genai
from google.genai import types

from app.config import ENV_PROJECT

gemini_client = genai.Client(api_key=ENV_PROJECT.GEMINI_API_KEY)


async def cache_file_with_system_prompt(uploaded_file, model, system_prompt):
    cache = await gemini_client.aio.caches.create(
        model=model,
        config=types.CreateCachedContentConfig(
            display_name=str(uuid4()),
            system_instruction=system_prompt,
            contents=[uploaded_file],
            ttl=f"{600}s",
        ),
    )
    print(f"Cache created successfully: {cache.name}")

    return cache.name
