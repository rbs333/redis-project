import settings
import openai


async def openai_chat(system_prompt: str, user_prompt: str, temp=0.1, seed=42):

    response = await openai.AsyncClient().chat.completions.create(
        model=settings.CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temp,
        seed=seed,
    )

    # return first response for now
    return response.choices[0].message.content
