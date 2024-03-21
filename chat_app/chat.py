from redisvl.query import VectorQuery
from redisvl.index import SearchIndex
from redisvl.utils.vectorize import HFTextVectorizer
from utilities.llm import openai_chat
from db.llm_memory import LLMMemoryLayer
import settings
import json

# check if this is in utils
hf = HFTextVectorizer(settings.VECTORIZER)


async def extract_user_info(user_input):
    """calls openai and decodes the information type"""

    system_prompt = "You are a pre-processing step for a recipe recommendation tool"

    user_prompt = f"""
    Your objective is to capture any relevant info expressed within the query that 
    will help us understand the type of recipes a user likes. Relevant info could be 
    "I don't love green beans" or any dietary restrictions such as gluten-free, vegetarian, 
    paleo, etc and/or any allergies or phrases like "I hate <thing>".  Make sure to 
    capture this information with a word that express sentiment such as "likes", 
    "dislikes", "allergic to", etc. to best understand the user's relationship with the 
    relevant info.

    Return the response as a json with the following attribute 
    "relevant_info": "list[string]". All relevant info should be returned as a simple
    list of strings. If there was no added relevant info return the attribute with an 
    empty list. Always return as a parsable JSON string object.
    
    Query: {user_input}
  """

    res = await openai_chat(system_prompt, user_prompt)

    return json.loads(res)


# this is when we want to respond to a message we first need to get the most relevant data to answer the question from the db
def retrieve_context(index, query, relevant_info):
    """When we hit the vector store we want matches not only on the query but also the relevant info"""

    query_embedding = hf.embed(
        f"{query}. Relevant user info: {','.join(relevant_info)}"
    )

    vector_query = VectorQuery(
        vector=query_embedding,
        vector_field_name="text_embedding",
        num_results=3,
        return_fields=["raw_content", "proposition"],
        return_score=True,
    )

    # we would need reference to the index to query it that makes sense
    res = index.query(vector_query)

    return "\n".join(
        [r["proposition"] for r in res]
    )  # just take the first one for now.


def gen_final_prompt(query, vector_context, chat_context, relevant_info) -> str:
    """put together the user's question and the context from the db and ask the generative AI to make an answer based in that world"""

    return f"""Use the provided context below to generate recipe(s) for the user
    and respond to questions as needed. Make use of the recipe starters, user relevant info, 
    and chat history to return a relevant response to the user's query. When generating a recipe 
    response, use the recipe starters as the base of your response but supplement as necessary.

    Recent chat history:

    {" ".join(chat_context)}

    User relevant info:

    {" ".join(relevant_info)}

    User query:

    {query}

    Recipe starters:

    {vector_context}

    Answer:
"""


async def vector_question(
    user_query: str,
    user_memory: dict,
    vector_index: SearchIndex,
) -> str:
    # make sure to embed relevant info as well as the query itself
    recipe_context = retrieve_context(
        vector_index, user_query, user_memory["relevantInfo"]
    )

    system_prompt = "You are a tool helping people pick recipes."
    final_prompt = gen_final_prompt(
        user_query,
        recipe_context,
        user_memory["recentChatHistory"],
        user_memory["relevantInfo"],
    )

    # does this need to be async?
    response = await openai_chat(system_prompt, final_prompt)

    # Response provided by LLM
    return response


async def gen_answer(memory: LLMMemoryLayer, user_id, msg, vector_index):
    memory.add_user_chat_msg(user_id, msg)

    input_meta = await extract_user_info(msg)

    if len(input_meta["relevant_info"]):
        memory.add_user_relevant_info(user_id, input_meta["relevant_info"])

    # if not recipe question (i.e. just statement) short-circuit and reduce num api calls
    # this was the problematic part of the code
    # if input_meta["answer"]:
    #     memory.add_user_chat_msg(user_id, f"bot:{input_meta['answer']}")
    #     return input_meta["answer"]

    user_memory = memory.fetch_user(user_id)

    answer = await vector_question(
        msg,
        user_memory,
        vector_index,
    )

    memory.add_user_chat_msg(user_id, f"bot:{answer}")
    return answer
