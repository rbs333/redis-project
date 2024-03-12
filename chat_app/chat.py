from redisvl.query import VectorQuery
from redisvl.index import SearchIndex
from redisvl.utils.vectorize import HFTextVectorizer
from utilities.llm import openai_chat
from db.llm_memory import LLMMemoryLayer
import settings
import json

# check if this is in utils
hf = HFTextVectorizer(settings.VECTORIZER)


async def categorize_user_input(user_input):
    """calls openai and decodes the information type"""

    system_prompt = "You are a pre-processing step for a recipe recommendation tool"

    user_prompt = f"""
    The first objective is to determine if the following query is one of 2 potential categories: 1) recipe_rec task 2) other. Label category recipe_rec if the goal of the query is to generate a recipe. If it is of category "other" you will generate a response to the query provided and add to the resulting json response.

    The second objective is to capture any relevant info expressed within the query regardless of it's category that will help us understand the type of recipes a user likes. Relevant info could be "I don't love green beans" or any dietary restrictions such as gluten-free, vegetarian, paleo, etc and/or any allergies or phrases like "I hate <thing>".  Make sure to capture this information with a word that express sentiment such as "likes", "dislikes", "allergic to", etc. to best understand the user's relationship with the relevant info.

    Return the response as a json with the following attributes "category": "objective 1 category", "relevant_info": "list[string]", "answer": "Response to query". The category field is either the string "recipe_rec" or "other" from goal 1. The preference field is a list of preferences saved as strings from the query. The requirements field is a list of requirements saves as string from the query. A preference can not be a requirement and vice versa. The answer field is a generated response to the provided query if the category is "other" else leave this field as an empty string.
    
    Query: {user_input}
  """

    res = await openai_chat(system_prompt, user_prompt)

    return json.loads(res)


# this is when we want to respond to a message we first need to get the most relevant data to answer the question from the db
def retrieve_context(index, query):

    query_embedding = hf.embed(query)

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

    return f"""Use the provided context below to generate recipe(s) for the user.
    Utilize the recipe starters, user relevant info, and chat history to return a relevant response
    to the user's query. Primarily utilize the recipe starters as the base of your response but supplement
    as necessary.

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
    recipe_context = retrieve_context(vector_index, user_query)

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

    input_meta = await categorize_user_input(msg)

    if len(input_meta["relevant_info"]):
        memory.add_user_relevant_info(user_id, input_meta["relevant_info"])

    # if not recipe question (i.e. just statement) short-circuit and reduce num api calls
    if input_meta["answer"]:
        memory.add_user_chat_msg(user_id, f"bot:{input_meta["answer"]}")
        return input_meta["answer"]

    user_memory = memory.fetch_user(user_id)

    answer = await vector_question(
        msg,
        user_memory,
        vector_index,
    )

    memory.add_user_chat_msg(user_id, f"bot:{answer}")
    return answer
