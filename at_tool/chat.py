from redisvl.query import VectorQuery
from redisvl.utils.vectorize import HFTextVectorizer
import settings
import openai

hf = HFTextVectorizer(settings.VECTORIZER)


class Chat:
    def __init__(self, index):
        self.index = index
        self.last_message: str = ""
        self.chat_messages: list[str] = []
        # might have other structures for holding data but none that I can think of right now

    def add_message(self, msg):
        self.chat_messages.append(msg)

    @staticmethod
    def embed_query(query):
        return hf.embed(query)

    # this is when we want to respond to a message we first need to get the most relevant data to answer the question from the db
    def retrieve_context(self, query):

        query_embedding = self.embed_query(query)

        vector_query = VectorQuery(
            vector=query_embedding,
            vector_field_name="text_embedding",
            num_results=3,
            return_fields=["raw_content", "proposition"],
            return_score=True,
        )

        # we would need reference to the index to query it that makes sense
        res = self.index.query(vector_query)

        return "\n".join(
            [r["proposition"] for r in res]
        )  # just take the first one for now.

    def promptify(self, query: str, context: str) -> str:
        """promptify takes the user's question and the context from the db and ask the generative AI to make an answer based in that world"""

        return f"""Use the provided context below derived from a Applachian trail guide pdf to answer the user's question.
      If you can't answer the user's question, based on the context; do not guess. If there is no context at all,
      respond with "I don't know".

      User's previous questions:

      {" ".join(self.chat_messages)}

      User question:

      {query}

      Helpful context:

      {context}

      Answer:
    """

    async def answer_question(self, query):
        context = self.retrieve_context(query)

        SYSTEM_PROMPT = "You are a tool assisting hikers find simple trail info."
        USER_PROMPT = self.promptify(query, context)

        response = await openai.AsyncClient().chat.completions.create(
            model=settings.CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT},
            ],
            temperature=0.1,
            seed=42,
        )

        # Response provided by LLM
        res = response.choices[0].message.content
        self.add_message(res)
        return res
