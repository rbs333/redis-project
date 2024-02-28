from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
import openai
import settings
import json
from redisvl.utils.vectorize import HFTextVectorizer

from redisvl.redis.utils import array_to_buffer


class Context:
    def __init__(self, docs):
        self.docs: list[str] = docs
        self.chunks: list[str] = self.get_chunks()
        self.metadata: list[str] = self.create_metadata()
        self.propositions: list[str] = self.create_propositions()
        self.embeddings = self.create_embeddings()

        if not (len(self.chunks) == len(self.propositions) == len(self.embeddings)):
            raise ValueError("Chunk, embedding, and proposition lengths do not match")

    def get_chunks(self):
        """Load and split data from docs into chunks"""

        """
        Note: right now for simplicity all chunks of documents go into one list 
        but this could be extended to different data structure. For example, could store 
        in dict where doc_name is the key to chunks etc
        """

        chunks = []

        for doc in self.docs:
            chunks.extend(self.load_and_split(doc))

        return chunks

    def create_metadata(self):
        """TODO: store metadata of chunks page number etc."""

    def create_propositions(self):
        """this takes the chunks and makes them better for later use"""
        return [self.create_proposition(chunk) for chunk in self.chunks]

    @staticmethod
    def create_proposition(chunk):
        """this takes the chunks and makes them better for later use"""

        SYSTEM_PROMPT = """
          You are a trail guide tool. Segments of a raw pdf containing tables, information text, and maps of the Appalachian Trail.

          Create a clear proposition from the data which includes all potentially important information on the location of shelters, mile markers, etc.

          Return the proposition as a single string with key proposition in a json like so: {"proposition": "single string"}
        """

        response = openai.OpenAI().chat.completions.create(
            model=settings.CHAT_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Decompose this raw content using the rules above:\n {chunk} ",
                },
            ],
        )

        res = response.choices[0].message.content
        return json.loads(res)["proposition"]

    def create_embeddings(self):
        """this converts the propositions into vectors"""

        hf = HFTextVectorizer(settings.VECTORIZER)

        return hf.embed_many([proposition for proposition in self.propositions])

    def get_redis_data(self):
        """this takes the data and makes it ready for inserting into DB"""

        return [
            {
                "chunk_id": f"{i}",
                "raw_content": chunk.page_content,
                "proposition": self.propositions[i],
                # convert embeddings to bytes for hash
                "text_embedding": array_to_buffer(self.embeddings[i]),
            }
            for i, chunk in enumerate(self.chunks)
        ]

    @staticmethod
    def load_and_split(
        doc, chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
    ):
        loader = UnstructuredFileLoader(doc, mode="single", strategy="fast")

        # providing some chunk_overlap for help with summarization
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        return loader.load_and_split(text_splitter)
