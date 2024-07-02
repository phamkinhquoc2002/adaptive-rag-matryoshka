from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI

from graphs import AdaptiveRAG
from agents import AgentState
from retriever import Retriever

from pprint import pprint
import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY=os.environ["GOOGLE_API_KEY"]
llm=ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=GOOGLE_API_KEY)
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=3))
retriever = Retriever(embed_model="BAAI/bge-small-en-v1.5", persist_directory=".chromadb", collection_name="database", api_key=os.environ["COHERE_API_KEY"], rerank=True)
faa=AdaptiveRAG(llm, retriever, wikipedia)
initial_state = {"question":"Should I invest money in Forex if I make 200$ dollars a month?"}
for output in faa.graph.stream(initial_state):
    for k, v in output.items():
        pprint(f"Node '{k}':")
    pprint("\n-----\n")
pprint(v["response"])