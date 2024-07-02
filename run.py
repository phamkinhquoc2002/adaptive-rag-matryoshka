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

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--auto_database', action="store_true")
parser.add_argument('--persist_dir', type=str, default="")
parser.add_argument('--collection_name', type=str, default="")
parser.add_argument('--rerank', action="store_true")

parser.add_argument('--embed_model', type=str, default="")
parser.add_argument('--rerank_api', type=str, default="")
parser.add_argument('llm', choices=['local', 'gpt-3.5', 'gemini'])
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--query', type=str, default="")

args = parser.parse_args()

if __name__ == '__main__':
    if args.embed_model == "bge":
        retriever = Retriever(embed_model="BAAI/bge-small-en-v1.5", persist_directory=args.persist_directory, collection_name=args.collection_name, api_key=args.rerank_api, rerank=args.rerank)
    elif args.embed_model == "matryoshka":
        retriever = Retriever(embed_model="phamkinhquoc2002/bge-base-financial-matryoshka", persist_directory=args.persist_directory, collection_name=args.collection_name, api_key=args.rerank_api, rerank=args.rerank)
    
    if args.llm == "local":
        llm = ChatOllama()
    elif args.llm == "gpt-3.5":
        llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    elif args.llm == "gemini":
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=GOOGLE_API_KEY)
    
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=args.k))
    rag = AdaptiveRAG(llm, retriever,  wikipedia)
    query = {"question": args.query}
    for output in rag.graph.stream(query):
        for k, v in output.items():
            pprint(f"Node '{k}:'")
        pprint("\n-----------\n")
    pprint(v["response"])

    