from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

class Retriever:
    def __init__(self, embed_model, api_key, persist_directory, collection_name, rerank=False):
        self.embed_model=HuggingFaceEmbeddings(model_name= embed_model)
        self.rerank_model=CohereRerank(cohere_api_key=api_key)
        self.retriever=Chroma(persist_directory=persist_directory,
                              collection_name=collection_name,
                              embedding_function=self.embed_model)
        self.retriever_model=ContextualCompressionRetriever(
                base_compressor=self.rerank_model,
                base_retriever=self.retriever.as_retriever())
    def search(self, query, k: int):
        if k:
            results = self.retriever.similarity_search(query, k=k)
        else:
            results = self.retriever.similarity_search(query, k=3)
        return results
    def rerank(self, query):
        results=self.retriever_model.get_relevant_documents(query=query)
        return results
    
