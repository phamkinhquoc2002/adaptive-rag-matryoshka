from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, StateGraph
from langchain_core.output_parsers import JsonOutputParser
from agents import RouterQuery, GradeDocuments, AgentState
from agents import create_router_chain, create_grade_chain, create_query_rewriter, create_generation_chain, hallucation_grader
from langchain.schema import Document

class AdaptiveRAG():
    def __init__(self, llm, retriever, search_tool):
        self.llm=llm
        self.retriever=retriever
        self.search_tool=search_tool
        
        workflow=StateGraph(AgentState)
        workflow.add_node("generate", self.generate)
        workflow.add_node("retriever", self.context_retriever) 
        workflow.add_node("search_tool", self.web_search)
        workflow.add_node("grader", self.grade_documents)
        workflow.add_node("query_rewriter", self.grade_documents)
        
        workflow.set_conditional_entry_point(
            self.route_query,
            {
                "retriever":"retriever",
                "web_search":"search_tool"
            }
        )
        
        workflow.add_edge("retriever", "grader")
        workflow.add_edge("grader", "generate")
        workflow.add_edge("query_rewriter", "retriever")
        workflow.add_conditional_edges(
            "generate",
            self.hallucination_grader,
            {
                "useful":END,
                "not useful":"query_rewriter"
            }
        )
        
        workflow.add_edge("search_tool", "generate")
        
        self.graph=workflow.compile()
        
    #Agent Nodes
    def context_retriever(self, state: AgentState):
        question=state["question"]
        documents=[]
        docs=self.retriever.rerank(query=question)
        for doc in docs:
            documents.append(docs.page_content)
        return {"question":question, "documents":documents}

    def web_search(self, state: AgentState):
        question=state["question"]
        documents=[]
        docs=self.search_tool.run(question)
        for doc in docs:
            doc = Document(page_content=doc)
            documents.append(doc)
        return {"question":question, "documents":documents}

    def grade_documents(self, state: AgentState):
        grader=create_grade_chain(self.llm, GradeDocuments)
        question=state["question"]
        documents=state["documents"]
        filtered_docs=[]

        for doc in documents:
            relevancy=grader.invoke({"document":doc, "question":question})
            if relevancy == "Relevant":
                filtered_docs.append(doc)
                documents.remove(doc)
            elif relevancy == "Irrelevant":
                continue
        return {'questions':question, "documents":filtered_docs}

    def generate(self, state: AgentState):
        question=state["question"]
        filtered_docs=state["documents"]
        generation=create_generation_chain(self.llm)
        if len(filtered_docs) == 0:
            response="NO RELEVANT INFORMATION WAS FOUND!"
        response=generation.invoke({"context":filtered_docs, "question":question})
        return {"question":question, "documents":filtered_docs, "response":response}
    
    def rewriter(self, state: AgentState):
        question=state["question"]
        rewriter=create_query_rewriter(self.llm)
        rewritten_question=rewriter.invoke({"question":question})
        return {"question":rewritten_question}
    
    #Edges
    def hallucination_grader(self, state:AgentState):
        question=state["question"]
        documents=state["documents"]
        response=state["response"]
        hallucation_grader=hallucation_grader(self.llm)
        score=hallucation_grader.invoke({"question":question, "documents":documents, "response":response})
        if score.binary_score == "yes":
            return "useful"
        else:
            return "not useful"            

    def route_query(self, state: AgentState):
        router=create_router_chain(self.llm, RouterQuery)
        question=state["question"]
        response=router.invoke({"question":question})
        
        if response.datasource=="web_search":
            print("---ROUTE TO WEB_SEARCH")
            return "web_search"
        elif response.datasource=="vectorstore":
            print("--ROUTE TO VECTORSTORE")
            return "vectorstore"