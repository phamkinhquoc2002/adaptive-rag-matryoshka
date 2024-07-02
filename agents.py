import torch
import operator
from langchain import hub
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage
from langchain.output_parsers import PydanticOutputParser, StructuredOutputParser
from typing import List, TypedDict, Literal, Optional
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langchain_core.prompts import ChatPromptTemplate

   
#MULTI-AGENT DEFINE:

#RouterAgent

class RouterQuery(BaseModel):
    
    """ Route a user query to the most relevant data source"""
    
    datasource: Literal["vectorstore", "web_search"] = Field(description="Given user query, choose to route it to vectorstore or web_search")

def create_router_chain(llm, router: RouterQuery):
    prompt=ChatPromptTemplate.from_messages([
        ("system", """You are an agentic AI expert at classifying user query into a datasource to answer it: vectorstore or web_search.
         Use vectorstore to answer questions, otherwise use web_search. Know that the vectorstore contains financial knowledge"""),
        ("human", "Question: {question}")
    ])
    router_llm= prompt | llm.with_structured_output(router)
    return router_llm
    
#GraderAgent

class GradeDocuments(BaseModel):
    
    """Grade the relevancy of the retrieved information with user query"""
    
    relevancy: Literal["Relevant", "Irrelevant"] = Field(description="Grade the relevancy score of the documents with the query")
    
def create_grade_chain(llm, grade: GradeDocuments):
    prompt=ChatPromptTemplate.from_messages([
        ("system", """You are an agentic AI expert at deciding of a retrieved document is relevant to the user question or not. If it's relevant, return 'Relevant', if not, return 'Irrelevant'"""),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")
    ])
    grade_llm=prompt | llm.with_structured_output(grade)
    return grade_llm


### Hallucination Grader

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

def hallucation_grader(llm):
    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
     
    hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)
    
    hallucination_grader = hallucination_prompt | llm.with_structured_input(GradeHallucinations)
    return hallucation_grader

#RewriteQueryAgent 
    
def create_query_rewriter(llm):
    system = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
    re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ])

    rewritten_query = re_write_prompt | llm | StructuredOutputParser()
    return rewritten_query
    
#GraphState

class AgentState(TypedDict):
    
    """The Graph State of Agentic Graph
    
    args:
    
    questions:  messages
    documents: retrieved information
    answer: answer of the llm
    """
    
    question=str
    documents=Optional[List[str]]
    response=Optional[str]
    
#Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
def create_generation_chain(llm):
    prompt=hub.pull("rlm/rag-prompt")
    chain=prompt | llm | StructuredOutputParser()
    return chain
