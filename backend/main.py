# main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import os
import json
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
from typing_extensions import TypedDict
from typing import List
from langchain_core.documents import Document
from pprint import pprint
from langgraph.graph import END, StateGraph
from websocket_manager import WebSocketManager

app = FastAPI()
# (Optional) Configure CORS if your frontend is on a different origin (port).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in development, allow all. Restrict in production.
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = WebSocketManager()



def get_secrets():
    with open('secrets.json') as secrets_file:
        return json.load(secrets_file)
    
secrets = get_secrets()
os.environ["LANGSMITH_API_KEY"]  = secrets.get("LANGSMITH_API_KEY")
os.environ["GOOGLE_API_KEY"] = secrets.get("GOOGLE_API_KEY")
os.environ["GOOGLE_CSE_ID"] = secrets.get("GOOGLE_CSE_ID")
os.environ["LANGSMITH_TRACING"] = "true"

persist_directory = "./chroma_langchain_db"
collection_name = "rag-chroma"
local_llm = "llama3.2"

### Retriever
oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")
vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=oembed,
    collection_name=collection_name
)
retriever = vectorstore.as_retriever(search_kwargs={'k': 4})  # will fetch top 4 relevant chunks


### Retrieval Grader
llm = ChatOllama(model=local_llm, format="json", temperature=0)

prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
    of a retrieved document to a user question. If the document contains keywords related to the user question, 
    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
     <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question", "document"],
)

retrieval_grader = prompt | llm | JsonOutputParser()

### Generate responses
prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
    Use only and EXCLUSIVELY the following pieces of retrieved context to answer the question. 
    If you don't know the answer or if the provided context is not enough to answer the question, just say that you don't know. 
    Keep the answer concise and coherent. <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question} 
    Context: {context} 
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "document"],
)

llm = ChatOllama(model=local_llm, temperature=0)
rag_chain = prompt | llm | StrOutputParser()


### Hallucination Grader

llm = ChatOllama(model=local_llm, format="json", temperature=0)

prompt = PromptTemplate(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether 
    an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate 
    whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
    single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here are the facts:
    \n ------- \n
    {documents} 
    \n ------- \n
    Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "documents"],
)

hallucination_grader = prompt | llm | JsonOutputParser()

### Answer Grader

llm = ChatOllama(model=local_llm, format="json", temperature=0)

prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an 
    answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
    useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
     <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
    \n ------- \n
    {generation} 
    \n ------- \n
    Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "question"],
)

answer_grader = prompt | llm | JsonOutputParser()

### Router

llm = ChatOllama(model=local_llm, format="json", temperature=0)

prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a 
    user question to a vectorstore or web search. Use the vectorstore for questions only and exclusively related to Curiosity or Perseverance missions to Mars.
    Otherwise, use web_search. 
    Note: Ignore any punctuation such as question marks or exclamation points when determining the appropriate datasource.
    You do not need to be stringent with the keywords in the question related to these topics. 
    Otherwise, use web-search. Give a binary choice 'web_search' or 'vectorstore' based on the question. 
    Return the a JSON with a single key 'datasource' and no premable or explanation. 
    
    Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question"],
)

question_router = prompt | llm | JsonOutputParser()

### Google Search Fallback

search = GoogleSearchAPIWrapper(k=3)
def search_results(query):
    return search.results(query, num_results=3)

web_search_tool = Tool(
    name="google_search",
    description="Search Google for recent results.",
    func=search_results
)


### State
MAX_ATTEMPTS = 2

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    
    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
        attempts: number of generation attempts
    """
    question: str
    generation: str
    web_search: str
    documents: List[str]
    attempts: int
    websocket: WebSocket

### Node Functions

def retrieve(state):
    print("---RETRIEVE---")
    websocket = state["websocket"]
    manager.send_message(websocket, "RETRIEVING")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question, "attempts": state.get("attempts", 0), "websocket": state["websocket"]}

def generate(state):
    print("---GENERATE---")
    websocket = state["websocket"]
    manager.send_message(websocket, "GENERATING")
    question = state["question"]
    documents = state["documents"]
    attempts = state.get("attempts", 0) + 1
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation, "attempts": attempts, "websocket": state["websocket"]}

def grade_documents(state):
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    websocket = state["websocket"]
    manager.send_message(websocket, "CHECK_DOCS_RELEVANCE_TO_QUESTION")
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    web_search = "No"
    
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score["score"]
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = "Yes"
            continue
    
    return {"documents": filtered_docs, "question": question, "web_search": web_search, "attempts": state.get("attempts", 0), "websocket": state["websocket"]}

def web_search(state):
    print("---WEB SEARCH---")
    websocket = state["websocket"]
    manager.send_message(websocket, "WEB_SEARCH")
    question = state["question"]
    documents = state.get("documents", [])
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["snippet"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    return {"documents": documents, "question": question, "attempts": state.get("attempts", 0), "websocket": state["websocket"]}

def route_question(state):
    print("---ROUTE QUESTION---")
    websocket = state["websocket"]
    manager.send_message(websocket, "ROUTING_QUESTION")
    question = state["question"]
    source = question_router.invoke({"question": question})
    print(source["datasource"])
    if source["datasource"] == "web_search":
        return "websearch"
    return "vectorstore"

def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")
    websocket = state["websocket"]
    manager.send_message(websocket, "ASSESS_GRADED_DOCUMENTS")
    if state["web_search"] == "Yes":
        print("---DECISION: INCLUDE WEB SEARCH---")
        return "websearch"
    print("---DECISION: GENERATE---")
    return "generate"

def grade_generation_v_documents_and_question(state):
    print("---CHECK HALLUCINATIONS---")
    websocket = state["websocket"]
    manager.send_message(websocket, "CHECK_HALLUCINATIONS")
    attempts = state.get("attempts", 0)
    if attempts > MAX_ATTEMPTS:
        return "stop"
    
    score = hallucination_grader.invoke({"documents": state["documents"], "generation": state["generation"]})
    grade = score["score"]
    
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED---")
        score = answer_grader.invoke({"question": state["question"], "generation": state["generation"]})
        if score["score"] == "yes":
            return "useful"
        return "not useful"
    return "not supported"

def stop(state):
    return {"generation": "I'm sorry, I was not able to produce an answer", "attempts": state.get("attempts", 0), "websocket": state["websocket"]}


### Assemble workflow

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("websearch", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate
workflow.add_node("stop", stop)  # stop

# Build graph
workflow.set_conditional_entry_point(
    route_question,
    {
        "websearch": "websearch",
        "vectorstore": "retrieve",
    },
)

workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
workflow.add_edge("websearch", "generate")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
        "stop": "stop"
    },
)

# Compile the workflow
agent = workflow.compile()

@app.get('/')
async def root():
    return {"message": "Hello World"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            inputs = {"question": data.question, "websocket": websocket, "attempts": 0}
            for event in agent.stream(inputs):
                for key, value in event.items():
                    pprint(f"Finished running: {key}:")
                    if key in ["generate", "stop"]:
                        pprint(value["generation"])
    except WebSocketDisconnect:
        manager.disconnect(websocket)





