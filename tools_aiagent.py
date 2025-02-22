import os 
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# # Set environment variables
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

#arxiv and wikipedia

from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper

# Example usage of WikipediaAPIWrapper
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

#RAG tools 
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader=WebBaseLoader('https://docs.smith.langchain.com/')
docs =loader.load()
documents = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(docs)
vectorstoredb=FAISS.from_documents(documents,OpenAIEmbeddings())
retriever=vectorstoredb.as_retriever()

from langchain.tools.retriever import create_retriever_tool
retriever_tool = create_retriever_tool(retriever,'Langsmith-Search-Internal',"Search any info regarding langsmith")

tools = [wiki,arxiv,retriever_tool]

from langchain_chatgroq import ChatGroq
llm =ChatGroq(model='Llama3-8b-8192')

## Prompt Template
from langchain import hub
prompt=hub.pull("hwchase17/openai-functions-agent")
prompt.messages

#Agents
from langchain.agents import create_openai_tools_agent
agent = create_openai_tools_agent(llm,tools,prompt)

#Agent Executor
from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent,tools=tools,verbose=True)
agent_executor.invoke({'input':'Tell me about Langsmith'}) #Invoke  Langsmith internal tool 
agent_executor.invoke({'input':'Tell me about AI'}) #Invoke wiki tool
agent_executor.invoke({'input':'1706.03762'}) # Invoke arxiv tool 