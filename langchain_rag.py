import os 
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# # Set environment variables
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'


#webbased loader
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://python.langchain.com/docs/integrations/document_loaders/web_base/")
docs = loader.load()

#chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
documents=text_splitter.split_documents(docs)


#embedding
from langchain_openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings()

from langchain_community.vectorstores import FAISS
vectorstoredb=FAISS.from_documents(documents,embedding)

query = "Load multiple urls concurrently"
result = vectorstoredb.similarity_search(query)


#integrated with LLM 
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on prompt context:
    <context>
    {context}
    </context>
    """
)
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model='gpt-4o')

from langchain.chains.combine_documents import create_stuff_documents_chain
document_chain = create_stuff_documents_chain(llm,prompt)


#retriver # create vector store as interface for document chain (runnable binding)
retriever = vectorstoredb.as_retriever()
from langchain.chains import create_retrieval_chain
retrieval_chain = create_retrieval_chain(retriever,document_chain)

result = retrieval_chain.invoke({"input":"Load multiple urls concurrently"})
print(result['answer'])
