import os 
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# # Set environment variables
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model='gpt-3.5-turbo')

prompt = ChatPromptTemplate(
    [
         ("system", "You are a helpful AI bot."),
         ("user", "{input}")
    ]
)

from langchain_core.output_parsers import StrOutputParser
output_parser = StrOutputParser()

chain = prompt | llm | output_parser
response = chain.invoke({"input":"can you tell me about Machine Learning?"})

print(response)
