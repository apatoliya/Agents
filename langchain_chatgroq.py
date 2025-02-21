import os 
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# # Set environment variables
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

#loading Groq
from langchain_groq import ChatGroq
llm = ChatGroq(model='gemma2-9b-it')


from langchain_core.messages import HumanMessage,AIMessage
llm.invoke([HumanMessage(content='Hi, my name is John  and I am AI engineer!!'),
            AIMessage(content='Hello John, Its Nice to meet you !!'),
            HumanMessage(content='what is my name and what do i do ?')])

#Chat Message History

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}
def get_session_history(session_id:str)->BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

with_message_history = RunnableWithMessageHistory(llm,get_session_history)


config = {
    "configurable": {"session_id":"John"}
}
response = with_message_history.invoke([HumanMessage(content='Hi, my name is John  and I am AI engineer!!')],config=config)
#print(response.content)

config = {
    "configurable": {"session_id":"John"}
}
response1 = with_message_history.invoke([HumanMessage(content='what is my name')],config=config)

get_session_history('John')



#prompt Template with message variable

from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",),
        MessagesPlaceholder(variable_name='messages')
    ]
)
chain = prompt |llm 
#chain.invoke({"messages":[HumanMessage(content='Hi My name is John!!')]})

with_message_history1 = RunnableWithMessageHistory(chain,get_session_history)
config2 = {'configurable': {'session_id':"chat5"}}

response2 = with_message_history.invoke([HumanMessage(content='what is my name')],config=config)
print(response2)


response3=chain.invoke({"messages":[HumanMessage(content="Hi My name is John")], "language": "French"})
response3.content

