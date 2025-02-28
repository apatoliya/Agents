from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessageGraph
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Define the state structure for the message graph
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Initialize the ChatOpenAI model with a specified temperature
model = ChatOpenAI(temperature=0)

def blog_generation_graph():
    """
    Create a state graph for blog generation workflow.
    """
    graph_workflow = StateGraph(State)

    def title_generation(state):
        """
        Generate a blog title based on the given topic.
        """
        prompt_title = SystemMessage(content='Please generate blog title based on given topic!')
        return {'messages': [model.invoke([prompt_title] + state['messages'])]}

    def content_generation(state):
        """
        Generate blog content based on the given topic.
        """
        prompt_content = SystemMessage(content='Please generate blog content based on given topic!')
        return {'messages': [model.invoke([prompt_content] + state['messages'])]}

    # Add nodes for title and content generation to the graph
    graph_workflow.add_node('Title', title_generation)
    graph_workflow.add_node('Blog', content_generation)

    # Define the edges of the graph workflow
    graph_workflow.add_edge(START, 'Title')
    graph_workflow.add_edge('Title', 'Blog')
    graph_workflow.add_edge('Blog', END)

    # Compile the agent from the graph workflow
    agent = graph_workflow.compile()
    return agent

# Create the blog generation agent
agent = blog_generation_graph()

 