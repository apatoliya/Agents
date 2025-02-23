import os 
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# # Set environment variables
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

from typing_extensions import TypedDict
class State(TypedDict):
    graph_state:str

def first_node(state):
    return{'graph_state': state['graph_state'] + ' I am Playing!!!'}
def second_node(state):
    return{'graph_state': state['graph_state'] + ' Basketball!!!'}
def third_node(state):
    return{'graph_state': state['graph_state'] + ' Soccer!!!'}

import random
from typing import Literal
def play(state) -> Literal['second_node', 'third_node']:
    graph_state = state['graph_state']

    # Generate random number between 0 and 1
    random_choice = random.random()
    
    # Return second_node or third_node based on random choice
    if random_choice < 0.5:
        return 'second_node'
    else:
        return 'third_node'

from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END

builder = StateGraph(State)
builder.add_node('first_node', first_node)
builder.add_node('second_node', second_node)
builder.add_node('third_node', third_node)

builder.add_edge(START, 'first_node')
builder.add_conditional_edges('first_node', play)
builder.add_edge('second_node', END)
builder.add_edge('third_node', END)

#compile
graph = builder.compile()

#display(Image(graph.get_graph().draw_mermaid_png()))
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))

# # Print the graph structure using the correct attributes
# print("Graph structure:")
# for node in graph.nodes:
#     print(f"Node: {node}")

# # If you need a basic visualization, we can create a simple representation
# def create_graph_representation(builder):
#     mermaid = ["graph TD"]
#     # Get edges from the builder and handle them as tuples
#     edges = builder.edges
#     for source, target in edges:
#         mermaid.append(f"    {source} --> {target}")
#     return "\n".join(mermaid)

# # Print the diagram
# print("\nGraph Diagram:")
# print(create_graph_representation(builder))

# response = graph.invoke({'graph_state':'Hi! my name is John'})
# response['graph_state']

