import os 
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# # Set environment variables
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'

# # Initialize ChatOpenAI with a valid model
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model='gpt-3.5-turbo')

#function to invoke llm
def invoke_llm(prompt):
    try:
        response = llm.invoke(prompt)
        return response
    except Exception as e:
        print('Error invoking LLM:',e)
        return None
def main():
    prompt = input('Enter your Prompt: ')
    response = invoke_llm(prompt)
    if response:
        print("LLM Response: ")
        print(response.content)
    else:
        print('No Response received from LLM.')
    
if __name__ == "__main__":
    main()    
