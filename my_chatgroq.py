from langchain_openai import ChatOpenAI

class ChatGroq:
    def __init__(self):
        self.llm = ChatOpenAI(model='gpt-4')

    def generate_response(self, prompt):
        return self.llm.invoke(prompt)

# Other functions or classes that do not import this module 