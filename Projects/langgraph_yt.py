import os
from dotenv import load_dotenv
from pprint import pprint
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from gtts import gTTS
from langchain_core.messages import SystemMessage
from youtube_transcript_api import YouTubeTranscriptApi
from google.cloud import texttospeech

from langgraph.graph import StateGraph , START, END 
from typing_extensions import TypedDict
from IPython.display import Image, display




load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")

model = ChatOpenAI(model="gpt-4o")


def transcribe_video(video_url):
    """
    Transcribe the YouTube video from the given URL.
    """
    try:
        video_id = video_url.split("watch?v=")[1].split("&")[0]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([entry['text'] for entry in transcript])

        prompt = SystemMessage(content=f'Transcribe the video: {text}.')
        t = model.invoke([prompt])
        text = t.content
        return text

    except Exception as e:
        print(f"An error occurred: {e}")

def summarize_video(transcribe_text):
    """
    Summarize the transcribed text.
    """
    try:
        prompt = SystemMessage(content=f'Summarize the text: {transcribe_text}.')
        t = model.invoke([prompt])  # Invoke the model
        text = t.content
        return text
    except Exception as e:
        print(f"An error occurred: {e}")

def generate_audio(text):
    """
    Generate audio from the given text and save it as an MP3 file.
    """
    try:
        language_code = 'en'
        output = gTTS(text=text, lang=language_code, slow=False)
        output.save("YT_agent.mp3")
        print("Audio file generated successfully: YT_agent.mp3")
    except Exception as e:
        print(f"An error occurred during audio generation: {e}")

class MessagesState_URL(TypedDict):
    url: str

# Example usage
video_url = 'https://www.youtube.com/watch?v=-UkJDpH3I2U&'
transcribed_text = transcribe_video(video_url)
if transcribed_text:
    summarized_text = summarize_video(transcribed_text)
    if summarized_text:
        generate_audio(summarized_text)

# State graph setup
def assistant(state: MessagesState_URL) -> MessagesState_URL:
    # Invoke the model and generate audio
    transcribed_text = transcribe_video(state["url"])
    summarized_text = summarize_video(transcribed_text)
    generate_audio(summarized_text)
    return {"messages": [summarized_text]}  # Return the summarized text

builder = StateGraph(MessagesState_URL)
builder.add_node("assistant", assistant)
builder.add_edge(START, "assistant")
builder.add_edge("assistant", END)

react_graph = builder.compile()
display(Image(react_graph.get_graph().draw_mermaid_png()))

