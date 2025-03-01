import os
from dotenv import load_dotenv
from pprint import pprint
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from gtts import gTTS
from langchain_core.messages import SystemMessage, HumanMessage
from youtube_transcript_api import YouTubeTranscriptApi
from google.cloud import texttospeech

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Dict, Any
from typing_extensions import TypedDict
from IPython.display import Image, display

import operator

# Load environment variables
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
#os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")



# Initialize the model
model = ChatOpenAI(model="gpt-4o")

# Define the state structure
class AgentState(TypedDict):
    url: str
    transcript: str
    summary: str
    audio_path: str
    messages: List[Dict[str, Any]]

def extract_video_id(url: str) -> str:
    """Extract the YouTube video ID from a URL."""
    if "watch?v=" in url:
        return url.split("watch?v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    else:
        raise ValueError("Invalid YouTube URL format")

def transcribe_video(state: AgentState) -> AgentState:
    """Transcribe the YouTube video from the given URL."""
    try:
        video_id = extract_video_id(state["url"])
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([entry['text'] for entry in transcript])
        
        # Update state with transcript
        state["transcript"] = text
        state["messages"].append(HumanMessage(content=f"I've transcribed the video with ID {video_id}"))
        
        return state
    except Exception as e:
        state["messages"].append(HumanMessage(content=f"Error transcribing video: {str(e)}"))
        return state

def summarize_video(state: AgentState) -> AgentState:
    """Summarize the transcribed text."""
    try:
        if not state.get("transcript"):
            state["messages"].append(HumanMessage(content="No transcript available to summarize."))
            return state
            
        prompt = SystemMessage(content=f'Summarize the following video transcript concisely: {state["transcript"]}')
        response = model.invoke([prompt])
        summary = response.content
        
        # Update state with summary
        state["summary"] = summary
        state["messages"].append(HumanMessage(content="I've created a summary of the video."))
        
        return state
    except Exception as e:
        state["messages"].append(HumanMessage(content=f"Error summarizing video: {str(e)}"))
        return state

def generate_audio(state: AgentState) -> AgentState:
    """Generate audio from the summary."""
    try:
        if not state.get("summary"):
            state["messages"].append(HumanMessage(content="No summary available to convert to audio."))
            return state
            
        # Use gTTS for audio generation
        language_code = 'en'
        output_filename = "YT_summary.mp3"
        tts = gTTS(text=state["summary"], lang=language_code, slow=False)
        tts.save(output_filename)
        
        # Update state with audio path
        state["audio_path"] = output_filename
        state["messages"].append(HumanMessage(content=f"I've generated an audio file at {output_filename}"))
        
        return state
    except Exception as e:
        state["messages"].append(HumanMessage(content=f"Error generating audio: {str(e)}"))
        return state

def should_end(state: AgentState) -> str:
    """Determine if the workflow should end."""
    if state.get("audio_path"):
        return "end"
    else:
        return "continue"

def create_agent_workflow():
    """Create and return the agent workflow graph."""
    # Initialize the workflow
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("transcribe", transcribe_video)
    workflow.add_node("summarize", summarize_video)
    workflow.add_node("generate_audio", generate_audio)
    
    # Add edges
    workflow.add_edge(START, "transcribe")
    workflow.add_edge("transcribe", "summarize")
    workflow.add_edge("summarize", "generate_audio")
    workflow.add_conditional_edges(
        "generate_audio",
        should_end,
        {
            "end": END,
            "continue": "transcribe"  # Loop back if needed
        }
    )
    
    # Compile the workflow
    graph = workflow.compile()
    display(Image(graph.get_graph().draw_mermaid_png()))
    return graph


def process_youtube_video(url: str) -> Dict[str, Any]:
    """Process a YouTube video through the entire workflow."""
    # Initialize the agent state
    initial_state = AgentState(
        url=url,
        transcript="",
        summary="",
        audio_path="",
        messages=[]
    )
    
    # Create and run the workflow
    workflow = create_agent_workflow()
    final_state = workflow.invoke(initial_state)
    
    return final_state

# Example usage
if __name__ == "__main__":
    video_url = 'https://www.youtube.com/watch?v=-UkJDpH3I2U'
    result = process_youtube_video(video_url)
    
    print("\nWorkflow completed!")
    print(f"Audio file generated: {result['audio_path']}")
    print("\nSummary of the video:")
    print(result['summary'])
