from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os
from dotenv import load_dotenv
import uuid
from gtts import gTTS
from pydub import AudioSegment
import speech_recognition as sr
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph_supervisor import create_supervisor
from langchain_core.prompts import ChatPromptTemplate
from agent import web_search_agent, stock_history_agent
from langchain.schema import HumanMessage, AIMessage

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
recognizer = sr.Recognizer()
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """You are a stock market assistant that coordinates between different specialized agents.
Your role is to:
1. Understand the user's request and focus on the key terms provided
2. Use the appropriate agent(s) to find relevant information
3. Provide a concise and focused response based on the agent results

Available agents:
- Google Search Agent: For finding stock-related news and information
- Stock History Agent: For fetching historical stock data

Always ensure the response is well-formatted and includes all relevant information.
Give a summary from the retrieved data."""
    ),
])

supervisor = create_supervisor(
    [web_search_agent, stock_history_agent],
    model=llm,
    output_mode="full_history",
    prompt=prompt,
)

app_agent = supervisor.compile()

def process_user_query(query: str):
    result = app_agent.invoke({"Message": [
        {
            "role":"user",
            "content":{query}
        }]})
    return result

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        os.makedirs("static", exist_ok=True)

        # Save uploaded audio file
        with open("temp.wav", "wb") as f:
            f.write(await file.read())

        # Convert audio to proper format
        sound = AudioSegment.from_file("temp.wav")
        sound.export("converted.wav", format="wav")

        # Transcribe
        with sr.AudioFile("converted.wav") as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)

        # Query LangGraph supervisor
        response = process_user_query(text)

        final_response = ""
        for message in reversed(response['messages']):
            if isinstance(message, AIMessage):
                final_response = message.content
                break

        # Generate audio
        audio_filename = f"response_{uuid.uuid4().hex}.mp3"
        audio_path = f"static/{audio_filename}"
        tts = gTTS(text=final_response, lang='en')
        tts.save(audio_path)

        return JSONResponse({
            "transcription": text,
            "response": final_response,
            "audio_url": f"/static/{audio_filename}"
        })

    except sr.UnknownValueError:
        return JSONResponse({"error": "Google Speech Recognition could not understand audio"})
    except sr.RequestError as e:
        return JSONResponse({"error": f"Could not request results from Google; {e}"})
    except Exception as e:
        return JSONResponse({"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)