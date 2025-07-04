import speech_recognition as sr
import pyttsx3
import webbrowser
import datetime
from llama_cpp import Llama

engine = pyttsx3.init()

llm = Llama(
    model_path="C:/Users/Aditya/models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf",
    n_ctx=512,
    n_threads=4
)

def speak(text):
    engine.say(text)
    engine.runAndWait()

def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Calibrating Mic for background noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        recognizer.energy_threshold = 300
        print("Listening...")
        audio = recognizer.listen(source) 
    try:
        command = recognizer.recognize_google(audio)
        print(f"You said: {command}")
        return command.lower()
    except sr.UnknownValueError:
        speak("Sorry, there's and issue with the recognizer service")
        return ""
    except sr.RequestError:
        speak("Sorry, there's an issue with the recognizer")
        return ""
    
def ask_llama(prompt):
    print(f"[LLM Input] {prompt}")
    output = llm(
        f"<s> {prompt}",
        max_tokens=100,
        stop=["</s>", "\n"]
    )    
    reply = output["choices"][0]["text"].strip()
    print(f"[LLM Output] {reply}")
    return reply

def handle_command(command):
    if "open youtube" in command:
        speak("Opening Youtube")
        webbrowser.open("https://www.youtube.com")
    elif "what is your name" in command:
        speak("I am your personal assistant, Call me JARVIS")
    elif "time" in command:
        now = datetime.datetime.now().strftime("%H:%M")   
        speak("The current time is {now}")
    elif "stop" in command or "exit" in command:
        speak("Goodbye")
        exit()
    else:
        response = llm(
            prompt = command,
            max_tokens=100,
            stop=["</s>", "\n"]
        )["choices"][0]["text"].strip()
        print(f"[LLM Output] {response}")
        speak(response)   
while True:
    command = listen()
    if command:
        handle_command(command)            
     




              
