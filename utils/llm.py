from PIL import Image
from io import BytesIO
import base64
from langchain_cohere.chat_models import ChatCohere
from langchain_core.messages import HumanMessage, AIMessage
import os
from dotenv import load_dotenv
load_dotenv()

HISTORY = []  # Global history to maintain conversation context

cohere_api_key = os.getenv("COHERE_API_KEY")




def maintain_history(question: str = "", response: str = "",
                     messages: list | None = None, max_length: int = 40):
    """Store finished turns or raw message segments."""
    global HISTORY
    messages = messages or []

    # 1. keep history short
    while len(HISTORY) > max_length - 2:
        HISTORY.pop(0)

    # 2. store a completed user ⇄ assistant turn
    if question and response:
        HISTORY.extend([
            {"type": "text", "text": question},          
            {"type": "text", "text": response},          
        ])

    # 3. store arbitrary segments passed in from vision functions
    if messages:
        while len(HISTORY) > max_length - len(messages):
            HISTORY.pop(0)
        HISTORY.extend(messages)
    


def clear_history():
    global HISTORY
    HISTORY = []
    

def summarize_scenary(image: Image.Image) -> str:
    chat = ChatCohere(model="c4ai-aya-vision-32b", temperature=0.3, cohere_api_key=cohere_api_key)
    
    # Convert PIL image to base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    message_content = [
        {"type": "text", "text": "Please summarize the following image of a captured by live feed:"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
    ]

    message = HumanMessage(content=message_content)
    response = chat.invoke([message])

    message_content.append({"type": "text", "text": response.content})

    maintain_history(messages = message_content)

    return response.content


def answer_question(image: Image.Image, question:str) -> str:
    chat = ChatCohere(model="c4ai-aya-vision-32b", temperature=0.3, cohere_api_key=cohere_api_key)
    
    # Convert PIL image to base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")



    message_content = [
        {"type": "text", "text": "Please answer the question using  the following image of a captured by live feed, Answer in less than 30 words."},

        {"type": "text", "text": f"question:{question}"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
    ]

    

    message = HumanMessage(content=message_content)
    response = chat.invoke([message])

    message_content.append({"type": "text", "text": response.content})

    maintain_history(messages=message_content)
    
    return response.content


def ask_followup(question: str) -> str:
    chat = ChatCohere(model="c4ai-aya-vision-32b", temperature=0.3, cohere_api_key=cohere_api_key)

    # Add the new question
    global HISTORY
    messages_content = HISTORY.copy()
    messages_content.append({"type": "text", "text": question})
    messages = HumanMessage(content=messages_content)

    response = chat.invoke([messages])
    
    maintain_history(question, response.content)
    
    return response.content