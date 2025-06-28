import cv2
import pytesseract
import pyttsx3
from ultralytics import YOLO
from PIL import Image
import torch
import speech_recognition as sr
import threading
import queue
from datetime import datetime
import cohere
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    CLIPProcessor,
    CLIPModel,
)


# Configuration Paths
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Model instances
model = YOLO("yolov8l.pt")  
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Scene Labels for CLIP
SCENE_CATEGORIES = [
    "office",
    "kitchen",
    "living room",
    "street",
    "garden",
    "shop",
    "bedroom",
    "hall",
    "outdoor",
    "indoor",
    "classroom",
    "lab",
    "bathroom",
    "restaurant",
    "library",
    "gym",
]

# Speech Recognition
recognizer = sr.Recognizer()
recognizer.energy_threshold = 200
recognizer.dynamic_energy_threshold = True
recognizer.pause_threshold = 0.8
recognizer.phrase_threshold = 0.3
recognizer.non_speaking_duration = 0.5

# Use default microphone
microphone = sr.Microphone()

# Threshold
CONF_THRESHOLD = 0.8

# Global variables
current_frame = None
frame_lock = threading.Lock()
voice_queue = queue.Queue()
is_listening = False
DEBUG_MODE = True  # Set to True to see what text is being detected
last_analysis = {
    "objects": [],
    "text": "",
    "caption": "",
    "scene": "",
    "timestamp": None
}

# Voice commands
VOICE_COMMANDS = {
    "describe": ["describe", "what do you see", "tell me what you see", "what's in front of me"],
    "objects": ["objects", "what objects", "detect objects", "find objects"],
    "read_text": ["read text", "what text", "text detection", "read", "read the text", "what does it say"],
    "scene": ["scene", "where am i", "location", "environment"],
    "help": ["help", "commands", "what can you do", "instructions"],
    "stop": ["stop", "quit", "exit", "goodbye"],
    "detailed": ["detailed", "more details", "explain", "elaborate"],
    "person": ["person", "people", "who is there", "describe person", "person description"],
    "comprehensive": ["comprehensive", "full description", "complete description", "everything", "all details"]
}

# Initialize Cohere client
co = cohere.Client('kEARoQwZWzviGO3OxVfwTkhCK2R65X2tR7mA6Qax')

# ==================================================
# UTILITY FUNCTIONS
# ==================================================
def detect_objects(img, conf_thresh=0.3):
    """Detect objects using YOLO and filter by confidence."""
    results = model(img)
    labels = []
    for result in results:
        for box in result.boxes:
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            
            # Apply confidence threshold
            if confidence >= conf_thresh:
                # Additional filtering for common false positives
                if should_include_object(class_name, confidence):
                    labels.append(class_name)
    return list(set(labels)), results


def should_include_object(class_name, confidence):
    """Filter out objects that are likely false positives."""
    # Common false positive objects that need higher confidence
    high_confidence_objects = ['laptop', 'computer', 'tv', 'monitor', 'cell phone']
    
    if class_name in high_confidence_objects:
        # Require higher confidence for these objects
        return confidence >= 0.75
    else:
        # Normal confidence threshold for other objects
        return True


def extract_text(img):
    """Perform OCR using Tesseract with enhanced preprocessing and validation."""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Multiple preprocessing attempts for better text detection
        results = []
        
        # Attempt 1: Basic preprocessing
        basic_text = pytesseract.image_to_string(gray, config='--psm 6')
        if basic_text.strip():
            results.append(basic_text.strip())
        
        # Attempt 2: Enhanced contrast
        enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
        enhanced_text = pytesseract.image_to_string(enhanced, config='--psm 6')
        if enhanced_text.strip():
            results.append(enhanced_text.strip())
        
        # Attempt 3: Denoised image
        denoised = cv2.fastNlMeansDenoising(gray)
        denoised_text = pytesseract.image_to_string(denoised, config='--psm 6')
        if denoised_text.strip():
            results.append(denoised_text.strip())
        
        # Attempt 4: For small text (like expiry dates)
        small_text = pytesseract.image_to_string(gray, config='--psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz/.-')
        if small_text.strip():
            results.append(small_text.strip())
        
        # Combine all results and validate
        all_text = ' '.join(results)
        
        # Clean and validate the text
        cleaned_text = clean_and_validate_text(all_text)
        
        return cleaned_text
        
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""


def clean_and_validate_text(text):
    """Clean and validate extracted text to remove false positives."""
    if not text:
        return ""
    
    import re
    
    # Remove single characters that are likely noise
    text = re.sub(r'\b[a-zA-Z]\b', '', text)
    
    # Remove lines with only numbers or symbols
    lines = text.split('\n')
    valid_lines = []
    for line in lines:
        line = line.strip()
        if len(line) > 2:  # Only keep lines with more than 2 characters
            # Check if line has meaningful content
            if re.search(r'[a-zA-Z]{2,}', line) or re.search(r'\d{2,}', line):
                valid_lines.append(line)
    
    # Rejoin valid lines
    cleaned_text = ' '.join(valid_lines)
    
    # Remove extra whitespace
    cleaned_text = ' '.join(cleaned_text.split())
    
    # Only return if we have meaningful text
    if len(cleaned_text) > 3:
        return cleaned_text
    else:
        return ""


def get_blip_caption(frame):
    """Get scene caption using BLIP (local)."""
    try:
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = blip_processor(pil_image, return_tensors="pt")
        with torch.no_grad():
            output = blip_model.generate(**inputs, max_length=30)
        caption = blip_processor.decode(output[0], skip_special_tokens=True)
        return caption
    except:
        return ""


def get_clip_classification(frame, labels=SCENE_CATEGORIES):
    """Classify scene using CLIP."""
    try:
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = clip_processor(text=labels, images=pil_image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
            top_prob, top_label_id = probs.max(dim=1)
        return labels[top_label_id]
    except:
        return ""


def speak(text):
    """Convert text to speech."""
    if text:
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"Speech Error: {e}")


def get_refined_response(user_query, context, detail_level="detailed"):
    try:
        # Validate context to ensure we have real data
        objects = context.get('objects', [])
        text = context.get('text', '')
        caption = context.get('caption', '')
        scene = context.get('scene', '')
        
        # If we have very little real data, be very conservative
        has_real_data = len(objects) > 0 or len(text.strip()) > 5 or len(caption.strip()) > 5
        
        if detail_level == "detailed":
            instruction = """IMPORTANT: Only describe what is actually detected. Do NOT make up or imagine details. Be factual and accurate. If you're not sure about something, say so. Include only:
            - Objects that are actually detected (if any)
            - Text that is actually readable (if any)
            - Scene description if clear and specific
            - Location if identifiable
            Use casual, friendly language but be honest about what you can and cannot see. Do not invent details."""
            max_tokens = 100
        elif detail_level == "brief":
            instruction = "Give a quick, factual summary. Only mention what is actually detected. If unsure, say so."
            max_tokens = 40
        elif detail_level == "person":
            instruction = """Describe people only if they are actually detected. If no people are detected, say so clearly. If people are detected, describe them naturally but factually. Do not invent details about appearance or actions."""
            max_tokens = 80
        elif detail_level == "scene":
            instruction = """Describe the scene only if you have clear information about it. If the scene is unclear or generic, say so. Do not invent details."""
            max_tokens = 80
        else:
            instruction = "Respond naturally and helpfully, but only mention what is actually detected. Be honest about limitations."
            max_tokens = 60

        # Create a more factual prompt
        prompt = (
            f"Here's what I can actually detect:\n"
            f"Objects detected: {objects if objects else 'None detected'}\n"
            f"Text detected: '{text if text else 'No readable text'}'"
        )
        
        if caption and len(caption.strip()) > 5:
            prompt += f"\nScene description: {caption}"
        if scene:
            prompt += f"\nLocation: {scene}"
            
        prompt += f"\n\nQuestion: {user_query}\n\n"
        prompt += f"Instructions: {instruction}\n\n"
        prompt += "Factual Response (only mention what is actually detected):"
        
        # Use much lower temperature for more conservative responses
        temperature = 0.1 if has_real_data else 0.05
        
        response = co.generate(
            model='command',
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,  # Much lower for more factual responses
            k=0,
            p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1
        )
        answer = response.generations[0].text.strip()
        
        # Validate the response
        if not answer or len(answer) < 5:
            return get_fallback_response(context, user_query)
        
        # Check if response contains hallucinated content
        if contains_hallucination(answer, context):
            return get_fallback_response(context, user_query)
        
        return answer
    except Exception as e:
        print(f"Cohere API Error: {e}")
        return get_fallback_response(context, user_query)


def contains_hallucination(response, context):
    """Check if the response contains hallucinated content."""
    response_lower = response.lower()
    objects = context.get('objects', [])
    text = context.get('text', '')
    
    # Check for common hallucination patterns
    hallucination_indicators = [
        'looks like', 'seems like', 'appears to be', 'maybe', 'probably',
        'i think', 'i believe', 'it seems', 'it appears', 'might be',
        'could be', 'possibly', 'perhaps', 'likely', 'probably'
    ]
    
    # If response has too many uncertain phrases, it might be hallucinating
    uncertain_count = sum(1 for indicator in hallucination_indicators if indicator in response_lower)
    if uncertain_count > 2:
        return True
    
    # Check if response mentions objects not in the detected list
    response_words = set(response_lower.split())
    detected_objects = set([obj.lower() for obj in objects])
    
    # Common objects that might be hallucinated
    common_hallucinations = ['person', 'man', 'woman', 'people', 'chair', 'table', 'wall', 'room']
    
    for hallucination in common_hallucinations:
        if hallucination in response_words and hallucination not in detected_objects:
            return True
    
    return False


def get_fallback_response(context, user_query):
    """Provide a fallback response when AI generation fails or is unreliable."""
    objects = context.get('objects', [])
    text = context.get('text', '')
    caption = context.get('caption', '')
    scene = context.get('scene', '')
    
    # Build a simple factual response
    parts = []
    
    if objects:
        parts.append(f"I can detect {', '.join(objects)}")
    
    if text and len(text.strip()) > 3:
        parts.append(f"I can read some text: {text[:100]}...")
    
    if caption and len(caption.strip()) > 5:
        parts.append(f"The scene looks like {caption}")
    
    if scene:
        parts.append(f"You appear to be in a {scene}")
    
    if parts:
        return ". ".join(parts) + "."
    else:
        return "I'm not detecting anything specific in the current view. Try moving the camera or adjusting the lighting."


def analyze_current_frame():
    """Analyze the current frame and update global results."""
    global current_frame, last_analysis
    
    with frame_lock:
        if current_frame is None:
            return
        
        frame = current_frame.copy()
    
    if DEBUG_MODE:
        print("\n=== FRAME ANALYSIS ===")
    
    # Perform all analyses (excluding text extraction)
    objects, results = detect_objects(frame, CONF_THRESHOLD)
    # text = extract_text(frame)  # Disabled text extraction
    caption = get_blip_caption(frame)
    scene = get_clip_classification(frame)
    
    if DEBUG_MODE:
        print(f"Objects detected: {objects}")
        # Print confidence levels for debugging
        print("Detailed detections:")
        for result in results:
            for box in result.boxes:
                confidence = box.conf[0]
                class_name = result.names[int(box.cls[0])]
                print(f"  - {class_name}: {confidence:.2f}")
        # print(f"Text extracted: '{text[:100]}{'...' if len(text) > 100 else ''}'")  # Disabled
        print(f"BLIP caption: '{caption}'")
        print(f"CLIP scene: '{scene}'")
        print("=====================\n")
    
    # Update last analysis (without text)
    last_analysis = {
        "objects": objects,
        "text": "",  # Always empty - no text detection
        "caption": caption,
        "scene": scene,
        "timestamp": datetime.now()
    }


def listen_for_commands():
    """Listen for voice commands in background."""
    global is_listening
    
    print("Starting voice recognition...")
    
    with microphone as source:
        # Adjust for ambient noise
        print("Calibrating microphone for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1.0)
        print(f"Microphone calibrated. Energy threshold: {recognizer.energy_threshold}")
    
    print("Voice recognition active - listening for commands...")
    
    while is_listening:
        try:
            with microphone as source:
                audio = recognizer.listen(
                    source, 
                    timeout=1.0,
                    phrase_time_limit=3.0,
                )
            
            try:
                # Use Google's speech recognition
                command = recognizer.recognize_google(
                    audio, 
                    language='en-US'
                ).lower()
                
                print(f"Heard: '{command}'")
                voice_queue.put(command)
                
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                print(f"Speech recognition service error: {e}")
                speak("Sorry, I'm having trouble with speech recognition")
                
        except sr.WaitTimeoutError:
            continue
        except Exception as e:
            print(f"Speech recognition error: {e}")
            continue


def process_voice_command(command):
    """Process voice commands and generate appropriate responses."""
    global last_analysis
    
    # Check for command matches
    for cmd_type, keywords in VOICE_COMMANDS.items():
        if any(keyword in command for keyword in keywords):
            return execute_command(cmd_type, command)
    
    # If no direct command match, treat as a question
    return handle_question(command)


def execute_command(cmd_type, command):
    """Execute specific voice commands."""
    global last_analysis
    
    # Handle stop command immediately without analysis
    if cmd_type == "stop":
        speak("Goodbye! Have a great day.")
        return "exit"
    
    # Handle help command without analysis
    elif cmd_type == "help":
        return get_help_message()
    
    # For all other commands that need visual analysis, check if analysis is needed
    if last_analysis["timestamp"] is None or (datetime.now() - last_analysis["timestamp"]).seconds > 3:
        speak("Let me take a look at what's around you...")
        analyze_current_frame()
    
    if cmd_type == "describe":
        return get_full_description()
    
    elif cmd_type == "objects":
        return get_objects_description()
    
    elif cmd_type == "read_text":
        return get_text_on_demand()
    
    elif cmd_type == "scene":
        return get_scene_description()
    
    elif cmd_type == "detailed":
        return get_detailed_description()
    
    elif cmd_type == "person":
        return get_person_description()
    
    elif cmd_type == "comprehensive":
        return get_comprehensive_description()
    
    return "I didn't quite catch that. Try saying 'help' to see what I can do for you."


def handle_question(question):
    """Handle natural language questions about the scene and general queries."""
    global last_analysis
    
    # it checks if it's a visual analysis question
    visual_keywords = [
        "see", "look", "detect", "find", "spot", "notice", "observe",
        "what", "where", "how many", "describe", "tell me about",
        "person", "people", "object", "text", "scene", "room", "place"
    ]
    
    is_visual_question = any(keyword in question.lower() for keyword in visual_keywords)
    
    if is_visual_question:
        # it handles visual analysis questions
        return handle_visual_question(question)
    else:
        # it handles general conversation questions
        return handle_general_question(question)


def handle_visual_question(question):
    """Handle questions specifically about visual analysis."""
    global last_analysis
    
    # it will analyse the current frame if needed
    if last_analysis["timestamp"] is None or (datetime.now() - last_analysis["timestamp"]).seconds > 3:
        speak("Let me take a look at what's around you...")
        analyze_current_frame()
    
    # it will determine detail level based on question type
    detail_level = "detailed"
    if any(word in question.lower() for word in ["person", "people", "who", "man", "woman", "boy", "girl"]):
        detail_level = "person"
    elif any(word in question.lower() for word in ["scene", "environment", "room", "place", "setting"]):
        detail_level = "scene"
    elif any(word in question.lower() for word in ["brief", "quick", "short"]):
        detail_level = "brief"
    
    # Create context from current analysis
    context = {
        "objects": last_analysis["objects"],
        "text": last_analysis["text"],
        "caption": last_analysis["caption"],
        "scene": last_analysis["scene"]
    }
    
    # Generate refined response using Cohere
    response = get_refined_response(question, context, detail_level)
    return response


def handle_general_question(question):
    """It will handle general questions."""
    try:
        # prompt for general conversation
        
        prompt = (
            f"Question: {question}\n\n"
            f"Instructions: You are Aurasight, a helpful voice assistant for visually impaired users. "
            f"Answer the question naturally and helpfully. Be friendly and conversational. "
            f"If the question is about something you can't help with, politely explain your limitations. "
            f"Keep responses concise and clear.\n\n"
            f"Response:"
        )
        
        response = co.generate(
            model='command',
            prompt=prompt,
            max_tokens=120,
            temperature=0.7,  # Slightly higher for general conversation
            k=0,
            p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1
        )
        
        answer = response.generations[0].text.strip()
        if not answer or len(answer) < 5:
            return "I'm not sure how to answer that. Could you rephrase your question?"
        
        return answer
        
    except Exception as e:
        print(f"General question error: {e}")
        return "I'm having trouble processing that question right now. Could you try asking something else?"


def get_general_response(user_input):
    """Generate a general response for any type of input."""
    try:
        # Determine if it's a question, statement, or command
        is_question = user_input.strip().endswith('?') or any(word in user_input.lower() for word in ['what', 'how', 'why', 'when', 'where', 'who', 'which'])
        
        if is_question:
            return handle_general_question(user_input)
        else:
            # Handle statements or casual conversation
            prompt = (
                f"User said: {user_input}\n\n"
                f"Instructions: You are Aurasight, a friendly voice assistant. "
                f"Respond naturally to what the user said. Be conversational and helpful. "
                f"If they're making a statement, acknowledge it appropriately. "
                f"If they need help, offer assistance.\n\n"
                f"Response:"
            )
            
            response = co.generate(
                model='command',
                prompt=prompt,
                max_tokens=80,
                temperature=0.7,
                k=0,
                p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
            
            answer = response.generations[0].text.strip()
            if not answer or len(answer) < 3:
                return "I understand. How can I help you?"
            
            return answer
            
    except Exception as e:
        print(f"General response error: {e}")
        return "I'm here to help! What would you like to know?"


def get_full_description():
    """Get complete scene description."""
    global last_analysis
    
    parts = []
    if last_analysis["objects"]:
        parts.append(f"I can see {', '.join(last_analysis['objects'])}")
    if last_analysis["caption"]:
        parts.append(f"It looks like {last_analysis['caption']}")
    if last_analysis["scene"]:
        parts.append(f"You're in what appears to be a {last_analysis['scene']}")
    
    if parts:
        return ". ".join(parts) + "."
    else:
        return "I'm not really seeing anything clear at the moment. Maybe try moving the camera around a bit?"


def get_objects_description():
    """Get objects-only description."""
    global last_analysis
    
    if last_analysis["objects"]:
        return f"I can spot {', '.join(last_analysis['objects'])} in front of you."
    else:
        return "I'm not picking up any objects right now. Try pointing the camera at something specific."


def get_text_on_demand():
    """Perform text detection only when specifically requested."""
    global current_frame
    
    if current_frame is None:
        return "I don't have a current image to analyze for text."
    
    speak("Let me try to read any text in the current view...")
    
    try:
        # Perform text extraction on demand
        text = extract_text(current_frame)
        
        if text and len(text.strip()) > 3:
            # Clean and validate the text
            cleaned_text = clean_and_validate_text(text)
            
            if cleaned_text:
                # Limit text length to avoid overwhelming responses
                text_preview = cleaned_text[:200]
                if len(cleaned_text) > 200:
                    text_preview += "..."
                return f"I can read this text: {text_preview}"
            else:
                return "I can see some text but it's not clear enough to read properly."
        else:
            return "I don't see any readable text in the current view. Try holding something with text closer to the camera or adjusting the lighting."
            
    except Exception as e:
        print(f"Text detection error: {e}")
        return "I'm having trouble reading text right now. Please try again."


def get_scene_description():
    """Get scene-only description."""
    global last_analysis
    
    parts = []
    if last_analysis["caption"]:
        parts.append(f"It looks like {last_analysis['caption']}")
    if last_analysis["scene"]:
        parts.append(f"You're in a {last_analysis['scene']}")
    
    if parts:
        return ". ".join(parts) + "."
    else:
        return "I'm having trouble figuring out what kind of place this is. Maybe try a different angle?"


def get_detailed_description():
    """Get very detailed description."""
    global last_analysis
    
    detailed_parts = []
    
    if last_analysis["objects"]:
        obj_count = len(last_analysis["objects"])
        detailed_parts.append(f"I've found {obj_count} different things: {', '.join(last_analysis['objects'])}")
    
    if last_analysis["caption"]:
        detailed_parts.append(f"The overall scene looks like {last_analysis['caption']}")
    
    if last_analysis["scene"]:
        detailed_parts.append(f"Based on the surroundings, you seem to be in a {last_analysis['scene']}")
    
    if detailed_parts:
        return ". ".join(detailed_parts) + "."
    else:
        return "I'm not getting a clear picture of what's around you. Maybe try adjusting the lighting or camera position?"


def get_help_message():
    """Get help message with available commands."""
    return """I'm Aurasight, your voice-controlled assistant! I can help you in many ways:

    VISUAL ANALYSIS:
    Say 'describe' or 'what do you see' for a full description
    Say 'objects' to hear what objects I can spot
    Say 'read text' to read any text in the current view (on-demand only)
    Say 'scene' or 'where am I' for location information
    Say 'detailed' for a more comprehensive description
    Say 'person' or 'who is there' for detailed person descriptions
    Say 'comprehensive' for complete scene analysis

    GENERAL CONVERSATION:
    You can ask me ANY question! I'm not limited to just visual analysis.
    Ask me about the weather, time, general knowledge, or just chat with me.
    I can help with information, answer questions, or just have a conversation.

    Say 'stop' to exit the program"""


def extract_expiry_date(text):
    """Extract expiry date from text using regex patterns."""
    return None


def extract_product_info(text):
    """Extract product information like ingredients, nutrition, etc."""
    return {}


def get_expiry_description():
    """Get expiry date information from detected text."""
    return "Text reading feature has been disabled. I cannot read expiry dates or product information."


def get_product_description():
    """Get detailed product information."""
    return "Text reading feature has been disabled. I cannot read product labels or information."


def get_person_description():
    """Get detailed person description using Cohere."""
    global last_analysis
    
    if not last_analysis["objects"]:
        return "I don't see any people in the current view."
    
    # Check if people are detected
    people_detected = any(obj in ["person", "people"] for obj in last_analysis["objects"])
    if not people_detected:
        return "I can't spot any specific people in the current view."
    
    # Use Cohere for detailed person description
    context = {
        "objects": last_analysis["objects"],
        "text": last_analysis["text"],
        "caption": last_analysis["caption"],
        "scene": last_analysis["scene"]
    }
    
    response = get_refined_response("Describe the people in detail", context, "person")
    return response


def get_comprehensive_description():
    """Get comprehensive scene description using Cohere."""
    global last_analysis
    
    # Use Cohere for comprehensive description
    context = {
        "objects": last_analysis["objects"],
        "text": last_analysis["text"],
        "caption": last_analysis["caption"],
        "scene": last_analysis["scene"]
    }
    
    response = get_refined_response("Tell me everything you can see in detail", context, "detailed")
    return response


# MAIN LOOP

def main():
    global current_frame, is_listening
    
    cap = cv2.VideoCapture(0)
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Aurasight Voice-Controlled Assistant")
    print("Designed for visually impaired users")
    print("Say 'help' for available commands")
    print("Press 'q' to quit")
    
    # Welcome message
    speak("Hi there! I'm Aurasight, your friendly voice-controlled visual assistant. Just say 'help' if you want to know what I can do for you.")
    
    # Start voice recognition
    is_listening = True
    voice_thread = threading.Thread(target=listen_for_commands, daemon=True)
    voice_thread.start()
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Update current frame
        with frame_lock:
            current_frame = frame.copy()
        
        # Process voice commands
        try:
            while not voice_queue.empty():
                command = voice_queue.get_nowait()
                
                response = process_voice_command(command)
                
                if response == "exit":
                    is_listening = False
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                
                print(f"Response: {response}")
                speak(response)
                
        except queue.Empty:
            pass
        
        # Visual display (for sighted users/debugging)
        if frame_count % 30 == 0:  # Update every 30 frames
            # Simple visualization
            display_frame = frame.copy()
            cv2.putText(
                display_frame,
                "Voice-Controlled Aurasight",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                display_frame,
                "Say commands or ask questions",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )
            cv2.imshow("Aurasight Voice Assistant", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    is_listening = False
    cap.release()
    cv2.destroyAllWindows()
    speak("Thanks for using Aurasight! Have a great day!")


def initialize_tts():
    """Initialize and test the TTS engine with optimal settings."""
    global engine
    
    try:
        # Try to initialize with SAPI5 (Windows default)
        engine = pyttsx3.init('sapi5')
        print("Using SAPI5 TTS engine")
    except:
        try:
            # Fallback to nsss (macOS)
            engine = pyttsx3.init('nsss')
            print("Using NSSS TTS engine")
        except:
            try:
                # Fallback to espeak (Linux)
                engine = pyttsx3.init('espeak')
                print("Using eSpeak TTS engine")
            except:
                # Final fallback to default
                engine = pyttsx3.init()
                print("Using default TTS engine")
    
    try:
        # Set basic properties
        engine.setProperty('rate', 140)
        engine.setProperty('volume', 0.9)
        
        # Get available voices
        voices = engine.getProperty('voices')
        if voices:
            # Try to find a clear, English voice
            selected_voice = None
            for voice in voices:
                voice_name = voice.name.lower()
                voice_id = voice.id.lower()
                if any(lang in voice_name or lang in voice_id for lang in ['english', 'en', 'us', 'uk']):
                    selected_voice = voice.id
                    break
            
            # If no English voice found, use the first available
            if not selected_voice:
                selected_voice = voices[0].id
            
            engine.setProperty('voice', selected_voice)
            print(f"Using voice: {selected_voice}")
        
        # Test the engine
        test_text = "Hello, I'm ready to help you."
        engine.say(test_text)
        engine.runAndWait()
        
        return True
        
    except Exception as e:
        print(f"TTS initialization error: {e}")
        return False


# Initialize TTS engine
if not initialize_tts():
    print("Warning: TTS initialization failed. Speech may not work properly.")


if __name__ == "__main__":
    main()