# 🎤 Aurasight - Voice-Controlled Visual Assistant

A powerful, accessible voice-controlled visual assistant designed specifically for visually impaired users. Aurasight uses advanced AI models to provide real-time scene descriptions, object detection, and intelligent responses through natural voice interaction.

## 🌟 Unique Features

## Features

- **Voice-activated commands**: Just say "start", "stop", "follow up", "clear history", "ok", or "restart"—no button clicks needed!
- **Image capture**: Takes a photo using your webcam.
- **Speech-to-text**: Converts your spoken questions into text.
- **AI-powered answers**: Uses a Cohere Multimodal LLM to answer your questions about the scene or follow up.
- **Text-to-speech**: Speaks the AI's answer back to you.
- **Modern Tkinter GUI**: Clean, accessible interface.


### 🎯 **Multi-Modal AI Integration**

- **YOLO Object Detection** - Real-time object recognition with confidence filtering
- **BLIP Image Captioning** - Natural language scene descriptions
- **CLIP Scene Classification** - Environment/location identification
- **Cohere AI** - Intelligent, contextual responses with hallucination detection
- **Tesseract OCR** - Text reading capabilities (on-demand)

### 🗣️ **Advanced Voice Commands**
- **"Describe"** or **"What do you see"** - Full scene description
- **"Objects"** or **"What objects"** - List detected objects
- **"Read text"** - Read visible text (on-demand only)
- **"Scene"** or **"Where am I"** - Environment description
- **"Person"** or **"Who is there"** - Describe people in detail
- **"Detailed"** or **"More details"** - Comprehensive analysis
- **"Comprehensive"** or **"Everything"** - Complete scene analysis
- **"Help"** - List available commands
- **"Stop"**, **"Quit"**, **"Exit"** - Exit the program

### 🔧 **Intelligent Features**
- **False Positive Prevention** - Advanced filtering for common misdetections
- **Hallucination Detection** - Prevents AI from making up details
- **Context-Aware Responses** - Uses actual detected data, not imagination
- **Detail Level Control** - Brief, detailed, person-specific, or comprehensive responses
- **Background Voice Recognition** - Continuous listening without interruption
- **Immediate Commands** - "Stop" works instantly without analysis delays

### 🎨 **Accessibility-Focused Design**
- **Voice-First Interface** - No physical interaction required
- **Natural Language Processing** - Conversational AI beyond visual analysis
- **Error Recovery** - Graceful handling when services fail
- **Debug Mode** - Visual feedback for sighted users/developers
- **Smart Caching** - Avoids redundant frame analysis

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Webcam
- Microphone
- Internet connection (for speech recognition and Cohere AI)
- 8GB+ RAM recommended for AI models

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd aurasight
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Tesseract OCR**
   - **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt-get install tesseract-ocr`

4. **Set up Cohere API**
   - Get API key from [Cohere](https://cohere.ai/)
   - Update the API key in `main.py` (line with `co = cohere.Client('your-api-key')`)

5. **Download YOLO model**
   - The `yolov8l.pt` file should be in the project directory
   - If missing, it will be downloaded automatically on first run

6. **Run the application**
   ```bash
   python main.py
   ```

## 📁 Project Structure

```
aurasight/
├── main.py                # Main application with all functionality
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── yolov8l.pt            # YOLO model file (YOLOv8 Large)
```

## 🎮 Usage

### Basic Commands
1. **Start the application** - Run `python main.py`
2. **Wait for initialization** - Models will load automatically
3. **Speak commands** - Use natural language or specific commands
4. **Get immediate responses** - No need to wait for analysis unless needed

### Example Interactions

**Scene Description:**
```
You: "What do you see?"
Aurasight: "I can see a chair, table, and coffee mug. It looks like you're in an office environment."
```

**Object Detection:**
```
You: "What objects are there?"
Aurasight: "I can spot a chair, table, and coffee mug in front of you."
```

**Text Reading (On-Demand):**
```
You: "Read the text"
Aurasight: "I can read this text: 'Welcome to Aurasight - Your Visual Assistant'"
```

**General Conversation:**
```
You: "What's the weather like?"
Aurasight: "I can't check the weather directly, but I can help you with visual analysis and general questions."
```

**Immediate Exit:**
```
You: "Stop"
Aurasight: "Goodbye! Have a great day." [Program exits immediately]
```

## ⚙️ Configuration

### Speech Recognition Settings
The system uses optimized settings for accessibility:
- **Energy threshold**: 200 (balanced sensitivity)
- **Dynamic energy threshold**: True (adapts to environment)
- **Pause threshold**: 0.8 seconds (natural speech patterns)
- **Phrase threshold**: 0.3 seconds (filters noise)
- **Non-speaking duration**: 0.5 seconds (quick response)

### AI Model Settings
- **Confidence threshold**: 0.8 (high accuracy)
- **False positive filtering**: Special handling for laptops, computers, TVs
- **Hallucination detection**: Prevents AI from inventing details
- **Response temperature**: 0.1 (conservative, factual responses)

### Debug Mode
Set `DEBUG_MODE = True` in the code to see:
- Detected objects with confidence levels
- BLIP captions
- CLIP scene classifications
- Detailed analysis information

## 🔧 Advanced Features

### False Positive Prevention
```python
# Objects requiring higher confidence
high_confidence_objects = ['laptop', 'computer', 'tv', 'monitor', 'cell phone']
# These require 75% confidence vs 60% for other objects
```

### Hallucination Detection
The system checks for:
- Uncertain phrases ("looks like", "maybe", "probably")
- Objects not actually detected
- Overly imaginative responses

### Smart Frame Analysis
- **Caching**: Only analyzes frames when needed (every 3+ seconds)
- **Multi-model fusion**: Combines YOLO, BLIP, and CLIP results
- **Confidence filtering**: Reduces false positives
- **Context preservation**: Maintains analysis results for responses

## 🔧 Troubleshooting

### Common Issues

**Microphone not working:**
- Check microphone permissions in system settings
- Ensure microphone is set as default input device
- Try different microphone if available
- Check if other applications can use the microphone

**Speech recognition issues:**
- Speak clearly and at normal volume
- Reduce background noise
- Check internet connection (required for Google Speech Recognition)
- Try adjusting energy threshold in code if too sensitive/not sensitive enough

**Models not loading:**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check if `yolov8l.pt` file exists in project directory
- Verify sufficient RAM (8GB+ recommended for AI models)
- Close other applications to free up memory

**Camera issues:**
- Check camera permissions in system settings
- Ensure no other application is using the camera
- Try different camera if available
- Check if camera works in other applications

**False object detections:**
- The system now has advanced filtering for common false positives
- Laptops, computers, TVs require higher confidence (75%)
- Check debug output to see confidence levels
- Adjust lighting for better detection

### Performance Tips
- **Close unnecessary applications** to free up RAM for AI models
- **Use SSD storage** for faster model loading
- **Good lighting** improves object detection accuracy
- **Stable internet connection** for speech recognition and Cohere AI
- **Quiet environment** improves voice command recognition

### Error Recovery
The system includes multiple fallback mechanisms:
- **Graceful degradation** when AI services fail
- **Fallback responses** when Cohere API is unavailable
- **Error handling** for camera, microphone, and model issues
- **Debug information** to help identify problems

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly with voice commands
5. Submit a pull request

### Development Guidelines
- Maintain accessibility focus
- Test with voice commands
- Ensure error handling
- Add debug information for new features
- Keep performance in mind (AI models are resource-intensive)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **YOLO (Ultralytics)** - Real-time object detection
- **BLIP (Salesforce)** - Image captioning and scene understanding
- **CLIP (OpenAI)** - Scene classification and visual-language understanding
- **Tesseract** - OCR capabilities for text reading
- **Cohere** - Natural language generation and conversational AI
- **SpeechRecognition** - Voice input processing
- **OpenCV** - Computer vision and camera handling
- **PyTorch** - Deep learning framework for AI models

## 🌟 What Makes Aurasight Unique

### Compared to Other Assistants:
- **Siri/Alexa**: Limited visual capabilities, no real-time object detection
- **Google Lens**: No voice interface, requires manual interaction
- **Be My Eyes**: Human volunteers, not AI-powered
- **Seeing AI**: Limited to specific use cases, less conversational

### Technical Innovations:
- **Multi-model fusion** of different AI technologies
- **Real-time confidence filtering** for accuracy
- **Contextual response generation** with hallucination prevention
- **Voice-first design** optimized for accessibility
- **Hybrid local/cloud architecture** for privacy and performance

### Accessibility Features:
- **Truly hands-free** operation
- **Natural language interface** - no memorizing commands
- **Comprehensive scene understanding** beyond just objects
- **Conversational AI** that can handle general questions
- **Graceful error handling** that doesn't leave users stranded

Aurasight represents a **next-generation accessibility tool** that combines cutting-edge AI with thoughtful user experience design, making it uniquely powerful for visually impaired users while remaining useful for everyone. 
