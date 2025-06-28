#!/usr/bin/env python3
"""
Test script to verify STT functionality using speech_recognition
"""

import speech_recognition as sr
import time

def test_stt():
    """Test speech-to-text functionality."""
    print("Testing STT (Speech-to-Text) functionality...")
    
    # Initialize recognizer
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 400
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8
    recognizer.phrase_threshold = 0.3
    recognizer.non_speaking_duration = 0.5
    
    # Initialize microphone
    microphone = sr.Microphone()
    
    try:
        # Calibrate microphone
        print("1. Calibrating microphone...")
        with microphone as source:
            print("   📢 Please remain quiet for 2 seconds...")
            recognizer.adjust_for_ambient_noise(source, duration=2.0)
            print(f"   ✓ Microphone calibrated. Energy threshold: {recognizer.energy_threshold}")
        
        # Test speech recognition
        print("2. Testing speech recognition...")
        print("   📢 Please say something (you have 5 seconds)...")
        
        with microphone as source:
            try:
                audio = recognizer.listen(source, timeout=5.0, phrase_time_limit=5.0)
                print("   ✓ Audio captured successfully")
                
                # Recognize speech
                print("   🔄 Processing speech...")
                text = recognizer.recognize_google(audio, language='en-US')
                print(f"   ✓ Recognized: '{text}'")
                
                return True
                
            except sr.WaitTimeoutError:
                print("   ⚠ No speech detected within timeout")
                return False
            except sr.UnknownValueError:
                print("   ⚠ Could not understand the audio")
                return False
            except sr.RequestError as e:
                print(f"   ❌ Speech recognition service error: {e}")
                return False
                
    except Exception as e:
        print(f"❌ STT test failed: {e}")
        return False

def test_microphone_devices():
    """Test available microphone devices."""
    print("3. Checking microphone devices...")
    
    try:
        microphone = sr.Microphone()
        print(f"   ✓ Microphone initialized")
        
        # List available devices (if possible)
        try:
            import pyaudio
            audio = pyaudio.PyAudio()
            info = audio.get_host_api_info_by_index(0)
            numdevices = info.get('deviceCount')
            
            input_devices = []
            for i in range(0, numdevices):
                device_info = audio.get_device_info_by_host_api_device_index(0, i)
                if device_info.get('maxInputChannels') > 0:
                    input_devices.append((i, device_info.get('name')))
                    print(f"   ✓ Input device {i}: {device_info.get('name')}")
            
            if not input_devices:
                print("   ⚠ No input devices found")
            else:
                print(f"   ✓ Found {len(input_devices)} input devices")
            
            audio.terminate()
            
        except ImportError:
            print("   ℹ PyAudio not available for device listing")
        except Exception as e:
            print(f"   ⚠ Could not list devices: {e}")
            
        return True
        
    except Exception as e:
        print(f"   ❌ Microphone test failed: {e}")
        return False

if __name__ == "__main__":
    print("🎤 STT (Speech-to-Text) Test\n")
    
    # Test microphone devices
    device_test = test_microphone_devices()
    
    # Test STT functionality
    stt_test = test_stt()
    
    print("\n" + "="*50)
    if device_test and stt_test:
        print("🎉 All STT tests passed!")
        print("The speech recognition should work properly now.")
    elif device_test:
        print("⚠️  Microphone works but speech recognition failed.")
        print("Check your internet connection and try again.")
    else:
        print("❌ STT tests failed. Check microphone permissions and settings.")
    
    print("\nVoice commands you can try:")
    print("- 'Start analysis'")
    print("- 'Help'")
    print("- 'Read text'")
    print("- 'Describe'")
    print("- 'Stop'") 