from flask import Flask, render_template_string, request, jsonify
import sounddevice as sd
import soundfile as sf
import tempfile
import os
import threading
import numpy as np
import time
from emotional_model import get_emotion_prediction, EmotionalModel
import torch
app = Flask(__name__)


class EmotionClassifier:
    def __init__(self, model):
        # Audio recording parameters
        self.sample_rate = 48000
        self.duration = 3
        self.is_recording = False
        self.temp_file = None
        self.model = model

    def record_audio(self):
        """Record audio for specified duration"""
        try:

            # Record audio
            audio_data = sd.rec(
                int(self.duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
            )

            sd.wait()  # Wait until recording is finished

            # Save to temporary file
            self.temp_file = tempfile.mktemp(suffix=".wav")
            sf.write(self.temp_file, audio_data, self.sample_rate)

            print(f"Audio saved to: {self.temp_file}")
            return self.temp_file

        except Exception as e:
            print(f"Failed to record audio: {str(e)}")
            return None

    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_file and os.path.exists(self.temp_file):
            os.remove(self.temp_file)
            self.temp_file = None


# Global classifier instance
model = EmotionalModel(8)
model.load_state_dict(torch.load("models/model_59.pth"))
classifier = EmotionClassifier(model)

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Emotion Classifier</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: white;
        }
        
        .container {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            border: 1px solid rgba(255, 255, 255, 0.18);
        }
        
        h1 {
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
            font-weight: 300;
        }
        
        .subtitle {
            text-align: center;
            margin-bottom: 30px;
            opacity: 0.8;
            font-size: 1.1em;
        }
        
        .record-section {
            text-align: center;
            margin: 30px 0;
        }
        
        .record-btn {
            background: linear-gradient(45deg, #ff6b6b, #ee5a24);
            border: none;
            padding: 15px 30px;
            font-size: 18px;
            color: white;
            border-radius: 50px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px 0 rgba(255, 107, 107, 0.3);
            margin: 10px;
        }
        
        .record-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px 0 rgba(255, 107, 107, 0.4);
        }
        
        .record-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        
        .status {
            text-align: center;
            margin: 20px 0;
            font-size: 1.2em;
            min-height: 30px;
        }
        
        .progress-container {
            width: 100%;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 25px;
            margin: 20px 0;
            overflow: hidden;
        }
        
        .progress-bar {
            height: 10px;
            background: linear-gradient(45deg, #4ecdc4, #44a08d);
            width: 0%;
            transition: width 0.1s ease;
            border-radius: 25px;
        }
        
        .result-section {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            text-align: center;
        }
        
        .emotion-result {
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        
        .emotion-happy { color: #feca57; }
        .emotion-sad { color: #48cae4; }
        .emotion-angry { color: #ff6b6b; }
        .emotion-fearful { color: #a55eea; }
        .emotion-disgusted { color: #26de81; }
        .emotion-surprised { color: #fd79a8; }
        .emotion-neutral { color: #ddd; }
        
        .instructions {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 15px;
            margin: 20px 0;
            text-align: center;
            font-size: 0.9em;
            opacity: 0.8;
        }
        
        .error {
            color: #ff6b6b;
            background: rgba(255, 107, 107, 0.1);
            border: 1px solid rgba(255, 107, 107, 0.3);
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Emotion Classifier</h1>
        <div class="subtitle">AI-powered emotion recognition from your voice</div>
        
        <div class="instructions">
            Click "Record Audio" to capture 3 seconds of audio for emotion analysis.
            Speak clearly into your microphone when recording starts.
        </div>
        
        <div class="record-section">
            <button id="recordBtn" class="record-btn" onclick="toggleRecording()">
                🎤 Record Audio (3s)
            </button>
        </div>
        
        <div class="status" id="status">Ready to record</div>
        
        <div class="progress-container">
            <div class="progress-bar" id="progressBar"></div>
        </div>
        
        <div class="result-section">
            <div style="opacity: 0.7; margin-bottom: 10px;">Predicted Emotion:</div>
            <div class="emotion-result" id="emotionResult">No prediction yet</div>
        </div>
        
        <div id="errorContainer"></div>
    </div>

    <script>
        let isRecording = false;
        let progressInterval;
        
        function toggleRecording() {
            if (!isRecording) {
                startRecording();
            }
        }
        
        async function startRecording() {
            const recordBtn = document.getElementById('recordBtn');
            const status = document.getElementById('status');
            const progressBar = document.getElementById('progressBar');
            const errorContainer = document.getElementById('errorContainer');
            
            // Clear any previous errors
            errorContainer.innerHTML = '';
            
            isRecording = true;
            recordBtn.disabled = true;
            recordBtn.textContent = '🎙️ Recording...';
            status.textContent = 'Recording... Speak now!';
            
            // Start progress animation
            let progress = 0;
            progressInterval = setInterval(() => {
                progress += (100 / 30); // 3 seconds = 30 intervals of 100ms
                progressBar.style.width = Math.min(progress, 100) + '%';
                
                if (progress >= 100) {
                    clearInterval(progressInterval);
                }
            }, 100);
            
            try {
                const response = await fetch('/record', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    }
                });
                
                const data = await response.json();
                
                if (data.success) {
                    displayEmotion(data.emotion);
                    status.textContent = 'Prediction complete!';
                } else {
                    showError('Recording failed: ' + data.error);
                    status.textContent = 'Recording failed';
                }
            } catch (error) {
                showError('Network error: ' + error.message);
                status.textContent = 'Error occurred';
            }
            
            // Reset UI
            isRecording = false;
            recordBtn.disabled = false;
            recordBtn.textContent = '🎤 Record Audio (3s)';
            progressBar.style.width = '0%';
            clearInterval(progressInterval);
        }
        
        function displayEmotion(emotion) {
            const emotionResult = document.getElementById('emotionResult');
            emotionResult.textContent = emotion;
            emotionResult.className = 'emotion-result emotion-' + emotion.toLowerCase();
        }
        
        function showError(message) {
            const errorContainer = document.getElementById('errorContainer');
            errorContainer.innerHTML = `<div class="error">⚠️ ${message}</div>`;
        }
        
        // Update status ready message after a delay
        setTimeout(() => {
            if (!isRecording) {
                document.getElementById('status').textContent = 'Ready to record';
            }
        }, 1000);
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    """Serve the main interface"""
    return render_template_string(HTML_TEMPLATE)


@app.route("/record", methods=["POST"])
def record_audio():
    """Handle audio recording and prediction"""
    try:
        # Record audio
        filename = classifier.record_audio()

        if filename is None:
            return jsonify({"success": False, "error": "Failed to record audio"})

        # Get prediction
        try:
            emotion = get_emotion_prediction(classifier.model, filename)
            print(emotion)
            classifier.cleanup()  # Clean up the temporary file

            return jsonify(
                {
                    "success": True,
                    "emotion": emotion,
                    "message": f"Predicted emotion: {emotion}",
                }
            )
        except Exception as e:
            classifier.cleanup()
            return jsonify({"success": False, "error": f"Prediction failed: {str(e)}"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


def main():
    """Main function to run the web application"""
    print("Starting Emotion Classifier Web Interface...")
    print("Open your browser and go to: http://localhost:5000")
    print("Press Ctrl+C to stop the server")

    # Check if required packages are available
    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError as e:
        print(f"Missing required audio package: {e}")
        print("Please install: pip install sounddevice soundfile")
        return

    # Run the Flask app
    app.run(debug=True, host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()
