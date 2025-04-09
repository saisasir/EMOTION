export interface EmotionResult {
  emotion: string;
  confidence: number;
  audioResponse?: string;
  text_response?: string; // ✅ Add this line if you need the text response
}

export interface EmotionResponse {
  text: string;
  audioUrl?: string;
}

export const EMOTIONS = [
  'happy',
  'sad',
  'angry',
  'neutral',
  'fear',
  'surprised',
  'disgusted'
];

export const getEmotionColor = (emotion: string): string => {
  switch (emotion.toLowerCase()) {
    case 'happy':
      return 'bg-emotion-happy text-black';
    case 'sad':
      return 'bg-emotion-sad text-white';
    case 'angry':
      return 'bg-emotion-angry text-white';
    case 'neutral':
      return 'bg-emotion-neutral text-black';
    case 'fear':
      return 'bg-emotion-fear text-white';
    case 'surprised':
      return 'bg-emotion-surprised text-white';
    case 'disgusted':
      return 'bg-emotion-disgusted text-white';
    default:
      return 'bg-gray-200 text-black';
  }
};

export const getEmotionIcon = (emotion: string): string => {
  switch (emotion.toLowerCase()) {
    case 'happy':
      return '😊';
    case 'sad':
      return '😢';
    case 'angry':
      return '😠';
    case 'neutral':
      return '😐';
    case 'fear':
      return '😨';
    case 'surprised':
      return '😲';
    case 'disgusted':
      return '🤢';
    default:
      return '🤔';
  }
};

export const getEmotionResponse = (emotion: string): EmotionResponse => {
  const responses: Record<string, EmotionResponse> = {
    happy: {
      text: "That's wonderful! I'm here to make your day even brighter."
    },
    sad: {
      text: "It's okay to feel down. Let's find something uplifting together."
    },
    angry: {
      text: "I hear your frustration. Let's work through it calmly."
    },
    neutral: {
      text: "Ready to assist. How can I help you today?"
    },
    fear: {
      text: "You're safe. I'm right here with you. Let's figure it out."
    },
    surprised: {
      text: "Wow! That caught you off guard, huh? Let's explore it."
    },
    disgusted: {
      text: "Hmm, that didn't sit well with you. I understand. Let me help."
    }
  };

  return responses[emotion.toLowerCase()] || {
    text: "I'm here to assist, whatever you feel. Let's begin."
  };
};

// Backend API URL - Update with your deployed backend URL
export const API_URL = "https://your-flask-app.onrender.com";


// Real prediction function that calls the backend API
export const predictEmotion = async (audio: Blob): Promise<EmotionResult> => {
  try {
    console.log("Sending audio to API:", API_URL);
    const formData = new FormData();
    formData.append('audio_file', audio, 'audio.wav');
    
    const response = await fetch(`${API_URL}/predict-emotion`, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    
    const data = await response.json();
    console.log("API response:", data);
    
    return {
      emotion: data.emotion,
      confidence: data.confidence,
      audioResponse: data.audio_base64  // Store the audio response from the backend
    };
  } catch (error) {
    console.error('Error calling emotion recognition API:', error);
    // Fallback to mock data if the API call fails
    return mockPredictEmotion(audio);
  }
};

// This is a mock function simulating the emotion recognition model
export const mockPredictEmotion = async (audio: Blob): Promise<EmotionResult> => {
  console.log("Using mock prediction");
  // In a real app, you would send the audio to a server for processing
  // For demo purposes, we'll return a random emotion with random confidence
  
  return new Promise((resolve) => {
    // Simulate processing time
    setTimeout(() => {
      const randomIndex = Math.floor(Math.random() * EMOTIONS.length);
      const emotion = EMOTIONS[randomIndex];
      const confidence = 70 + Math.random() * 30; // Random confidence between 70-100%
      
      // For testing, create a mock base64 audio response
      // This is a very short MP3 file encoded as base64
      const mockAudioBase64 = "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4Ljc2LjEwMAAAAAAAAAAAAAAA//tAwAAAAAAAAAAAAAAAAAAAAAAAWGluZwAAAA8AAAACAAAFRgDMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzM//////////////////////////////////////////////////////////////////8AAAAATGF2YzU4LjEzAAAAAAAAAAAAAAAAJAYHkgAAAAAAAAAAAAAAAAAAAP/7UMQAAAesTXWUEQAB0CG7xnJAAIAmSR1e6v+/1S0UjOlKJYZA+KAhO4YDYBgobydPRN6JYeTsA8AH/g8HP4P/93oIH/cH/qB8Hw/8EAQD/8H/8ugJn+C4If//qBj/+JLHLUWWGWSWaTI//QokrcHhUHIZFMXQUdBt9Q2a3/7UMQMgAgZeXmvPHPZBiVu9veWsDYaVxwKYUiTDPI6E+JkR1HaE0QM0uJojTJZclx4P4fxGh///+7s3////r+GaRWZZ5JJprAcZcTqlAaGfACrOIjUJ0xseuClqWCmW9Wt6tburlz63/9fXqx3a30v/7UMQMgAfdG2elpLGJC6PtdKyWMXev/6/pXfX99f+v6sy/1f////6/n//9f/r//r/r/X/31//1/R2/+++v/X/W/T9H9U/zdMQU1FMy45OS41VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVcf/7UMQPAAf9SXnnnTGo8KTtvPSOoFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV";
      
      resolve({
        emotion,
        confidence,
        audioResponse: mockAudioBase64
      });
    }, 2000);
  });
};
