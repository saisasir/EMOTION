export interface EmotionResult {
  emotion: string;
  confidence: number;
  audioResponse?: string;
  text_response?: string; // ✅ API will return this
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

// 🎨 Emotion color for each mood
export const getEmotionColor = (emotion: string): string => {
  switch (emotion.toLowerCase()) {
    case 'happy': return 'bg-emotion-happy text-black';
    case 'sad': return 'bg-emotion-sad text-white';
    case 'angry': return 'bg-emotion-angry text-white';
    case 'neutral': return 'bg-emotion-neutral text-black';
    case 'fear': return 'bg-emotion-fear text-white';
    case 'surprised': return 'bg-emotion-surprised text-white';
    case 'disgusted': return 'bg-emotion-disgusted text-white';
    default: return 'bg-gray-200 text-black';
  }
};

// 😊 Icon for each emotion
export const getEmotionIcon = (emotion: string): string => {
  switch (emotion.toLowerCase()) {
    case 'happy': return '😊';
    case 'sad': return '😢';
    case 'angry': return '😠';
    case 'neutral': return '😐';
    case 'fear': return '😨';
    case 'surprised': return '😲';
    case 'disgusted': return '🤢';
    default: return '🤔';
  }
};

// 💬 Friendly text response per emotion
export const getEmotionResponse = (emotion: string): EmotionResponse => {
  const responses: Record<string, EmotionResponse> = {
    happy:     { text: "That's wonderful! I'm here to make your day even brighter." },
    sad:       { text: "It's okay to feel down. Let's find something uplifting together." },
    angry:     { text: "I hear your frustration. Let's work through it calmly." },
    neutral:   { text: "Ready to assist. How can I help you today?" },
    fear:      { text: "You're safe. I'm right here with you. Let's figure it out." },
    surprised: { text: "Wow! That caught you off guard, huh? Let's explore it." },
    disgusted: { text: "Hmm, that didn't sit well with you. I understand. Let me help." }
  };

  return responses[emotion.toLowerCase()] || {
    text: "I'm here to assist, whatever you feel. Let's begin."
  };
};

// 🌍 Backend API URL from Vite env
export const API_URL = import.meta.env.VITE_API_BASE_URL;

// 🔗 Main prediction function
export const predictEmotion = async (audio: Blob): Promise<EmotionResult> => {
  try {
    console.log("📡 Sending audio to API:", API_URL);
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
    console.log("✅ API response:", data);

    return {
      emotion: data.emotion,
      confidence: data.confidence,
      audioResponse: data.audio_base64,
      text_response: data.text_response,
    };
  } catch (error) {
    console.error('❌ Error calling API, using mock data:', error);
    return mockPredictEmotion(audio);
  }
};

// 🧪 Mock fallback for offline testing
export const mockPredictEmotion = async (audio: Blob): Promise<EmotionResult> => {
  return new Promise((resolve) => {
    setTimeout(() => {
      const randomIndex = Math.floor(Math.random() * EMOTIONS.length);
      const emotion = EMOTIONS[randomIndex];
      const confidence = 70 + Math.random() * 30;

      // Base64 dummy (1-sec silent audio, plays nothing)
      const mockAudioBase64 =
        "UklGRiQAAABXQVZFZm10IBAAAAABAAEAIlYAAESsAAACABAAZGF0YQAAAAA=";

      resolve({
        emotion,
        confidence,
        audioResponse: mockAudioBase64,
        text_response: getEmotionResponse(emotion).text
      });
    }, 2000);
  });
};
