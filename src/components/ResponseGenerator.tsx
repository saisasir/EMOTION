import React, { useState, useEffect, useRef } from 'react';
import { EmotionResult, getEmotionResponse, getEmotionIcon } from '@/utils/emotionUtils';
import { Button } from '@/components/ui/button';
import { Volume2, VolumeX } from 'lucide-react';
import { toast } from 'sonner';

interface ResponseGeneratorProps {
  emotionResult: EmotionResult | null;
  audioData?: string | null;
  text?: string | null;
}

const ResponseGenerator: React.FC<ResponseGeneratorProps> = ({ emotionResult, audioData, text }) => {
  const [response, setResponse] = useState<string>('');
  const [isPlaying, setIsPlaying] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);

  // Handle the text response
  useEffect(() => {
    if (emotionResult) {
      setResponse(text ?? getEmotionResponse(emotionResult.emotion).text);
    }
  }, [emotionResult, text]);

  // Handle the base64 audio response
  useEffect(() => {
    if (audioData) {
      try {
        const blob = base64ToBlob(audioData, 'audio/mp3');
        const url = URL.createObjectURL(blob);
        setAudioUrl(url);

        const newAudio = new Audio(url);
        audioRef.current = newAudio;

        newAudio.addEventListener(
          'canplaythrough',
          () => {
            if (!newAudio.played.length) {
              newAudio
                .play()
                .then(() => {
                  console.log('✅ Auto-played AI response');
                  setIsPlaying(true);
                })
                .catch(() => {
                  console.warn('⚠️ Auto-play blocked by browser');
                  toast.warning('Click play to hear the AI response');
                });
            }
          },
          { once: true }
        );

        return () => {
          URL.revokeObjectURL(url);
          newAudio.pause();
          newAudio.src = '';
          audioRef.current = null;
        };
      } catch (error) {
        console.error('❌ Error processing base64 audio:', error);
        toast.error('Failed to load audio response');
      }
    }
  }, [audioData]);

  // Convert base64 to Blob
  const base64ToBlob = (base64: string, mimeType: string): Blob => {
    try {
      const byteString = atob(base64);
      const arrayBuffer = new ArrayBuffer(byteString.length);
      const intArray = new Uint8Array(arrayBuffer.byteLength);
      for (let i = 0; i < byteString.length; i++) {
        intArray[i] = byteString.charCodeAt(i);
      }
      return new Blob([intArray], { type: mimeType });
    } catch (e) {
      console.error("❌ base64 decode failed", e);
      throw new Error("Invalid base64 input");
    }
  };

  // Handle play/stop button
  const handlePlayAudio = () => {
    if (!audioUrl) {
      toast.error('No audio response available');
      return;
    }

    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
        setIsPlaying(false);
      } else {
        audioRef.current
          .play()
          .then(() => setIsPlaying(true))
          .catch((err) => {
            console.error('❌ Error playing audio:', err);
            toast.error('Failed to play audio');
          });
      }
    }
  };

  if (!emotionResult) return null;

  const emotionIcon = getEmotionIcon(emotionResult.emotion);

  return (
    <div className="w-full glassmorphism-card animate-fade-up">
      <div className="flex items-start space-x-4">
        <div className="flex-shrink-0 w-12 h-12 rounded-full bg-blue-100 flex items-center justify-center">
          <span className="text-2xl">{emotionIcon}</span>
        </div>

        <div className="flex-1">
          <h3 className="text-lg font-medium mb-2">AI Response</h3>
          <p className="text-gray-700 mb-4">{response}</p>

          <div className="flex justify-end">
            <Button
              variant="outline"
              size="sm"
              className="flex items-center space-x-2"
              onClick={handlePlayAudio}
              disabled={!audioUrl}
            >
              {isPlaying ? (
                <>
                  <VolumeX className="h-4 w-4" />
                  <span>Stop</span>
                </>
              ) : (
                <>
                  <Volume2 className="h-4 w-4" />
                  <span>Play</span>
                </>
              )}
            </Button>
          </div>
        </div>
      </div>

      <audio
        ref={audioRef}
        src={audioUrl || undefined}
        onEnded={() => setIsPlaying(false)}
        className="hidden"
      />
    </div>
  );
};

export default ResponseGenerator;
