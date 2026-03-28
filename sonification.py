import numpy as np

class Sonifier:
    def __init__(self, sample_rate=16000, duration_per_point=0.05):
        """
        sample_rate: 16000Hz (16kHz) for Gemini Live compatibility
        duration_per_point: in seconds, the length of tone for each simulated data point
        """
        self.sample_rate = sample_rate
        self.duration_per_point = duration_per_point
        
    def generate_tone(self, frequency, duration):
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        # Create a smooth tone (sine wave), add a subtle harmonic so it sounds less harsh
        sine_wave = np.sin(frequency * t * 2 * np.pi) + 0.2 * np.sin(2 * frequency * t * 2 * np.pi)
        
        # Apply gentle envelope to avoid clicking sounds between notes
        fade_samples = int(self.sample_rate * 0.01)
        if len(sine_wave) > 2 * fade_samples:
            env = np.ones_like(sine_wave)
            env[:fade_samples] = np.linspace(0, 1, fade_samples)
            env[-fade_samples:] = np.linspace(1, 0, fade_samples)
            sine_wave *= env
            
        return sine_wave

    def sonify(self, series, min_freq=300, max_freq=1200):
        """
        Maps a 1D sequence of numbers to PCM audio.
        Assumes 'series' is a list or numpy array.
        """
        if len(series) == 0:
            return b""
            
        val_min = np.min(series)
        val_max = np.max(series)
        
        audio_stream = []
        for val in series:
            # Map the value linearily into the frequency spectrum
            if val_max == val_min:
                freq = min_freq
            else:
                ratio = (val - val_min) / (val_max - val_min)
                freq = min_freq + ratio * (max_freq - min_freq)
            
            wave = self.generate_tone(freq, self.duration_per_point)
            audio_stream.extend(wave)
            
        # Convert to 16-bit PCM integer format
        audio_stream = np.array(audio_stream)
        # Normalize and scale to int16 maximum
        if np.max(np.abs(audio_stream)) > 0:
             audio_stream = audio_stream / np.max(np.abs(audio_stream))
        
        pcm_data = np.int16(audio_stream * 32767)
        return pcm_data.tobytes()
