import numpy as np
from scipy import signal

class EEGSonifier:
    def __init__(self, sampling_rate=44100, eeg_sampling_rate=256):
        # Audio sampling rate set to 44.1kHz (CD quality) to ensure
        # full audible spectrum representation (20Hz-20kHz) while keeping
        # file sizes manageable compared to higher rates like 48kHz or 96kHz
        self.sampling_rate = sampling_rate
        
        # Default EEG sampling rate of 256Hz matches clinical standards
        # and captures frequencies up to 128Hz (Nyquist limit),
        # which includes all neurophysiologically relevant oscillations
        self.eeg_sampling_rate = eeg_sampling_rate
        
    def simple_tone_mapping(self, eeg_data, duration=10.0):
        """
        Map EEG amplitude to frequency of a simple sine wave.
        
        This direct amplitude-to-frequency mapping leverages the human auditory
        system's sensitivity to pitch variations (better than ~0.5% difference detection)
        to represent subtle EEG amplitude changes that might be missed visually.
        
        Sine waves are used because they contain no harmonics, creating a pure
        representation of the signal without additional spectral artifacts.
        """
        # Interpolation necessary to bridge the large gap between EEG sampling rate
        # (typically 256Hz) and audio rate (44100Hz) while preserving temporal dynamics
        t_eeg = np.arange(len(eeg_data)) / self.eeg_sampling_rate
        t_audio = np.arange(int(duration * self.sampling_rate)) / self.sampling_rate
        eeg_interpolated = np.interp(t_audio, t_eeg, eeg_data)
        
        # Normalization to 0-1 range ensures consistent mapping regardless of
        # absolute EEG amplitude, which can vary widely between subjects/recordings
        normalized_eeg = (eeg_interpolated - np.min(eeg_interpolated)) / \
                        (np.max(eeg_interpolated) - np.min(eeg_interpolated))
        
        # 100-1000Hz range chosen because:
        # - Below 100Hz is difficult to perceive on normal speakers
        # - Above 1000Hz can become shrill and fatiguing
        # - This range falls within the peak sensitivity of human hearing (2-5kHz)
        # - Provides 3.3 octaves of range (log2(1000/100)) for good perceptual discrimination
        freq_mapped = 100 + normalized_eeg * 900
        
        # Frequency modulation implementation via phase accumulation
        # rather than direct sine generation preserves phase continuity
        # which prevents audible clicks/pops at transition points
        phase = np.cumsum(2 * np.pi * freq_mapped / self.sampling_rate)
        
        # Amplitude of 0.5 prevents clipping while allowing headroom
        # for combining multiple sounds in more complex sonifications
        audio = 0.5 * np.sin(phase)
        
        return audio
        
    def multi_band_sonification(self, band_data):
        """
        Sonify multiple frequency bands with different timbres.
        
        This approach leverages auditory scene analysis principles:
        human ability to distinguish and track multiple sound sources simultaneously
        is much better than visual ability to track multiple moving objects.
        
        Different waveforms for different bands creates distinct auditory streams
        that can be selectively attended to, similar to how we can focus on
        a single conversation in a crowded room (cocktail party effect).
        """
        # Fixed 10-second duration provides consistent output length
        # regardless of input data length, allowing easier comparison
        audio_data = np.zeros(int(10 * self.sampling_rate))
        
        # Each band gets its own distinct timbre and pitch range
        # to create auditory scene segregation (ASA)
        for i, (band_name, band_signal) in enumerate(band_data.items()):
            # Interpolation as in simple mapping, but for each band separately
            t_eeg = np.arange(len(band_signal)) / self.eeg_sampling_rate
            t_audio = np.arange(len(audio_data)) / self.sampling_rate
            band_interpolated = np.interp(t_audio, t_eeg, band_signal)
            
            # Normalization per band emphasizes the relative dynamics
            # within each frequency range, regardless of absolute power
            band_normalized = (band_interpolated - np.min(band_interpolated)) / \
                            (np.max(band_interpolated) - np.min(band_interpolated))
            
            # Staggered base frequencies create spectral separation between bands
            # to improve auditory stream segregation, following principles
            # from Bregman's Auditory Scene Analysis
            base_freq = 100 * (i + 1)
            
            # Timbre selection for each band considers:
            # 1. Perceptual associations with the cognitive states linked to each band
            # 2. Spectral characteristics that minimize masking between bands
            # 3. Amplitude levels balanced to compensate for perceptual loudness differences
            
            if band_name == 'delta':
                # Low-frequency sine for delta (0.5-4Hz) reflects its slow, 
                # smooth oscillatory nature associated with deep sleep
                # Small frequency deviation (50Hz) matches its subtle, slow character
                phase = np.cumsum(2 * np.pi * (base_freq + band_normalized * 50) / self.sampling_rate)
                wave = 0.15 * np.sin(phase)
            elif band_name == 'theta':
                # Triangle wave (implemented as 0.5 duty cycle sawtooth) for theta (4-8Hz)
                # contains odd harmonics but with faster roll-off than square waves,
                # creating a softer sound that reflects theta's association with 
                # drowsiness and meditative states
                phase = np.cumsum(2 * np.pi * (base_freq + band_normalized * 100) / self.sampling_rate)
                wave = 0.1 * signal.sawtooth(phase, 0.5)
            elif band_name == 'alpha':
                # Square wave for alpha (8-13Hz) has strong harmonics creating a hollow sound
                # that stands out in the mix, reflecting alpha's prominence during
                # relaxed wakefulness and its larger amplitude over occipital regions
                phase = np.cumsum(2 * np.pi * (base_freq + band_normalized * 200) / self.sampling_rate)
                wave = 0.1 * signal.square(phase, duty=0.5)
            elif band_name == 'beta':
                # Sawtooth for beta (13-30Hz) creates a bright, buzzy sound appropriate for
                # its association with active thinking and alertness
                # With rich harmonic content that cuts through the mix
                phase = np.cumsum(2 * np.pi * (base_freq + band_normalized * 300) / self.sampling_rate)
                wave = 0.05 * signal.sawtooth(phase)
            elif band_name == 'gamma':
                # Noise-modulated sine for gamma (30-100Hz) creates complexity that
                # reflects gamma's association with higher cognitive processing
                # Noise component represents gamma's often transient and spatially specific nature
                phase = np.cumsum(2 * np.pi * (base_freq + band_normalized * 400) / self.sampling_rate)
                noise = np.random.normal(0, 0.5, len(phase))
                wave = 0.05 * np.sin(phase) * (0.5 + 0.5 * noise)
                
            # Simple additive synthesis combines all bands
            # More complex spatialization or filtering could be added here
            audio_data += wave
            
        # Final normalization prevents clipping while maximizing dynamic range
        # The 0.9 factor provides headroom to account for intersample peaks
        audio_data = audio_data / np.max(np.abs(audio_data)) * 0.9
        
        return audio_data
        
    def play_audio(self, audio_data):
        """
        Play audio data through sounddevice.
        Real-time playback allows immediate auditory feedback,
        very essential for developing intuitive understanding of EEG patterns.
        """
        sd.play(audio_data, self.sampling_rate)
        sd.wait()
        
    def save_audio(self, audio_data, filename):
        """
        Save audio data to WAV file.
        16-bit PCM encoding is good for quality and compatibility
        """
        from scipy.io import wavfile
        # Scale to 16-bit integer range (-32768 to 32767)
        # and convert to int16 data type required for standard WAV format
        wavfile.write(filename, self.sampling_rate, (audio_data * 32767).astype(np.int16))