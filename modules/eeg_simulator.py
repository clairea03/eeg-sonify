import numpy as np
from scipy import signal

class EEGSimulator:
    def __init__(self, sampling_rate=256, duration=10):
        # 256 Hz is a standard clinical EEG sampling rate providing adequate 
        # frequency resolution (up to 128 Hz via Nyquist) while minimizing storage requirements
        self.sampling_rate = sampling_rate
        # 10 seconds is long enough to capture multiple cycles of all relevant EEG rhythms
        # (even the slowest delta waves at 0.5 Hz complete 5 cycles in this duration)
        self.duration = duration
        # Pre-compute time array for efficiency in all generation methods
        self.time = np.arange(0, duration, 1/sampling_rate)
        
    def generate_band_limited_noise(self, min_freq, max_freq, amplitude=1.0):
        """
        Generate band-limited noise for a specific frequency band.
        
        Neurophysiologically, EEG rhythms arise from synchronized oscillations 
        of neuronal populations with some variability, making filtered noise
        a more realistic model than pure sinusoids. This approach creates
        signals with natural-appearing spectral characteristics that
        better represent the quasi-periodic nature of brain oscillations.
        """
        # Start with Gaussian white noise which has equal power at all frequencies,
        # providing a neutral substrate for filtering
        noise = np.random.normal(0, 1, len(self.time))
        
        # Design 4th-order Butterworth bandpass filter
        # Butterworth chosen for its maximally flat frequency response in the passband,
        # creating minimal distortion of the underlying spectral characteristics
        nyquist = 0.5 * self.sampling_rate
        low = min_freq / nyquist
        high = max_freq / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Apply zero-phase filtering with filtfilt to prevent temporal shifts
        # that would disrupt the time-domain morphology of the signal
        filtered_noise = signal.filtfilt(b, a, noise) * amplitude
        return filtered_noise
        
    def generate_normal_eeg(self):
        """
        Generate normal awake EEG with dominant alpha rhythm.
        
        Awake EEG is characterized by prominent alpha (8-13 Hz) activity,
        especially when eyes are closed, with subdominant contributions from
        other bands. Alpha amplitude is set higher to model its dominance
        in the posterior regions of the brain during relaxed wakefulness.
        """
        # Delta (0.5-4 Hz) - present but attenuated during wakefulness
        delta = self.generate_band_limited_noise(0.5, 4, 0.3)
        # Theta (4-8 Hz) - moderate during quiet wakefulness
        theta = self.generate_band_limited_noise(4, 8, 0.4)
        # Alpha (8-13 Hz) - dominant during relaxed wakefulness, especially with eyes closed
        # Amplitude set higher (1.0) to model its prominence in posterior regions
        alpha = self.generate_band_limited_noise(8, 13, 1.0)
        # Beta (13-30 Hz) - present during active thinking but lower amplitude
        beta = self.generate_band_limited_noise(13, 30, 0.2)
        
        # Simple linear summation models the additive nature of electrical fields
        # from different neuronal populations measured at the scalp
        eeg_signal = delta + theta + alpha + beta
        return eeg_signal

    def generate_sleep_stage_n1(self):
        """
        Generate Stage N1 sleep EEG with vertex waves and theta activity.
        
        N1 sleep is characterized by a shift from alpha to theta dominance,
        along with occasional vertex waves - sharp, high-amplitude waveforms
        occurring near the vertex of the scalp during transition to sleep.
        These transient events are modeled as Gaussian pulses to capture
        their characteristic morphology.
        """
        # Theta becomes dominant in light sleep
        theta = self.generate_band_limited_noise(4, 8, 1.0)
        # Alpha attenuates as alertness decreases
        alpha = self.generate_band_limited_noise(8, 13, 0.3)
        
        # Add occasional vertex waves, characteristic of sleep onset
        # Randomly selecting 5 positions to model their sporadic occurrence
        vertex_wave_times = np.random.choice(len(self.time) - 50, 5, replace=False)
        eeg_signal = theta + alpha
        
        for t in vertex_wave_times:
            # Create a vertex wave shape using a Gaussian pulse
            # which approximates the morphology of these sharp transient events
            wave = np.zeros(len(self.time))
            # Width of 50 samples and sigma of 10 creates a brief (~200ms) sharp wave
            # similar to clinical observations of vertex waves
            wave[t:t+50] = signal.gaussian(50, 10) * 2
            eeg_signal += wave
            
        return eeg_signal
        
    def generate_sleep_stage_n3(self):
        """
        Generate Slow Wave Sleep (N3) with dominant delta.
        
        N3 (deep sleep) is characterized by high-amplitude, low-frequency
        delta waves (0.5-4 Hz) making up >20% of the EEG. This pattern reflects
        widespread cortical synchronization during restorative sleep, with
        reduced higher frequency activity. The amplitude is markedly increased
        compared to wakefulness, reflecting stronger neuronal synchrony.
        """
        # Delta dominates in deep sleep with much higher amplitude (2.0)
        # reflecting the strong synchronization of cortical neurons
        delta = self.generate_band_limited_noise(0.5, 4, 2.0)
        # Some theta activity remains but at lower amplitude
        theta = self.generate_band_limited_noise(4, 8, 0.3)
        
        # Higher frequencies are minimal in deep sleep
        return delta + theta
        
    def generate_rem_sleep(self):
        """
        Generate REM sleep with mixed frequencies and sawtooth waves.
        
        REM sleep EEG resembles wakefulness with mixed frequencies but
        includes characteristic sawtooth waves - serrated waves at 2-6 Hz
        that often precede rapid eye movements. These distinctive patterns
        are modeled as actual sawtooth waveforms superimposed on the
        background of desynchronized activity.
        """
        # REM sleep has prominent theta activity
        theta = self.generate_band_limited_noise(4, 8, 1.0)
        # Alpha is present but reduced compared to wakefulness
        alpha = self.generate_band_limited_noise(8, 13, 0.2)
        # Beta activity reflects the activated brain state during REM
        beta = self.generate_band_limited_noise(13, 30, 0.3)
        
        # Add occasional sawtooth waves - a defining feature of REM sleep
        # 8 occurrences randomly distributed to model their intermittent nature
        sawtooth_times = np.random.choice(len(self.time) - 100, 8, replace=False)
        eeg_signal = theta + alpha + beta
        
        for t in sawtooth_times:
            # Create a sawtooth wave using scipy's sawtooth function
            # which accurately models the asymmetric ramp-like morphology
            wave = np.zeros(len(self.time))
            # 5 Hz frequency matches clinical observations
            # 100 samples duration (~400ms) is typical for these events
            wave[t:t+100] = signal.sawtooth(2 * np.pi * 5 * np.linspace(0, 1, 100)) * 0.5
            eeg_signal += wave
            
        return eeg_signal
        
    def generate_seizure_activity(self):
        """
        Generate seizure-like activity with rhythmic spikes.
        
        Epileptiform activity often manifests as rhythmic spike-wave discharges
        around 3 Hz, particularly in absence seizures. The simulation models
        seizure evolution with three phases: onset (building amplitude),
        established seizure (maintained high amplitude), and termination (fading).
        The spike-wave morphology is approximated by altering Gaussian pulses.
        """
        # Create background activity with broad frequency content
        base = self.generate_band_limited_noise(0.5, 30, 0.2)
        
        # Create rhythmic spike-wave discharges at around 3 Hz
        # which is the characteristic frequency of absence seizures
        t = self.time
        seizure_component = np.zeros_like(t)
        
        # Model seizure evolution with a three-phase amplitude envelope:
        # 1. Onset - gradual build-up (20% of duration)
        # 2. Established seizure - sustained activity (60% of duration)
        # 3. Termination - gradual offset (20% of duration)
        # This models the natural evolution of many seizure types
        amplitude_envelope = np.concatenate([
            np.linspace(0, 1, int(len(t)/5)),     # Onset
            np.ones(int(len(t)/5*3)),             # Established seizure
            np.linspace(1, 0, int(len(t)/5))      # Termination
        ])
        
        # Handle edge cases in envelope length to match time array
        if len(amplitude_envelope) < len(t):
            amplitude_envelope = np.pad(amplitude_envelope, 
                                       (0, len(t) - len(amplitude_envelope)), 
                                       'constant')
        elif len(amplitude_envelope) > len(t):
            # Truncate if too long
            amplitude_envelope = amplitude_envelope[:len(t)]
        
        # Generate 3 Hz spike-wave discharges - the classic pattern in absence seizures
        spike_freq = 3  # Hz
        for i in range(int(self.duration * spike_freq)):
            # Calculate center position for each spike
            center = int(i / spike_freq * self.sampling_rate)
            # Width of approximately 50ms (sampling_rate/20) creates realistic spike duration
            width = int(self.sampling_rate // 20)
            
            # Safety check to avoid array bounds errors
            if center - width < 0 or center + width >= len(seizure_component):
                continue
            
            # Create spike-wave morphology using modified Gaussian
            # The "* 2 - 1" creates the negative wave following the positive spike
            # which is characteristic of spike-wave complexes
            if width > 0:
                spike = signal.gaussian(2*width, width/5) * 2 - 1
                # Ensure spike length matches expected window
                if len(spike) == 2*width:
                    seizure_component[center-width:center+width] += spike
        
        # Apply amplitude envelope to model seizure evolution
        seizure_component *= amplitude_envelope
        
        # Seizure activity is typically much higher amplitude than background
        # The factor of 3 creates the dramatic amplitude increase seen clinically
        return base + seizure_component * 3