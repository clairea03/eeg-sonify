# signal_processor.py
import numpy as np
from scipy import signal
import mne

class SignalProcessor:
    def __init__(self, sampling_rate=256):
        self.sampling_rate = sampling_rate
        
    def apply_bandpass_filter(self, eeg_data, low_freq, high_freq):
        """Apply bandpass filter to EEG data"""
        nyquist = 0.5 * self.sampling_rate
        low = low_freq / nyquist
        high = high_freq / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_data = signal.filtfilt(b, a, eeg_data)
        return filtered_data
        
    def extract_frequency_bands(self, eeg_data):
        """Extract different frequency bands from EEG data"""
        bands = {
            'delta': self.apply_bandpass_filter(eeg_data, 0.5, 4),
            'theta': self.apply_bandpass_filter(eeg_data, 4, 8),
            'alpha': self.apply_bandpass_filter(eeg_data, 8, 13),
            'beta': self.apply_bandpass_filter(eeg_data, 13, 30),
            'gamma': self.apply_bandpass_filter(eeg_data, 30, 100)
        }
        return bands
        
    def compute_power_spectral_density(self, eeg_data):
        """Compute power spectral density of EEG data"""
        freqs, psd = signal.welch(eeg_data, fs=self.sampling_rate, 
                                  nperseg=self.sampling_rate)
        return freqs, psd
        
    def compute_hjorth_parameters(self, eeg_data):
        """Compute Hjorth parameters: Activity, Mobility, and Complexity"""
        # First derivative
        diff1 = np.diff(eeg_data, 1)
        # Second derivative
        diff2 = np.diff(eeg_data, 2)
        
        # Pad the derivatives to match original data length
        diff1 = np.pad(diff1, (0, 1), 'constant')
        diff2 = np.pad(diff2, (0, 2), 'constant')
        
        # Calculate Activity (variance of the signal)
        activity = np.var(eeg_data)
        
        # Calculate Mobility
        mobility = np.sqrt(np.var(diff1) / activity)
        
        # Calculate Complexity
        complexity = np.sqrt(np.var(diff2) / np.var(diff1)) / mobility
        
        return activity, mobility, complexity
