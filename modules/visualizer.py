import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy import signal

class EEGVisualizer:
    def __init__(self, sampling_rate=256):
        # Default to 256 Hz as it's a common clinical EEG sampling rate that 
        # captures frequencies up to 128 Hz (Nyquist), covering all relevant brain rhythms
        self.sampling_rate = sampling_rate
        
    def plot_time_series(self, eeg_data, time_window=10):
        """
        Plot EEG data in time domain to observe temporal patterns.
        Time domain visualization is essential for identifying transient events 
        like spikes, vertex waves, and other morphological features that
        frequency analysis might obscure.
        """
        fig, ax = plt.subplots(figsize=(10, 4))
        # Limit samples to prevent memory issues with large datasets
        # while maintaining temporal resolution appropriate for visual analysis
        samples = min(len(eeg_data), time_window * self.sampling_rate)
        time = np.arange(samples) / self.sampling_rate
        ax.plot(time, eeg_data[:samples])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (μV)')
        ax.set_title('EEG Time Series')
        ax.grid(True)
        return fig
        
    def plot_power_spectrum(self, freqs, psd):
        """
        Plot power spectral density using logarithmic scale to visualize 
        frequency distribution of EEG power.
        
        Logarithmic scale is crucial for EEG as power follows roughly 1/f distribution,
        and clinical bands span several orders of magnitude in power.
        
        Limited to 50 Hz as higher frequencies rarely contain clinically
        relevant information and often contain EMG artifacts.
        """
        fig, ax = plt.subplots(figsize=(10, 4))
        # Focus on frequencies below 50 Hz where most clinically 
        # relevant EEG activity occurs
        mask = freqs <= 50
        ax.semilogy(freqs[mask], psd[mask])
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power Spectral Density (μV²/Hz)')
        ax.set_title('Power Spectral Density')
        
        # Add frequency band shading to highlight canonical EEG rhythms
        # that correspond to different physiological and cognitive states
        bands = [
            ('Delta', 0.5, 4, 'lightblue'),    # Deep sleep, unconsciousness
            ('Theta', 4, 8, 'lightgreen'),     # Drowsiness, meditation
            ('Alpha', 8, 13, 'lightsalmon'),   # Relaxed wakefulness
            ('Beta', 13, 30, 'lightpink'),     # Active thinking, focus
            ('Gamma', 30, 50, 'lightyellow')   # Cognitive processing, perception
        ]
        
        for name, fmin, fmax, color in bands:
            # Span visualization helps clinicians quickly identify 
            # which frequency bands show abnormal activity
            ax.axvspan(fmin, fmax, color=color, alpha=0.3, label=name)
            
        ax.legend()
        ax.grid(True)
        return fig
        
    def create_spectrogram(self, eeg_data):
        """
        Create a time-frequency representation via spectrogram to reveal
        how frequency content changes over time.
        
        This visualization is essential for identifying transient spectral events
        like sleep spindles, seizure evolution, and other non-stationary phenomena
        that would be averaged out in simple spectral analysis.
        """
        fig, ax = plt.subplots(figsize=(10, 4))
        # Segment size balances frequency resolution vs temporal precision
        # Shorter segments give better temporal resolution but poorer frequency resolution
        nperseg = min(256, len(eeg_data) // 10)
        f, t, Sxx = signal.spectrogram(eeg_data, fs=self.sampling_rate, nperseg=nperseg)
        
        # Plot only up to 50 Hz to focus on clinically relevant frequencies
        # and reduce computational load
        mask = f <= 50
        # Use logarithmic color scale (dB) to visualize wide dynamic range
        # of EEG power across frequencies
        pcm = ax.pcolormesh(t, f[mask], 10 * np.log10(Sxx[mask]), shading='gouraud', cmap='viridis')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (s)')
        ax.set_title('EEG Spectrogram')
        fig.colorbar(pcm, ax=ax, label='Power (dB)')
        return fig
        
    def plot_animated_eeg(self, eeg_data, window_size=5):
        """
        Create animated visualization function for real-time EEG monitoring.
        
        The windowed approach mimics clinical EEG display systems where
        a moving window of data is shown to track brain activity over time
        while maintaining temporal resolution necessary for event detection.
        
        Returns a callable function that Streamlit can use for animation.
        """
        # Pre-calculate window size in samples for efficiency
        samples_per_window = int(window_size * self.sampling_rate)
        
        def update_plot(time_idx):
            # Calculate window boundaries based on current position
            # This sliding window approach emulates clinical EEG review workflow
            start_idx = max(0, time_idx - samples_per_window)
            end_idx = min(len(eeg_data), time_idx)
            
            # Handle edge case of empty window
            if start_idx == end_idx:
                return None
                
            fig, ax = plt.subplots(figsize=(10, 3))
            time = np.arange(start_idx, end_idx) / self.sampling_rate
            ax.plot(time, eeg_data[start_idx:end_idx])
            # Fixed width window maintains consistent temporal scale
            # for accurate pattern recognition
            ax.set_xlim(time[0], time[0] + window_size)
            
            # Dynamic y-axis scaling based on overall signal amplitude
            # prevents clipping while maintaining sensitivity to smaller features
            signal_range = np.max(np.abs(eeg_data)) * 1.2
            ax.set_ylim(-signal_range, signal_range)
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude (μV)')
            ax.set_title('Real-time EEG Visualization')
            ax.grid(True)
            return fig
            
        return update_plot
