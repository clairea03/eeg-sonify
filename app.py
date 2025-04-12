import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import threading
import sounddevice as sd

# Import custom modules - modular design allows separation of concerns
# and better maintainability as each component handles a specific aspect
from modules.eeg_simulator import EEGSimulator
from modules.signal_processor import SignalProcessor
from modules.sonifier import EEGSonifier
from modules.visualizer import EEGVisualizer

# Wide layout better accommodates complex visualizations and controls
# without crowding the interface, improving clinical utility
st.set_page_config(page_title="EEG Sonification Explorer", layout="wide")

# Initialize session state to persist data between Streamlit reruns
# This is critical for maintaining application state during interactive sessions
# since Streamlit reruns the entire script on each interaction
if 'eeg_data' not in st.session_state:
    st.session_state.eeg_data = None
if 'audio_thread' not in st.session_state:
    st.session_state.audio_thread = None

# Title and description
st.title("EEG Sonification Explorer")
st.markdown("""
This application allows you to explore the sonification of EEG signals. 
Generate simulated EEG patterns for different brain states, visualize them, and listen to their sonified representations.
""")

# Sidebar keeps controls separate from visualization area
# improving cognitive ergonomics for clinical and research use
st.sidebar.header("EEG Simulation Controls")

# EEG parameters 
# 256 Hz is standard clinical rate, but higher rates improve 
# frequency resolution for advanced analysis
sampling_rate = st.sidebar.slider("Sampling Rate (Hz)", 100, 1000, 256)
# Duration affects both computational load and ability to observe
# longer-term patterns like sleep cycles or seizure evolution
duration = st.sidebar.slider("Duration (seconds)", 5, 60, 10)

# Brain state selection represents key neurophysiological states
# studied in clinical neurophysiology and sleep medicine
brain_state = st.sidebar.selectbox(
    "Select Brain State",
    ["Normal Awake", "Sleep Stage N1", "Slow Wave Sleep (N3)", 
     "REM Sleep", "Seizure Activity"]
)

# Sonification parameters section
# Different auditory mapping strategies highlight different
# aspects of the underlying neural dynamics
st.sidebar.header("Sonification Controls")
sonification_method = st.sidebar.selectbox(
    "Sonification Method",
    ["Simple Tone Mapping", "Multi-band Sonification"]
)

# Volume control essential for auditory comfort and perceptual tuning
# as some brain patterns may produce harsh sounds at full volume
volume = st.sidebar.slider("Volume", 0.0, 1.0, 0.5)

# Initialize specialized processing classes with consistent sampling rate
# to ensure temporal alignment across all processing stages
simulator = EEGSimulator(sampling_rate=sampling_rate, duration=duration)
processor = SignalProcessor(sampling_rate=sampling_rate)
sonifier = EEGSonifier(eeg_sampling_rate=sampling_rate)
visualizer = EEGVisualizer(sampling_rate=sampling_rate)

# Audio playback function runs in separate thread to prevent
# UI blocking, allowing continued interaction during playback
def play_audio_thread(audio_data, sample_rate):
    sd.play(audio_data, sample_rate)
    sd.wait()

# Generate button initiates the EEG simulation process
# This is separated from visualization to allow users to
# experiment with different parameters before committing computational resources
if st.sidebar.button("Generate EEG Data"):
    with st.spinner("Generating EEG data..."):
        # Each brain state requires different generation algorithms
        # to accurately represent its distinct neurophysiological characteristics
        if brain_state == "Normal Awake":
            eeg_data = simulator.generate_normal_eeg()
        elif brain_state == "Sleep Stage N1":
            eeg_data = simulator.generate_sleep_stage_n1()
        elif brain_state == "Slow Wave Sleep (N3)":
            eeg_data = simulator.generate_sleep_stage_n3()
        elif brain_state == "REM Sleep":
            eeg_data = simulator.generate_rem_sleep()
        elif brain_state == "Seizure Activity":
            eeg_data = simulator.generate_seizure_activity()
        
        # Store in session state for persistence across UI interactions
        st.session_state.eeg_data = eeg_data

# Tabs organize the interface according to user goals:
# visualization, analysis, and educational content
tab1, tab2, tab3 = st.tabs(["Visualization", "Analysis", "About"])

with tab1:
    if st.session_state.eeg_data is not None:
        eeg_data = st.session_state.eeg_data
        
        # Controls organized in columns for intuitive grouping
        # following standard audio player paradigms
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Play Audio"):
                # Different sonification methods highlight different
                # aspects of brain activity patterns
                if sonification_method == "Simple Tone Mapping":
                    # Simpler method focuses on overall amplitude variations
                    audio_data = sonifier.simple_tone_mapping(eeg_data, duration)
                else:  # Multi-band Sonification
                    # More complex method differentiates frequency bands
                    # linked to different cognitive and physiological processes
                    bands = processor.extract_frequency_bands(eeg_data)
                    audio_data = sonifier.multi_band_sonification(bands)
                
                # Volume adjustment for user comfort and perceptual tuning
                audio_data = audio_data * volume
                
                # Always stop current playback before starting new
                # to prevent audio overlap and cacophony
                sd.stop()
                
                # Threaded playback prevents UI freezing
                # crucial for maintaining interactive experience
                st.session_state.audio_thread = threading.Thread(
                    target=play_audio_thread, 
                    args=(audio_data, sonifier.sampling_rate)
                )
                st.session_state.audio_thread.start()
        
        with col2:
            if st.button("Stop Audio"):
                # Immediate audio cessation important for
                # user control, especially with disturbing sounds
                sd.stop()
        
        with col3:
            if st.button("Save Audio"):
                # Audio export for offline analysis or sharing
                # follows the same sonification process as playback
                if sonification_method == "Simple Tone Mapping":
                    audio_data = sonifier.simple_tone_mapping(eeg_data, duration)
                else:  # Multi-band Sonification
                    bands = processor.extract_frequency_bands(eeg_data)
                    audio_data = sonifier.multi_band_sonification(bands)
                
                audio_data = audio_data * volume
                # Filename encodes brain state for easy identification
                filename = f"eeg_sonification_{brain_state.replace(' ', '_')}.wav"
                sonifier.save_audio(audio_data, filename)
                st.success(f"Audio saved as {filename}")
        
        # Multiple visualization modes provide complementary views
        # of the same underlying neural data
        
        # Time series shows amplitude variations directly
        # critical for identifying transient events and morphology
        st.subheader("EEG Time Series")
        st.pyplot(visualizer.plot_time_series(eeg_data))
        
        # Spectrogram reveals time-frequency relationships
        # essential for identifying non-stationary phenomena
        st.subheader("EEG Spectrogram")
        st.pyplot(visualizer.create_spectrogram(eeg_data))
        
        # Power spectrum shows frequency distribution
        # highlighting dominant rhythms characteristic of different brain states
        freqs, psd = processor.compute_power_spectral_density(eeg_data)
        st.subheader("Power Spectral Density")
        st.pyplot(visualizer.plot_power_spectrum(freqs, psd))
    else:
        # Clear instruction for first-time users
        # reduces cognitive load in learning the application
        st.info("Generate EEG data using the controls in the sidebar to begin.")

with tab2:
    if st.session_state.eeg_data is not None:
        eeg_data = st.session_state.eeg_data
        
        # Analysis section provides quantitative metrics
        # to complement visual and auditory exploration
        st.subheader("Signal Analysis")
        
        # Extract frequency bands that correspond to
        # different cognitive and physiological processes
        bands = processor.extract_frequency_bands(eeg_data)
        
        # Calculate band powers to quantify relative contribution
        # of different neural oscillations
        band_powers = {}
        for band_name, band_data in bands.items():
            # PSD calculation for each isolated frequency band
            _, band_psd = processor.compute_power_spectral_density(band_data)
            # Mean power provides a single metric for comparison
            band_powers[band_name] = np.mean(band_psd)
        
        # Bar chart visualization of band powers provides intuitive
        # comparison of different frequency components
        band_df = pd.DataFrame({
            'Band': list(band_powers.keys()),
            'Relative Power': list(band_powers.values())
        })
        
        st.bar_chart(band_df.set_index('Band'))
        
        # Hjorth parameters provide complementary time-domain metrics
        # that capture signal complexity and variability
        activity, mobility, complexity = processor.compute_hjorth_parameters(eeg_data)
        
        st.subheader("Hjorth Parameters")
        hjorth_df = pd.DataFrame({
            'Parameter': ['Activity', 'Mobility', 'Complexity'],
            'Value': [activity, mobility, complexity]
        })
        
        # Table format for precise numeric values
        st.table(hjorth_df)
        
        # Basic statistics provide context and scaling information
        # for interpreting other metrics
        st.subheader("Basic Statistics")
        stats_df = pd.DataFrame({
            'Statistic': ['Mean', 'Std Dev', 'Min', 'Max', 'Range'],
            'Value': [
                np.mean(eeg_data),
                np.std(eeg_data),
                np.min(eeg_data),
                np.max(eeg_data),
                np.ptp(eeg_data)  # Peak to peak (max - min)
            ]
        })
        
        st.table(stats_df)
    else:
        st.info("Generate EEG data using the controls in the sidebar to begin.")

# Tab 3 contains detailed educational content about EEG sonification
# and the scientific principles underlying the application
with tab3:
    st.header("About EEG Sonification")
    st.markdown("""
    ## What is EEG Sonification?
    
    EEG sonification is the process of converting electroencephalogram (EEG) data into sound. This technique allows 
    users to "hear" brain activity patterns that might be difficult to perceive visually.
    
    ## Brain States in This Application
    
    This application allows you to explore several simulated brain states:
    
    - **Normal Awake**: Characterized by prominent alpha rhythm (8-13 Hz), especially when eyes are closed
    - **Sleep Stage N1**: Light sleep with slowing of alpha rhythm and appearance of theta activity
    - **Slow Wave Sleep (N3)**: Deep sleep with high-amplitude delta waves
    - **REM Sleep**: Rapid eye movement sleep with mixed frequency activity similar to wakefulness
    - **Seizure Activity**: Abnormal, synchronized electrical discharges that may present as rhythmic spike-wave patterns
    
    ## Sonification Methods
    
    - **Simple Tone Mapping**: Maps the EEG amplitude directly to the frequency of a sine wave
    - **Multi-band Sonification**: Extracts different frequency bands and maps each to a different sound character
    
    ## Technical Details
    
    ### EEG Simulation
    
    The simulated EEG data in this application is generated using a combination of band-limited noise and specific waveforms characteristic of each brain state. Our implementation is inspired by the simulation capabilities in MNE-Python [1].
    
    #### Band-limited noise generation
    - White noise is generated and filtered to specific frequency bands
    - A Butterworth bandpass filter is applied using SciPy's signal processing module
    
    #### Brain state-specific patterns
    - **Normal Awake**: Dominant alpha (8-13 Hz) with subdominant theta, beta, and delta
    - **N1 Sleep**: Dominant theta (4-8 Hz) with vertex waves
    - **N3 Sleep**: High-amplitude delta (0.5-4 Hz)
    - **REM Sleep**: Mixed frequencies with sawtooth waves
    - **Seizure**: 3 Hz spike-wave complexes with evolving amplitude
    
    ### Signal Processing
    
    Our signal processing approach draws from established methods in EEG analysis, including:
    
    - Frequency band extraction using bandpass filters
    - Power spectral density estimation using Welch's method
    - Computation of Hjorth parameters (Activity, Mobility, Complexity)
    - Spectrogram visualization with Short-Time Fourier Transform
    
    ## References
    
    [1] MNE-Python: MNE 1.9.0 documentation. Available at: https://mne.tools/stable/auto_examples/simulation/generate_simulated_raw_data.html
    
    [2] Introduction to MNE-Python: Overview of MEG/EEG analysis with MNE-Python. Available at: https://mne.tools/stable/auto_tutorials/intro/10_overview.html
    
    [3] Vavra, P., et al. (2023). SleepEEGpy: a Python-based package for the preprocessing and analysis of sleep EEG. bioRxiv. Available at: https://www.biorxiv.org/content/10.1101/2023.03.08.531747v1
    
    [4] Hermann, T., Hunt, A., & Neuhoff, J. G. (Eds.). (2011). The Sonification Handbook. Logos Verlag Berlin.
    
    [5] Niedermeyer, E., & da Silva, F. L. (Eds.). (2005). Electroencephalography: Basic Principles, Clinical Applications, and Related Fields. Lippincott Williams & Wilkins.
    """)