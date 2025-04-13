# EEG Sonification Explorer

A Streamlit application for exploring the sonification of EEG (electroencephalogram) signals. This application allows users to generate simulated EEG patterns for different brain states, visualize them in various ways, and listen to their sonified representations.

View this project at: https://eeg-sonification.streamlit.app

## Features

- **EEG Simulation**: Generate realistic EEG data for different brain states:
  - Normal Awake (with dominant alpha rhythm)
  - Sleep Stage N1 (light sleep)
  - Slow Wave Sleep (N3) (deep sleep with delta waves)
  - REM Sleep (with mixed frequencies)
  - Seizure Activity (with rhythmic discharges)

- **Visualizations**:
  - Time series display
  - Power spectral density with frequency band highlighting
  - Spectrogram for time-frequency analysis
  - Animated real-time EEG display

- **Sonification**:
  - Simple tone mapping (EEG amplitude → sound frequency)
  - Multi-band sonification (different waveforms for each frequency band)
  - Audio playback and saving capabilities

- **Analysis**:
  - Frequency band power extraction and visualization
  - Hjorth parameters calculation
  - Basic signal statistics

## Project Structure

- **app.py**: Main Streamlit application with user interface
- **modules/**:
  - **eeg_simulator.py**: Generates synthetic EEG data
  - **signal_processor.py**: Processes and analyzes EEG signals
  - **visualizer.py**: Creates visualizations of the EEG data
  - **sonifier.py**: Converts EEG signals to sound

## Installation

1. Clone this repository
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   streamlit run app.py
   ```

## Requirements

- Python 3.7+
- NumPy
- SciPy
- Matplotlib
- Streamlit
- Pandas
- MNE 

## Technical Details

The application uses sophisticated signal processing techniques to:
- Generate realistic EEG signals using filtered noise and specific waveforms
- Extract frequency bands using Butterworth filters
- Calculate power spectral densities using Welch's method
- Create spectrograms using the Short-Time Fourier Transform
- Sonify signals through frequency mapping and waveform synthesis

## Usage

1. Select a brain state and parameters in the sidebar
2. Generate EEG data
3. Explore the visualizations in the "Visualization" tab
4. Listen to the sonified representation using the playback controls
5. View detailed signal analysis in the "Analysis" tab
6. Learn about EEG sonification in the "About" tab


## Acknowledgments & Resources

This project is based on established methods in EEG research and signal processing.

    [1] MNE-Python: MNE 1.9.0 documentation. Available at: https://mne.tools/stable/auto_examples/simulation/generate_simulated_raw_data.html
    
    [2] Introduction to MNE-Python: Overview of MEG/EEG analysis with MNE-Python. Available at: https://mne.tools/stable/auto_tutorials/intro/10_overview.html
    
    [3] Vavra, P., et al. (2023). SleepEEGpy: a Python-based package for the preprocessing and analysis of sleep EEG. bioRxiv. Available at: https://www.biorxiv.org/content/10.1101/2023.03.08.531747v1
    
    [4] Hermann, T., Hunt, A., & Neuhoff, J. G. (Eds.). (2011). The Sonification Handbook. Logos Verlag Berlin.
    
    [5] Niedermeyer, E., & da Silva, F. L. (Eds.). (2005). Electroencephalography: Basic Principles, Clinical Applications, and Related Fields. Lippincott Williams & Wilkins.
