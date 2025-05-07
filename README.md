The "Audio to MIDI" project aims to recognize sounds in monophonic (or polyphonic) tracks using advanced sound analysis methods. We use a variety of techniques, from classical transformation methods (such as STFT and CQT) to more advanced approaches using neural networks. Our goal is to convert audio into MIDI format, allowing for precise reproduction of sounds in various musical compositions.

### Main Scripts:
1. **STFT** - Fourier transform implementation in the script `audio_to_midi_fourier_test.py`.
2. **CQT/Chroma** - Sound analysis using CQT transformation, implemented in `audio_to_midi_chroma.py`.
3. **CQT/Chroma + Harmonic Correlation** - An enhanced version with harmonic analysis, also in `audio_to_midi_chroma.py`.
4. **Neural Networks** - Advanced models for detecting octaves and identifying sounds within the corresponding octaves, found in the script `neural_network_cqt.py`.

For segmenting individual sounds in audio recordings, we use the `onset_detect` function from the **Librosa** library.

### Content:
- A video demonstrating the project: [YouTube](https://youtu.be/2y9CeWTdmJc)
- Full project description: `opis_projektu.pdf`

This project is part of a larger effort on audio processing, located in the `monophonic_music/audio_to_midi` folder.

