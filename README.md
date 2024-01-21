# Audio_to_midi

Głównym celem projektu jest rozpoznanie dźwięków w utworach monofonicznych (lub polifonicznych) z użyciem różnych metod. Większość oparte na bibiotekach librosa, numpy, scipy itp.
Programy znajdują się w folderze monophonic/audio_to_midi.

1. Stft - audio_to_midi_fourier_test.py
2. Cqt/chroma - audio_to_midi_chroma.py
3. Cqt/chroma + korelacja alikwotów
4. Sieci neuronowe:
   Wykrywanie oktawy oraz wykrycie dźwięku w tej oktawie.
   
Do segmentacji poszczególnych dźwięków została użyta funkcja onset_detect z biblioteki Librosa.

Nagranie dotyczące działania projektu - https://youtu.be/2y9CeWTdmJc
