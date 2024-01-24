# Audio_to_midi

Głównym celem projektu jest rozpoznanie dźwięków w utworach monofonicznych (lub polifonicznych) z użyciem różnych metod. Większość programów korzysta z bibiotek librosa, numpy, scipy itp.
Główne programy projektu znajdują się w folderze monophonic_music/audio_to_midi.

1. Stft - audio_to_midi_fourier_test.py
2. Cqt/chroma - audio_to_midi_chroma.py
3. Cqt/chroma + korelacja alikwotów - audio_to_midi_chroma.py
4. Sieci neuronowe:
   Wykrywanie oktawy oraz wykrycie dźwięku w tej oktawie.
   
Do segmentacji poszczególnych dźwięków została użyta funkcja onset_detect z biblioteki Librosa.

Nagranie dotyczące działania projektu - https://youtu.be/2y9CeWTdmJc
Projekt znajduje się w monophonic/audio_to_midi/neural_network_cqt.py
Opis projektu - opis_projektu.pdf
