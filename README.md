# Audio_to_midi

Głównym celem projektu jest rozpoznanie dźwięków w utworach polifonicznych z użyciem różnych metod. Większość oparte na bibiotekach librosa, numpy, scipy itp.

Do tej pory udało mi się zrobić dwiema metodami:

1. Stft - audio_to_midi_fourier_test.py - Łatwo się wywala
1. Cqt/chroma - audio_to_midi_chroma.py - precyzyjna (przynajmniej dla monofonicznych nagrań)

Do segmentacji poszczególnych dźwięków używam onset_detect.

