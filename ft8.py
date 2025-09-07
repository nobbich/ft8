# -*- coding: utf-8 -*-
"""
Spyder-Editor

Dies ist eine temporäre Skriptdatei.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

class FT8_FSK8:
    def __init__(self, base_freq=1500, sample_rate=12000):
        """
        FT8 FSK-8 Implementation
        
        FT8 Parameter:
        - 8 Töne (0-7) mit je 6.25 Hz Abstand
        - Symbol-Dauer: 0.16 Sekunden (160ms)
        - Gesamt-Dauer: 79 Symbole × 0.16s = 12.64s (plus Pausen = ~13s)
        - Baud-Rate: 6.25 Baud
        - Bandbreite: ~50 Hz
        """
        self.base_freq = base_freq  # Basis-Frequenz in Hz
        self.tone_spacing = 6.25    # Frequenz-Abstand zwischen Tönen
        self.symbol_duration = 0.16 # Dauer eines Symbols in Sekunden
        self.sample_rate = sample_rate
        self.num_tones = 8
        
        # Die 8 FSK-Töne berechnen
        self.tones = [base_freq + i * self.tone_spacing for i in range(8)]
        
        # FT8 verwendet 79 Symbole pro Nachricht
        self.message_length = 79
        
    def symbol_to_tone(self, symbol):
        """
        Konvertiert ein Symbol (0-7) zur entsprechenden Frequenz
        
        Symbol 0 = niedrigste Frequenz
        Symbol 7 = höchste Frequenz
        """
        if 0 <= symbol <= 7:
            return self.tones[symbol]
        else:
            raise ValueError("Symbol muss zwischen 0 und 7 liegen")
    
    def generate_tone(self, frequency, duration):
        """
        Generiert einen Sinuston mit gegebener Frequenz und Dauer
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        return np.sin(2 * np.pi * frequency * t)
    
    def encode_message(self, symbols):
        """
        Kodiert eine Liste von Symbolen (0-7) zu einem Audio-Signal
        
        Args:
            symbols: Liste von Integers (0-7), z.B. [3, 1, 4, 1, 5, 9, 2, 6]
        
        Returns:
            numpy array mit dem Audio-Signal
        """
        if len(symbols) > self.message_length:
            symbols = symbols[:self.message_length]
        
        audio_signal = np.array([])
        
        for symbol in symbols:
            tone_freq = self.symbol_to_tone(symbol)
            tone_audio = self.generate_tone(tone_freq, self.symbol_duration)
            audio_signal = np.concatenate([audio_signal, tone_audio])
        
        return audio_signal
    
    def decode_message(self, audio_signal):
        """
        Vereinfachte Dekodierung - findet die dominante Frequenz pro Symbol-Zeit
        (In der Realität viel komplexer mit FFT und Fehlerkorrektur)
        """
        symbols = []
        samples_per_symbol = int(self.sample_rate * self.symbol_duration)
        
        for i in range(0, len(audio_signal), samples_per_symbol):
            if i + samples_per_symbol > len(audio_signal):
                break
                
            # Audio-Segment für dieses Symbol
            segment = audio_signal[i:i + samples_per_symbol]
            
            # FFT um dominante Frequenz zu finden
            fft = np.fft.fft(segment)
            freqs = np.fft.fftfreq(len(segment), 1/self.sample_rate)
            
            # Nur positive Frequenzen betrachten
            positive_freqs = freqs[:len(freqs)//2]
            positive_fft = np.abs(fft[:len(fft)//2])
            
            # Dominante Frequenz finden
            dominant_freq_idx = np.argmax(positive_fft)
            dominant_freq = positive_freqs[dominant_freq_idx]
            
            # Nächste FSK-Frequenz finden
            closest_symbol = 0
            min_diff = float('inf')
            
            for j, tone_freq in enumerate(self.tones):
                diff = abs(dominant_freq - tone_freq)
                if diff < min_diff:
                    min_diff = diff
                    closest_symbol = j
            
            symbols.append(closest_symbol)
        
        return symbols
    
    def visualize_spectrum(self, symbols, title="FSK-8 Spektrum"):
        """
        Visualisiert das Frequenzspektrum der FSK-8 Übertragung
        """
        audio = self.encode_message(symbols)
        
        # Zeit-Frequenz-Darstellung
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Zeitsignal
        plt.subplot(3, 1, 1)
        t = np.linspace(0, len(audio)/self.sample_rate, len(audio))
        plt.plot(t, audio)
        plt.title(f"{title} - Zeitsignal")
        plt.xlabel("Zeit (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        
        # Subplot 2: Spektrogramm
        plt.subplot(3, 1, 2)
        plt.specgram(audio, Fs=self.sample_rate, cmap='viridis')
        plt.title("Spektrogramm (Zeit-Frequenz)")
        plt.xlabel("Zeit (s)")
        plt.ylabel("Frequenz (Hz)")
        plt.colorbar(label="Leistung (dB)")
        
        # Die 8 FSK-Töne als horizontale Linien einzeichnen
        for i, freq in enumerate(self.tones):
            plt.axhline(y=freq, color='red', linestyle='--', alpha=0.7, 
                       label=f'Ton {i}' if i == 0 else "")
        
        # Subplot 3: Symbol-Sequenz
        plt.subplot(3, 1, 3)
        plt.step(range(len(symbols)), symbols, where='post', linewidth=2)
        plt.title("Symbol-Sequenz")
        plt.xlabel("Symbol-Nummer")
        plt.ylabel("Ton (0-7)")
        plt.grid(True)
        plt.ylim(-0.5, 7.5)
        
        plt.tight_layout()
        plt.show()
        
        return audio

# Beispiel-Verwendung
if __name__ == "__main__":
    # FT8 FSK-8 Encoder/Decoder erstellen
    ft8 = FT8_FSK8(base_freq=1500)  # Typische FT8 Frequenz
    
    print("FT8 FSK-8 Parameter:")
    print(f"Basis-Frequenz: {ft8.base_freq} Hz")
    print(f"Ton-Abstand: {ft8.tone_spacing} Hz")
    print(f"Symbol-Dauer: {ft8.symbol_duration} s")
    print(f"Die 8 Töne: {[f'{f:.2f} Hz' for f in ft8.tones]}")
    print()
    
    # Beispiel-Nachricht (erste paar Symbole einer typischen FT8-Nachricht)
    # In der Realität sind das kodierte Daten mit Fehlerkorrektur
    example_symbols = [3, 1, 4, 1, 5, 4, 2, 6, 5, 3, 5, 1, 5, 7, 2, 3, 2, 3, 5, 4][:20]
    
    print(f"Beispiel-Symbol-Sequenz: {example_symbols}")
    print(f"Entsprechende Frequenzen:")
    for i, symbol in enumerate(example_symbols):
        freq = ft8.symbol_to_tone(symbol)
        print(f"  Symbol {symbol} -> {freq:.2f} Hz")
    
    # Audio generieren
    audio_signal = ft8.encode_message(example_symbols)
    print(f"\nGeneriertes Audio-Signal: {len(audio_signal)} Samples")
    print(f"Dauer: {len(audio_signal)/ft8.sample_rate:.2f} Sekunden")
    
    # Dekodierung testen
    decoded_symbols = ft8.decode_message(audio_signal)
    print(f"\nOriginal:  {example_symbols}")
    print(f"Dekodiert: {decoded_symbols}")
    print(f"Fehler: {sum(o != d for o, d in zip(example_symbols, decoded_symbols))}")
    
    # Visualisierung
    audio = ft8.visualize_spectrum(example_symbols, "FT8 FSK-8 Beispiel")
    
    # Audio als WAV-Datei speichern (optional)
    # wavfile.write("ft8_example.wav", ft8.sample_rate, 
    #               (audio * 32767).astype(np.int16))
    
    print("\nWichtige Erkenntnisse:")
    print("- Jedes Symbol dauert 160ms")
    print("- 8 verschiedene Töne mit 6.25 Hz Abstand")
    print("- Sehr schmale Bandbreite (~50 Hz)")
    print("- Langsame Übertragung ermöglicht schwache Signale")
    print("- Gesamte FT8-Nachricht: 79 Symbole in ~13 Sekunden")
