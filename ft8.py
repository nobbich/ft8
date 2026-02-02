#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FT8 FSK-8 Encoder/Decoder mit Visualisierung
"""

import numpy as np
import matplotlib.pyplot as plt

class FT8_FSK8:
    FT8_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 /"
    
    def __init__(self, base_freq=1500, sample_rate=12000):
        self.base_freq = base_freq
        self.tone_spacing = 6.25
        self.symbol_duration = 0.16
        self.sample_rate = sample_rate
        self.num_tones = 8
        self.tones = [base_freq + i*self.tone_spacing for i in range(self.num_tones)]
        self.message_length = 79

    # --- FT8 Codierung ---
    def char_to_6bit(self, c):
        if c not in self.FT8_ALPHABET:
            raise ValueError(f"Zeichen '{c}' nicht im FT8_ALPHABET")
        return self.FT8_ALPHABET.index(c)

    def bit6_to_char(self, val):
        if 0 <= val < len(self.FT8_ALPHABET):
            return self.FT8_ALPHABET[val]
        else:
            return "?"

    def ft8_encode_full(self, message):
        message = message.upper().ljust(13)
        bits = ""
        for c in message:
            val = self.char_to_6bit(c)
            bits += format(val, '06b')
        return bits

    def crc16_ccitt(self, bits):
        n = 8
        byte_array = [int(bits[i:i+n].ljust(8,'0'),2) for i in range(0,len(bits),8)]
        crc = 0xFFFF
        for b in byte_array:
            crc ^= b << 8
            for _ in range(8):
                if crc & 0x8000:
                    crc = ((crc << 1) ^ 0x1021) & 0xFFFF
                else:
                    crc = (crc << 1) & 0xFFFF
        return format(crc, '016b')

    def ft8_encode_full_with_crc(self, message):
        bits = self.ft8_encode_full(message)
        crc_bits = self.crc16_ccitt(bits)
        return bits + crc_bits

    def ft8_bits_to_symbols(self, bitstream):
        symbols = []
        for i in range(0,len(bitstream),3):
            chunk = bitstream[i:i+3].ljust(3,'0')
            symbols.append(int(chunk,2))
        return symbols[:79]

    def ft8_symbols_to_bits(self, symbols):
        bits = ""
        for s in symbols:
            bits += format(s,'03b')
        return bits

    # --- FSK-8 Audio ---
    def symbol_to_tone(self, symbol):
        if 0 <= symbol <= 7:
            return self.tones[symbol]
        else:
            raise ValueError("Symbol muss zwischen 0 und 7 liegen")
    
    def generate_tone(self, frequency, duration):
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        return np.sin(2 * np.pi * frequency * t)
    
    def encode_message(self, symbols):
        if len(symbols) > self.message_length:
            symbols = symbols[:self.message_length]
        
        audio_signal = np.array([])
        for symbol in symbols:
            tone_freq = self.symbol_to_tone(symbol)
            tone_audio = self.generate_tone(tone_freq, self.symbol_duration)
            audio_signal = np.concatenate([audio_signal, tone_audio])
        
        return audio_signal
    
    def decode_message(self, audio_signal):
        symbols = []
        samples_per_symbol = int(self.sample_rate * self.symbol_duration)
        
        for i in range(0, len(audio_signal), samples_per_symbol):
            if i + samples_per_symbol > len(audio_signal):
                break
            segment = audio_signal[i:i + samples_per_symbol]
            fft = np.fft.fft(segment)
            freqs = np.fft.fftfreq(len(segment), 1/self.sample_rate)
            
            positive_freqs = freqs[:len(freqs)//2]
            positive_fft = np.abs(fft[:len(fft)//2])
            
            dominant_freq_idx = np.argmax(positive_fft)
            dominant_freq = positive_freqs[dominant_freq_idx]
            
            closest_symbol = 0
            min_diff = float('inf')
            for j, tone_freq in enumerate(self.tones):
                diff = abs(dominant_freq - tone_freq)
                if diff < min_diff:
                    min_diff = diff
                    closest_symbol = j
            symbols.append(closest_symbol)
        
        return symbols

    # --- Spektrum visualisieren ---
    def visualize_spectrum(self, symbols, decoded_symbols=None, title="FSK-8 Spektrum"):
        audio = self.encode_message(symbols)
        t = np.linspace(0, len(audio)/self.sample_rate, len(audio))
        
        plt.figure(figsize=(12, 8))
        
        # Zeitsignal
        plt.subplot(3,1,1)
        plt.plot(t, audio)
        plt.title(f"{title} - Zeitsignal")
        plt.xlabel("Zeit (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        
        # Spektrogramm
        plt.subplot(3,1,2)
        Pxx, freqs, bins, im = plt.specgram(audio, Fs=self.sample_rate, cmap='viridis')
        plt.title("Spektrogramm (Zeit-Frequenz)")
        plt.xlabel("Zeit (s)")
        plt.ylabel("Frequenz (Hz)")
        plt.colorbar(label="Leistung (dB)")
        
        # FSK-Frequenzen als Referenzlinien
        for i, freq in enumerate(self.tones):
            plt.axhline(y=freq, color='red', linestyle='--', alpha=0.7)
        
        # Dekodierte Symbole als Punkte überlagern
        if decoded_symbols is not None:
            for i, sym in enumerate(decoded_symbols):
                time_pos = i * self.symbol_duration + self.symbol_duration/2
                freq_pos = self.tones[sym]
                plt.plot(time_pos, freq_pos, 'wo', markersize=6, markeredgecolor='black')
        
        # Symbol-Sequenz
        plt.subplot(3,1,3)
        plt.step(range(len(symbols)), symbols, where='post', linewidth=2)
        if decoded_symbols is not None:
            plt.step(range(len(decoded_symbols)), decoded_symbols, where='post', linewidth=1, linestyle='--', color='orange')
        plt.title("Symbol-Sequenz")
        plt.xlabel("Symbol-Nummer")
        plt.ylabel("Ton (0-7)")
        plt.grid(True)
        plt.ylim(-0.5, 7.5)
        
        plt.tight_layout()
        plt.show()
        return audio
    
    # --- Bitstream Decodierung ---
    def ft8_decode_bits_with_crc(self, full_bits):
        message_bits = full_bits[:-16]
        crc_bits = full_bits[-16:]
        computed_crc = self.crc16_ccitt(message_bits)
        crc_ok = (computed_crc == crc_bits)
        recovered_message = ""
        for i in range(0, len(message_bits), 6):
            val = int(message_bits[i:i+6], 2)
            recovered_message += self.bit6_to_char(val)
        return recovered_message.strip(), crc_ok

# --- Beispiel ---
if __name__ == "__main__":
    ft8 = FT8_FSK8()
    
    msg = "CQ HB9ABC FN31"
    full_bits = ft8.ft8_encode_full_with_crc(msg)
    symbols = ft8.ft8_bits_to_symbols(full_bits)
    
    print("Original-Symbole: ", symbols)
    
    audio = ft8.encode_message(symbols)
    
    decoded_symbols = ft8.decode_message(audio)
    print("Dekodierte Symbole: ", decoded_symbols)
    
    recovered_bits = ft8.ft8_symbols_to_bits(decoded_symbols)
    recovered_message, crc_ok = ft8.ft8_decode_bits_with_crc(full_bits)
    
    print("Nachricht: ", recovered_message)
    print("CRC OK: ", crc_ok)
    
    # Spektrum visualisieren + dekodierte Symbole einblenden
    ft8.visualize_spectrum(symbols, decoded_symbols)
