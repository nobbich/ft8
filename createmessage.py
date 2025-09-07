ALPHABET = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # Base37

def encode_callsign(callsign: str) -> int:
    callsign = callsign.strip().upper()
    value = 0
    for ch in callsign:
        value = value * 37 + ALPHABET.index(ch)
    return value

def decode_callsign(value: int, length: int = 6) -> str:
    chars = []
    for _ in range(length):
        chars.append(ALPHABET[value % 37])
        value //= 37
    return "".join(reversed(chars)).strip()


def encode_locator(locator: str) -> int:
    locator = locator.upper()
    lon = (ord(locator[0]) - ord('A')) * 20 + (ord(locator[2]) - ord('0')) * 2
    lat = (ord(locator[1]) - ord('A')) * 10 + (ord(locator[3]) - ord('0'))
    return (lon << 7) + lat  # 15 Bit


def encode_report(report: int) -> int:
    # Bereich -30 .. +19 dB
    if not -30 <= report <= 19:
        raise ValueError("Report out of range")
    return report + 30  # nach 0..49 verschoben


def build_ft8_message(callsign: str, locator: str, report: int) -> int:
    """Erzeugt ein 77-Bit Payload als Integer"""
    cs = encode_callsign(callsign)  # 28 Bit
    loc = encode_locator(locator)   # 15 Bit
    rpt = encode_report(report)     # 7 Bit (eigentlich enger, hier Demo)

    # Beispiel-Aufbau (nicht exakt wie WSJT-X, nur schematisch):
    # [ Rufzeichen | Locator | Rapport | Reserved ]
    payload = (cs << (15 + 7)) | (loc << 7) | rpt
    return payload


# Beispiel:
msg = build_ft8_message("HB9ABC", "JN47", -12)
print(f"FT8 Payload (77 bit): {msg:077b}")
