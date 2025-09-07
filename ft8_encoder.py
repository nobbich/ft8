import numpy as np
from ft8_ldpc_matrix import H_MATRIX, N, K

def ldpc_encode(payload_bits):
    """
    payload_bits: Liste oder np.array von K=77 Bits
    Gibt ein Codewort mit N=174 Bits zurück
    """
    if len(payload_bits) != K:
        raise ValueError(f"Payload muss {K} Bits haben")

    codeword = np.zeros(N, dtype=np.uint8)
    codeword[:K] = payload_bits

    # Paritätsbits so setzen, dass c*H^T = 0 gilt (über GF(2))
    for row in H_MATRIX:
        s = sum(codeword[i] for i in row) % 2
        if s != 0:
            j = row[-1]       # einfachen Back-Substitution-Trick
            codeword[j] ^= 1

    return codeword


# --- Demo ---
if __name__ == "__main__":
    # Zufällige FT8-Nutzlast (77 Bits)
    payload = np.random.randint(0, 2, size=K, dtype=np.uint8)

    cw = ldpc_encode(payload)

    print("Payload (77 Bits):")
    print("".join(map(str, payload)))

    print("\nCodewort (174 Bits):")
    print("".join(map(str, cw)))
