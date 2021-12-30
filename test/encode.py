import numpy as np


def hash_encode(text: str):
    hashed = hash(text)
    out = []
    dim = 36
    for i in range(1, dim + 1):
        out.append(hashed // i + 0.12 * hashed % i)
    out = np.array(out)
    return out / np.clip(np.max(np.abs(out)), 1e-10, None)


def encoder(text):
    o_a = ord("a")
    o_z = ord("z")
    ord_list = list(range(o_a, o_z)) + [ord("."), ord(" "), ord(",")]
    c2i = {o: i for i, o in enumerate(ord_list)}
    arr = np.zeros(shape=len(ord_list) + 1)
    for s in text:
        o = ord(s)
        if o in ord_list:
            arr[c2i[o]] += 1
        else:
            arr[-1] += 1
    return arr
