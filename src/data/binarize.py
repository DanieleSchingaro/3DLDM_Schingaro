#src/data/binarize.py
"""
Codifica bit-plane della maschera di segmentazione per la condizione della ControlNet.
Estratta da utils.py di NV-Generate-CTMR (MAISI), logica invariata.

La maschera [B,1,X,Y,Z] con valori interi {0,1,2,3} diventa [B,bits,X,Y,Z], dove ogni
canale e' un bit-plane. Con bits=8 (=conditioning_embedding_in_channels) le 3 classi
usano 2 bit; i 6 canali alti restano a zero. La ControlNet vuole la condizione come
tensore multi-canale, non come mappa di interi.
"""

import torch

def binarize_labels(x:torch.Tensor, bits:int=8)->torch.Tensor:
    """
    Args:
        x: [B,1,X,Y,Z] interi (torch.long)
        bits: numero di bit-plane in output (=conditioning_embedding_in_channels)
    Returns:
        [B,bits,X,Y,Z] uint8 coi bit-plane della maschera
    """
    mask=2**torch.arange(bits).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte().squeeze(1).permute(0, 4, 1, 2, 3)