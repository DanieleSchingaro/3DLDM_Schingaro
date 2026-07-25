#src/data/list_volumes.py

"""
Elenca i path dei volumi MRI del dataset a partire dal file degli split.
Utility di preparazione dati: stampa su stdout un path per riga, cosi' da
poter essere consumata da script di shell (es. la segmentazione FSL FAST in
batch) senza dover parsare il JSON in bash.

Gli split inclusi sono configurabili (default: training + validation + test),
i duplicati vengono rimossi mantenendo l'ordine di prima apparizione.

IMPORTANTE: i path vanno su stdout, i log su stderr (comportamento di default
del logging di Python). Cosi' lo script chiamante puo' catturare la sola lista
dei path senza che i messaggi di log la sporchino.

Lancio:
    python3 -m src.data.list_volumes
    python3 -m src.data.list_volumes --splits_path data/splits/dataset.json
    python3 -m src.data.list_volumes --splits training validation
"""

import json
import logging
import argparse
from typing import List


def setup_logging()->logging.Logger:
    """
    Configura il logger per l'utility. I messaggi vanno su stderr (default di
    logging.basicConfig con StreamHandler), lasciando stdout pulito per i path.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger=logging.getLogger("list_volumes")
    return logger


def list_volumes(
    splits_path:str,
    splits:List[str],
    logger:logging.Logger,
)->List[str]:
    """
    Legge il file degli split e ritorna la lista dei path immagine per gli
    split richiesti, senza duplicati (ordine di prima apparizione preservato).
    Ogni item dello split puo' essere un dict con chiave "image" oppure
    direttamente una stringa col path.
    """
    with open(splits_path, "r") as f:
        data=json.load(f)

    volumes=[]
    seen=set()
    for split in splits:
        if split not in data:
            logger.error(f"Split '{split}' non presente in {splits_path}, salto.")
            continue
        for item in data[split]:
            path=item["image"] if isinstance(item, dict) else item
            if path not in seen:
                seen.add(path)
                volumes.append(path)

    logger.info(f"Volumi raccolti: {len(volumes)} dagli split {splits}")
    return volumes


def main():
    """
    Script principale: legge gli split e stampa i path dei volumi su stdout,
    uno per riga. Pensato per essere consumato da script di shell.
    """
    parser=argparse.ArgumentParser(description="Elenca i path dei volumi del dataset dagli split")
    parser.add_argument(
        "--splits_path",
        type=str,
        default="data/splits/dataset.json",
        help="Path al file degli split",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["training", "validation", "test"],
        help="Quali split includere (default: tutti)",
    )
    args=parser.parse_args()

    logger=setup_logging()

    volumes=list_volumes(
        splits_path=args.splits_path,
        splits=args.splits,
        logger=logger,
    )

    #stampa i path su stdout (uno per riga) per lo script chiamante
    for path in volumes:
        print(path)


if __name__=="__main__":
    main()