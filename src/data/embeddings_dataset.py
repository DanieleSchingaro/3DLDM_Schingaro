#src/data/embeddings_dataset.py

"""
Crea un file .json per splittare gli embeddings in training, validation e test
seguendo l'ordine e la struttra di dataset.json usato per l'addestramento del VAE.
Sostituisce i path MRI raw con i path degli embedding .npz
"""

import os 
import json
from collections import Counter

SPLITS_PATH="data/splits/dataset.json"
EMBEDDINGS_DIR="data/processed/embeddings_v5"
OUTPUT_PATH="data/splits/embeddings_dataset_v5.json"

#mapping per fonte dataset
DATASET_SOURCES={
    "hc_adni_brain_mask": "adni",
    "hc_nifd_brain_mask": "nifd",
    "hc_oasis1_brain_mask": "oasis1",
    "hc_oasis2_brain_mask": "oasis2",
    "hc_oasis3_brain_mask": "oasis3",
    "hc_ppmi_brain_mask": "ppmi",
}

def get_source(path:str)->str:
    """Identifica la fonte del dataset dal path"""
    for key, source in DATASET_SOURCES.items():
        if key in path:
            return source
    return "unknown"

def raw_to_embedding_path(raw_path:str)->str:
    """
    Converte il path della MRI raw nel path dell'embedding.
    Gestisce .nii.gz, .nifti.nii.gz e .nii
    """
    rel=raw_path.replace("data/raw/", "")

    if rel.endswith(".nii.gz"):
        base=rel[:-7]
    elif rel.endswith(".nii"):
        base=rel[:-4]
    else:
        base=os.path.splitext(rel)[0]
    
    emb=base+"_emb.npz"
    return os.path.join(EMBEDDINGS_DIR, emb)

def main():
    with open(SPLITS_PATH, "r") as f:
        splits=json.load(f)
    
    new_splits={}
    total_files=0
    total_missing=0
    global_missing=0

    for split_name, items in splits.items():
        new_items=[]
        missing=0
        domain_count=Counter()

        for item in items:
            total_files+=1
            raw_path=item["image"]
            emb_path=raw_to_embedding_path(raw_path)
            source=get_source(raw_path)

            if not os.path.exists(emb_path):
                print(f"Mancante: {emb_path}")
                missing+=1
                global_missing+=1
                continue
            
            #salva path+source per domain analysis
            new_items.append({
                "image":emb_path,
                "source":source,
            })
            domain_count[source]+=1
        
        new_splits[split_name]=new_items
        total_missing+=missing

        #report per split
        print(f"\n{split_name}: {len(new_items)} embedding ({missing} mancanti)")
        print(f"Distribuzione per dataset:")
        for source, count in sorted(domain_count.items()):
            print(f"-{source}: {count}")
    
    #report finale
    print(f"\n{'='*50}")
    print(f"Totale file: {total_files}")
    print(f"Embedding trovati: {total_files-global_missing}")
    print(f"Embedding mancanti: {global_missing}/{total_files}")

    if global_missing>0:
        pct=global_missing/total_files*100
        print(f"{pct:.1f}% degli embedding mancanti - verifica encode_dataset.py")
    else:
        print("Tutti gli embedding presenti")
    
    with open(OUTPUT_PATH, "w") as f:
        json.dump(new_splits, f, indent=2)
    print(f"\nSalvato in {OUTPUT_PATH}")

if __name__=="__main__":
    main()