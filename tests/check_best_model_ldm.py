#tests/check_best_model_ldm.py
"""
Verifica il checkpoint del miglior LDM (Unet di diffusione) e le metriche su MLFlow.
Esegui con: python3 tests/check_best_model_ldm.py

NOTA METODOLOGICA IMPORTANTE:
Questo script controlla la SALUTE TECNICA del checkpoint (NaN, scale_factor,
parametri, loss). NON va usato per scegliere il modello finale: per i modelli
di diffusione la val_loss e' un indicatore DEBOLE della qualita' di generazione.
La scelta del checkpoint migliore va fatta con le metriche sui campioni generati
(FID, MMD, MS-SSIM in eval.py), tipicamente confrontando i checkpoint periodici
ldm_unet_epoch{N}.pt.
"""

import torch
import mlflow 
from pathlib import Path 

#checkpoint 
checkpoint_path=Path("outputs/models_v4/ldm_unet_best.pt")
if not checkpoint_path.exists():
    print(f"File non trovato: {checkpoint_path}")
    exit(1)

file_size_mb=checkpoint_path.stat().st_size/(1024*1024)
print(f"Caricamento checkpoint: {checkpoint_path}")
print(f"Dimensione file: {file_size_mb:.2f} MB")
checkpoint=torch.load(checkpoint_path, map_location="cpu", weights_only=False)

print("\n===INFO CHECKPOINT===")
if "epoch" in checkpoint:
    print(f"Epoca salvata: {checkpoint['epoch']}")
if "loss" in checkpoint:
    print(f"Loss: {checkpoint['loss']:.6f}")

print("\n===CHIAVI PRESENTI===")
for key in checkpoint.keys():
    print(f"-{key}")

#scale factor
print("\n===SCALE FACTOR===")
if "scale_factor" in checkpoint:
    sf=checkpoint["scale_factor"]
    sf_val=sf.item() if isinstance(sf, torch.Tensor) else sf
    print(f"scale_factor: {sf_val:.6f}")
    #validà -> finito e positivo
    if not torch.isfinite(torch.tensor(float(sf_val))):
        print("Attenzione: scale_factor NON finito (NaN/Inf) -> sampling corrotto")
    elif sf_val <=0:
        print("Attenzione: scale_factor <=0 -> valore anomalo")
    else:
        print("scale_factor valido (finito e positivo)")
else:
    print("Attenzione: scale_factor ASSENTE -> il sampling non potrà de-normalizzare")

#num_train_timesteps: serve al sampling
if "num_train_timesteps" in checkpoint:
    print(f"num_train_timesteps: {checkpoint['num_train_timesteps']}")
else:
    print("Attenzione: num_train_timesteps assente (servirà al sampling)")

#Unet
if "unet_state_dict" in checkpoint:
    total_params=sum(
        v.numel() for v in checkpoint["unet_state_dict"].values()
    )
    print(f"\nParametri UNet diffusione: {total_params:,}")
 
    #controllo NaN dei pesi
    nan_layers=[
        k for k, v in checkpoint["unet_state_dict"].items()
        if torch.is_tensor(v) and torch.isnan(v).any()
    ]
    if nan_layers:
        print(f"NaN trovati in {len(nan_layers)} layer:")
        for l in nan_layers:
            print(f"-{l}")
    else:
        print("Nessun NaN nei pesi della UNet")
 
    #controllo Inf dei pesi (un LDM divergente puo' produrre inf)
    inf_layers=[
        k for k, v in checkpoint["unet_state_dict"].items()
        if torch.is_tensor(v) and torch.isinf(v).any()
    ]
    if inf_layers:
        print(f"Inf trovati in {len(inf_layers)} layer:")
        for l in inf_layers:
            print(f"-{l}")
    else:
        print("Nessun Inf nei pesi della UNet")

#MLFlow
print("\n===METRICHE MLFLOW===")
try:
    client=mlflow.tracking.MlflowClient()
    experiments=client.search_experiments(filter_string="name='LDM_training'")
    if experiments:
        runs=client.search_runs(
            experiment_ids=[experiments[0].experiment_id],
            order_by=["start_time DESC"],
        )
        if runs:
            #ultimo run
            run=runs[0]
            print(f"Run ID: {run.info.run_id}")
            print(f"Run name: {run.info.run_name}")
            print(f"Status: {run.info.status}")
            print(f"\nMetriche:")
            metric_names=[
                "train_loss",
                "val_loss",
                "lr",
                "total_time_hours",
            ]
            metrics=run.data.metrics
            for key in metric_names:
                if key in metrics:
                    print(f"{key:<25}: {metrics[key]:.6f}")
 
            #stampa anche metriche non in lista
            extra={k: v for k, v in metrics.items() if k not in metric_names}
            if extra:
                print(f"\nAltre metriche:")
                for key, value in sorted(extra.items()):
                    print(f"{key:<25}: {value:.6f}")
 
            print(f"\nParametri:")
            for key, value in sorted(run.data.params.items()):
                print(f"{key:<25}: {value}")
        else:
            print("Nessun run trovato")
    else:
        print("Esperimento LDM_training non trovato")
except Exception as e:
    print(f"Errore MLFlow: {e}")
 
print("\n"+"="*55)
print("PROMEMORIA: la scelta del modello finale va fatta con FID/MMD/MS-SSIM")
print("sui campioni generati (eval.py), NON con la val_loss qui sopra.")
print("="*55)