#tests/check_best_model_vae.py

"""
Verifica il checkpoint del miglior VAE e le metriche salvate su MLFlow.
Esegui con: python3 tests/check_best_model_vae.py
"""
import torch
import mlflow 
from pathlib import Path 

#Checkpoint
checkpoint_path=Path("outputs/models_v4/autoencoder_best.pt")
if not checkpoint_path.exists():
    print(f"File non trovato: {checkpoint_path}")
    exit(1)

file_size_mb=checkpoint_path.stat().st_size/(1024*1024)
print(f"Caricamento checkpoint: {checkpoint_path}")
print(f"Dimensione file: {file_size_mb:.2f} MB")
checkpoint=torch.load(checkpoint_path, map_location="cpu")

print("\n===INFO CHECKPOINT===")
if "epoch" in checkpoint:
    print(f"Epoca salvata: {checkpoint['epoch']}")
if "val_loss" in checkpoint:
    print(f"Validation loss: {checkpoint['val_loss']:.6f}")

print("\n===CHIAVI PRESENTI===")
for key in checkpoint.keys():
    print(f"-{key}")

if "autoencoder_state_dict" in checkpoint:
    total_params=sum(
        v.numel() for v in checkpoint["autoencoder_state_dict"].values()
    )
    print(f"\nParametri autoencoder: {total_params:,}")

    #controllo NaN dei pesi
    nan_layers=[
        k for k,v in checkpoint["autoencoder_state_dict"].items()
        if torch.isnan(v).any()
    ]
    if nan_layers:
        print(f"NaN trovati in {len(nan_layers)} layer:")
        for l in nan_layers:
            print(f"-{l}")
    else:
        print("Nessun NaN nei pesi autoencoder")

if "discriminator_state_dict" in checkpoint:
    total_params_d=sum(
        v.numel() for v in checkpoint["discriminator_state_dict"].values()
    )
    print(f"\nParametri discriminator: {total_params_d:,}")

    #controllo NaN discriminator
    nan_layers_d=[
        k for k,v in checkpoint["discriminator_state_dict"].items()
        if torch.isnan(v).any()
    ]
    if nan_layers_d:
        print(f"NaN trovati in {len(nan_layers_d)} layer discriminator")
    else:
        print("Nessun NaN nei pesi del discriminator")

#MLFlow
print("\n===METRICHE MLFLOW===")
try:
    client=mlflow.tracking.MlflowClient()
    experiments=client.search_experiments(filter_string="name='VAE_training'")
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
                "train_recon_loss",
                "train_kl_loss",
                "train_gen_loss",
                "train_disc_loss",
                "val_recon_loss",
                "lr",
                "total_time_hours",
            ]
            metrics=run.data.metrics
            for key in metric_names:
                if key in metrics:
                    print(f"{key:<25}: {metrics[key]:.6f}")
            
            #stampa anche metriche non in lista
            extra={k:v for k,v in metrics.items() if k not in metric_names}
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
        print("Esperimento VAE_training non trovato")
except Exception as e:
    print(f"Errore MLFlow: {e}")