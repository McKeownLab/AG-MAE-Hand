import os
import torch
from torch.utils.data import DataLoader
from dataset.dataset import PD_Dataset
from model.stmae import STMAE, Encoder
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
import pickle

def load_pretrained_stmae(cfg, device):
    """Load pre-trained STMAE model from a checkpoint."""
    stmae_args = cfg.stmae
    encoder = Encoder(
        patch_num=stmae_args.num_joints * stmae_args.window_size,
        patch_dim=stmae_args.coords_dim,
        window_size=stmae_args.window_size,
        num_classes=stmae_args.coords_dim,
        dim=stmae_args.encoder_embed_dim,
        depth=stmae_args.encoder_depth,
        heads=stmae_args.num_heads,
        mlp_dim=stmae_args.mlp_dim,
        pool="cls",
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    )

    stmae = STMAE(
        encoder=encoder,
        decoder_dim=stmae_args.decoder_dim,
        decoder_depth=stmae_args.decoder_depth,
        masking_strategy=stmae_args.masking_strategy,
        spatial_masking_ratio=stmae_args.spatial_masking_ratio,
        temporal_masking_ratio=stmae_args.temporal_masking_ratio,
    ).to(device)

    checkpoint_path = os.path.join(
        cfg.save_folder_path, cfg.dataset, cfg.exp_name, "weights", "best_stmae_model.pth"
    )
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    stmae.load_state_dict(checkpoint["state_dict"])
    print(f"Loaded STMAE checkpoint from epoch {checkpoint['epoch']}")
    stmae.eval()
    return stmae

def generate_embeddings_with_metadata(stmae, dataloader, device):
    """Generate embeddings for the dataset along with labels and indexes."""
    embeddings = []
    labels = []
    indexes = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Embeddings"):
            sequences = batch["Sequence"].to(device).float()
            encoded = stmae.inference(sequences)

            embeddings.append(encoded.cpu().numpy())
            labels.extend(batch["Label"].cpu().numpy())
            indexes.extend(batch["Index"])  # Assuming `Index` is a list or dictionary
    return (
        np.concatenate(embeddings, axis=0),
        np.array(labels),
        indexes,
    )

def save_embeddings_with_metadata(embeddings, labels, indexes, output_path):
    """Save embeddings, labels, and indexes to a file using pickle."""
    data_to_save = {
        "embeddings": embeddings,
        "labels": labels,
        "indexes": indexes,
    }
    with open(output_path, "wb") as f:
        pickle.dump(data_to_save, f, protocol=4)  # Use protocol 4 for large files
    print(f"Embeddings, labels, and indexes saved to {output_path}")

if __name__ == "__main__":
    # Configuration
    cfg_path = "configs/pd.yaml"  
    cfg = OmegaConf.load(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    dataset = PD_Dataset(
        data_dir=cfg.data.data_dir,
        label_csv_path=cfg.data.label_csv_path,
        window_size=cfg.stmae.window_size,
        step=cfg.data.step,
        normalize=cfg.data.normalize,
        random_rot=True,
    )
    dataloader = DataLoader(dataset, batch_size=cfg.stmae.batch_size, shuffle=False)

    # Load STMAE model
    stmae = load_pretrained_stmae(cfg, device)

    # Generate embeddings with metadata
    embeddings, labels, indexes = generate_embeddings_with_metadata(stmae, dataloader, device)

    # Save embeddings with metadata
    output_path = os.path.join(cfg.save_folder_path, cfg.dataset, cfg.exp_name, "stmae_embeddings_pd_2.npy")
    save_embeddings_with_metadata(embeddings, labels, indexes, output_path)
