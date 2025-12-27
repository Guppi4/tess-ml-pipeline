"""Sync TIC data from catalog.json back to checkpoint file."""
import json

def sync_tic_to_checkpoint(session_id: str = "s0070_1-1"):
    """Copy TIC fields from catalog.json to checkpoint's star_catalog."""

    catalog_path = f"streaming_results/{session_id}_catalog.json"
    checkpoint_path = f"streaming_results/checkpoints/{session_id}_checkpoint.json"

    # Load catalog with TIC data
    print(f"Loading catalog: {catalog_path}")
    with open(catalog_path, 'r') as f:
        catalog = json.load(f)

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    with open(checkpoint_path, 'r') as f:
        checkpoint = json.load(f)

    # Sync TIC fields
    tic_fields = ['tic_id', 'tmag', 'sep_arcsec', 'n_candidates', 'match_quality']
    updated = 0

    for star_id, star_data in catalog['stars'].items():
        if star_id in checkpoint['star_catalog']:
            for field in tic_fields:
                if field in star_data:
                    checkpoint['star_catalog'][star_id][field] = star_data[field]
            updated += 1

    # Save updated checkpoint
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, indent=2, default=str)

    print(f"Synced TIC data for {updated} stars to checkpoint")
    print(f"Saved: {checkpoint_path}")

if __name__ == "__main__":
    sync_tic_to_checkpoint()
