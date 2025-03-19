from training.champion_transformer import ChampionEmbeddingTransformer
import umap
import matplotlib.pyplot as plt
import torch
import numpy as np
from utils.scrapers import LoLScraper
from utils.champion import CHAMPION_IDS
import argparse

parser = argparse.ArgumentParser(description="Visualize champion embeddings using UMAP.")
parser.add_argument("--checkpoint", type=str, default="training_checkpoints/checkpoint_epoch_5.pth",
                    help="Path to the trained model checkpoint.")
parser.add_argument("-n", "--neighbors", type=int, default=15,
                    help="Number of neighbors for UMAP.")
args = parser.parse_args()


# Define an RGB mapping for each tag.
# Colors are defined as tuples: (R, G, B) in [0, 255]
TAG_COLOR_MAP = {
    "Mage": (128, 0, 128),       # Purple
    "Fighter": (255, 0, 0),      # Red
    "Tank": (0, 128, 0),         # Green
    "Support": (255, 255, 0),    # Yellow
    "Marksman": (255, 165, 0),   # Orange
    "Assassin": (0, 0, 0),       # Black
}

DEFAULT_COLOR = (0, 0, 255)      # Blue, default if no matching tag

def average_rgb(colors):
    """Average a list of RGB tuples (ints) component-wise."""
    if not colors:
        return DEFAULT_COLOR
    colors = np.array(colors, dtype=np.float32)
    avg = colors.mean(axis=0)
    return tuple(np.round(avg).astype(np.int32))

# Load champion data from ddragon.
scrapper = LoLScraper()
champions_data = scrapper.ddragon.champions.get_all_champions_data()["data"]

# Initialize the model.
model = ChampionEmbeddingTransformer(
    feature_dim=159,          # Adjust as needed.
    d_model=256,
    num_champions=200,
    champion_embedding_dim=128,
    num_layers=4,
    num_heads=8,
    dropout=0.1,
)
# Load the saved state dictionary.
model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device("cpu")))
model.eval()

# Extract champion embeddings.
champion_embeddings = model.champion_embedding.weight.detach().cpu().numpy()
champion_embeddings = champion_embeddings[:170]  # e.g., take first 170 champions

champion_labels = []
champion_colors = []  # We'll store RGB tuples
for champ_idx in range(170):
    champ_id = CHAMPION_IDS[champ_idx]
    champ_name = None
    champion_tags = []
    # Find champion info from ddragon.
    for name, data in champions_data.items():
        if data["key"] == str(champ_id):
            champ_name = name
            champion_tags = data.get("tags", [])
            break
    if champ_name is None:
        champ_name = f"Unknown_{champ_id}"
    champion_labels.append(champ_name)
    
    # For each tag in champion_tags, get the corresponding color.
    colors = []
    for tag in champion_tags:
        if tag in TAG_COLOR_MAP:
            colors.append(TAG_COLOR_MAP[tag])
    # If multiple colors, average them; if none, use default.
    final_color = average_rgb(colors) if colors else DEFAULT_COLOR
    champion_colors.append(final_color)

# Convert RGB tuples to normalized RGB for matplotlib scatter (scale to 0-1).
champion_colors_norm = [tuple(np.array(c)/255.0) for c in champion_colors]

# Use UMAP to reduce the champion embeddings to 2D.
n = args.neighbors  # Number of neighbors
umap_reducer = umap.UMAP(
    n_neighbors=n,      # Try between 5 and 15
    n_components=2, 
    metric="cosine",     # "cosine" is often a good choice for embeddings
    min_dist=0.1,        # Adjust to control tightness of clusters
    spread=1.0,          # Overall scale of the embedding
    random_state=42
)
champion_embeddings_umap = umap_reducer.fit_transform(champion_embeddings)

# Use kMeans clustering.
from sklearn.cluster import KMeans
num_clusters = 15 # Adjust as needed.
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(champion_embeddings)

# Plot UMAP projection.
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    champion_embeddings_umap[:, 0],
    champion_embeddings_umap[:, 1],
    c=clusters,          # color by cluster
    cmap="viridis",
    s=50,
)
plt.colorbar(scatter, label="Cluster")
plt.title("UMAP Projection of Champion Embeddings with kMeans Clusters")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
for i, label in enumerate(champion_labels):
    plt.annotate(
        label,
        (champion_embeddings_umap[i, 0], champion_embeddings_umap[i, 1]),
        textcoords="offset points",
        xytext=(0, 5),
        ha="center",
        fontsize=8,
    )
plt.savefig(f"viz/champion_embeddings_umap_nmeans_{n}.png")
# Plot champion points with their averaged RGB colors.
plt.figure(figsize=(10, 8))
plt.scatter(
    champion_embeddings_umap[:, 0],
    champion_embeddings_umap[:, 1],
    c=champion_colors_norm,  # Use normalized RGB values.
    s=50,
)
plt.title("UMAP Projection of Champion Embeddings (Averaged RGB by Tags)")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")

# Annotate points with champion names.
for i, label in enumerate(champion_labels):
    plt.annotate(
        label,
        (champion_embeddings_umap[i, 0], champion_embeddings_umap[i, 1]),
        textcoords="offset points",
        xytext=(0, 5),
        ha="center",
        fontsize=8,
    )

plt.savefig(f"viz/champion_embeddings_umap_{n}.png")
