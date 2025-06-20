from pathlib import Path
import matplotlib.pyplot as plt
import torch
import numpy as np
import umap
from training.model import PlayerTimelineSummaryModel
from utils.scrapers import LoLScraper
from utils.champion import CHAMPION_IDS
import argparse

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from training.models.joint_embedding_training_models import ChampionEncoder

parser = argparse.ArgumentParser(description="Visualize champion embeddings using t-SNE.")
# parser.add_argument("--checkpoint", type=str, default="timeline_transformer",
#                     help="Path to the trained model checkpoint.")
parser.add_argument("-n", "--neighbors", type=int, default=15,
                    help="(Not used for t-SNE, but kept for compatibility.)")
parser.add_argument("-e", type=int, default=4,
                    help="Choose epoch")

args = parser.parse_args()


# Define an RGB mapping for each tag.
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
emb_dim = 256

model = ChampionEncoder(
    num_champions=170,
    champion_emb_dim=emb_dim,
    d_model=512,
    dropout=0.1
)


# Load the saved state dictionary.
checkpoint = "best_models/e10_p3_d01_film/joint_embedding_best_model_champion_encoder.pth"
model.load_state_dict(torch.load(checkpoint, map_location=torch.device("cpu")))
model.eval()
print("model loaded...")

# Extract champion embeddings.
champion_ids = torch.arange(170, dtype=int)
champion_embeddings = model(champion_ids).detach().numpy()
print("embeddings extracted...")

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
    # Here we simply take the first tag's color.
    final_color = TAG_COLOR_MAP.get(champion_tags[0], DEFAULT_COLOR)
    champion_colors.append(final_color)

# Convert RGB tuples to normalized RGB for matplotlib scatter (scale to 0-1).
champion_colors_norm = [tuple(np.array(c)/255.0) for c in champion_colors]

print("umap start...")

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
print("umap end")

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
plt.savefig(f"ce_umap_nmeans_{n}_e5_d01_film.png")
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

plt.savefig(f"ce_umap_{n}_e5_d01_film.png")
