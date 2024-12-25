import matplotlib.pyplot as plt
import numpy as np


tier_colors = {
    "IRON": "gray",
    "BRONZE": "saddlebrown",
    "SILVER": "silver",
    "GOLD": "gold",
    "PLATINUM": "cyan",
    "EMERALD": "green",
    "DIAMOND": "blue",
    "MASTER": "purple",
    "GRANDMASTER": "magenta",
    "CHALLENGER": "red",
}


def plot_histogram(
    data: dict[str, int], file_name: str, title: str = "Histogram"
) -> None:
    total_players = sum(data.values())

    data = {k: (v / total_players) * 100 for k, v in data.items()}

    colors = []
    for rank in data:
        # Assuming rank format like "IRON_IV" or "GOLD_I":
        tier = rank.split("_")[0]
        color = tier_colors.get(tier, "black")  # Default to black if not found
        colors.append(color)

    plt.figure(figsize=(12, 6))
    plt.bar(data.keys(), data.values(), color=colors)

    plt.title(title)
    plt.xlabel("Rank")
    plt.ylabel("Reverse Cumulative Percentage (%)")

    # Rotate x ticks for better readability
    plt.xticks(rotation=45, ha="right")

    # # Set y-ticks at every 10% and enable grid
    plt.yticks(range(0, 11))
    plt.grid(True, which="major", axis="y", linestyle="--", color="gray", alpha=0.7)

    plt.tight_layout()
    plt.savefig(file_name)


def plot_ccdf(data: dict[str, int], file_name: str) -> None:
    categories = list(data.keys())
    values = list(data.values())

    total = sum(values)
    middle = total // 2
    # Skumulowana suma częstości
    cumulative_values = np.cumsum(values)
    mode = max(data, key=data.get)

    median = None
    for cat, val in zip(categories, cumulative_values):
        if val >= middle:
            median = cat
            break

    cdf = cumulative_values / total
    ccdf = 100 * (1 - cdf)

    # Rysowanie wykresu CCDF
    plt.figure(figsize=(12, 6))

    colors = []
    for rank in categories:
        # Assuming rank format like "IRON_IV" or "GOLD_I":
        tier = rank.split("_")[0]
        color = tier_colors.get(tier, "black")  # Default to black if not found
        colors.append(color)

    # Wykres schodkowy – bo dystrybuanty często się tak przedstawia
    plt.bar(
        categories,
        ccdf,
        color=colors,
        width=0.7,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.9,
    )

    plt.axvline(x=median, color="green", linestyle="--", label=f"Median: {median}")

    plt.axvline(x=mode, color="orange", linestyle="--", label=f"Mode: {mode}")

    plt.grid(True, which="major", axis="y", linestyle="--", color="gray", alpha=0.7)
    plt.axhline(y=50, color="red", linestyle="--", linewidth=1, label="50%")
    plt.xlabel("Kategoria")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("CCDF (%)")
    plt.title("CCDF of Ranks")

    plt.legend()
    plt.tight_layout()

    plt.savefig(file_name)
