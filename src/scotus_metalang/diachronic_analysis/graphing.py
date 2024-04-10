from matplotlib import pyplot as plt
from pathlib import Path


def save_and_show(fig: plt.Figure, filename: str, prefix: str = None, show: bool = False) -> None:
    """Saves and optionally shows a figure."""
    if prefix:
        save_dir = Path("figures", prefix)
    else:
        save_dir = Path("figures")
    Path.mkdir(save_dir, exist_ok=True, parents=True)
    save_path = Path(save_dir, f"{filename}.pdf")
    fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
