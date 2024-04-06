from matplotlib import pyplot as plt


def save_and_show(fig: plt.Figure, title: str, show: bool = False) -> None:
    """Saves and optionally shows figure."""
    fig.savefig(f"figures/{title}.pdf", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
