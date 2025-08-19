import numpy as np
import matplotlib.pyplot as plt


def create_target_image(size=64):
    """Erstellt ein einfaches Testbild (eine Matrix) mit einem Kreuz."""
    image = np.zeros((size, size))
    center = size // 2
    thickness = size // 16

    # Vertikale Linie
    image[:, center - thickness : center + thickness] = 1.0
    # Horizontale Linie
    image[center - thickness : center + thickness, :] = 1.0
    return image


def approximate_with_low_rank(target_matrix, rank):
    """
    Approximiert eine Matrix mit einer gegebenen Rangzahl r.
    Dies simuliert den Kern von LoRA: W ≈ B @ A
    Wir benutzen SVD für die bestmögliche Approximation.
    """
    # 1. Singulärwertzerlegung (SVD) durchführen
    # U enthält die linken Singulärvektoren, S die Singulärwerte, Vh die rechten.
    U, S, Vh = np.linalg.svd(target_matrix)

    # 2. Die Matrizen auf den gewünschten Rang 'r' kürzen
    U_r = U[:, :rank]
    S_r = np.diag(S[:rank])  # S ist ein Vektor, wir brauchen eine Diagonalmatrix
    Vh_r = Vh[:rank, :]

    # 3. Die Approximation rekonstruieren
    # Dies ist die Low-Rank-Approximation der Originalmatrix
    reconstructed_matrix = U_r @ S_r @ Vh_r

    # Optional: Um die LoRA-Struktur B*A explizit zu zeigen
    # Man kann die Wurzel der Singulärwerte aufteilen
    # B = U_r @ np.sqrt(S_r)
    # A = np.sqrt(S_r) @ Vh_r
    # reconstructed_matrix_lora = B @ A  # Das Ergebnis ist identisch

    return reconstructed_matrix

def create_complex_matrix(size=64):
    """ Erstellt eine Matrix mit einem 'X', die einen hohen Rang hat. """
    matrix = np.zeros((size, size))
    for i in range(size):
        matrix[i, i] = 1  # Hauptdiagonale
        matrix[i, size - 1 - i] = 1  # Gegendiagonale
    return matrix

def main():
    image_size = 64
    target_image = create_complex_matrix(image_size)

    # Ränge, die wir visualisieren wollen
    ranks_to_test = [1, 2, 4, 8, 16, image_size]

    num_ranks = len(ranks_to_test)
    fig, axes = plt.subplots(2, num_ranks, figsize=(num_ranks * 3, 7))

    for i, rank in enumerate(ranks_to_test):
        # Approximation berechnen
        approximated_image = approximate_with_low_rank(target_image, rank)

        # Fehler berechnen (Mean Squared Error)
        error = np.mean((target_image - approximated_image) ** 2)

        # Originalbild-Approximation plotten
        ax = axes[0, i]
        im = ax.imshow(approximated_image, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"Rank = {rank}\nError = {error:.4f}")
        ax.set_xticks([])
        ax.set_yticks([])

        # Differenz (Fehlerbild) plotten
        ax_err = axes[1, i]
        err_im = ax_err.imshow(np.abs(target_image - approximated_image), cmap="hot", vmin=0, vmax=1)
        ax_err.set_title(f"Differenz zum Original")
        ax_err.set_xticks([])
        ax_err.set_yticks([])

    fig.suptitle("Visualisierung der Low-Rank-Approximation (LoRA-Prinzip)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    plt.savefig("lora_approximation_visualization.png", dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()
