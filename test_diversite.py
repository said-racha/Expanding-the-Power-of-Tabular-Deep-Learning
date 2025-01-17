import torch
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

def compute_correlation(preds):
    """
    Calcule la corrélation (Spearman) entre les prédictions des sous-modèles pour chaque exemple.
    Args:
        preds (torch.Tensor): Prédictions des sous-modèles de taille (batch_size, k, output_dim).
    Returns:
        torch.Tensor: Corrélation moyenne sur le batch (Spearman).
    """
    batch_size, _, _ = preds.size()
    spearman_corrs = []

    for i in range(batch_size):
        # Flatten predictions pour comparer les modèles entre eux
        preds_i = preds[i].detach().cpu().numpy()
        spearman_corr = stats.spearmanr(preds_i).statistic
        spearman_corrs.append(spearman_corr)

    return torch.tensor(spearman_corrs).mean()



def plot_predictions_distribution(preds):
    """
    Visualise la distribution des prédictions des sous-modèles (boxplots).
    Args:
        preds (torch.Tensor): Prédictions des sous-modèles de taille (batch_size, k, output_dim).
    """
    preds_mean = preds.mean(dim=0).detach().cpu().numpy()
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=preds_mean.T)
    plt.title(f"Distribution des prédictions des sous-modèles")
    plt.xlabel("Sous-modèles")
    plt.ylabel("Valeurs de prédiction")
    plt.show()


from scipy.special import kl_div
import numpy as np

def compute_kl_divergence(preds):
    """
    Calcule la KL-Divergence entre les distributions des prédictions des sous-modèles.
    Args:
        preds (torch.Tensor): Prédictions des sous-modèles de taille (batch_size, k, output_dim).
    Returns:
        float: Moyenne de la KL-Divergence sur le batch.
    """
    batch_size, k, output_dim = preds.size()
    kl_divergences = np.empty((batch_size,k,output_dim))

    for i in range(batch_size):
        example_preds = preds[i].detach().cpu().numpy()
        mean_dist = example_preds.mean(axis=0)

        for j in range(k):
            kl = 0.5 * (kl_div(example_preds[j], mean_dist)**2 + kl_div(mean_dist, example_preds[j])**2) # c'est un peu douteux comme formule car normalement on compare deux distributions, pas k
            kl_divergences[i,j] = kl
    return torch.tensor(kl_divergences).mean()


if __name__ == "__main__":

    import tabm_raph as raph
    from test_wine import get_wine_data
    from test_wine import train_multiclass_classification

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")


    # modèle
    model = raph.TabM([13, 64, 32, 16, 3]).to(device)

    # entraînement
    BATCH_SIZE = 128
    train_loader, test_loader = get_wine_data(split=.2, batch_size=BATCH_SIZE, seed=42)
    train_multiclass_classification(model, train_loader, test_loader, "runs/wine/raph/TabM", device, verbose=False)

    # données test
    model.mean_over_heads = False
    X_test = next(iter(test_loader))[0].to(device) # les tests sont effectués sur le premier batch
    preds = model(X_test)

    # Corrélations
    spearman_corr = compute_correlation(preds)
    print(f"Corrélation Spearman moyenne : {spearman_corr:.4f}")

    # Visualisation
    plot_predictions_distribution(preds)

    # KL-Divergence
    kl_divergence = compute_kl_divergence(preds)
    print(f"KL-Divergence moyenne : {kl_divergence:.4f}")

