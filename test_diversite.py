import torch
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn

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



def plot_predictions_distribution(preds,titre=""):
    """
    Visualise la distribution des prédictions des sous-modèles (boxplots).
    Args:
        preds (torch.Tensor): Prédictions des sous-modèles de taille (batch_size, k, output_dim).
    """
    preds_mean = preds.mean(dim=0).detach().cpu().numpy()
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=preds_mean.T)
    if titre=="":
        plt.title("Distribution des prédictions des sous-modèles")
    else:
        plt.title("Distribution des prédictions des sous-modèles (" + titre + ")")
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
    import tabm_luc as luc
    from test_wine import get_wine_data
    from test_wine import train_multiclass_classification

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")


    # modèle
    tabm = raph.TabM([13, 64, 32, 16, 3]).to(device)
    tabm_naive = raph.TabM_Naive([13, 64, 32, 16, 3]).to(device)
    mlpk = luc.EnsembleModel(luc.MLPk, 13, [64, 32, 16], 3, dropout_rate=0)



    # entraînement
    BATCH_SIZE = 200
    train_loader, test_loader = get_wine_data(split=.2, batch_size=BATCH_SIZE, seed=42)
    train_multiclass_classification(tabm, train_loader, test_loader, "runs/wine/raph/TabM", device, verbose=False)
    train_multiclass_classification(tabm_naive, train_loader, test_loader, "runs/wine/raph/TabM", device, verbose=False)
    train_multiclass_classification(mlpk, train_loader, test_loader, "runs/wine/raph/TabM", device, verbose=False)


    # données test
    tabm.mean_over_heads = False
    tabm_naive.mean_over_heads = False
    mlpk.mean_over_heads = False
    X_test = next(iter(test_loader))[0].to(device) # les tests sont effectués sur le premier batch
    preds_tabm = tabm(X_test)
    preds_tabm_naive = tabm_naive(X_test)
    preds_mlpk = mlpk(X_test)

    # Corrélations
    spearman_tabm = compute_correlation(preds_tabm)
    spearman_tabm_naive = compute_correlation(preds_tabm_naive)
    spearman_mlpk = compute_correlation(preds_mlpk)
    print(f"Corrélation Spearman moyenne (TabM) : {spearman_tabm:.4f}")
    print(f"Corrélation Spearman moyenne (TabM Naive) : {spearman_tabm_naive:.4f}")
    print(f'Corrélation Spearman moyenne (MLP_k) : {spearman_mlpk:.4f}')

    # Visualisation
    plot_predictions_distribution(preds_tabm,"TabM")
    plot_predictions_distribution(preds_tabm_naive, "TabM Naive")
    plot_predictions_distribution(preds_mlpk, "MLP_k")


    # KL-Divergence
    kl_tabm = compute_kl_divergence(preds_tabm)
    kl_tabm_naive = compute_kl_divergence(preds_tabm_naive)
    kl_mlpk = compute_kl_divergence(preds_mlpk)
    print(f"KL-Divergence moyenne (TabM): {kl_tabm:.4f}")
    print(f"KL-Divergence moyenne (TabM Naive): {kl_tabm_naive:.4f}")
    print(f"KL-Divergence moyenne (MLP_k): {kl_mlpk:.4f}")

