import torch
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn


def compute_correlation(preds):

    batch_size, _, _ = preds.size()
    spearman_corrs = []

    for i in range(batch_size):
        # Flatten predictions pour comparer les modèles entre eux
        preds_i = preds[i].detach().cpu().numpy()
        spearman_corr = stats.spearmanr(preds_i).statistic
        spearman_corrs.append(spearman_corr)

    return torch.tensor(spearman_corrs).mean()



def plot_predictions_distribution(preds,titre=""):

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

    
    # modèles
    layers = [64, 32, 16]
    tabM_naive = luc.EnsembleModel(luc.TabM_naive, 13, layers, 3, dropout_rate=0)
    tabM_mini = luc.EnsembleModel(luc.TabM_mini, 13, layers, 3, dropout_rate=0)
    tabM = luc.EnsembleModel(luc.TabM, 13, layers, 3, dropout_rate=0)
    mlpk = luc.EnsembleModel(luc.MLPk, 13, layers, 3, dropout_rate=0)
    


    # entraînement
    BATCH_SIZE = 32
    train_loader, test_loader = get_wine_data(split=.2, batch_size=BATCH_SIZE, seed=42)
    train_multiclass_classification(tabM, train_loader, test_loader, "runs/wine/raph/TabM", device, verbose=False)
    train_multiclass_classification(tabM_naive, train_loader, test_loader, "runs/wine/raph/TabM_naive", device, verbose=False)
    train_multiclass_classification(mlpk, train_loader, test_loader, "runs/wine/raph/mlpk", device, verbose=False)
    

    # données test
    tabM.mean_over_heads = False
    tabM_naive.mean_over_heads = False
    mlpk.mean_over_heads = False
    

    X_test = []
    for inputs, labels in test_loader:  # Itérez sur le DataLoader pour récupérer toutes les données
        X_test.append(inputs)
    X_test = torch.cat(X_test, dim=0).to(device)

    
    preds_tabm = tabM(X_test)
    preds_tabm_naive = tabM_naive(X_test)
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
    






    # test des initialisations
    perturbation_scales = [0.1, 0.5, 1.0, 2.0]
    distributions = ['uniform', 'normal', 'laplace']

    results = []

    for scale in perturbation_scales:
        for dist in distributions:
            model = raph.TabM(
                [13, 64, 32, 16, 3],
                amplitude=scale,
                init=dist
            )

            train_multiclass_classification(model, train_loader, test_loader, f"runs/wine/{dist}_{scale}", device)
            model.mean_over_heads = False
            preds = model(X_test)
            spearman = compute_correlation(preds)
            kl_divergence = compute_kl_divergence(preds)

            results.append({
                'scale': scale,
                'distribution': dist,
                'spearman': spearman.item()
            })

    import pandas as pd
    df = pd.DataFrame(results)
    print(df)

