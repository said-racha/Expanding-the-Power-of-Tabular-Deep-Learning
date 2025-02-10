import pandas as pd
import numpy as np
import torch
from torch import nn

# import excel_former
# import ft_transformer
# import t2g_former


from datasets import *
from models import *
from backbones import *
from utils import *
from training import *

if __name__ == "__main__":
    BATCH_SIZE = 32
    device = initialize_device()

    datasets = [
        # format : dataset_name, loader_func, task_type, lr, nb_epochs,embed_dim, num_heads
        ("breast_cancer", get_breast_cancer_data, "binary", 1e-4, 26, 24, 4),
        ("adult_income", get_adult_income_data, "binary", 1e-4, 20, 48, 6),
        ("california_housing", get_california_housing_data, "regression", 1e-7, 12, 10, 1),
        ("iris", get_iris_data, "multiclass", 1e-4, 85, 24, 4),
        ("wine_quality", get_quality_wine_data, "multiclass", 1e-2, 45, 192, 16)
    ]

    results = []

    for dataset_name, loader_func, task_type, lr, nb_epochs, embed_dim, num_heads in datasets:

        print(f"\n\n *** Processing {dataset_name} dataset...")
        train_loader, test_loader, shape_x, shape_y = loader_func(split=0.2, batch_size=BATCH_SIZE, seed=42)

        # Convert DataLoader to numpy arrays for XGBoost and LightGBM
        X_train, y_train = dataloader_to_numpy(train_loader)
        X_test, y_test = dataloader_to_numpy(test_loader)

        # Set hidden layers and criterion based on task type (multiclass, binary, regression)
        if task_type == "multiclass":
            layers = [64, 32, 16]
            criterion = nn.CrossEntropyLoss()
            out_features = shape_y
        elif task_type == "binary":
            layers = [64, 32]
            criterion = nn.BCEWithLogitsLoss()
            out_features = 1
        else:
            layers = [64, 32]
            criterion = nn.MSELoss()
            out_features = 1

        # Compare models
        input_dim = shape_x
        hidden_sizes = layers
        output_dim = out_features

        tabM_naive = EnsembleModel(TabM_naive, input_dim, hidden_sizes, output_dim).to(device)
        simple_MLP = MLP(input_dim, hidden_sizes, output_dim).to(device)
        tabM_mini = EnsembleModel(TabM_mini, input_dim, hidden_sizes, output_dim).to(device)
        tabM = EnsembleModel(TabM, input_dim, hidden_sizes, output_dim).to(device)
        mlpk = EnsembleModel(MLPk, input_dim, hidden_sizes, output_dim).to(device)

        embed_dim = embed_dim
        tabM_attention = TabMWithAttention(input_dim, hidden_sizes, embed_dim, output_dim=output_dim,
                                           num_heads=num_heads, k=32.3).to(device)

        mlpk_conf = EnsembleModel(MLPk, input_dim, hidden_sizes, output_dim, head_aggregation="conf").to(device)
        tabM_conf = EnsembleModel(TabM, input_dim, hidden_sizes, output_dim, head_aggregation="conf").to(device)
        tabM_naive_conf = EnsembleModel(TabM_naive, input_dim, hidden_sizes, output_dim, head_aggregation="conf").to(device)
        tabM_mini_conf = EnsembleModel(TabM_mini, input_dim, hidden_sizes, output_dim, head_aggregation="conf").to(device)

        # Define baselines (ExcelFormer, FT-Transformer, T2g-Former)
        model_config = {
            "d_numerical": input_dim,
            "categories": [],
            "token_bias": True,
            "n_layers": 3,
            "d_token": 256,
            "n_heads": 32,
            "attention_dropout": 0.3,
            "ffn_dropout": 0.0,
            "residual_dropout": 0.0,
            "prenormalization": True,
            "kv_compression": None,
            "kv_compression_sharing": None,
            "d_out": output_dim,
        }

        # ft = ft_transformer.FTTransformer(**model_config, d_ffn_factor=.6, activation="gelu", initialization="xavier").to(device)
        # excel = excel_former.ExcelFormer(**model_config).to(device)
        # t2g = t2g_former.T2GFormer(**model_config, d_ffn_factor=.6, activation="gelu", initialization="xavier").to(device)

        NB_ITER = nb_epochs

        # Training
        models = {
            "TabM_naive": tabM_naive,
            "Simple_MLP": simple_MLP,
            "TabM_mini": tabM_mini,
            "TabM": tabM,
            "MLPk": mlpk,
            "TabM_Attention": tabM_attention,
            # "FT-Transformer": ft,
            # "Excel-Former": excel,
            # "T2g-Former": t2g,
            "MLPk_conf": mlpk_conf,
            "TabM_conf": tabM_conf,
            "TabM_naive_conf": tabM_naive_conf,
            "TabM_mini_conf": tabM_mini_conf
        }

        for model_name, model in models.items():
            print(f"Training {model_name}...")
            result = train_model(model, train_loader, test_loader, f'runs/{dataset_name}/{model_name}', device, task_type=task_type, nb_iter=NB_ITER)
            results.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "Accuracy": result["accuracy"],
                "Loss": result["loss"],
                "F1-Score": result["f1_score"],
                "Precision": result["precision"],
                "Recall": result["recall"],
                "AUC Score": result["auc_score"],
                "RMSE": result["rmse"],
                "R²": result["r2"],
                "Execution Time": result["execution_time"]
            })

        print("Training XGBoost...")
        xgb_result = train_xgboost(X_train, y_train, X_test, y_test, task_type=task_type)
        results.append({
            "Dataset": dataset_name,
            "Model": "XGBoost",
            "Accuracy": xgb_result["accuracy"],
            "Loss": xgb_result["loss"],
            "F1-Score": xgb_result["f1_score"],
            "Precision": xgb_result["precision"],
            "Recall": xgb_result["recall"],
            "AUC Score": xgb_result["auc_score"],
            "RMSE": xgb_result["rmse"],
            "R²": xgb_result["r2"],
            "Execution Time": xgb_result["execution_time"]
        })

        print("Training LightGBM...")
        lgb_result = train_lightgbm(X_train, y_train, X_test, y_test, task_type=task_type)
        results.append({
            "Dataset": dataset_name,
            "Model": "LightGBM",
            "Accuracy": lgb_result["accuracy"],
            "Loss": lgb_result["loss"],
            "F1-Score": lgb_result["f1_score"],
            "Precision": lgb_result["precision"],
            "Recall": lgb_result["recall"],
            "AUC Score": lgb_result["auc_score"],
            "RMSE": lgb_result["rmse"],
            "R²": lgb_result["r2"],
            "Execution Time": lgb_result["execution_time"]
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv("results/result_all_models.csv")
