import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import lines as mlines
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error

from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.kernel_ridge import KernelRidge

from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge

def feature_selection(selection_method, X_train, y_train,
                      n_features=None, scoring=None, alpha=None, min_features=None,
                      X_internal_test=None, X_external_test=None):

    X_internal_test_selected = None
    X_external_test_selected = None

    if selection_method == 'select_k_best':
        selector = SelectKBest(score_func=scoring or f_regression, k=n_features)

    elif selection_method == 'rfe_cv':
        estimator = Ridge(alpha=alpha or 0.3)
        selector = RFECV(
            estimator=estimator,
            step=1,
            cv=KFold(5, shuffle=True, random_state=42),
            scoring= scoring or 'neg_mean_squared_error',
            min_features_to_select=min_features or 2
        )

    elif selection_method == 'rfe':
        estimator = Ridge(alpha=alpha or 0.3)
        selector = RFE(estimator, n_features_to_select=n_features)
    
    else:
        raise ValueError("selection_method must be 'select_k_best', 'rfe_cv' or 'rfe' ")


    X_train_selected = selector.fit_transform(X_train, y_train)
    if selection_method == 'rfe_cv':
        print("Optimal number of features:", selector.n_features_)
    if X_internal_test is not None:
        X_internal_test_selected = selector.transform(X_internal_test)
    if X_external_test is not None:
        X_external_test_selected = selector.transform(X_external_test)

    selected_mask = selector.get_support()
    selected_features = X_train.columns[selected_mask]
    print("Selected feature names:", selected_features.tolist())

    return X_train_selected, X_internal_test_selected, X_external_test_selected, selected_features

def regression(method, X_train, y_train, y_train_labels, kernel=None,
               n_neighbors=None, weights=None, p=None, alpha=None,
               X_internal_test=None, y_internal_test=None, y_internal_test_labels=None,
               X_external_test=None, y_external_test=None, y_external_test_labels=None):
    
    dfs = [] 
    #general function to get metrics 
    def compute_metrics_df(y_true, y_pred, labels, dataset_name, model_type):
        r2 = np.nan if y_true is None else np.corrcoef(y_true, y_pred)[0,1]**2
        mae = np.nan if y_true is None else np.mean(np.abs(y_true - y_pred))
        print(f"{model_type} {dataset_name} R2: {r2:.3f}, MAE: {mae:.3f}")
        col_name = f"predicted_value_{model_type.lower()}"
        return pd.DataFrame({
            'ligandID': labels,
            'true_value': y_true,
            col_name: y_pred,
            'set': dataset_name
        }), r2, mae

    # KNN regression functionality 
    if method == 'knn-r':
        model = KNeighborsRegressor(
            n_neighbors=n_neighbors or 10,
            weights=weights or 'distance',
            p=p or 1
        )
        model_type = "knn"
        model.fit(X_train, y_train)

        #get predictions and metrics 
        y_train_pred = model.predict(X_train)
        y_internal_pred = model.predict(X_internal_test) if X_internal_test is not None else None
        y_external_pred = model.predict(X_external_test) if X_external_test is not None else None

        df_train, _, _ = compute_metrics_df(y_train, y_train_pred, y_train_labels, 'train', model_type)
        dfs.append(df_train)

        if y_internal_pred is not None:
            df_internal, _, _ = compute_metrics_df(y_internal_test, y_internal_pred, y_internal_test_labels, 'internal_test', model_type)
            dfs.append(df_internal)

        if y_external_pred is not None:
            df_external, _, _ = compute_metrics_df(y_external_test, y_external_pred, y_external_test_labels, 'external_test', model_type)
            dfs.append(df_external)

    # KRR functionality 
    elif method == 'krr':
        model_type = "krr"
        model = KernelRidge(kernel=kernel, alpha=alpha)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_internal_pred = model.predict(X_internal_test) if X_internal_test is not None else None
        y_external_pred = model.predict(X_external_test) if X_external_test is not None else None

        col_name = f'predicted_value_{model_type.lower()}'

        df_train = pd.DataFrame({'ligandID': y_train_labels,
                                'true_value': y_train,
                                col_name: y_train_pred,
                                'set': 'train'})
        dfs.append(df_train)

        if y_internal_pred is not None:
            df_internal = pd.DataFrame({'ligandID': y_internal_test_labels,
                                        'true_value': y_internal_test,
                                        col_name: y_internal_pred,
                                        'set': 'internal_test'})
            dfs.append(df_internal)

        if y_external_pred is not None:
            df_external = pd.DataFrame({'ligandID': y_external_test_labels,
                                        'true_value': y_external_test,
                                        col_name: y_external_pred,
                                        'set': 'external_test'})
            dfs.append(df_external)

    # make sure df_all is always defined
    if len(dfs) > 0:
        df_all = pd.concat(dfs, ignore_index=True)
    else:
        df_all = pd.DataFrame()  # return empty dataframe if nothing was predicted

    return df_all

def model_selection_knn(X_train, y_train, y_train_labels,
                       X_internal_test, y_internal_test, y_internal_test_labels,
                       scoring_list, k_list, neighbors_list):

    best_r2 = -np.inf
    best_mae = np.inf

    best_params_r2 = None
    best_params_mae = None

    best_df_r2 = None
    best_df_mae = None

    best_features_r2 = None
    best_features_mae = None

    for scoring in scoring_list:
        for k in k_list:
            X_train_sel, X_int_sel, _, selected_features = feature_selection(
                selection_method='select_k_best',
                X_train=X_train,
                y_train=y_train,
                n_features=k,
                scoring=scoring,
                X_internal_test=X_internal_test,
                X_external_test=None
            )

            for n in neighbors_list:
                print(f"\nTesting: scoring={scoring.__name__}, k={k}, n_neighbors={n}")

                df_results = regression(
                    method='knn-r',
                    X_train=X_train_sel,
                    y_train=y_train,
                    y_train_labels=y_train_labels,
                    n_neighbors=n,
                    X_internal_test=X_int_sel,
                    y_internal_test=y_internal_test,
                    y_internal_test_labels=y_internal_test_labels,
                )

                subset_internal = df_results[df_results['set'] == 'internal_test']

                if len(subset_internal) == 0:
                    continue

                y_true = subset_internal['true_value']
                y_pred = subset_internal['predicted_value_knn']

                r2_internal = np.corrcoef(y_true, y_pred)[0,1]**2
                mae_internal = np.mean(np.abs(y_true - y_pred))

                print(f"--> Internal R2: {r2_internal:.3f}, MAE: {mae_internal:.3f}")

                # params that give best r2
                if r2_internal > best_r2:
                    best_r2 = r2_internal
                    best_params_r2 = {
                        'scoring': scoring.__name__,
                        'k': k,
                        'n_neighbors': n
                    }
                    best_df_r2 = df_results.copy()
                    best_features_r2 = selected_features.tolist()

                # params that give best mae
                if mae_internal < best_mae:
                    best_mae = mae_internal
                    best_params_mae = {
                        'scoring': scoring.__name__,
                        'k': k,
                        'n_neighbors': n
                    }
                    best_df_mae = df_results.copy()
                    best_features_mae = selected_features.tolist()

    print("\n Best Model by Internal Test R2")
    print(best_params_r2)
    print(f"R2: {best_r2:.3f}")
    print("Selected Features:", best_features_r2)

    print("\n Best Model by Internal Test MAE")
    print(best_params_mae)
    print(f"MAE: {best_mae:.3f}")
    print("Selected Features:", best_features_mae)

    # plot best models
    def plot_results(df, title):
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.set_xlabel('Experimental $ΔΔG^{‡}$', fontsize=12)
        ax.set_ylabel('Predicted $ΔΔG^{‡}$ (kcal/mol)', fontsize=12)
        ax.set_title(title)

        for subset_name, color in zip(['train', 'internal_test'],
                                     ['black', 'blue']):
            subset = df[df['set'] == subset_name]
            ax.scatter(subset['true_value'],
                       subset['predicted_value_knn'],
                       color=color,
                       label=subset_name.replace('_', ' ').title(),
                       alpha=0.7)

        ax.legend()
        plt.show()

    plot_results(best_df_r2, "Best Model (R²)")
    plot_results(best_df_mae, "Best Model (MAE)")

    return (best_df_r2, best_params_r2, best_features_r2,
            best_df_mae, best_params_mae, best_features_mae)


def model_selection_krr(X_train, y_train, y_train_labels,
                        X_internal_test, y_internal_test, y_internal_test_labels,
                        ridge_alpha_list=[0.1, 0.3, 1.0],
                        krr_alpha_list=[0.3, 0.7, 1.0],
                        kernel_list=['linear', 'polynomial', 'rbf'],
                        n_features_list=[2, 3, 4, 5],
                        scoring=None):

    best_r2 = -np.inf
    best_mae = np.inf

    best_params_r2 = None
    best_params_mae = None

    best_df_r2 = None
    best_df_mae = None

    best_features_r2 = None
    best_features_mae = None

    # Only plain RFE
    selection_method = 'rfe'

    for ridge_alpha in ridge_alpha_list:
        for n_features in n_features_list:
            X_train_sel, X_int_sel, _, selected_features = feature_selection(
                selection_method=selection_method,
                X_train=X_train,
                y_train=y_train,
                n_features=n_features,
                alpha=ridge_alpha,
                X_internal_test=X_internal_test,
                X_external_test=None,
                scoring=scoring
            )

            for krr_alpha in krr_alpha_list:
                for kernel in kernel_list:
                    print(f"\nTesting: ridge_alpha={ridge_alpha}, krr_alpha={krr_alpha}, "
                          f"kernel={kernel}, n_features={n_features}")

                    df_results = regression(
                        method='krr',
                        X_train=X_train_sel,
                        y_train=y_train,
                        y_train_labels=y_train_labels,
                        alpha=krr_alpha,
                        kernel=kernel,
                        X_internal_test=X_int_sel,
                        y_internal_test=y_internal_test,
                        y_internal_test_labels=y_internal_test_labels
                    )

                    subset_internal = df_results[df_results['set'] == 'internal_test']
                    if len(subset_internal) == 0:
                        continue

                    y_true = subset_internal['true_value']
                    y_pred = subset_internal['predicted_value_krr']

                    r2_internal = np.corrcoef(y_true, y_pred)[0,1]**2
                    mae_internal = np.mean(np.abs(y_true - y_pred))

                    print(f"--> Internal R2: {r2_internal:.3f}, MAE: {mae_internal:.3f}")

                    # Best R2
                    if r2_internal > best_r2:
                        best_r2 = r2_internal
                        best_params_r2 = {
                            'ridge_alpha': ridge_alpha,
                            'krr_alpha': krr_alpha,
                            'kernel': kernel,
                            'n_features': n_features
                        }
                        best_df_r2 = df_results.copy()
                        best_features_r2 = selected_features.tolist()

                    # Best MAE
                    if mae_internal < best_mae:
                        best_mae = mae_internal
                        best_params_mae = {
                            'ridge_alpha': ridge_alpha,
                            'krr_alpha': krr_alpha,
                            'kernel': kernel,
                            'n_features': n_features
                        }
                        best_df_mae = df_results.copy()
                        best_features_mae = selected_features.tolist()

    print("\nBest Model by Internal Test R2")
    print(best_params_r2)
    print(f"R2: {best_r2:.3f}")
    print("Selected Features:", best_features_r2)

    print("\nBest Model by Internal Test MAE")
    print(best_params_mae)
    print(f"MAE: {best_mae:.3f}")
    print("Selected Features:", best_features_mae)

    # plot best models
    def plot_results(df, title):
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.set_xlabel('Experimental $ΔΔG^{‡}$', fontsize=12)
        ax.set_ylabel('Predicted $ΔΔG^{‡}$ (kcal/mol)', fontsize=12)
        ax.set_title(title)

        for subset_name, color in zip(['train', 'internal_test'],
                                     ['black', 'blue']):
            subset = df[df['set'] == subset_name]
            col = 'predicted_value_krr'
            ax.scatter(subset['true_value'],
                       subset[col],
                       color=color,
                       label=subset_name.replace('_', ' ').title(),
                       alpha=0.7)

        ax.legend()
        plt.show()

    plot_results(best_df_r2, "Best Model (R²)")
    plot_results(best_df_mae, "Best Model (MAE)")

    return (best_df_r2, best_params_r2, best_features_r2,
            best_df_mae, best_params_mae, best_features_mae)