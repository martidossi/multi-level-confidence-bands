import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Experiment
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This notebook details the process of **selecting data points** (houses) for inclusion in the survey and preparing the corresponding **experimental interface**. To see the `code` behind, you can click on the “Show code” toggle button in the top right corner of the app interface:

    <img src="https://raw.githubusercontent.com/martidossi/multi-level-confidence-bands/main/pics/see_code.png" width="300">

    ⚠️ _**Note**: this notebook is best viewed on a desktop screen for full functionality and layout, some elements may not display properly on mobile._
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Index

    **Part 1**: [Data selection](#1-data-selection)

    - [1.1 _California housing_ dataset](#11-california-housing-dataset)
    - [1.2 Model fit](#12-model-fit)
    - [1.3 House selection](#13-house-selection)
    - [1.4 Choropleth map](#14-choropleth-map)

    **Part 2**: [Survey interface](#2-survey-interface)

    - [2.1 Data](#21-data)
    - [2.2 Ranged dot plot](#22-ranged-dot-plot)
    - [2.3 Heatmaps](#23-heatmaps)

    ---
    """)
    return


@app.cell
def _():
    # Libraries

    import marimo as mo

    import sys
    import os
    import glob

    import yaml

    import pandas as pd
    import numpy as np

    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    import matplotlib.ticker as mticker

    import geoplot as gplt
    import geopandas as gpd
    import geoplot.crs as gcrs

    from sklearn.datasets import fetch_california_housing

    from sklearn.model_selection import train_test_split
    from interpret.glassbox import ExplainableBoostingRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

    from datetime import datetime

    from interpret.glassbox import ExplainableBoostingClassifier
    from interpret import show
    return (
        ExplainableBoostingRegressor,
        datetime,
        fetch_california_housing,
        gcrs,
        glob,
        gpd,
        gplt,
        mean_squared_error,
        mo,
        mticker,
        np,
        os,
        pd,
        plt,
        sys,
        train_test_split,
        yaml,
    )


@app.cell
def _(os, sys):
    sys.path.append(os.path.abspath("src/"))
    from utils import calculate_coverage, get_conformalized_interval
    return (get_conformalized_interval,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1. Data selection
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.1 _California housing_ dataset
    """)
    return


@app.cell
def _(fetch_california_housing, pd):
    df_california = fetch_california_housing(as_frame=True)

    df = df_california.frame
    df['Longitude'] = df['Longitude'].round(2)
    df['Latitude'] = df['Latitude'].round(2)
    df['MedHouseVal'] = df['MedHouseVal'].round(5)

    X = pd.DataFrame(df_california.data, columns=df_california.feature_names)
    y = df_california.target
    n = len(y)
    return X, df, y


@app.cell
def _(df):
    df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ------
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.2 Model fit
    """)
    return


@app.cell
def _():
    # Random seed to ensure reproducibility of the data selection process (dataset split and model estimation):
    seed = 42
    return (seed,)


@app.cell
def _(X, seed, train_test_split, y):
    # Dataset split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.6, random_state=seed)
    X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, train_size=0.5, random_state=seed)

    #print(f"Proportion training set: {len(X_train)/len(X):.2f}")
    #print(f"Proportion test set: {len(X_test)/len(X):.2f}")
    #print(f"Proportion calibration set: {len(X_calib)/len(X):.2f}")
    return X_calib, X_test, X_train, y_calib, y_test, y_train


@app.cell(hide_code=True)
def _(X, X_calib, X_test, X_train, mo, seed):
    train_prop = len(X_train) / len(X)
    test_prop = len(X_test) / len(X)
    calib_prop = len(X_calib) / len(X)

    mo.md(f"""
    ### 1.2.1 Dataset split
    Proportions of training, test, and calibration sets:

    - `training set`: {train_prop:.0%}  
    - `test set`: {test_prop:.0%}  
    - `calibration set`: {calib_prop:.0%}

    For reproducibility, we set the random seed to **{seed}**.
    """)
    return


@app.cell
def _(
    ExplainableBoostingRegressor,
    X_calib,
    X_test,
    X_train,
    mean_squared_error,
    mo,
    np,
    seed,
    y_test,
    y_train,
):
    # Model fit and predict with default hyperparameters
    ebm_model = ExplainableBoostingRegressor(random_state=seed)
    ebm_model.fit(X_train, y_train)

    # Predictions
    y_pred_train = ebm_model.predict(X_train)
    y_pred_calib = ebm_model.predict(X_calib)
    y_pred_test = ebm_model.predict(X_test)

    # RMSE
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    mo.md(f"""
    ### 1.2.2 Model performance
    | Dataset | RMSE |
    |:--------|------:|
    | **Train** | {rmse_train:.3f} |
    | **Test**  | {rmse_test:.3f} |
    """)
    return ebm_model, y_pred_calib, y_pred_test


@app.cell
def _(plt, y_pred_test, y_test):
    # Residual analysis

    fig_resid, ax_resid = plt.subplots(1, 2, figsize=(10, 4))
    ax_resid[0].plot(
        y_test, (y_test-y_pred_test), ls='', marker='o', markersize=5, alpha=0.7, color='royalblue',
        markeredgecolor="white", markeredgewidth=0.8
    )
    ax_resid[0].axhline(0, color='red', ls='-')
    ax_resid[0].grid()
    ax_resid[0].set_xlabel('y test')
    ax_resid[0].set_ylabel('residuals')
    ax_resid[0].set_title('Residuals vs true values')

    ax_resid[1].hist((y_test-y_pred_test), bins=30, edgecolor="black", alpha=0.7, color='royalblue')
    ax_resid[1].axvline(0, color='red', ls='-')
    ax_resid[1].grid()
    ax_resid[1].set_xlabel('residuals')
    ax_resid[1].set_title('Residual distribution')

    plt.suptitle('Residual analysis', fontsize=14)
    plt.tight_layout()
    fig_resid
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Note:** The _California housing_ dataset is known to be *capped* at a maximum median house value of 5.0 (≈ \$500 000). As a result, observations at this upper bound show larger residuals, since the true prices exceed the recorded limit. This censoring effect explains the visible concentration of residuals around 5 in the plot.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ------
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.2.3 Feature importance
    The visualization below shows **global feature importance** as derived from the EBM model. Feature importance in EBM does _not_ represent simple correlation with the target variable. Instead, it measures each feature’s **contribution to the model’s predictions** (how much the predicted value changes when the feature varies, according to the model’s learned shape functions). This plot shows which features the model relies on most across the entire dataset, providing a **global explanation** of the model’s behavior.
    """)
    return


@app.cell
def _(ebm_model, pd, plt):
    # Global explanation

    # Interactive dashboard
    # global_explanation = ebm_model.explain_global()
    # show(global_explanation)

    global_exp = ebm_model.explain_global()
    df_ebm_feat_imp = pd.DataFrame(global_exp.data()).sort_values(by='scores')

    # Plot feature importance
    fig_feat_imp, ax_feat_imp = plt.subplots(figsize=(8, 5))

    ax_feat_imp.barh(df_ebm_feat_imp["names"], df_ebm_feat_imp["scores"], color="royalblue", alpha=0.8)
    ax_feat_imp.set_xlabel("feature importance")
    ax_feat_imp.set_ylabel("feature")
    ax_feat_imp.set_title("EBM Feature importance")
    ax_feat_imp.grid(alpha=0.5)

    fig_feat_imp.tight_layout()
    fig_feat_imp
    return


@app.cell
def _():
    # Local explanation (why a single prediction was made)
    # Interactive dashboard
    # local_explanation = ebm_model.explain_local(X_test_houses, y_test_houses)
    # show(local_explanation)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ------
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.3 House selection

    To select houses for the survey that span a range of **prediction accuracies**, we first rank them by their absolute residuals (`y_true − y_pred`). Our goal here is to define three difficulty levels: `easy`, `medium`, and `hard`, based on model accuracy. We look at the **quantiles** of the residual distribution to guide this process. The histogram below shows the distribution across the test set, with the **75th** and **95th percentiles** (dashed lines) marking the thresholds for house selection.
    """)
    return


@app.cell
def _(np, plt, y_pred_test, y_test):
    # Compute quantile thresholds
    q1 = (np.abs(y_test-y_pred_test)).quantile(0.75)
    q2 = (np.abs(y_test-y_pred_test)).quantile(.95)

    # Plot
    fig_err, ax_err = plt.subplots(figsize=(6, 4))
    ax_err.hist((np.abs(y_test-y_pred_test)), bins=50, color="royalblue", edgecolor="black", alpha=0.7)

    # Add threshold lines
    ax_err.axvline(q1, color="orange", linestyle="--", linewidth=2,
                   label=f"easy – medium threshold (75th pct = {q1:.2f})")
    ax_err.axvline(q2, color="red", linestyle="--", linewidth=2,
                   label=f"medium – hard threshold (95th pct = {q2:.2f})")

    ax_err.set_title("Distribution of (absolute) residuals", fontsize=14)
    ax_err.set_xlabel("residuals", fontsize=14)
    ax_err.set_ylabel("count", fontsize=14)
    ax_err.legend(fontsize=9)
    ax_err.grid(alpha=0.5, linestyle="--")
    ax_err.tick_params(axis="both", which="major", labelsize=12)

    fig_err.tight_layout()
    fig_err
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Rather than random sampling, we prefered a **targeted approach** to also ensure geographic diversity and balanced coverage across the target variable range. Below is the dataset of the nine selected houses, each described by its name, location, true value, and difficulty level.
    """)
    return


@app.cell
def _(pd, yaml):
    # Load yaml houses file
    # difficulty_level refers to the prediction error magnitude, classifies into three possible classes
    with open("src/houses.yml", "r") as f:
        dict_houses = yaml.safe_load(f)

    df_houses = pd.DataFrame(dict_houses)

    df_houses['longitude'] = df_houses['longitude'].round(2)
    df_houses['latitude'] = df_houses['latitude'].round(2)
    df_houses['true_value'] = df_houses['true_value'].round(5)
    return df_houses, dict_houses


@app.cell
def _(X_test, df, df_houses, pd, y_test):
    # Merge with complete df
    df_house_match = pd.merge(
        df,
        df_houses,
        left_on=["MedHouseVal", "Longitude", "Latitude"],
        right_on=["true_value", "longitude", "latitude"],
        how="left"
    )

    # Indexes of selected houses
    idx_house_match = df_house_match.index[df_house_match['name'].notna()].tolist()

    # Filtered y and X
    y_test_houses = y_test[y_test.index.isin(idx_house_match)]
    X_test_houses = X_test[X_test.index.isin(idx_house_match)]
    df_house_match[df_house_match.name.notna()][list(df_houses.columns)] #.sort_values(by='difficulty_level')
    return X_test_houses, idx_house_match, y_test_houses


@app.cell
def _(idx_house_match, np, pd, y_pred_test, y_test):
    df_test_pred = pd.DataFrame({
        'y_test': y_test,
        'y_pred': y_pred_test
    })
    df_test_pred['pred_error'] = np.abs(
        df_test_pred['y_test'] - df_test_pred['y_pred']
    )

    df_test_pred["sel_houses"] = df_test_pred["y_test"].index.isin(idx_house_match)
    df_test_pred = df_test_pred.sort_values(by='pred_error')#.reset_index(drop=True)
    sort_idx_true = df_test_pred.index[df_test_pred["sel_houses"] == True]
    df_test_pred.loc[sort_idx_true]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.4 Visualization of the selected house
    """)
    return


@app.cell
def _(
    X_test,
    datetime,
    df_houses,
    get_conformalized_interval,
    pd,
    y_calib,
    y_pred_calib,
    y_pred_test,
    y_test,
):
    # Save data

    alphas = [0.001, 0.01, 0.05, 0.1, 0.2][::-1]
    save_dfs = False

    if save_dfs == True:
        now = datetime.now()
        print("Last run:", now)
        for alpha in alphas: 
            conf_lower_bound, conf_upper_bound = get_conformalized_interval(
                y_test_pred=y_pred_test, 
                y_calib_pred=y_pred_calib,
                y_calib=y_calib,
                alpha=alpha
            )
            df_test = pd.DataFrame(data={
                'lower_bound': conf_lower_bound,
                'upper_bound': conf_upper_bound,
                'y_test': y_test,
                'y_test_pred': y_pred_test
            })

            # Adding house attributes
            df_test = X_test.join(df_test)

            # Adding info about houses
            df_test_house = pd.merge(
                df_test,
                df_houses,
                left_on=["y_test", "Longitude", "Latitude"],
                right_on=["true_value", "longitude", "latitude"],
                how="inner"
            )

            # Drop duplicated columns
            df_test_house = df_test_house.drop(["true_value", "longitude", "latitude"], axis=1)

            df_to_save = df_test_house #[df_test_house.name.notna()]

            if len(df_to_save)==9:
                print(f'Saving dataset for alpha {alpha}')
                df_to_save.to_csv(f'./data/output/houses_conformal_predictions/houses_conformal_{str(alpha).replace(r'0.', '')}.csv', index=False)
    return (alphas,)


@app.cell
def _(glob, os, pd):
    # Read data

    IN_PYODIDE = os.environ.get("PYODIDE") == "1"

    def load_house_predictions(folder="./data/output/houses_conformal_predictions/"):
        dataframes = {}

        if IN_PYODIDE:
            # Pyodide (browser) – must fetch via URLs
            from pyodide.http import open_url
            base_url = "https://martidossi.github.io/multi-level-confidence-bands/data/output/houses_conformal_predictions"
            for alpha in ["001", "01", "05", "10", "20"]:
                url = f"{base_url}/houses_conformal_{alpha}.csv"
                df = pd.read_csv(open_url(url))
                dataframes[alpha] = df
        else:
            # Local environment – read from filesystem
            csv_files = glob.glob(os.path.join(folder, "houses_conformal_*.csv"))
            for file in csv_files:
                basename = os.path.basename(file)
                alpha = basename.replace("houses_conformal_", "").replace(".csv", "")
                df = pd.read_csv(file)
                dataframes[alpha] = df

        return dataframes
    return (load_house_predictions,)


@app.cell
def _(load_house_predictions):
    dataframes = load_house_predictions()
    # dataframes.keys()
    return (dataframes,)


@app.cell
def _(dataframes, pd):
    pd.DataFrame(dataframes['001'])
    return


@app.cell
def _(np, pd):
    # Houses and their classification in terms of prediction error with standard confidence level of 0.05

    df_plot_houses = pd.read_csv("data/output/houses_conformal_predictions/houses_conformal_05.csv")

    # Convert the column to a categorical with the specified order
    df_plot_houses['category'] = pd.Categorical(df_plot_houses['category'], categories=['easy', 'medium', 'hard'], ordered=True)
    df_plot_houses = df_plot_houses.sort_values(by='category', ascending=False)

    df_plot_houses['lower_diff'] = df_plot_houses['y_test_pred'] - df_plot_houses['lower_bound']
    df_plot_houses['upper_diff'] = df_plot_houses['upper_bound'] - df_plot_houses['y_test_pred']

    df_plot_houses['abs_error'] = np.abs(df_plot_houses['y_test'] - df_plot_houses['y_test_pred'])
    df_plot_houses = df_plot_houses.sort_values(by=['abs_error', 'category'], ascending=[False, False])

    #df_plot_houses
    return (df_plot_houses,)


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2. Survey interface
    """)
    return


@app.cell
def _(mo):
    list_model_houses = [
        'Artemisstreet', 'Aphroditestreet', 'Apollostreet', 
        'Hephaestusstreet', 'Athenastreet', 'Poseidonstreet',
        'Aresstreet', 'Hestiastreet', 'Zeusstreet'
    ]

    dropdown_model_house = mo.ui.dropdown(
        options=list_model_houses, value='Artemisstreet', label="Which house?"
    )

    dropdown_model_house
    return dropdown_model_house, list_model_houses


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Choose the house
    """)
    return


@app.cell(hide_code=True)
def _(dict_houses, mo):
    dropdown_house = mo.ui.dropdown(
        options=[el['name'] for el in dict_houses], value='Artemisstreet', label="Which house?"
    )
    dropdown_house
    return (dropdown_house,)


@app.cell
def _(df_houses, dropdown_house):
    pred_error = df_houses[df_houses.name==dropdown_house.value].category.values[0]
    print(f'The selected house has a {pred_error} prediction error.')
    return


@app.cell
def _(pd, train_test_split):
    url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
    df_url = pd.read_csv(url)
    X_url = df_url.drop('median_house_value', axis=1)
    y_url = df_url['median_house_value']
    X_train_url, x_, y_train_url, y_ = train_test_split(X_url, y_url, test_size=0.4, random_state=43)
    X_train_url.describe()
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell(hide_code=True)
def _(
    X_train,
    df_plot_houses,
    dict_houses,
    dropdown_house,
    gcrs,
    gpd,
    gplt,
    mticker,
    np,
    plt,
    y_train,
):
    import warnings
    warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS")

    cali = gpd.read_file(gplt.datasets.get_path('california_congressional_districts'))
    # cali = cali.to_crs(epsg=3310) # # Reproject to a suitable projected CRS (California Albers)
    cali = cali.assign(area=cali.geometry.area) # area in square meters
    # If you want square kilometers:
    # cali["area_km2"] = cali["area"] / 1e6

    # General config
    cols = ['MedHouseVal', 'HouseAge', 'MedInc', 'Population']
    cols_renaming = {
        'MedHouseVal': ' Median house value ($)', 
        'HouseAge': 'Housing median age',
        'MedInc': 'Median income per year ($)',
        'Population': 'Population'
    }
    df_train = X_train.join(y_train)
    df_train['MedInc'] = df_train['MedInc']*10 # scaled

    arrow_vals = df_plot_houses[df_plot_houses.name==dropdown_house.value].rename(columns={'y_test': 'MedHouseVal'})[cols]
    arrow_vals['MedInc'] = arrow_vals['MedInc']*10
    arrow_vals_rounded = [round(float(v), 2) for v in arrow_vals.values.flatten()]
    arrow_vals_rounded[0] = None

    # Define single house data
    house = [el for el in dict_houses if el['name'] == dropdown_house.value][0]

    house_gdf = gpd.GeoDataFrame(
        [house],
        geometry=gpd.points_from_xy([house["longitude"]], [house["latitude"]]),
        crs="EPSG:4326"
    )

    # input coordinates are in lat/lon (EPSG:4326):
    gdf = gpd.GeoDataFrame(
        df_train.copy(), 
        geometry=gpd.points_from_xy(df_train.Longitude, df_train.Latitude),
        crs="EPSG:4326"
    ) # GeoDataFrame
    proj = gcrs.AlbersEqualArea(central_latitude=37.16611, central_longitude=-119.44944)

    fig, axes = plt.subplots(2, 2, figsize=(18, 18), subplot_kw={'projection': proj})
    axes = axes.flatten()

    for j, col in enumerate(cols):
        ax = axes[j]
        tgdf = gdf.sort_values(by=col)
        gplt.polyplot(cali, projection=proj, ax=ax)
        pointplot = gplt.pointplot(tgdf, ax=ax, hue=col, cmap='viridis', legend=True, alpha=0.4, s=3)
        house_proj = house_gdf.to_crs(ax.projection)

        ax.scatter(
            house_proj.geometry.x,
            house_proj.geometry.y,
            color='red',
            s=100,
            marker='o',
            edgecolor='black',
            label='Target house',
            zorder=5
        )

        cbar = pointplot.get_figure().axes[-1]

        if col == "MedHouseVal":
            formatter = mticker.FuncFormatter(lambda x, pos: f'{int(x)*100}k')
            cbar_format_ax = cbar 
            cbar_format_ax.yaxis.set_major_formatter(formatter)  

        if col == "MedInc":
            formatter = mticker.FuncFormatter(lambda x, pos: f'{int(x)}k')
            cbar_format_ax = cbar 
            cbar_format_ax.yaxis.set_major_formatter(formatter)  

        if arrow_vals_rounded[j] is not None:
            cbar = plt.gcf().axes[-1]
            val = arrow_vals_rounded[j]
            norm = plt.Normalize(vmin=tgdf[col].min(), vmax=tgdf[col].max())
            frac = np.clip(norm(val), 0, 1)
            if col == "MedInc":
                label_val = f'{int(val)}k'
            else:
                label_val = f'{val:.0f}'
            cbar.annotate(
                label_val,
                xy=(-0.05, frac),
                xycoords='axes fraction',
                xytext=(-20, 0),
                textcoords='offset points',
                ha='right',
                va='center',
                fontsize=12,
                color='black',
                arrowprops=dict(arrowstyle="->", color='black', linewidth=1)
            )
        ax.legend(loc='upper right', fontsize=12)
        ax.set_title(cols_renaming[col], fontsize=16)

    #plt.suptitle("Housing features across California", fontsize=20, y=1.02)

    plt.suptitle(f"{dropdown_house.value}\nHousing features across California", fontsize=20, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(f"pics/house_feature_{dropdown_house.value}.png", dpi=600, bbox_inches="tight")
    plt.show()
    return


@app.cell(hide_code=True)
def _(alphas, np):
    # Setup

    bar_height = 1.0  # Height of each confidence level block

    # Setup the updated grayscale range
    start_gray_updated = 0.99 # 0.99
    end_gray_final = 0.88 # 0.88

    # Calculate normalized confidence deltas
    confidences = [1 - alpha for alpha in alphas]
    conf_deltas_norm = (np.array(confidences) - min(confidences)) / (max(confidences) - min(confidences))

    # Calculate final adjusted grayscale values
    grayscale_final_adjusted = start_gray_updated - conf_deltas_norm * (start_gray_updated - end_gray_final)
    return bar_height, confidences, grayscale_final_adjusted


@app.cell
def _(dataframes):
    dataframes
    return


@app.cell
def _(
    alphas,
    bar_height,
    confidences,
    dataframes,
    dropdown_house,
    grayscale_final_adjusted,
    np,
    plt,
):
    #fig_1, ax_1 = plt.subplots(1, 1, figsize=(8, 6))
    fig_1, ax_1 = plt.subplots(1, 1, figsize=(6, 4.5))

    for i, (alpha_sel, gray_value) in enumerate(zip(alphas, grayscale_final_adjusted)):

        # Filtering df
        df_alpha = dataframes[str(alpha_sel).replace('0.', '')]
        df_plot = df_alpha[df_alpha.name==dropdown_house.value]
        y_pred = df_plot["y_test_pred"].values[0] * 100
        lower_bound = df_plot["lower_bound"].values[0] * 100
        upper_bound = df_plot["upper_bound"].values[0] * 100

        # Use the new refined grayscale color
        color = (gray_value, gray_value, gray_value)

        # Teken shaded confidence interval blok
        ax_1.fill_betweenx(
            [i * bar_height, (i + 1) * bar_height],
            lower_bound,
            upper_bound,
            color=color,
            edgecolor="lightgray",
            linewidth=0.7
        )
        # Y-as labels: confidence levels
        confidence_labels = []
        for cl in confidences:
            if abs(cl - 0.999) < 0.0001:
                confidence_labels.append("0.999")
            else:
                confidence_labels.append(f"{cl:.2f}")

    # Plot the true value as a vertical red line
    y_true_scaled = df_plot.y_test.values[0] * 100
    #plt.axvline(
    #    x=y_true_scaled, 
    #    color='red', linestyle='--', linewidth=1.5, 
    #    label=f'True value: {y_true_scaled:.1f}k'
    #)

    plt.ylabel("Confidence level", fontsize=14)
    plt.xlabel("Predicted value ($)", fontsize=14)
    plt.yticks(
        np.arange(len(alphas)) * bar_height + bar_height / 2, 
        labels=[f'{np.around(float(el)*100, 1)}%' for el in confidence_labels],
        fontsize=12
    )
    plt.ylim(0, len(alphas) * bar_height)
    plt.xlim(0, 800)
    plt.xticks(
        ticks=np.arange(0, 801, 100),
        labels=[f"{int(x)}k" for x in np.arange(0, 801, 100)],
        fontsize=12
    )

    #plt.title(f'{dropdown_house.value}\nConfidence bands') #\n\n({pred_error} prediction error)')
    plt.title(f'Multiple confidence bands', fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.5)

    #plt.legend()
    plt.tight_layout()

    #plt.savefig(f"pictures/conf_bands_{dropdown_house.value}.png", dpi=600, bbox_inches="tight")
    plt.savefig(f"pics/example_conf_bands.png", dpi=600, bbox_inches="tight")

    plt.show()
    return


@app.cell
def _(alphas, dataframes, dropdown_house, np, plt):
    fig_2, ax_2 = plt.subplots(1, 1, figsize=(7, 3))


    colors = plt.cm.Blues(np.linspace(0.3, 1, len(alphas)))
    fixed_y = 0  # All error bars are plotted on the same y-axis value
    scale_factor = 100_000  # 1 unit = 100k

    for i_2, alpha_2 in enumerate(alphas[::-1]):

        confidence_level = round(1 - alpha_2, 6)
        df_alpha_2 = dataframes[str(alpha_2).replace('0.','')]
        df_plot_2 = df_alpha_2[df_alpha_2.name==dropdown_house.value]
        y_pred_2 = df_plot_2["y_test_pred"].values[0]
        lower = df_plot_2["lower_bound"].values[0]
        upper = df_plot_2["upper_bound"].values[0]

        # Convert values from 0-8 scale to 0k-800k scale
        y_pred_scaled = y_pred_2 * scale_factor
        lower_scaled = lower * scale_factor
        upper_scaled = upper * scale_factor

        # Horizontal error bars
        plt.errorbar(
            y_pred_scaled, fixed_y,
            xerr=[[y_pred_scaled - lower_scaled], [upper_scaled - y_pred_scaled]],
            fmt='o', capsize=4,
            color=colors[i_2],
            label=f'{confidence_level * 100:.1f}%' #label=f'{confidence_level:.3f}'
        )

    # Plot red dot at the same y position
    y_true_scaled_2 = df_plot_2.y_test.values[0] * scale_factor
    #plt.scatter(y_true_scaled_2, fixed_y, color="red", s=40, zorder=3) #, label="True value")

    #plt.title(f'{dropdown_house.value}\nError bars') #\n({pred_error} prediction error)')
    plt.title(f'Error bars', fontsize=14)
    plt.yticks([fixed_y], [""])  # Hide y-axis label
    plt.xlabel("Predicted value ($)", fontsize=14)
    plt.xlim(0, 800_000)
    plt.xticks(
        np.arange(0, 800_001, 100_000), 
        [f"{int(x/1_000)}k" for x in np.arange(0, 800_001, 100_000)],
        fontsize=12
    )
    plt.grid(axis='x', linestyle='--', alpha=0.5)

    legend = plt.legend(
        title="Confidence level",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.45),  # Still use a lower y
        ncol=5,
        title_fontsize=14,
        fontsize=12,
        handletextpad=0.4,   # space between handle and text
        columnspacing=0.8,   # space between columns
        borderaxespad=0.2    # space between legend and plot
    )
    ax_2.tick_params(axis='y', length=0) 
    plt.subplots_adjust(bottom=0.6)

    plt.tight_layout()
    #plt.savefig(f"pictures/error_bars_{dropdown_house.value}.png", dpi=600, bbox_inches="tight")
    plt.savefig(f"pics/example_error_bars.png", dpi=600, bbox_inches="tight")
    plt.show()
    return


@app.cell
def _(
    X_test_houses,
    dropdown_model_house,
    ebm_model,
    list_model_houses,
    np,
    pd,
    plt,
    y_test_houses,
):
    dict_model_houses = dict(zip(list_model_houses, range(9)))
    local_exp = ebm_model.explain_local(X_test_houses, y_test_houses)
    idx = dict_model_houses[dropdown_model_house.value]

    instance_local = local_exp.data(idx)

    feature_names = instance_local['names'] 
    contributions = instance_local['scores']

    # Compute predicted value as sum of contributions (if you want it)
    predicted_value = ebm_model.predict(
        pd.DataFrame(X_test_houses.iloc[idx]).transpose()
    )

    # Sort contributions for plotting
    sorted_idx = np.argsort(np.abs(contributions))
    sorted_features = np.array(feature_names)[sorted_idx]
    sorted_contribs = np.array(contributions)[sorted_idx]

    # Plot waterfall-like chart
    plt.figure(figsize=(8,6))
    plt.barh(sorted_features, sorted_contribs, color='skyblue')
    plt.axvline(x=0, color='black', linewidth=0.8)
    plt.title(f'Local Explanation for {dropdown_model_house.value}\nPredicted value = {predicted_value[0]:.2f}, True value = {y_test_houses.iloc[idx]:.2f}')
    plt.xlabel('Contribution to prediction')
    plt.grid(alpha=0.5)
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
