import marimo

__generated_with = "0.15.1"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _(mo):
    mo.md(r"""# `Real data` exploration""")
    return


@app.cell
def _():
    import marimo as mo

    import pandas as pd
    import numpy as np

    import seaborn as sns
    import matplotlib.pyplot as plt

    from interpret import show
    from interpret.data import Marginal

    from sklearn.datasets import fetch_california_housing
    from sklearn.datasets import load_diabetes
    return (
        Marginal,
        fetch_california_housing,
        load_diabetes,
        mo,
        np,
        pd,
        plt,
        show,
        sns,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Load data

    🔗 `fetch_california_housing`: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html

    🔗 `load_diabetes`: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html
    """
    )
    return


@app.cell
def _(fetch_california_housing, load_diabetes):
    df_california = fetch_california_housing(as_frame=True)
    df_diabetes = load_diabetes(as_frame=True)
    return df_california, df_diabetes


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Data description""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Diabetes

    - Samples total: 442
    - Dimensionality: 10
    - Features: real, -.2 < x < .2
    - Target: integer, 25 - 346
    - No missing values

    Original df (prior transformation): https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt
    """
    )
    return


@app.cell
def _(df_diabetes):
    df_diabetes.frame.info()
    return


@app.cell
def _(df_diabetes):
    print(df_diabetes.DESCR)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## California housing
    - Samples total: 20640
    - Dimensionality: 8
    - Features: real, 
    - Target: real,
    - No missing values

    ---

    The data contains information from the 1990 California census. It does provide an **accessible introductory dataset** for teaching people about the basics of machine learning. 
    _Note_: there are no explicit cleaning steps because this dataset arrives fully cleaned: it's all numerical, has no missing values, and requires no preprocessing in that sense.

    - The target variable is the median house value for California districts, in hundreds of thousands of dollars.


    🔗 **Useful links**:

    - https://www.kaggle.com/datasets/camnugent/california-housing-prices (slightly different dataset, the original one?)
    - https://inria.github.io/scikit-learn-mooc/python_scripts/datasets_california_housing.html (Scikit-learn MOOC)
    - https://medium.com/data-science/metrics-for-uncertainty-evaluation-in-regression-problems-210821761aa (very similar study on same data)
    - https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html
    """
    )
    return


@app.cell
def _(df_california):
    df_california.frame.info()
    return


@app.cell
def _(df_california):
    print(df_california.DESCR)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Data exploration""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Choose df""")
    return


@app.cell(hide_code=True)
def _(mo):
    dropdown_df = mo.ui.dropdown(
        options=['California housing', 'Diabetes'], value='California housing', label="Select df: "
    )
    dropdown_df
    return (dropdown_df,)


@app.cell
def _(df_california, df_diabetes, dropdown_df, pd):
    if dropdown_df.value == 'California housing':
        df = df_california
        X = pd.DataFrame(df_california.data, columns=df_california.feature_names)
        y = df_california.target
        n = len(y)
    else:
        df = df_diabetes
        X = pd.DataFrame(df_diabetes.data, columns=df_diabetes.feature_names)
        y = df_diabetes.target
        n = len(y)
    return X, df, y


@app.cell
def _():
    # (X.age*X.age).sum()
    return


@app.cell
def _(X):
    X
    return


@app.cell
def _(y):
    y
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Feature distribution""")
    return


@app.cell
def _(Marginal, X, dropdown_df, show, y):
    # InterpretML
    exploring_data = Marginal().explain_data(X, y, name=dropdown_df.value)
    show(exploring_data)
    return


@app.cell
def _(df, plt, sns):
    # Correlation heatmap
    correlation_matrix = df.frame.corr()
    plt.figure(figsize=(7, 5))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation matrix')
    plt.show()
    return


@app.cell
def _(X):
    X.describe()
    return


@app.cell
def _(df, plt):
    df.frame.hist(figsize=(12, 10), bins=30, edgecolor="black")
    plt.subplots_adjust(hspace=0.7, wspace=0.4)
    plt.show()
    return


@app.cell
def _(df, dropdown_df, np, pd, plt, sns):
    if dropdown_df.value == 'California housing':
        rng = np.random.RandomState(0)
        indices = rng.choice(
            np.arange(df.frame.shape[0]), size=500, replace=False
        )

        columns_drop = ["Longitude", "Latitude"]
        subset = df.frame.iloc[indices].drop(columns=columns_drop)

        # Quantize the target and keep the midpoint for each interval
        subset["MedHouseVal"] = pd.qcut(subset["MedHouseVal"], 6, retbins=False)

        _ = sns.pairplot(data=subset, hue="MedHouseVal", palette="viridis")
        plt.show()
    return


@app.cell(column=1)
def _():
    return


if __name__ == "__main__":
    app.run()
