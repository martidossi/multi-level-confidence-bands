import marimo

__generated_with = "0.15.1"
app = marimo.App(width="columns")


@app.cell
def _():
    import marimo as mo
    import pandas as pd

    import os
    IN_PYODIDE = os.environ.get("PYODIDE") == "1"

    if IN_PYODIDE:
        from pyodide.http import open_url
        df = pd.read_csv(open_url("not_shared/results.csv"))
    else:
        df = pd.read_excel("not_shared/results.xlsx", sheet_name="survey data")

    #df = pd.read_excel('not_shared/results.xlsx', sheet_name='survey data')
    df['condition'] = df['condition'].replace({
        'baseline': 'No AI',
        'conbands': 'Confidence bands',
        'errorbars': 'Error bars'
    })
    return (df,)


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
