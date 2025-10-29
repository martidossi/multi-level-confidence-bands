import marimo

__generated_with = "0.15.1"
app = marimo.App(width="columns")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# `Survey results` exploration – setup""")
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    return mo, pd


@app.cell
def _(pd):
    df = pd.read_excel('not_shared/results.xlsx', sheet_name='survey data')
    df['condition'] = df['condition'].replace({
        'baseline': 'No AI',
        'conbands': 'Confidence bands',
        'errorbars': 'Error bars'
    })
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
