import json
from pathlib import Path

import numpy as np
from numpy.polynomial import polynomial
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from scotus_metalang.diachronic_analysis import authors


def load_predictions(op_paths_to_pred_paths: dict[Path, Path]) -> pd.DataFrame:
    rows = []
    for opinion_path, prediction_path in op_paths_to_pred_paths.items():
        author = opinion_path.parent.name
        with open(opinion_path, "r") as f:
            case = json.load(f)
            term = case["term"]
            opinion_type = case["type"]
            docket_number = case["docket"]
        scores = np.loadtxt(prediction_path)
        threshold = .5
        num_tokens = len(scores)
        predictions = scores > threshold
        ft, mc, dq, les = np.sum(predictions, axis=0)
        row = [docket_number, author, opinion_type, term, num_tokens, ft, mc, dq, les]
        rows.append(row)
        columns = ["docket_number", "author", "opinion_type", "term", "tokens", "ft", "mc", "dq", "les"]
    df_all = pd.DataFrame(rows, columns=columns)
    df_18 = df_all[df_all["term"].astype(int) < 2019]  # Exclude 2019 data because that's training data
    return df_18


def plot_si_vs_nsi_ops(si_df, nsi_df):
    fig, ax = plt.subplots()
    terms = list(range(1986, 2019))
    si_vals = [len(si_df[si_df["term"] == str(t)]) for t in terms]
    nsi_vals = [len(nsi_df[nsi_df["term"] == str(t)]) for t in terms]
    totals = [s + n for n, s in zip(si_vals, nsi_vals)]
    si_ratios = [s / t for s, t in zip(si_vals, totals)]
    nsi_ratios = [1 - s for s in si_ratios]
    width = .5
    bottom = np.zeros(len(terms))
    ax.bar(terms, si_ratios, width, label="SI", bottom=bottom, color="blue")
    bottom += si_ratios
    ax.bar(terms, nsi_ratios, width, label="NSI", bottom=bottom, color="orange")

    ax.tick_params(axis='x', labelrotation=90)
    ax.legend(loc="upper right")
    ax.set_title("Statutory Interpretation Opinions vs Non-Statutory Interpretation Opinions")
    return fig


def plot_opinion_length_per_term(df: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots()
    tokens_per_term = dict(df.groupby("term")["tokens"].mean())
    ax.plot(tokens_per_term.keys(), tokens_per_term.values())
    ax.set_title("Wordpiece Tokens per Opinion")
    ax.tick_params(axis='x', labelrotation=90)
    return fig


def plot_frequency_by_author(df: pd.DataFrame, category: str) -> Figure:
    fig, ax = plt.subplots()
    cat_by_author = dict(df.groupby(["author"])[category].sum())
    tokens_by_author = dict(df.groupby(["author"])["tokens"].sum())
    frequencies_by_author = [cat_by_author[a] / tokens_by_author[a] for a in authors.ORDERED_JUSTICES]
    colors = ["blue" if authors.JUSTICE_TO_IDEOLOGY[a] == "liberal" else "red" for a in authors.ORDERED_JUSTICES]
    ax.tick_params(axis='x', labelrotation=90)
    ax.bar(authors.ORDERED_JUSTICES.keys(), frequencies_by_author, color=colors)
    ax.set_title(f"Rates of {category} by Author")
    return fig


def plot_frequency_by_term(df: pd.DataFrame, category: str) -> Figure:
    fig, ax = plt.subplots()
    cat_by_term = dict(df.groupby(["term"])[category].sum())
    tokens_by_term = dict(df.groupby(["term"])["tokens"].sum())
    frequencies_by_term = [cat_by_term[term] / tokens_by_term[term] for term in cat_by_term]
    ax.tick_params(axis='x', labelrotation=90)
    ax.bar(cat_by_term.keys(), frequencies_by_term)
    ax.set_title(f"Rates of {category} by Term")
    return fig


def plot_frequency_line_with_trend(df: pd.DataFrame, category: str,
                                   ci: bool = False) -> Figure:
    df1 = df.copy()
    df1[f"{category}_rate"] = df1[category] / df1.tokens
    df_grouped = df1[["term", f"{category}_rate"]].groupby(["term"]).agg(["mean", "std", "count"])
    df_grouped = df_grouped.droplevel(axis=1, level=0).reset_index()
    term = df_grouped["term"].astype(float)
    trend = polynomial.polyfit(term, df_grouped["mean"], 1)
    p = polynomial.Polynomial(trend)
    fig, ax = plt.subplots()
    ax.plot(term, df_grouped["mean"])
    ax.plot(term, p(term))

    if ci:
        df_grouped['ci'] = 1.96 * df_grouped['std'] / np.sqrt(df_grouped['count'])
        df_grouped['ci_lower'] = df_grouped['mean'] - df_grouped['ci']
        df_grouped['ci_upper'] = df_grouped['mean'] + df_grouped['ci']
        ax.fill_between(term, df_grouped['ci_lower'], df_grouped['ci_upper'],
                        color='b', alpha=.15)
    ax.set_ylim(ymin=0)
    ax.set_title("Rate of " + category)
    fig.autofmt_xdate(rotation=90)
    return fig


def plot_frequency_line_all_cats(df: pd.DataFrame, ci: bool = False,) -> Figure:
    fig, axs = plt.subplots(2, 2, figsize=(12, 6))
    categories = ["ft", "mc", "dq", "les"]
    for category, ax in zip(categories, axs.flatten()):
        df1 = df.copy()
        df1[f"{category}_rate"] = df1[category] / df1.tokens
        df_grouped = df1[["term", f"{category}_rate"]].groupby(["term"]).agg(["mean", "std", "count"])
        df_grouped = df_grouped.droplevel(axis=1, level=0).reset_index()
        term = df_grouped["term"].astype(float)
        trend = polynomial.polyfit(term, df_grouped["mean"], 1)
        p = polynomial.Polynomial(trend)
        ax.plot(term, df_grouped["mean"])
        ax.plot(term, p(term))

        if ci:
            df_grouped['ci'] = 1.96 * df_grouped['std'] / np.sqrt(df_grouped['count'])
            df_grouped['ci_lower'] = df_grouped['mean'] - df_grouped['ci']
            df_grouped['ci_upper'] = df_grouped['mean'] + df_grouped['ci']
            ax.fill_between(term, df_grouped['ci_lower'], df_grouped['ci_upper'],
                            color='b', alpha=.15)

        ax.set_ylim(ymin=0)
        ax.set_title("Rate of " + category)
    return fig


def plot_frequency_line_all_cats_ideology(df: pd.DataFrame, ci: bool = False,) -> Figure:
    fig, axs = plt.subplots(2, 2, figsize=(12, 6))
    for ideology in ["liberal", "conservative"]:
        if ideology == "liberal":
            color = "blue"
        else:
            color = "red"
        ideology_justices = [k for k, v in authors.JUSTICE_TO_IDEOLOGY.items() if v == ideology]
        sample = df[df["author"].isin(ideology_justices)]
        categories = ["ft", "mc", "dq", "les"]
        for category, ax in zip(categories, axs.flatten()):
            df1 = sample.copy()
            df1[f"{category}_rate"] = df1[category] / df1.tokens
            df_grouped = df1[["term", f"{category}_rate"]].groupby(["term"]).agg(["mean", "std", "count"])
            df_grouped = df_grouped.droplevel(axis=1, level=0).reset_index()
            term = df_grouped["term"].astype(float)
            trend = polynomial.polyfit(term, df_grouped["mean"], 1)
            p = polynomial.Polynomial(trend)
            ax.plot(term, df_grouped["mean"], color=color)
            ax.plot(term, p(term), color=color)

            if ci:
                df_grouped['ci'] = 1.96 * df_grouped['std'] / np.sqrt(df_grouped['count'])
                df_grouped['ci_lower'] = df_grouped['mean'] - df_grouped['ci']
                df_grouped['ci_upper'] = df_grouped['mean'] + df_grouped['ci']
                ax.fill_between(term, df_grouped['ci_lower'], df_grouped['ci_upper'],
                                color='b', alpha=.15)

            ax.set_ylim(ymin=0)
            ax.set_title("Rate of " + category)
    return fig
