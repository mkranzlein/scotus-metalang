import json
from pathlib import Path

import numpy as np
from numpy.polynomial import polynomial
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from scotus_metalang.diachronic_analysis import authors


def load_data(op_paths_to_pred_paths: dict[Path, Path]) -> pd.DataFrame:
    rows = []
    for opinion_path, prediction_path in op_paths_to_pred_paths.items():
        author = opinion_path.parent.name
        with open(opinion_path, "r") as f:
            case = json.load(f)
            term = case["term"]
            opinion_type = case["type"]
            docket_number = case["docket"]
        scores = np.loadtxt(prediction_path)
        threshold = .60
        num_tokens = len(scores)
        predictions = scores > threshold
        ft, mc, dq, les = np.sum(predictions, axis=0)
        row = [docket_number, author, opinion_type, term, num_tokens, ft, mc, dq, les]
        rows.append(row)
        columns = ["docket_number", "author", "opinion_type", "term", "tokens", "ft", "mc", "dq", "les"]
    df_all = pd.DataFrame(rows, columns=columns)
    df_18 = df_all[df_all["term"].astype(int) < 2019]  # Exclude 2019 data because that's training data
    return df_18


def plot_opinion_length_per_term(df: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots()
    tokens_per_term = dict(df.groupby("term")["tokens"].mean())
    ax.plot(tokens_per_term.keys(), tokens_per_term.values())
    ax.title("Wordpiece Tokens per Opinion")
    ax.xticks(rotation=90)
    return fig, ax


def plot_frequency_by_author(category: str, df: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots()
    cat_by_author = dict(df.groupby(['author'])[category].sum())
    tokens_by_author = dict(df.groupby(['author'])["tokens"].sum())
    frequencies_by_author = [cat_by_author[a] / tokens_by_author[a] for a in authors.ORDERED_JUSTICES]
    ax.xticks(rotation=90)
    ax.bar(authors.ORDERED_JUSTICES.keys(), frequencies_by_author)
    ax.title(f"Rates of {category} by Author")
    return fig, ax


def plot_frequency_by_term(category: str, df: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots()
    cat_by_term = dict(df.groupby(["term"])[category].sum())
    tokens_by_term = dict(df.groupby(["term"])["tokens"].sum())
    frequencies_by_term = [cat_by_term[term] / tokens_by_term[term] for term in cat_by_term]
    ax.xticks(rotation=90)
    ax.bar(cat_by_term.keys(), frequencies_by_term)
    ax.title(f"Rates of {category} by Term")
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


def plot_frequency_line_all_cats(df: pd.DataFrame, ci_interval: bool = False,) -> Figure:
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

        if ci_interval:
            df_grouped['ci'] = 1.96 * df_grouped['std'] / np.sqrt(df_grouped['count'])
            df_grouped['ci_lower'] = df_grouped['mean'] - df_grouped['ci']
            df_grouped['ci_upper'] = df_grouped['mean'] + df_grouped['ci']
            ax.fill_between(term, df_grouped['ci_lower'], df_grouped['ci_upper'],
                            color='b', alpha=.15)

        ax.set_ylim(ymin=0)
        ax.set_title("Rate of " + category)
    return fig


def plot_frequency_by_type(df: pd.DataFrame, category: str, op_type: str) -> Figure:
    fig, ax = plt.subplots()
    df_sample = df[df["opinion_type"] == op_type]
    cat_by_term = dict(df_sample.groupby(["term"])[category].sum())
    tokens_by_term = dict(df_sample.groupby(["term"])["tokens"].sum())
    frequencies_by_term = [cat_by_term[term] / tokens_by_term[term] for term in cat_by_term]
    ax.xticks(rotation=90)
    ax.bar(cat_by_term.keys(), frequencies_by_term)
    title = f"Rates of {category} by {op_type}"
    ax.title(title)
    ax.show()
    return fig
