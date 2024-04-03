import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from numpy.polynomial import polynomial
import pandas as pd
from matplotlib import pyplot as plt

from scotus_metalang.diachronic_analysis import authors


def load_data(op_paths_to_pred_paths: dict[Path, Path]):
    rows = []
    for opinion_path, prediction_path in op_paths_to_pred_paths.items():
        author = opinion_path.parent
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


def plot_cases_per_term(df):
    fig, ax = plt.subplot()
    cases_per_term = dict(df.groupby('term')["docket_number"].nunique())
    ax.plot(cases_per_term.keys(), cases_per_term.values())
    ax.xticks(rotation=90)
    title = "Cases per SCOTUS Term"
    ax.title(title)
    return fig


def plot_avg_opinions_per_case(df):
    fig, ax = plt.subplot()
    cases_per_term = dict(df.groupby('term')["docket_number"].nunique())
    opinions_per_term = dict(df.groupby("term").size())
    average_per_term = [opinions_per_term[term] / cases_per_term[term] for term in cases_per_term]
    ax.plot(cases_per_term.keys(), average_per_term)
    title = "Opinions per Case"
    ax.xticks(rotation=90)
    ax.ylim(1, 3)
    ax.title(title)
    return fig

def plot_cases_per_term(df):
    cases_per_term = dict(df.groupby('term')["docket_number"].nunique())
    ax.plot(cases_per_term.keys(), cases_per_term.values())
    title = "Cases per SCOTUS Term"
    .title(title)
    .xticks(rotation=90)
    .savefig(f"figures/{title}.pdf")
    .show()


def plot_avg_opinions_per_case(df):
    cases_per_term = dict(df.groupby('term')["docket_number"].nunique())
    opinions_per_term = dict(df.groupby("term").size())
    average_per_term = [opinions_per_term[term] / cases_per_term[term] for term in cases_per_term]
    plt.plot(cases_per_term.keys(), average_per_term)
    title = "Opinions per Case"
    plt.title(title)
    plt.xticks(rotation=90)
    plt.ylim(1, 3)
    plt.savefig(f"figures/{title}.pdf")
    plt.show()


def plot_opinion_length_per_term(df):
    tokens_per_term = dict(df.groupby("term")["tokens"].mean())
    plt.plot(tokens_per_term.keys(), tokens_per_term.values())
    title = "Wordpiece Tokens per Opinion"
    plt.title(title)
    plt.xticks(rotation=90)
    plt.savefig(f"figures/{title}.pdf")
    plt.show()

def plot_frequency_by_author(category, df):
    cat_by_author = dict(df.groupby(['author'])[category].sum())
    tokens_by_author = dict(df.groupby(['author'])["tokens"].sum())
    frequencies_by_author = [cat_by_author[a] / tokens_by_author[a] for a in authors.ORDERED_JUSTICES]
    plt.xticks(rotation=90)
    plt.bar(authors.ORDERED_JUSTICES.keys(), frequencies_by_author)
    title = f"Rates of {category} by Author"
    plt.title(title)
    plt.savefig(f"figures/{title}.pdf")
    plt.show()
plot_frequency_by_author("dq", df_18)


def plot_frequency_by_term(category, df):
    cat_by_term = dict(df.groupby(["term"])[category].sum())
    tokens_by_term = dict(df.groupby(["term"])["tokens"].sum())
    frequencies_by_term = [cat_by_term[term] / tokens_by_term[term] for term in cat_by_term]
    plt.xticks(rotation=90)
    plt.bar(cat_by_term.keys(), frequencies_by_term)
    title = f"Rates of {category} by Term"
    plt.title(title)
    plt.savefig(f"figures/{title}.pdf")
    plt.show()
plot_frequency_by_term("ft", df_18)