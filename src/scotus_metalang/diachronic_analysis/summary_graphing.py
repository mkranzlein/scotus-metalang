import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from scotus_metalang.diachronic_analysis import authors


def load_data(opinion_paths) -> list[dict]:
    opinions = []   
    for filepath in opinion_paths:
        with open(filepath, "r") as f:
            opinion = json.load(f)
            opinions.append(opinion)
            

def get_opinion_types_by_author(opinions: list[dict]) -> dict:
    opinion_types_by_author = defaultdict(Counter)
    for opinion in opinions:
        author = opinion["author"]
        op_type = opinion["type"]
        if op_type in ["majority", "concurrence", "dissent", "concurring-in-part-and-dissenting-in-part"]:
            opinion_types_by_author[author][op_type] += 1
    return opinion_types_by_author


def get_opinion_types_by_term(opinions: list[dict]) -> dict:
    opinion_types_by_term = defaultdict(Counter)
    for opinion in opinions:
        term = opinion["term"]
        op_type = opinion["type"]
        if op_type in ["majority", "concurrence", "dissent", "concurring-in-part-and-dissenting-in-part"]:
            opinion_types_by_term[term][op_type] += 1
    return opinion_types_by_term


def plot_opinion_types_abs(opinion_types_by_author: dict) -> Figure:
    width = .5
    bottom = np.zeros(len(authors.ORDERED_JUSTICES))
    auths = list(authors.ORDERED_JUSTICES.keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    for category in ["majority", "concurrence", "dissent", "concurring-in-part-and-dissenting-in-part"]:
        vals = [opinion_types_by_author[author].get(category, 0) for author in auths]
        ax.bar(auths, vals, width, label=category, bottom=bottom)
        bottom += vals

    ax.tick_params(axis='x', labelrotation=90)
    ax.legend(loc="upper right")
    ax.set_title("Opinion Types by author (absolute)")
    return fig


def plot_opinion_types_norm(opinion_types_by_author) -> Figure:
    width = .5
    bottom = np.zeros(len(authors.ORDERED_JUSTICES))
    auths = list(authors.ORDERED_JUSTICES.keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    for category in ["majority", "concurrence", "dissent", "concurring-in-part-and-dissenting-in-part"]:
        totals = [sum(opinion_types_by_author[author].values()) for author in authors.ORDERED_JUSTICES]
        vals = [opinion_types_by_author[author].get(category, 0) for author in auths]
        ratios = [(v / t) for v, t in zip(vals, totals)]
        plt.bar(auths, ratios, width, label=category,
                bottom=bottom)
        bottom += ratios

    ax.tick_params(axis='x', labelrotation=90)
    ax.legend()
    ax.set_title("Opinion Types by author (normalized)")
    return fig


def plot_opinion_types_by_term_abs(opinion_types_by_term: dict) -> Figure:
    width = .5
    bottom = np.zeros(len(opinion_types_by_term.keys()))
    terms = sorted(opinion_types_by_term.keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    for category in ["majority", "concurrence", "dissent", "concurring-in-part-and-dissenting-in-part"]:
        vals = [opinion_types_by_term[term].get(category, 0) for term in terms]
        ax.bar(terms, vals, width, label=category, bottom=bottom)
        bottom += vals

    ax.tick_params(axis='x', labelrotation=90)
    ax.legend(loc="upper right")
    ax.set_title("Opinion Types by term (absolute)")
    return fig


def plot_opinion_types_by_term_norm(opinion_types_by_term) -> Figure:
    width = .5
    bottom = np.zeros(len(opinion_types_by_term.keys()))
    terms = sorted(opinion_types_by_term.keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    for category in ["majority", "concurrence", "dissent", "concurring-in-part-and-dissenting-in-part"]:
        totals = [sum(opinion_types_by_term[term].values()) for term in terms]
        vals = [opinion_types_by_term[term].get(category, 0) for term in terms]
        ratios = [(v / t) for v, t in zip(vals, totals)]
        plt.bar(terms, ratios, width, label=category,
                bottom=bottom)
        bottom += ratios

    ax.tick_params(axis='x', labelrotation=90)
    ax.legend()
    ax.set_title("Opinion Types by term (normalized)")
    return fig


def plot_opinions_per_case_by_term(opinions):
    cases_per_term = defaultdict(set)
    num_opinions_per_term = Counter()
    for opinion in opinions:
        term = opinion["term"]
        cases_per_term[term].add(opinion["scdb_id"])
        num_opinions_per_term[term] += 1
    num_cases_per_term = {term: len(s) for term, s in cases_per_term.items()}
    averages = [num_opinions_per_term[term] / num_cases_per_term[term] for term in sorted(num_cases_per_term.keys())]
    fig, ax = plt.subplots()
    ax.bar(sorted(num_cases_per_term.keys()), averages)
    ax.tick_params(axis="x", labelrotation=90)
    return averages, fig