from datetime import date


def get_term_year(decision_date: date) -> int:
    """Gets the Supreme Court term year of a case based on decision date.

    Supreme Court terms begin in October, so a case decided in June 2020 belongs
    to the 2019 term."""
    if decision_date.month >= 10:
        term_year = decision_date.year
    else:
        term_year = decision_date.year - 1
    return term_year
