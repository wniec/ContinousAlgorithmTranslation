"""Portfolio of available sub-optimizers.

Add new algorithm classes here and they become selectable by name in train.py
/ evaluate.py. A new optimizer also needs a ``StateSpec`` in
``cat/state/schema.py`` before it can participate in translation.
"""

from cat.optimizers.PSO import SPSO, SPSOL, IPSO, CPSO
from cat.optimizers.ES import CMAES
from cat.optimizers.DE import MADDE
from cat.optimizers.TR import BOBYQA

PORTFOLIO: dict = {
    "SPSO": SPSO,
    "SPSOL": SPSOL,
    "IPSO": IPSO,
    "CPSO": CPSO,
    "PSO": SPSO,  # convenient alias: the standard global-best PSO
    "CMAES": CMAES,
    "MADDE": MADDE,
    "BOBYQA": BOBYQA,
}


def get_portfolio(names: list[str]) -> list:
    """Return optimizer classes for the given names.

    Raises ValueError for any unknown name so errors surface early.
    """
    unknown = [n for n in names if n not in PORTFOLIO]
    if unknown:
        raise ValueError(
            f"Unknown optimizer(s): {unknown}. Available: {list(PORTFOLIO)}"
        )
    return [PORTFOLIO[n] for n in names]
