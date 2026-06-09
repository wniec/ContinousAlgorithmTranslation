from cat.suite.ioh_suite import IOHSuite, IOHProblemWrapper
from cat.suite.bbob_splits import (
    ALL_DIMS,
    ALL_FUNCTIONS,
    INSTANCE_IDS,
    EASY_TRAIN_FUNCTIONS,
    build_problem_ids,
    get_train_test_split,
)

__all__ = [
    "IOHSuite",
    "IOHProblemWrapper",
    "ALL_DIMS",
    "ALL_FUNCTIONS",
    "INSTANCE_IDS",
    "EASY_TRAIN_FUNCTIONS",
    "build_problem_ids",
    "get_train_test_split",
]
