"""TrainLeak: find the data-leakage mistakes that make ML results too good to be true.

import trainleak
trainleak.demo()                                     # see it work first, no data needed
trainleak.check(df)                                  # before you split
trainleak.check(train_df, test_df, target="label")   # after you split
trainleak.check(X_train, X_test, y_train, y_test)    # sklearn style
trainleak.check_code("my_notebook.ipynb")            # SMOTE / scaling before the split
"""

from .code import check_code
from .data import check, template_groups
from .demo import demo
from .report import Finding, Report

__all__ = ["Finding", "Report", "check", "check_code", "demo", "template_groups"]
__version__ = "0.1.0"
