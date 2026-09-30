"""In-memory fault injection for the design tests and package-suite experiment."""

import os

import numpy as np

MUTATION = os.environ.get("NORMALIZER_MUTATION", "")


def apply_mutation(module):
    if not MUTATION or getattr(module, "_mutation_applied", False):
        return
    module._mutation_applied = True
    if MUTATION == "old_midpoint":

        def old_midpoint(high, low):
            integer = np.floor(high)
            a, b = module._two_sum(high - integer - 0.5, low)
            near = np.abs(a) - np.abs(b) <= module.EPS_Q * high
            return (integer + ((a > 0) & ~near)) * module.QUANTUM, near

        module._round_quanta = old_midpoint
    elif MUTATION == "float_classification":
        original = module.normalize_proto

        def preserve_float_normal(values, stats=None):
            weights, normalizer = original(values, stats)
            branch, _ = module.normalize_branch(values)
            normal = branch >= module.TINY
            weights[normal] = branch[normal]
            return weights, normalizer

        module.normalize_proto = preserve_float_normal
    elif MUTATION == "no_dd_interval":
        module.EPS_Q = 0.0
    elif MUTATION == "collapsed_decimal_interval":

        def collapsed(context, argument):
            rounded = context.exp(argument)
            return rounded, rounded

        module._exp_interval = collapsed
    else:
        raise ValueError(f"unknown mutation: {MUTATION}")


class MutationPlugin:
    def pytest_collection_modifyitems(self, items):
        for item in items:
            if "normalizer-r3" in str(item.path):
                module = getattr(item.module, "p", None)
                if module is not None:
                    apply_mutation(module)
