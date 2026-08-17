"""The project's declared bit generator, constructed explicitly.

Preregistered evidence records *which* bit generator produced its streams.
``numpy.random.default_rng`` documents its choice as an implementation detail
that may change between releases, so archived seeds would silently stop
reproducing archived observations. Every stream the study depends on is built
here instead, so the recorded name and the running code cannot drift apart.
"""

import numpy as np

BIT_GENERATOR_NAME = "PCG64"


def study_generator(seed: object = None) -> np.random.Generator:
    """Return a Generator over the project's declared bit generator."""
    return np.random.Generator(np.random.PCG64(seed))
