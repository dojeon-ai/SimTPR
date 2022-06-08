import numpy as np
import random
import gym
import torch
import inspect
import sys
import os
import pkgutil
from importlib import import_module
from pathlib import Path


def set_global_seeds(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def save__init__args(values, underscore=False, overwrite=False, subclass_only=False):
    """
    Use in `__init__()` only; assign all args/kwargs to instance attributes.
    To maintain precedence of args provided to subclasses, call this in the
    subclass before `super().__init__()` if `save__init__args()` also appears
    in base class, or use `overwrite=True`.  With `subclass_only==True`, only
    args/kwargs listed in current subclass apply.
    """
    prefix = "_" if underscore else ""
    self = values['self']
    args = list()
    Classes = type(self).mro()
    if subclass_only:
        Classes = Classes[:1]
    for Cls in Classes:  # class inheritances
        if '__init__' in vars(Cls):
            args += inspect.getfullargspec(Cls.__init__).args[1:]
    for arg in args:
        attr = prefix + arg
        if arg in values and (not hasattr(self, attr) or overwrite):
            setattr(self, attr, values[arg])


def all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)])


def import_all_subclasses(_file, _name, _class):
    modules = get_all_submodules(_file, _name)
    for m in modules:
        for i in dir(m):
            attribute = getattr(m, i)
            if inspect.isclass(attribute) and issubclass(attribute, _class):
                setattr(sys.modules[_name], i, attribute)


def get_all_submodules(_file, _name):
    modules = []
    _dir = os.path.dirname(_file)
    for _, name, ispkg in pkgutil.iter_modules([_dir]):
        module = import_module('.' + name, package=_name)
        modules.append(module)
        if ispkg:
            modules.extend(get_all_submodules(module.__file__, module.__name__))
    return modules
