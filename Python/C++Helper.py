import numpy as np
import accelerate as acc


def passFullSys(fullSys):
    cFullSys = acc.fullSys()
    acc.fill_dict(cFullSys, fullSys.Params)
    for i, obj in enumerate(fullSys.subObjs):
