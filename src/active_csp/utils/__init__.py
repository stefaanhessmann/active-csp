class Properties:
    e_rel = "relaxation_energy"
    e = "energy"
    f = "forces"
    s = "stress"
    u = "uncertainty"
    e_u = "energy_uncertainty"
    f_u = "force_uncertainty"
    s_u = "stress_uncertainty"
    struc = "structure"
    prediction_properties = [e_rel]
    relaxation_properties = [e, f, s]


from .misc import *
from .db import *
from .state_tracker import *
