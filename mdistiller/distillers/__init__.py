from ._base import Vanilla
from .KD import KD,SDD_KD
from .AT import AT
from .OFD import OFD
from .RKD import RKD
from .FitNet import FitNet
from .KDSVD import KDSVD
from .CRD import CRD
from .NST import NST
from .PKT import PKT
from .SP import SP
from .VID import VID
from .ReviewKD import ReviewKD
from .DKD import DKD
from .SDD_DKD import SDD_DKD
from .SDD_DKD_LSKD import SDD_DKD_LSKD
from .SDD_KD_LSKD import SDD_KD_LSKD
from .nkd import NKDLoss
from .SDD_nkd import SDD_NKDLoss

distiller_dict = {
    "NONE": Vanilla,
    "KD": KD,
    "AT": AT,
    "OFD": OFD,
    "RKD": RKD,
    "FITNET": FitNet,
    "KDSVD": KDSVD,
    "CRD": CRD,
    "NST": NST,
    "PKT": PKT,
    "SP": SP,
    "VID": VID,
    "REVIEWKD": ReviewKD,
    "DKD": DKD,
    "SDD_DKD": SDD_DKD,
    "SDD_DKD_LSKD": SDD_DKD_LSKD,
    "SDD_KD_LSKD": SDD_KD_LSKD,
    "SDD_KD": SDD_KD,
    "NKD": NKDLoss,
    "SDD_NKD": SDD_NKDLoss
}
