from .MULT.manager import MULT
from .MAG_BERT.manager import MAG_BERT
from .MISA.manager import MISA
from .MMIM.manager import MMIM
from .TCL_MAP.manager import TCL_MAP
from .SDIF.manager import SDIF
from .TEXT.manager import TEXT
method_map = {
    'mult': MULT,
    'mag_bert': MAG_BERT,
    'misa': MISA,
    'mmim': MMIM,
    'tcl_map': TCL_MAP,
    'sdif': SDIF,
    'text':TEXT
    # 'gobi':Pretrain_Text,
    
}