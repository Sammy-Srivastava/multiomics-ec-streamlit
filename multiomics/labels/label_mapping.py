from __future__ import annotations
from typing import Optional

non_ec = {'ART', 'HC', 'Viremic', 'Viremic', 'suppressed', 'viremic'}

def to_ec_binary(group: str) -> Optional[int]:
    #1 for EC, 0 for non-EC
    if group is None:
        return None
    g = str(group).strip()
    
    if g.upper() == 'EC':
        return 1
    
    if g in non_ec:
        return 0
    
    if g.lower().replace(' ', '') in {'elitecontroller', 'elite_controller'}:
        return 1
    
    if g.lower() in {'unknown', 'na', 'nan'}:
        return None
    
    return None
