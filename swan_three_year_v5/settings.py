"""Frozen configuration helpers for isolated v5 runs."""
import json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent

def load():
    a=json.loads((HERE/'config.json').read_text())
    if a['models']!=['fno','ffno'] or a['seeds']!=[42,43,44]:raise ValueError('This v5 package freezes FNO/FFNO, seeds 42/43/44 only')
    if a['budget']!='same_successful_updates_as_frozen_two_year_models':raise ValueError('Unsupported budget')
    g=a['gpus']
    if not g or len(set(g))!=len(g) or any(not isinstance(x,int) or x<0 for x in g):raise ValueError('Invalid GPUs')
    for key in ('training_workers','evaluation_workers'):
        if not 1<=a[key]<=len(g):raise ValueError('Invalid '+key)
    return a

def paths(a):return Path(a['server_root']).resolve(),Path(a['result_root']).resolve()
def environment(a):
    server,root=paths(a)
    os.environ.update(SWAN_SERVER_ROOT=str(server),SWAN_DATA_PATH=str(root/'data/training_2019_2021.nc'))
    for year in (2019,2020,2021):os.environ[f'SWAN_BND_DIR_{year}']=str(server/f'bnd_{year}_v2')
