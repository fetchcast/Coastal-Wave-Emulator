#!/usr/bin/env python3
"""Freeze existing two-year FNO/FFNO candidates before accessing 2022."""
from pathlib import Path
import campaign as c
from settings import load,paths
from common import context,original_selected,verified_entry

def main():
    a=load();server,root=paths(a)
    if Path(a['source_2019_2020']).resolve()!=(server/'wavm-Waves_2019_2020_v2.nc').resolve():raise ValueError('Use the original verified 2019-2020 source')
    r,protocol,bases,ref=context(server)
    c.freeze(root/'config.json',a)
    values=[]
    for model in a['models']:
        rows=original_selected(server,model,r,bases)
        if rows is None:raise RuntimeError(f'{model}: all three original selected seeds must be complete')
        entries=[verified_entry(s,r) for s in rows]
        budgets={s['repair_audit']['target_updates'] for s in rows}
        if len(budgets)!=1 or any(s['repair_audit']['updates']!=s['repair_audit']['target_updates'] for s in rows):raise ValueError('Original update budgets differ/incomplete')
        intervals={s['repair_audit']['signature']['interval'] for s in rows}
        if len(intervals)!=1:raise ValueError('Original validation intervals differ')
        c.freeze(root/'controls'/f'{model}.json',dict(entries=entries,package=str(server/'swan_repaired_v1'),
            max_updates=next(iter(budgets)),eval_every_updates=next(iter(intervals)),selection='Existing v3 validation-only choice'))
        values.extend(rows)
    r.compare_protocols(values)
    c.freeze(root/'original_protocol.json',protocol)
    print('[FROZEN] Two-year controls: FNO/FFNO, three seeds each. No 2022 data read.')
if __name__=='__main__':main()
