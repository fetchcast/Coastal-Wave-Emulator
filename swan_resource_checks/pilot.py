"""Optional short-schedule LR pilot, isolated from all original run roots."""
import argparse,sys
from pathlib import Path
from common import read,inventory
p=argparse.ArgumentParser()
for name in ('server-root','job','root'):p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--gpu',type=int,required=True)
a=p.parse_args()
if inventory()[a.gpu]['busy']:raise RuntimeError('GPU became busy')
package=a.server_root/'swan_repaired_v1';sys.path.insert(0,str(package))
import run_repaired as runner
import repair_support as repair
job=read(a.job)
if repair.code_hashes(package)!=job['code_hashes']:raise ValueError('Training source hash mismatch')
runner.launch(a.root,[job],[a.gpu])
if runner.checked_summary(a.root,job) is None:raise RuntimeError('Pilot did not complete successfully')
