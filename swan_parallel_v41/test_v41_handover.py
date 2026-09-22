"""Exercise handover against an isolated temporary supervisor, never real jobs."""
import json,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from handover import stop_v4
class Handover(unittest.TestCase):
    def test_only_owned_tree_stops(self):
        with tempfile.TemporaryDirectory() as d:
            server=Path(d);pkg=server/'swan_parallel_v4';pkg.mkdir()
            state=server/'runs/iclr_bc_typhoon_v3';state.mkdir(parents=True)
            (state/'status.json').write_text(json.dumps(dict(stage='B_and_C')))
            code=pkg/'run.py'
            code.write_text("import signal,subprocess,sys,time\nfrom pathlib import Path\np=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])\ndef stop(*a):\n p.terminate();p.wait();raise SystemExit(0)\nsignal.signal(signal.SIGTERM,stop)\nPath('ready').write_text(str(p.pid))\nwhile True:time.sleep(.1)\n")
            parent=subprocess.Popen([sys.executable,'run.py'],cwd=pkg)
            unrelated=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])
            try:
                until=time.monotonic()+5
                while not (pkg/'ready').exists():
                    if time.monotonic()>until:raise RuntimeError('Test supervisor failed startup')
                    time.sleep(.02)
                # Reap the test parent in a helper thread so it cannot remain a zombie.
                import threading
                waiter=threading.Thread(target=parent.wait);waiter.start()
                stop_v4(server)
                waiter.join(timeout=5)
                self.assertEqual(parent.returncode,0);self.assertIsNone(unrelated.poll())
            finally:
                for p in (parent,unrelated):
                    if p.poll() is None:p.terminate()
                    p.wait(timeout=5)
if __name__=='__main__':unittest.main()
