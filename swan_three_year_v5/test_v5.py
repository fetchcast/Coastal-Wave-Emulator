"""CPU contract tests, including real synthetic NetCDF I/O when dependencies exist."""
import importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from prepare_data import choose_times,time_audit
from evaluate_2022 import year_indices
from events import parse_jma

class Contracts(unittest.TestCase):
    def test_future_year_excluded(self):
        times=np.array(['2021-12-31T23','2022-01-01T00'],dtype='datetime64[h]')
        ids,chosen=choose_times(times,[2021]);self.assertEqual(ids.tolist(),[0])
    def test_duplicates_rejected(self):
        with self.assertRaises(ValueError):choose_times(['2021-01-01','2021-01-01'],[2021])
    def test_gap_record(self):
        r=time_audit(np.array(['2020-12-31T22','2021-01-01T01'],dtype='datetime64[h]'))
        self.assertEqual(r[0]['missing_hours'],2)
    def test_2022_calendar(self):
        t=np.arange(np.datetime64('2022-01-01','h'),np.datetime64('2023-01-01T01','h'))
        ids,targets=year_indices(t,12);self.assertEqual(len(ids),8760);self.assertEqual(len(targets),8748)
        with self.assertRaises(ValueError):year_indices(t[:-2],12)
    def test_track_year(self):
        source='66666 2101 1\n21010100 002 5 250 1300 980\n66666 2201 1\n22010100 002 5 250 1300 980\n'
        events=parse_jma(source);self.assertEqual([e['id'] for e in events],['2201'])

@unittest.skipUnless(importlib.util.find_spec('netCDF4'),'netCDF4 is required for I/O integration')
class NetCDFIntegration(unittest.TestCase):
    def test_merge_and_reuse_with_year_endpoint(self):
        import netCDF4 as nc
        import prepare_data as module
        from settings import load
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);root=base/'runs/v5';(root/'controls').mkdir(parents=True)
            p0=base/'two.nc';p1=base/'one.nc'
            for p,start,end in [(p0,'2019-01-01','2021-01-01'),(p1,'2021-01-01','2022-01-01T01')]:
                times=np.arange(np.datetime64(start,'h'),np.datetime64(end,'h'))
                with nc.Dataset(p,'w') as ds:
                    ds.createDimension('time',len(times));ds.createDimension('nmax',2);ds.createDimension('mmax',3)
                    t=ds.createVariable('time','f8',('time',));t.units='hours since 1970-01-01';t[:]=times.astype(int)
                    for name in module.STATIC:
                        v=ds.createVariable(name,'f4',('nmax','mmax'));v[:]=1
                    for name in module.VARS:
                        v=ds.createVariable(name,'f4',('time','nmax','mmax'));v.units='test_units';v[:]=2 if p==p1 else 1
            for m in ('fno','ffno'):(root/'controls'/f'{m}.json').write_text(json.dumps(dict(entries=[dict(job=dict(time_steps=17544))])))
            cfg=load();cfg.update(server_root=str(base),result_root=str(root),source_2019_2020=str(p0),source_2021=str(p1))
            with patch.object(module,'load',return_value=cfg):module.main();module.main()
            manifest=json.loads((root/'data/prepared.json').read_text());self.assertEqual(manifest['frames'],26304)
            with nc.Dataset(root/'data/training_2019_2021.nc') as ds:
                self.assertEqual(ds.variables['hsign'].shape,(26304,2,3))
                self.assertEqual(float(ds.variables['hsign'][17543,0,0]),1)
                self.assertEqual(float(ds.variables['hsign'][17544,0,0]),2)
                self.assertEqual(len(ds.variables['time']),26304)
class ComparisonIntegration(unittest.TestCase):
    def test_paired_report_and_truth_mismatch(self):
        import csv
        from evaluate import compare
        from report_events import write_csv
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for model in ('fno','ffno'):
                for cohort in ('two_year','three_year'):
                    base=root/'evaluation'/f'{cohort}_{model}';base.mkdir(parents=True)
                    error=.1 if cohort=='two_year' else .08
                    write_csv(base/'event_metrics_by_seed.csv',[
                        dict(model=model,seed=seed,scope='full_year',hs_mae=error,hs_pooled_rmse=error*2)
                        for seed in (42,43,44)])
                    for seed in (42,43,44):
                        dest=base/'models'/f'{model}_s{seed}';dest.mkdir(parents=True)
                        write_csv(dest/'hourly.csv',[dict(time='2022-01-01T12:00:00',true_hs_max=3.,true_hs_mean=1.)])
            compare(root)
            with (root/'comparison_summary.csv').open() as f:rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),4)
            self.assertAlmostEqual(float(rows[0]['difference_three_minus_two']),-.02)
            bad=root/'evaluation/three_year_ffno/models/ffno_s44/hourly.csv'
            write_csv(bad,[dict(time='2022-01-01T12:00:00',true_hs_max=4.,true_hs_mean=1.)])
            with self.assertRaises(ValueError):compare(root)

if __name__=='__main__':unittest.main()

