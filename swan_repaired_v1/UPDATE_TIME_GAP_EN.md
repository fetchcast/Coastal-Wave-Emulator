# Time-gap update

English | [Original Korean note](UPDATE_TIME_GAP.md)

The supplied time axis omits 23 hourly records between 2019-12-31 00:00 and 2020-01-01 00:00. Ordered timestamps with integer-hour gaps are accepted, but any sequence crossing a gap between input start and target is excluded from every split and direction-calibration candidate set. Half-hour spacing, duplicate/reversed timestamps, and NaT are rejected.

The source NetCDF, model, loss, and training budget are unchanged. Blocks retain the original unit of 168 stored samples. A block containing a gap must not be described as exactly 168 calendar hours.

For the supplied gap, existing embargo exclusions already covered the affected sequences, so the reproduction check retained all split indices. Train/validation/test counts were 9770/1980/1980 at full fraction, 5150/1980/1980 at 50%, and 2640/1980/1980 at 25%. A separate gap-inside-block test excluded only affected sequences. Nanosecond/microsecond time units, invalid axes, and ten existing CPU checks were also tested. Other real-server preprocessing and GPU training were not verified in that patch-development environment.

## Historical patch installation

The original `SWAN_TimeGap_Update_v1.zip` was extracted over `swan_repaired_v1`. Since code fingerprints changed, earlier failed results were preserved and a new root was used. The repository now contains the updated source; the commands below document that patch-era workflow, not a requirement to download an unavailable archive.

```bash
cd /home/jovyan/swan
unzip -o -q SWAN_TimeGap_Update_v1.zip
nohup bash swan_repaired_v1/START_REPAIRED.sh --gpus 0,1,2 --root /home/jovyan/swan/runs/repaired_timegap_v1 > repaired_timegap_run.log 2>&1 &
tail -f repaired_timegap_run.log
```

Do not modify an existing `protocol.json` or disable verification to bypass fingerprint changes. Do not replace code used by active jobs.

```bash
python3 swan_repaired_v1/run_repaired.py --root /home/jovyan/swan/runs/repaired_timegap_v1 --report
```
