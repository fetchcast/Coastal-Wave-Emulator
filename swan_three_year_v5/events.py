#!/usr/bin/env python3
"""Define reproducible JMA track-based event windows without model predictions."""
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile
import numpy as np
from campaign import atomic, freeze, read

URL = 'https://www.jma.go.jp/jma/jma-eng/jma-center/rsmc-hp-pub-eg/Besttracks/bst_all.zip'
FORMAT_URL = 'https://www.jma.go.jp/jma/jma-eng/jma-center/rsmc-hp-pub-eg/Besttracks/e_format_bst.html'


def parse_jma(text):
    lines = text.splitlines()
    storms = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        header = lines[i]
        fields = header.split()
        if fields[0] != '66666':
            raise ValueError(f'Invalid JMA header at line {i+1}')
        sid, count = fields[1], int(fields[2])
        block = lines[i+1:i+1+count]
        if len(block) != count:
            raise ValueError('Truncated JMA track')
        if sid.startswith('22'):
            points = []
            for line in block:
                p = line.split()
                if len(p) < 6 or p[1] != '002':
                    raise ValueError('Invalid JMA track line')
                ts = p[0]
                stamp = f'20{ts[:2]}-{ts[2:4]}-{ts[4:6]}T{ts[6:8]}:00:00'
                points.append(dict(time=stamp, grade=int(p[2]), lat=int(p[3])/10,
                                   lon=int(p[4])/10))
            if not points:
                raise ValueError('Empty track')
            storms.append(dict(id=sid, name=header[30:50].strip() or sid,
                               lifetime_typhoon=any(p['grade'] == 5 for p in points), points=points))
        i += count + 1
    if not storms:
        raise ValueError('No 2022 storms found in JMA source')
    return storms


def verify_source(directory):
    """Download once; a user-supplied official jma_besttrack.txt also works."""
    directory = Path(directory)
    target = directory/'jma_besttrack.txt'
    if not target.exists():
        try:
            request = urllib.request.Request(URL, headers={'User-Agent':'SWAN-research-evaluation/3.0'})
            with urllib.request.urlopen(request, timeout=45) as response:
                data = response.read(20*1024*1024+1)
            if len(data) > 20*1024*1024:
                raise ValueError('Unexpected download size')
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                names = [n for n in z.namelist() if n.lower().endswith('.txt')]
                if len(names) != 1:
                    raise ValueError('Expected one JMA text file')
                text = z.read(names[0]).decode('ascii')
            parse_jma(text)
            tmp = target.with_suffix('.tmp')
            tmp.write_text(text, encoding='ascii')
            tmp.replace(target)
        except Exception as exc:
            raise RuntimeError('JMA download failed. Existing training was not stopped. '
                               'Download official bst_all.zip, extract its text file as '
                               f'{target}, and run again. Source: {URL}') from exc
    storms = parse_jma(target.read_text(encoding='ascii'))
    return dict(url=URL, format_url=FORMAT_URL,
                sha256=hashlib.sha256(target.read_bytes()).hexdigest(), storm_count=len(storms))


def sphere(lat, lon):
    lat, lon = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)], axis=-1)


def build_events(nc, source_dir, protocol, output):
    import xarray as xr
    from scipy.spatial import cKDTree
    provenance = verify_source(source_dir)
    storms = parse_jma((Path(source_dir)/'jma_besttrack.txt').read_text(encoding='ascii'))
    with xr.open_dataset(nc) as ds:
        wet = ds.kcs.values > 0
        lon, lat = ds.x.values, ds.y.values
        if lon.shape != wet.shape or lat.shape != wet.shape:
            raise ValueError('Expected matching 2D geographic coordinates')
        if not (np.isfinite(lon[wet]).all() and np.isfinite(lat[wet]).all()
                and np.abs(lat[wet]).max() <= 90 and np.abs(lon[wet]).max() <= 360):
            raise ValueError('Invalid geographic coordinates for event selection')
        tree = cKDTree(sphere(lat[wet], lon[wet]))
    events = []
    for storm in storms:
        pts = storm['points']
        times = np.array([p['time'] for p in pts], dtype='datetime64[h]').astype('int64')
        if len(times) < 2 or np.any(np.diff(times) <= 0) or np.max(np.diff(times)) > 12:
            raise ValueError(f'Invalid track spacing: {storm["id"]}')
        hours = np.arange(times[0], times[-1]+1)
        lat = np.interp(hours, times, [p['lat'] for p in pts])
        unwrapped = np.unwrap(np.radians([p['lon'] for p in pts]))
        lon = np.degrees(np.interp(hours, times, unwrapped))
        grades = np.array([p['grade'] for p in pts])[np.searchsorted(times, hours, side='right')-1]
        distance = 2*6371.0088*np.arcsin(np.clip(tree.query(sphere(lat, lon))[0]/2, 0, 1))
        near = (distance <= protocol['event_distance_km']) & np.isin(grades, protocol['tc_grades'])
        if not near.any():
            continue
        selected = hours[near]
        padding = int(protocol['event_padding_hours'])
        start = max(int(selected[0])-padding, int(np.datetime64('2022-01-01T12','h').astype('int64')))
        end = min(int(selected[-1])+padding, int(np.datetime64('2022-12-31T23','h').astype('int64')))
        if start > end:
            continue
        events.append(dict(id=storm['id'], name=storm['name'], lifetime_typhoon=storm['lifetime_typhoon'],
            start=str(np.datetime64(start,'h')), end=str(np.datetime64(end,'h')),
            min_center_distance_km=float(distance[near].min())))
    if not events:
        raise ValueError('No events found; inspect tracks and grid before evaluation')
    events.sort(key=lambda e:(e['start'],e['id']))
    value=dict(source=provenance, protocol=protocol, events=events,
               note='Track proximity defines evaluation windows, not causal attribution of all waves to a cyclone.')
    freeze(output, value)
    return value


def freeze_snapshots(prepared, event_manifest, output):
    cache = Path(read(prepared)['cache'])
    y = np.load(cache/'targets.npy', mmap_mode='r')
    wet = np.load(cache/'mask.npy')
    times = np.load(cache/'times.npy').astype('datetime64[h]')
    events = read(event_manifest)['events']
    out = []
    for event in events:
        ids = np.flatnonzero((times >= np.datetime64(event['start'])) & (times <= np.datetime64(event['end'])))
        ids = ids[ids >= 12]
        if not len(ids):
            raise ValueError('Empty event window')
        peaks = [float(np.max(y[i,0][wet])) for i in ids]
        i = int(ids[int(np.argmax(peaks))])
        out.append(dict(event_id=event['id'], target_index=i, time=str(times[i])))
    freeze(output, out)
    return out
