"""One pipeline run: newest frames from five satellites -> stitched global maps + motion -> site/.

  python pipeline/run.py <work dir> <site dir>        (env BACKFILL = how many missing replay frames to add, default 2)

work dir is kept between runs (Actions cache): reference layers, 26 h of small maps for motion, and the
replay frames (one every 30 min for the last 24 h, backfilled from the satellite archives).
Published (site/):
  clouds_4096.jpg, clouds_2048.jpg   cloud map (natural view), equirectangular, lon -180 at the left
  bt_4096.jpg, bt_2048.jpg           infrared temperature: 0 = no data, 1..255 = -90..+50 C
  sst_2048.png                       sea-surface temperature: 0 = land/ice, 1..255 = -2..+35 C (daily)
  flow.png                           64x32 RG motion, deg/hour (R east, G north), 128 = still
  meta.json                          observation time etc.
  frames.json + frames/<t>_c.jpg, <t>_bt.jpg, <t>_f.png   replay: last 24 h, every 30 min, f = motion to next frame
"""
import os, sys, json, glob, shutil, subprocess, time
import concurrent.futures as cf
from datetime import datetime, timedelta, timezone
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fetch_slot, flow

WORK, SITE = sys.argv[1], sys.argv[2]
BACKFILL = int(os.environ.get('BACKFILL', '2') or 2)
PARALLEL = int(os.environ.get('PARALLEL', '3') or 3)
STATIC, HIST, FR = f'{WORK}/static', f'{WORK}/history', f'{WORK}/frames'
for d in (WORK, SITE, HIST, FR, STATIC): os.makedirs(d, exist_ok=True)
t_start = time.time()
now = datetime.now(timezone.utc)
ms = lambda t: int(t.timestamp() * 1000)
H = 3.6e6

def stitch_slot(times, obs, tag):
    """Fetch + stitch one moment. Returns the output prefix, or None."""
    slot = f'{WORK}/slot_{tag}'
    shutil.rmtree(slot, ignore_errors=True)
    fetch_slot.download(times, slot, STATIC)
    out = f'{WORK}/out_{tag}'
    try:
        r = subprocess.run([sys.executable, f'{HERE}/stitch.py', slot, obs.strftime('%Y-%m-%dT%H:%MZ'), day_tex, polar, out, STATIC],
                           capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            print(f'stitch {tag} failed:\n', r.stdout[-800:], r.stderr[-1500:]); return None
        if tag == 'now': print(r.stdout.strip())
        return out
    finally:
        shutil.rmtree(slot, ignore_errors=True)

def add_frame(out, t_ms):
    shutil.copy(f'{out}_2048.jpg', f'{FR}/{t_ms}_c.jpg')
    shutil.copy(f'{out}_bt_2048.jpg', f'{FR}/{t_ms}_bt.jpg')
    shutil.copy(f'{out}_512.png', f'{FR}/{t_ms}_s.png')

def frame_times():
    return sorted(int(os.path.basename(p).split('_')[0]) for p in glob.glob(f'{FR}/*_s.png'))

def small(path): return np.array(Image.open(path).convert('L'), np.float32) / 255

def save_flow(fl, path):
    Image.fromarray(np.concatenate([fl, np.zeros((32, 64, 1), np.uint8)], -1), 'RGB').save(path)

# ------------------------------------------------------------------ newest frames
times = fetch_slot.find_times(now)
print('newest frames:', {k: (v.strftime('%H:%M') if v else None) for k, v in times.items()})
ir_keys = ['goes_e', 'goes_w', 'hima', 'mtg', 'iodc']
ir_times = [times[k] for k in ir_keys if times.get(k)]
if len(ir_times) < 4:
    print('fewer than 4 infrared sources available, skipping this run'); sys.exit(0)
obs = min(ir_times)
obs_ms = ms(obs)

month = now.strftime('%m')
day_tex = f'{STATIC}/day_{month}.jpg'
if not os.path.exists(day_tex):
    fetch_slot.fetch(f'https://kckbytes.github.io/earth-web/tex/day/{month}.jpg', day_tex)
polar = f'{WORK}/polar.jpg'
if fetch_slot.fetch('https://clouds.matteason.co.uk/images/2048x1024/clouds.jpg', polar) is None and not os.path.exists(polar):
    Image.new('L', (2048, 1024), 97).save(polar)

prev_meta = json.load(open(f'{WORK}/last_meta.json')) if os.path.exists(f'{WORK}/last_meta.json') else {}
fresh = not (prev_meta.get('time') == obs_ms and prev_meta.get('flow_hours', 0) > 0
             and os.path.exists(f'{WORK}/last/clouds_4096.jpg') and os.path.exists(f'{WORK}/last/bt_2048.jpg'))

if fresh:
    out = stitch_slot(times, obs, 'now')
    if out is None: sys.exit(1)
    # motion for the live map: against a stored small map ~1 h earlier, else backfill that one from the archive
    hist = sorted((int(os.path.basename(p)[:-4]), p) for p in glob.glob(f'{HIST}/*.png'))
    cands = hist + [(t, f'{FR}/{t}_s.png') for t in frame_times()]
    best = None
    for t, p in cands:
        age = (obs_ms - t) / H
        if 0.66 <= age <= 2.0 and (best is None or abs(age - 1.0) < abs(best[0] - 1.0)): best = (age, p)
    if best is None:
        back = {k: (v - timedelta(hours=1) if v else None) for k, v in times.items()}
        print('backfilling the map from one hour earlier')
        bo = stitch_slot(back, obs - timedelta(hours=1), 'back')
        if bo:
            p = f'{HIST}/{obs_ms - int(H)}.png'; shutil.copy(f'{bo}_512.png', p); best = (1.0, p)
    if best:
        fl = flow.estimate(small(best[1]), small(f'{out}_512.png'), best[0])
        m_, p90, mx = flow.stats(fl)
        print(f'motion over {best[0]:.2f} h: mean {m_:.2f}, p90 {p90:.2f}, max {mx:.2f} deg/h'); flow_hours = round(best[0], 3)
    else:
        fl = flow.still(); flow_hours = 0
    save_flow(fl, f'{out}_flow.png')
    shutil.copy(f'{out}_512.png', f'{HIST}/{obs_ms}.png')
    for t, p in hist:
        if obs_ms - t > 26 * H and os.path.exists(p): os.remove(p)
    os.makedirs(f'{WORK}/last', exist_ok=True)
    for suf, dst in [('_4096.jpg', 'clouds_4096.jpg'), ('_2048.jpg', 'clouds_2048.jpg'), ('_bt_4096.jpg', 'bt_4096.jpg'),
                     ('_bt_2048.jpg', 'bt_2048.jpg'), ('_sst_2048.png', 'sst_2048.png'), ('_flow.png', 'flow.png')]:
        shutil.copy(out + suf, f'{WORK}/last/{dst}')
    ft = frame_times()                                       # a replay frame every 30 min
    if not ft or obs_ms - ft[-1] >= 25 * 60e3:
        add_frame(out, obs_ms); print('added replay frame', obs.strftime('%H:%M'))
    meta = {'time': obs_ms, 'time_iso': obs.strftime('%Y-%m-%dT%H:%M:%SZ'), 'generated': int(time.time() * 1000),
            'flow_hours': flow_hours, 'sources': {k: (v.strftime('%Y-%m-%dT%H:%M:%SZ') if v else None) for k, v in times.items()},
            'credit': 'Contains modified EUMETSAT data; NASA GIBS (GOES via NOAA, Himawari via JMA); GHRSST MUR; SRTM.'}
    json.dump(meta, open(f'{WORK}/last_meta.json', 'w'))
else:
    print('no newer frames than last run')
    meta = prev_meta

# ------------------------------------------------------------------ backfill replay frames (30-min slots, last 24 h)
have = frame_times()
want = []
t0 = obs.replace(minute=(obs.minute // 30) * 30, second=0, microsecond=0)
for k in range(49):
    slot = t0 - timedelta(minutes=30 * k)
    if obs_ms - ms(slot) > 24 * H: break
    if not any(abs(ms(slot) - h) < 20 * 60e3 for h in have): want.append(slot)
todo = want[:BACKFILL]
if todo:
    print(f'replay: {len(have)} frames, {len(want)} missing, backfilling {len(todo)}')
    def job(slot):
        tt = {k: slot for k in fetch_slot.GIBS_LAYERS}
        tt.update({k: slot for k in fetch_slot.EUM_LAYERS})
        o = stitch_slot(tt, slot, f'bf{ms(slot)}')
        if o: add_frame(o, ms(slot))
        return slot, o is not None
    with cf.ThreadPoolExecutor(PARALLEL) as ex:
        for slot, ok in ex.map(job, todo): print('   ', slot.strftime('%m-%d %H:%M'), 'ok' if ok else 'failed')

ft = frame_times()                                           # drop frames older than 24 h
for t in ft:
    if obs_ms - t > 24.5 * H:
        for f in glob.glob(f'{FR}/{t}_*'): os.remove(f)
ft = frame_times()
for a, b in zip(ft, ft[1:]):                                 # motion between consecutive frames
    fp = f'{FR}/{a}_f.png'
    if not os.path.exists(fp) or os.path.getmtime(fp) < os.path.getmtime(f'{FR}/{b}_s.png'):
        save_flow(flow.estimate(small(f'{FR}/{a}_s.png'), small(f'{FR}/{b}_s.png'), (b - a) / H), fp)
if ft and os.path.exists(f'{WORK}/last/flow.png'):
    shutil.copy(f'{WORK}/last/flow.png', f'{FR}/{ft[-1]}_f.png')
for g in glob.glob(f'{WORK}/out_*'): os.remove(g)

# ------------------------------------------------------------------ publish
for f in glob.glob(f'{WORK}/last/*'): shutil.copy(f, SITE)
os.makedirs(f'{SITE}/frames', exist_ok=True)
for t in ft:
    for k in ('c.jpg', 'bt.jpg', 'f.png'):
        if os.path.exists(f'{FR}/{t}_{k}'): shutil.copy(f'{FR}/{t}_{k}', f'{SITE}/frames/{t}_{k}')
json.dump({'frames': [{'t': t} for t in ft], 'every_min': 30, 'kinds': ['c', 'bt']}, open(f'{SITE}/frames.json', 'w'))
json.dump(meta, open(f'{SITE}/meta.json', 'w'), indent=1)
span = (ft[-1] - ft[0]) / H if len(ft) > 1 else 0
print(f'published {meta.get("time_iso")}; replay {len(ft)} frames over {span:.1f} h; run took {time.time() - t_start:.0f}s')
