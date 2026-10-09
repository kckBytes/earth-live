"""One pipeline run: newest frames from five satellites -> stitched global cloud map + motion field -> site/.

  python pipeline/run.py <work dir> <site dir>

work dir keeps state between runs (restored from the Actions cache): reference layers and a 26-hour
history of small 512x256 maps used to measure cloud motion.
Published files (site/):
  clouds_4096.jpg, clouds_2048.jpg   global cloud map, equirectangular, lon -180 at the left edge
  flow.png                           64x32 RG motion field, deg/hour (R = east, G = north), 128 = still
  meta.json                          {"time": observation time (ms), "sources": {...}, "flow_hours": ...}
"""
import os, sys, json, glob, shutil, subprocess, time
from datetime import datetime, timezone
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fetch_slot, flow

WORK, SITE = sys.argv[1], sys.argv[2]
os.makedirs(WORK, exist_ok=True); os.makedirs(SITE, exist_ok=True)
STATIC, SLOT, HIST = f'{WORK}/static', f'{WORK}/slot', f'{WORK}/history'
os.makedirs(HIST, exist_ok=True)
t_start = time.time()
now = datetime.now(timezone.utc)

times = fetch_slot.find_times(now)
print('newest frames:', {k: (v.strftime('%H:%M') if v else None) for k, v in times.items()})
ir_keys = ['goes_e', 'goes_w', 'hima', 'mtg', 'iodc']
ir_times = [times[k] for k in ir_keys if times.get(k)]
if len(ir_times) < 4:
    print('fewer than 4 infrared sources available, skipping this run'); sys.exit(0)
obs = min(ir_times)                                     # conservative: the oldest frame in the mix
obs_ms = int(obs.timestamp() * 1000)

prev_meta = {}
if os.path.exists(f'{WORK}/last_meta.json'):
    prev_meta = json.load(open(f'{WORK}/last_meta.json'))
if prev_meta.get('time') == obs_ms and prev_meta.get('flow_hours', 0) > 0 and os.path.exists(f'{WORK}/last/clouds_4096.jpg'):
    print('no newer frames than last run; republishing')
    for f in glob.glob(f'{WORK}/last/*'): shutil.copy(f, SITE)
    sys.exit(0)

shutil.rmtree(SLOT, ignore_errors=True)
fetch_slot.download(times, SLOT, STATIC)

# day texture (surface brightness for the visible test) and the 3-hourly composite (polar fill)
month = now.strftime('%m')
day_tex = f'{STATIC}/day_{month}.jpg'
if not os.path.exists(day_tex):
    fetch_slot.fetch(f'https://kckbytes.github.io/earth-web/tex/day/{month}.jpg', day_tex)
polar = f'{WORK}/polar.jpg'
if fetch_slot.fetch('https://clouds.matteason.co.uk/images/2048x1024/clouds.jpg', polar) is None and not os.path.exists(polar):
    Image.new('L', (2048, 1024), 97).save(polar)

out = f'{WORK}/out'
subprocess.run([sys.executable, f'{HERE}/stitch.py', SLOT, obs.strftime('%Y-%m-%dT%H:%MZ'), day_tex, polar, out, STATIC], check=True)

# motion: compare with the stored map closest to 60 minutes earlier (40-120 min window)
small = np.array(Image.open(f'{out}_512.png').convert('L'), np.float32) / 255
hist = sorted((int(os.path.basename(p)[:-4]), p) for p in glob.glob(f'{HIST}/*.png'))
best = None
for t, p in hist:
    age = (obs_ms - t) / 3.6e6
    if 0.66 <= age <= 2.0 and (best is None or abs(age - 1.0) < abs(best[0] - 1.0)):
        best = (age, p)
if best is None:
    # No stored map ~1 h old (first run, or the schedule skipped): the satellites' archives keep past frames,
    # so fetch and stitch the slot one hour earlier right now instead of waiting.
    from datetime import timedelta
    back = {k: (v - timedelta(hours=1) if v else None) for k, v in times.items()}
    obs_b = obs - timedelta(hours=1)
    print('backfilling the map from one hour earlier:', obs_b.strftime('%H:%M'))
    shutil.rmtree(SLOT, ignore_errors=True)
    fetch_slot.download(back, SLOT, STATIC)
    try:
        subprocess.run([sys.executable, f'{HERE}/stitch.py', SLOT, obs_b.strftime('%Y-%m-%dT%H:%MZ'), day_tex, polar,
                        f'{WORK}/back', STATIC], check=True)
        p = f'{HIST}/{int(obs_b.timestamp() * 1000)}.png'
        shutil.copy(f'{WORK}/back_512.png', p)
        best = (1.0, p)
    except subprocess.CalledProcessError as e:
        print('backfill failed:', e)
if best:
    older = np.array(Image.open(best[1]).convert('L'), np.float32) / 255
    fl = flow.estimate(older, small, best[0])
    mean, p90, mx = flow.stats(fl)
    print(f'motion over {best[0]:.2f} h: mean {mean:.2f}, p90 {p90:.2f}, max {mx:.2f} deg/h')
    flow_hours = round(best[0], 3)
else:
    fl = flow.still(); flow_hours = 0
    print('no map ~1 h old yet: publishing still motion')
Image.fromarray(np.concatenate([fl, np.zeros((32, 64, 1), np.uint8)], -1), 'RGB').save(f'{out}_flow.png')
shutil.copy(f'{out}_512.png', f'{HIST}/{obs_ms}.png')
for t, p in hist:                                       # keep 26 h
    if (obs_ms - t) > 26 * 3.6e6: os.remove(p)

meta = {'time': obs_ms, 'time_iso': obs.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'generated': int(time.time() * 1000), 'flow_hours': flow_hours,
        'sources': {k: (v.strftime('%Y-%m-%dT%H:%M:%SZ') if v else None) for k, v in times.items()},
        'credit': 'Contains modified EUMETSAT data; NASA GIBS (GOES-East/West, Himawari via NOAA/JMA); GHRSST MUR; SRTM.'}
os.makedirs(f'{WORK}/last', exist_ok=True)
for src, dst in [(f'{out}_4096.jpg', 'clouds_4096.jpg'), (f'{out}_2048.jpg', 'clouds_2048.jpg'), (f'{out}_flow.png', 'flow.png')]:
    shutil.copy(src, f'{SITE}/{dst}'); shutil.copy(src, f'{WORK}/last/{dst}')
json.dump(meta, open(f'{SITE}/meta.json', 'w'), indent=1)
json.dump(meta, open(f'{WORK}/last/meta.json', 'w'), indent=1)
json.dump(meta, open(f'{WORK}/last_meta.json', 'w'))
shutil.rmtree(SLOT, ignore_errors=True)
print(f'published {meta["time_iso"]} ({(time.time() - obs.timestamp()) / 60:.0f} min after observation), run took {time.time() - t_start:.0f}s')
