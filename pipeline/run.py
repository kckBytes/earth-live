"""One pipeline run: newest frames from five satellites -> stitched global maps + motion -> site/.

  python pipeline/run.py <work dir> <site dir>        (env BACKFILL = how many missing replay frames to add, default 12)

work dir is kept between runs (Actions cache): reference layers, 26 h of small maps for motion, and the
replay frames (one every 30 min for the last 24 h, backfilled from the satellite archives).
Published (site/):
  clouds_4096.jpg, clouds_2048.jpg   cloud map (natural view), equirectangular, lon -180 at the left
  bt_4096.jpg, bt_2048.jpg           infrared temperature: 0 = no data, 1..255 = -90..+50 C
  sst_2048.png                       sea-surface temperature: 0 = land/ice, 1..255 = -2..+35 C (daily)
  flow.png                           64x32 RG motion, deg/hour (R east, G north), 128 = still
  meta.json                          observation time etc.
  storms.json                        globe labels: cyclones, lows/highs, alerts, launches
  wx.json + wx/<t>_a|b|c|w.png       weather layers from NOAA GFS (see weather.py)
  frames.json + frames/<t>_c.jpg, <t>_bt.jpg, <t>_f.png   replay: last 24 h, every 30 min, f = motion to next frame

Replay frames sit on an exact 30-minute grid (all from the archives, so every satellite is at the same moment).
Low cloud that only visible light can see (coastal stratus, fog) is carried through the night: each frame's
daylight-only layer is moved along the measured motion and slowly fades (fast over land, where such cloud really
does clear at night), instead of vanishing at sunset and popping back at sunrise.
"""
import os, sys, json, glob, shutil, subprocess, time, hashlib
import concurrent.futures as cf
from datetime import datetime, timedelta, timezone
import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fetch_slot, flow, storms, weather, sky

WORK, SITE = sys.argv[1], sys.argv[2]
BACKFILL = int(os.environ.get('BACKFILL', '12') or 12)
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

KEEP_H, PUB_H = 36, 24          # frames kept (night history for the low-cloud carry) / published
def add_frame(out, t_ms):
    for src, dst in [('_ir_2048.jpg', '_ir.jpg'), ('_lo_2048.png', '_lo.png'), ('_dw_512.png', '_dw.png'),
                     ('_bt_2048.jpg', '_bt.jpg'), ('_512.png', '_s.png')]:
        shutil.copy(out + src, f'{FR}/{t_ms}{dst}')

def frame_times():
    return sorted(int(os.path.basename(p).split('_')[0]) for p in glob.glob(f'{FR}/*_lo.png'))

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
             and os.path.exists(f'{WORK}/last/clouds_4096.jpg') and os.path.exists(f'{WORK}/last/layer_lo_4096.png'))

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
                     ('_bt_2048.jpg', 'bt_2048.jpg'), ('_sst_2048.png', 'sst_2048.png'), ('_flow.png', 'flow.png'),
                     ('_ir_4096.jpg', 'layer_ir_4096.jpg'), ('_lo_4096.png', 'layer_lo_4096.png'), ('_dw_512.png', 'layer_dw_512.png')]:
        shutil.copy(out + suf, f'{WORK}/last/{dst}')
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
for k in range(2 * KEEP_H + 1):
    slot = t0 - timedelta(minutes=30 * k)
    if obs_ms - ms(slot) > KEEP_H * H: break
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

for f in glob.glob(f'{FR}/*'):                              # drop frames past the history window (any format)
    try:
        if obs_ms - int(os.path.basename(f).split('_')[0]) > (KEEP_H + 0.5) * H: os.remove(f)
    except ValueError: os.remove(f)
ft = frame_times()
for a, b in zip(ft, ft[1:]):                                 # motion between consecutive frames
    fp = f'{FR}/{a}_f.png'
    if not os.path.exists(fp) or os.path.getmtime(fp) < os.path.getmtime(f'{FR}/{b}_s.png'):
        save_flow(flow.estimate(small(f'{FR}/{a}_s.png'), small(f'{FR}/{b}_s.png'), (b - a) / H), fp)
if ft and os.path.exists(f'{WORK}/last/flow.png'):
    shutil.copy(f'{WORK}/last/flow.png', f'{FR}/{ft[-1]}_f.png')
for g in glob.glob(f'{WORK}/out_*'): os.remove(g)

# ------------------------------------------------------------------ compose: carry daylight-only low cloud through the night
DEC_SEA, DEC_LAND = 0.985, 0.80          # kept per 30 min: ~70 % after 12 h over sea, gone within ~3 h over land
def enc(i):
    i = np.clip((i - 0.08) / 0.92, 0, 1)
    return np.where(i > 0, 0.40 + 0.56 * i ** 0.7, 0.38).astype(np.float32)
def load(path, size=None):
    im = Image.open(path).convert('L')
    if size and im.size != size: im = im.resize(size, Image.BILINEAR)
    return np.asarray(im, np.float32) / 255
def advect(P, flow_png, hours):
    h, w = P.shape
    f = np.asarray(Image.open(flow_png).convert('RGB').resize((w, h), Image.BILINEAR), np.float32)
    ve, vn = (f[..., 0] - 128) / 127 * flow.FLOW_MAX, (f[..., 1] - 128) / 127 * flow.FLOW_MAX
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return ndimage.map_coordinates(P, [yy + vn * hours / 180 * h, xx - ve * hours / 360 * w], order=1, mode='grid-wrap')
sea_p = f'{STATIC}/sea_2048.png'
SEA2 = load(sea_p, (2048, 1024)) > 0.5 if os.path.exists(sea_p) else np.ones((1024, 2048), bool)
P, prev = None, None
newest = ft[-1] if ft else 0
for t in ft:
    lo, dw = load(f'{FR}/{t}_lo.png', (2048, 1024)), load(f'{FR}/{t}_dw.png', (2048, 1024))
    hrs = (t - prev) / H if prev else 0
    if P is not None and 0 < hrs <= 3 and os.path.exists(f'{FR}/{prev}_f.png'):
        P = advect(P, f'{FR}/{prev}_f.png', hrs) * np.where(SEA2, DEC_SEA, DEC_LAND) ** (hrs / 0.5)
    else:
        P = np.zeros_like(lo)
    P = dw * lo + (1 - dw) * P
    prev = t
    if newest - t <= PUB_H * H + 60e3:
        c8 = (np.maximum(load(f'{FR}/{t}_ir.jpg'), enc(P)) * 255).astype(np.uint8)
        dig = hashlib.md5(c8.tobytes()).hexdigest()           # rewrite only when it changed, so phones re-fetch only then
        if not os.path.exists(f'{FR}/{t}_c.jpg') or not os.path.exists(f'{FR}/{t}_c.md5') or open(f'{FR}/{t}_c.md5').read() != dig:
            Image.fromarray(c8).save(f'{FR}/{t}_c.jpg', quality=90); open(f'{FR}/{t}_c.md5', 'w').write(dig)
# the live map gets the same carry, so the replay's last frame and the still globe agree
if P is not None and ft and os.path.exists(f'{WORK}/last/layer_lo_4096.png') and os.path.exists(f'{WORK}/last/flow.png'):
    lt = meta.get('time', newest)
    hrs = max(0.0, (lt - newest) / H)
    Pl = advect(P, f'{WORK}/last/flow.png', hrs) * np.where(SEA2, DEC_SEA, DEC_LAND) ** (hrs / 0.5) if hrs > 0 else P
    Pl = np.asarray(Image.fromarray(Pl.astype(np.float32)).resize((4096, 2048), Image.BILINEAR))
    lo, dw = load(f'{WORK}/last/layer_lo_4096.png'), load(f'{WORK}/last/layer_dw_512.png', (4096, 2048))
    c = np.maximum(load(f'{WORK}/last/layer_ir_4096.jpg'), enc(dw * lo + (1 - dw) * Pl))
    im = Image.fromarray((c * 255).astype(np.uint8))
    im.save(f'{WORK}/last/clouds_4096.jpg', quality=90); im.resize((2048, 1024), Image.LANCZOS).save(f'{WORK}/last/clouds_2048.jpg', quality=90)
    print(f'night carry: low cloud kept over {(enc(P) > 0.45).mean() * 100:.1f}% of the newest frame')

# ------------------------------------------------------------------ publish
for f in glob.glob(f'{WORK}/last/*'):
    if not os.path.basename(f).startswith('layer_'): shutil.copy(f, SITE)
os.makedirs(f'{SITE}/frames', exist_ok=True)
ft = [t for t in ft if ft[-1] - t <= PUB_H * H + 60e3]
for t in ft:
    for k in ('c.jpg', 'bt.jpg', 'f.png'):
        if os.path.exists(f'{FR}/{t}_{k}'): shutil.copy(f'{FR}/{t}_{k}', f'{SITE}/frames/{t}_{k}')
# how much the clouds change from each frame to the next, on an 8 x 4 grid of the world (lon x lat, x1000):
# the phone slows the replay down where the half of the Earth it shows is busy, and speeds through calm spells
act = {}
for a, b in zip(ft, ft[1:]):
    d = np.abs(small(f'{FR}/{b}_s.png') - small(f'{FR}/{a}_s.png'))
    act[a] = [int(round(v * 1000)) for v in d.reshape(4, 64, 8, 64).mean(axis=(1, 3)).ravel()]
json.dump({'frames': [dict({'t': t, 'v': int(os.path.getmtime(f'{FR}/{t}_c.jpg') * 1000)}, **({'a': act[t]} if t in act else {}))
                      for t in ft], 'every_min': 30, 'kinds': ['c', 'bt'], 'activity_grid': [8, 4]},
          open(f'{SITE}/frames.json', 'w'))
try:
    sky.build(WORK, SITE)                                    # stations, camera satellites, aurora, fires, lightning
except Exception as e:
    print('sky failed:', e)
try:
    weather.build(WORK, SITE)                                # Ventusky-style layers, only when a new model run is out
except Exception as e:
    print('weather failed:', e)
try:
    json.dump(storms.fetch_all(cache_dir=WORK), open(f'{SITE}/storms.json', 'w'))
except Exception as e:
    print('storms failed:', e); json.dump({'generated': int(time.time() * 1000), 'storms': []}, open(f'{SITE}/storms.json', 'w'))
json.dump(meta, open(f'{SITE}/meta.json', 'w'), indent=1)
span = (ft[-1] - ft[0]) / H if len(ft) > 1 else 0
print(f'published {meta.get("time_iso")}; replay {len(ft)} frames over {span:.1f} h; run took {time.time() - t_start:.0f}s')
