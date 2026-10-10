"""Active tropical cyclones with their recent track, for the storm labels on the globe.

Sources (free, no key): GDACS (UN/EU Global Disaster Alert and Coordination System) for every ocean basin,
and NOAA's National Hurricane Center for the current class and wind of Atlantic / East Pacific storms.
Output: {"generated": ms, "storms": [{"name", "kind", "detail", "last_obs": ms, "track": [[ms, lat, lon, forecast], ...]}]}
"""
import json, re, urllib.request
from datetime import datetime, timedelta, timezone

UA = {'User-Agent': 'earth-live (https://github.com/kckBytes/earth-live)'}
GDACS = 'https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH?eventlist=TC&fromDate={a}&toDate={b}'
NHC = 'https://www.nhc.noaa.gov/CurrentStorms.json'
NHC_KIND = {'HU': 'Hurricane', 'TS': 'Tropical storm', 'TD': 'Tropical depression', 'STS': 'Subtropical storm',
            'STD': 'Subtropical depression', 'PTC': 'Post-tropical', 'PC': 'Potential cyclone', 'TY': 'Typhoon'}

def get(url, timeout=40):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return json.loads(r.read().decode('utf-8'))

def ms(dt): return int(dt.timestamp() * 1000)

def strong_name(lat, lon):
    if lat >= 0 and 100 <= lon <= 180: return 'Typhoon'
    if lat >= 0 and (-180 <= lon < -20): return 'Hurricane'
    return 'Cyclone'

def gdacs_kind(sev, lat, lon):
    s = (sev or '').lower()
    if s.startswith('tropical depression'): return 'Tropical depression'
    if s.startswith('tropical storm'): return 'Tropical storm'
    return strong_name(lat, lon)

def category(kt):
    return next((c for c, lo in ((5, 137), (4, 113), (3, 96), (2, 83), (1, 64)) if kt >= lo), 0)

def track(ev, now):
    """Track points (6-hourly, observed then forecast) and the storm class along each segment."""
    p = ev['properties']
    g = get(p['url']['geometry'])
    last_obs = datetime.fromisoformat(p['todate']).replace(tzinfo=timezone.utc)
    pts, seg = [], {}
    for f in g.get('features', []):
        fp, geo = f.get('properties', {}), f.get('geometry', {})
        cls = str(fp.get('Class', ''))
        if cls.startswith('Line_Line_'):
            seg[int(cls.rsplit('_', 1)[1])] = str(fp.get('polygonlabel', ''))
            continue
        if not cls.startswith('Point_Polygon_Point'): continue
        m = re.match(r'(\d\d)/(\d\d) (\d\d):(\d\d)', fp.get('polygonlabel', ''))
        if not m: continue
        if geo.get('type') == 'Point': lon, lat = geo['coordinates'][:2]
        elif geo.get('type') == 'Polygon':                      # a small circle drawn around the centre
            ring = geo['coordinates'][0]; lon = sum(q[0] for q in ring) / len(ring); lat = sum(q[1] for q in ring) / len(ring)
        else: continue
        d, mo, hh, mi = map(int, m.groups())
        t = datetime(last_obs.year, mo, d, hh, mi, tzinfo=timezone.utc)
        if t > last_obs + timedelta(days=20): t = t.replace(year=t.year - 1)
        pts.append([ms(t), round(lat, 2), round(lon, 2), 1 if t > last_obs else 0])
    pts.sort()
    obs = [i for i, q in enumerate(pts) if not q[3]]
    cls_now = seg.get(obs[-1] - 1, seg.get(obs[-1], '')) if obs else ''
    return [q for q in pts if q[0] <= ms(now + timedelta(hours=12))], ms(last_obs), cls_now

def fetch(now=None):
    now = now or datetime.now(timezone.utc)
    out = []
    try:
        evs = get(GDACS.format(a=(now - timedelta(days=4)).strftime('%Y-%m-%d'), b=(now + timedelta(days=1)).strftime('%Y-%m-%d')))
    except Exception as e:
        print('storms: GDACS unavailable', e); evs = {'features': []}
    try:
        nhc = {s['name'].lower(): s for s in get(NHC).get('activeStorms', [])}
    except Exception as e:
        print('storms: NHC unavailable', e); nhc = {}
    seen = set()
    for ev in evs.get('features', []):
        p = ev.get('properties', {})
        try:
            to = datetime.fromisoformat(p['todate']).replace(tzinfo=timezone.utc)
            if now - to > timedelta(hours=30) or p['eventid'] in seen: continue
            seen.add(p['eventid'])
            name = p['eventname'].split('-')[0].strip().title()
            pts, last_obs, cls = track(ev, now)
            if len(pts) < 1: continue
            lon, lat = ev['geometry']['coordinates'][:2]
            kind = {'HU': strong_name(lat, lon), 'TS': 'Tropical storm', 'TD': 'Tropical depression'}.get(
                cls.upper(), gdacs_kind(p.get('severitydata', {}).get('severitytext'), lat, lon))
            detail = ''
            n = nhc.get(name.lower())
            if n:
                kind = NHC_KIND.get(n.get('classification', ''), kind)
                kt = float(n.get('intensity') or 0)
                cat = category(kt) if kind == 'Hurricane' else 0
                detail = (f'Cat {cat} · ' if cat else '') + f'{round(kt * 1.15078 / 5) * 5:.0f} mph'
            out.append({'name': name, 'kind': kind, 'detail': detail, 'last_obs': last_obs, 'track': pts})
        except Exception as e:
            print('storms: skipped', p.get('eventname'), e)
    print('storms:', ', '.join(f"{s['kind']} {s['name']} ({len(s['track'])} pts)" for s in out) or 'none active')
    return {'generated': ms(now), 'storms': out}

if __name__ == '__main__':
    print(json.dumps(fetch(), indent=1)[:2000])


# ---------------------------------------------------------------------------------------------------------
# Everything else on the globe's labels, each type capped so it never gets busy:
#   big pressure systems (L / H) from NOAA GFS, volcano eruptions / large wildfires / major earthquakes and
#   floods (GDACS orange or red alerts), and rocket launches (The Space Devs' Launch Library 2).
import os, math
import numpy as np
from scipy import ndimage

GFS = ('https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_1p00.pl?dir=%2Fgfs.{d}%2F{c:02d}%2Fatmos'
       '&file=gfs.t{c:02d}z.pgrb2.1p00.f{f:03d}&var_PRMSL=on')
GDACS_EV = 'https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH?eventlist=EQ;VO;WF;FL&alertlevel=Orange;Red&fromDate={a}&toDate={b}'
LL2 = 'https://ll.thespacedevs.com/2.3.0/launches/{w}/?limit=12'

def raw(url, timeout=60):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()

def grib_field(url):
    import eccodes
    g = eccodes.codes_new_from_message(raw(url))
    try: return eccodes.codes_get_values(g).reshape(181, 360)          # lat 90..-90, lon 0..359
    finally: eccodes.codes_release(g)

def prmsl(d, c, f):
    return grib_field(GFS.format(d=d, c=c, f=f)) / 100.0                 # hPa

_oro = {}
def orography():
    """GFS surface height (m), fetched once per run: sea-level pressure over high ground is an extrapolation,
    so highs and lows there (Tibet, the Rockies, Greenland, Antarctica) are not real weather systems."""
    if 'z' not in _oro:
        now = datetime.now(timezone.utc) - timedelta(hours=8)
        c = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=now.hour % 6)
        _oro['z'] = grib_field(GFS.format(d=c.strftime('%Y%m%d'), c=c.hour, f=0).replace('var_PRMSL=on', 'var_HGT=on&lev_surface=on'))
    return _oro['z']

def extrema(v):
    """Deep lows and strong highs on a 1-degree sea-level pressure field: (kind, lat, lon, hPa).
    Ranked by how much they stand out from their surroundings; a few per hemisphere so one ocean can't take all."""
    lat = 90 - np.arange(181)[:, None] * np.ones((1, 360)); lon = np.arange(360)[None, :] * np.ones((181, 1))
    lon = np.where(lon > 180, lon - 360, lon)
    try: low_ground = orography() < 800
    except Exception: low_ground = ~((lat > 25) & (lat < 45) & (lon > 70) & (lon < 105)) & (np.abs(lat) < 60)
    mean = ndimage.uniform_filter(v, size=(21, 21), mode=('nearest', 'wrap'))
    lo = (v == ndimage.minimum_filter(v, size=(15, 15), mode=('nearest', 'wrap'))) & (v < 1000) & (mean - v > 6) \
         & (np.abs(lat) < 72) & low_ground
    hi = (v == ndimage.maximum_filter(v, size=(15, 15), mode=('nearest', 'wrap'))) & (v > 1026) & (np.abs(lat) < 65) & low_ground
    out = []
    for kind, m, cap in (('L', lo, (4, 3)), ('H', hi, (2, 2))):
        pts = [(kind, float(lat[y, x]), float(lon[y, x]), float(v[y, x]), abs(float(mean[y, x] - v[y, x]))) for y, x in zip(*np.where(m))]
        pts.sort(key=lambda q: -q[4])
        out += [q[:4] for q in pts if q[1] >= 0][:cap[0]] + [q[:4] for q in pts if q[1] < 0][:cap[1]]
    return out

def km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    return 6371 * math.acos(max(-1, min(1, math.sin(la1) * math.sin(la2) + math.cos(la1) * math.cos(la2) * math.cos(lo1 - lo2))))

def systems(now):
    """Lows / highs at the last five GFS analyses (24 h, for the replay) plus +6 / +12 h of the newest run, linked
    into tracks by nearest neighbour."""
    snaps = []
    cyc = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=now.hour % 6)
    newest = None
    for k in range(7):
        c = cyc - timedelta(hours=6 * k)
        try:
            v = prmsl(c.strftime('%Y%m%d'), c.hour, 0)
        except Exception:
            continue
        snaps.append((c, extrema(v), 0)); newest = newest or c
        if len(snaps) >= 5: break
    if newest:
        for f in (6, 12):
            try: snaps.append((newest + timedelta(hours=f), extrema(prmsl(newest.strftime('%Y%m%d'), newest.hour, f)), 1))
            except Exception: pass
    snaps.sort(key=lambda s: s[0])
    tracks = []                                    # each: kind, [[ms, lat, lon, fc]], last hPa
    for t, pts, fc in snaps:
        used = set()
        for tr in tracks:
            if tr['end'] != 'open': continue
            best = None
            for i, p in enumerate(pts):
                if i in used or p[0] != tr['kind']: continue
                d = km(tr['track'][-1][1:3], p[1:3])
                if d < 1100 and (best is None or d < best[0]): best = (d, i)
            if best is None: tr['end'] = 'closed'; continue
            p = pts[best[1]]; used.add(best[1])
            tr['track'].append([ms(t), round(p[1], 1), round(p[2], 1), fc]); tr['hpa'] = p[3] if not fc else tr['hpa']
        for i, p in enumerate(pts):
            if i not in used:
                tracks.append({'kind': p[0], 'track': [[ms(t), round(p[1], 1), round(p[2], 1), fc]], 'hpa': p[3], 'end': 'open'})
    out = []
    for tr in tracks:
        obs = [q for q in tr['track'] if not q[3]]
        if not obs: continue
        out.append({'type': 'low' if tr['kind'] == 'L' else 'high', 'name': tr['kind'], 'kind': '',
                    'detail': f"{tr['hpa']:.0f}", 'last_obs': obs[-1][0], 'track': tr['track'],
                    'from': tr['track'][0][0] - 3 * 3600_000, 'until': tr['track'][-1][0] + 3 * 3600_000})
    print(f"pressure systems: {sum(1 for o in out if o['type'] == 'low')} lows, {sum(1 for o in out if o['type'] == 'high')} highs"
          f" from {len(snaps)} GFS fields")
    return out

def events(now):
    out = []
    try:
        ev = get(GDACS_EV.format(a=(now - timedelta(days=7)).strftime('%Y-%m-%d'), b=(now + timedelta(days=1)).strftime('%Y-%m-%d')))
    except Exception as e:
        print('events: GDACS unavailable', e); return out
    caps = {'VO': 3, 'WF': 3, 'EQ': 3, 'FL': 3}
    for f in sorted(ev.get('features', []), key=lambda f: -float(f['properties'].get('alertscore') or 0)):
        p = f['properties']; et = p.get('eventtype')
        if caps.get(et, 0) <= 0: continue
        try:
            t0 = datetime.fromisoformat(p['fromdate']).replace(tzinfo=timezone.utc)
            t1 = datetime.fromisoformat(p['todate']).replace(tzinfo=timezone.utc)
        except Exception: continue
        if p.get('iscurrent') != 'true' and now - t1 > timedelta(days=2): continue
        lon, lat = f['geometry']['coordinates'][:2]
        sev = (p.get('severitydata') or {}).get('severitytext', '')
        country = (p.get('country') or '').split(',')[0].strip()
        if et == 'EQ':
            m = re.search(r'(\d+(?:\.\d+))\s*M', sev)
            name, kind, until = (f"M {m.group(1)}" if m else 'Earthquake'), 'Earthquake' + (f' · {country}' if country else ''), t0 + timedelta(hours=36)
        elif et == 'VO':
            name, kind, until = re.sub(r'^\s*Eruption\s*', '', p.get('name', '')).strip() or 'Volcano', 'Eruption', t1 + timedelta(days=2)
        elif et == 'WF':
            name, kind, until = 'Wildfire', country, t1 + timedelta(days=1)
        else:
            name, kind, until = 'Flood', country, t1 + timedelta(days=1)
        caps[et] -= 1
        out.append({'type': {'EQ': 'quake', 'VO': 'volcano', 'WF': 'fire', 'FL': 'flood'}[et], 'name': name, 'kind': kind,
                    'detail': '', 'from': ms(t0), 'until': ms(until), 'last_obs': ms(t1),
                    'track': [[ms(t0), round(lat, 2), round(lon, 2), 0]]})
    print('events:', ', '.join(f"{o['type']} {o['name']}" for o in out) or 'none')
    return out

def launches(now, cache):
    """Launches from 24 h ago to 12 h ahead (Launch Library 2 allows ~15 calls an hour: cache for an hour)."""
    data = None
    if cache and os.path.exists(cache):
        try:
            c = json.load(open(cache))
            if ms(now) - c['at'] < 3600_000: data = c['results']
        except Exception: pass
    if data is None:
        data = []
        for w in ('upcoming', 'previous'):
            try: data += get(LL2.format(w=w)).get('results', [])
            except Exception as e: print('launches:', w, e)
        if cache and data: json.dump({'at': ms(now), 'results': data}, open(cache, 'w'))
    out, seen = [], set()
    for L in data:
        try:
            if L.get('id') in seen: continue
            seen.add(L.get('id'))
            t = datetime.fromisoformat(L['net'].replace('Z', '+00:00'))
            if not (now - timedelta(hours=24) <= t <= now + timedelta(hours=12)): continue
            pad = L.get('pad') or {}
            lat, lon = float(pad['latitude']), float(pad['longitude'])
            rocket = ((L.get('rocket') or {}).get('configuration') or {}).get('name') or L.get('name', '').split('|')[0].strip()
            mission = ((L.get('mission') or {}).get('name') or L.get('name', '').split('|')[-1]).strip()
            status = ((L.get('status') or {}).get('abbrev') or '')
            out.append({'type': 'launch', 'name': rocket, 'kind': 'Launch', 'detail': mission[:28], 'status': status,
                        'from': ms(t - timedelta(hours=6)), 'until': ms(t + timedelta(minutes=90)), 'last_obs': ms(t),
                        'track': [[ms(t), round(lat, 3), round(lon, 3), 0]]})
        except Exception: continue
    out = sorted(out, key=lambda o: o['last_obs'])[:6]
    print('launches:', ', '.join(f"{o['name']} ({o['detail']})" for o in out) or 'none')
    return out

def fetch_all(now=None, cache_dir=None):
    now = now or datetime.now(timezone.utc)
    d = fetch(now)
    for s in d['storms']: s.setdefault('type', 'storm')
    tcs = [s['track'][-1][1:3] for s in d['storms']]
    for f, args in ((systems, (now,)), (events, (now,)), (launches, (now, os.path.join(cache_dir, 'launches.json') if cache_dir else None))):
        try:
            add = f(*args)
            if f is systems:                                       # a tropical cyclone already has its own label
                add = [o for o in add if o['type'] != 'low' or all(km(o['track'][-1][1:3], c) > 700 for c in tcs)]
            d['storms'] += add
        except Exception as e: print('labels:', f.__name__, 'failed', e)
    return d
