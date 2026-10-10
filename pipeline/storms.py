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
