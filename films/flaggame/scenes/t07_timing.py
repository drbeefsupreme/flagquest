"""t07 timing: the 100 raisings (count curve), the hero cuts, and the sound exports.  Owner: T07Raised.
All times GLOBAL seconds (locked cues: raise_1 143.944 'RAISED', raise_100 146.769 'time')."""
import math

R1 = 143.944          # raise_1  ('RAISED')
R50 = 145.14          # '50'  -> the page of 50
THEN = 145.86         # 'then' -> counting resumes
R100 = 146.769        # raise_100 ('time')
A1, A2 = 5.0, 3.0     # exponential acceleration of the two runs


def count(T):
    """number of raisings so far (continuous, for the rolling counter)"""
    if T < R1:
        return 0.0
    if T < R50:
        u = (T - R1) / (R50 - R1)
        return 1 + 49 * (math.exp(A1 * u) - 1) / (math.exp(A1) - 1)
    if T < THEN:
        return 50.0
    if T < R100:
        u = (T - THEN) / (R100 - THEN)
        return 50 + 50 * (math.exp(A2 * u) - 1) / (math.exp(A2) - 1)
    return 100.0


def raise_time(k):
    """global time of raising k (1..100)"""
    if k <= 1:
        return R1
    if k <= 50:
        u = math.log(1 + (k - 1) / 49 * (math.exp(A1) - 1)) / A1
        return R1 + u * (R50 - R1)
    u = math.log(1 + (k - 50) / 50 * (math.exp(A2) - 1)) / A2
    return THEN + u * (R100 - THEN)


# hero cuts (full frame): (global start, style). The raising shown is int(count(start)).
CUTS = [
    (R1, "soviet"),        # 1   'RAISED'
    (144.31, "iwo"),       # 2   'AGAIN'
    (144.50, "delacroix"),
    (144.64, "moon"),
    (144.765, "lascaux"),
    (144.87, "egypt"),
    (144.965, "bayeux"),
    (145.055, "ukiyoe"),
    (R50, "PAGE50"),       # 50  '50'  the page of fifty panels
    (THEN, "flaghack"),    # 'then'
    (146.09, "woodcut"),   # '100'
    (146.30, "seurat"),
    (146.48, "blueprint"),
    (146.62, "PAGE100"),   # the page of one hundred: the last panels pop, #100 on 'time'
]


def cut_at(T):
    cur = None
    for i, (t0, st) in enumerate(CUTS):
        if T >= t0:
            cur = (i, t0, st)
    return cur


def export(S):
    from vx.foley import export_events
    ev = []
    hero = {max(1, int(count(t0) + 1e-6)): st for t0, st in CUTS if not st.startswith("PAGE")}
    for k in range(1, 101):
        t = raise_time(k)
        st = hero.get(k)
        ev.append(dict(t=t, kind="raise", strength=1.0 if (st or k in (50, 100)) else 0.55,
                       pan=0.0, desc=f"raising #{k}" + (f" ({st} full-frame cut)" if st else "")
                       + (" - cymbal: every 10th" if k % 10 == 0 else "")))
    for t0, st in CUTS:
        if st.startswith("PAGE"):
            ev.append(dict(t=t0, kind="page", strength=1.0, pan=0.0,
                           desc="the frame becomes a comic page of %s panels" % ("50" if st == "PAGE50" else "100")))
    return ev
