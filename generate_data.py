"""
generate_data.py - reads wnba_ratings_with_standings.csv and writes JSON for the LOBO web frontend.
Run after wnba.py. Outputs to docs/data/.

Mirrors the ZIDANE site architecture, simplified for a single league.
"""

import pandas as pd
import numpy as np
import json
import os
import re
from bisect import bisect_right
from datetime import datetime, timezone


# ── WNBA conference mapping (covers all team names since 1997) ───────────────
# Per-team-name (not per-franchise lineage): Detroit Shock was Eastern when
# they existed; the Tulsa Shock / Dallas Wings successor is Western. Each
# distinct team name gets its conference for the years it existed.
TEAM_CONFERENCE = {
    # Eastern Conference
    'Atlanta Dream':            'East',
    'Charlotte Sting':          'East',  # 1997-2006
    'Chicago Sky':              'East',
    'Cleveland Rockers':        'East',  # 1997-2003
    'Connecticut Sun':          'East',  # 2003+ (was Orlando Miracle)
    'Detroit Shock':            'East',  # 1998-2009 (became Tulsa, then Dallas)
    'Indiana Fever':            'East',
    'Miami Sol':                'East',  # 2000-2002
    'New York Liberty':         'East',
    'Orlando Miracle':          'East',  # 1999-2002 (became Connecticut Sun)
    'Toronto Tempo':            'East',  # 2026 expansion
    'Washington Mystics':       'East',

    # Western Conference
    'Dallas Wings':             'West',  # 2016+ (was Tulsa Shock)
    'Golden State Valkyries':   'West',  # 2025 expansion
    'Houston Comets':           'West',  # 1997-2008
    'Las Vegas Aces':           'West',  # 2018+ (was San Antonio Silver Stars)
    'Los Angeles Sparks':       'West',
    'Minnesota Lynx':           'West',
    'Phoenix Mercury':          'West',
    'Portland Fire':            'West',  # 2000-2002
    'Sacramento Monarchs':      'West',  # 1997-2009
    'San Antonio Silver Stars': 'West',  # 2003-2013 (was Utah Starzz)
    'Seattle Storm':            'West',
    'Tulsa Shock':              'West',  # 2010-2015 (was Detroit Shock, became Dallas Wings)
    'Utah Starzz':              'West',  # 1997-2002 (became San Antonio)
}


def conf(team):
    return TEAM_CONFERENCE.get(team, 'Other')


os.makedirs('docs/data/teams', exist_ok=True)
os.makedirs('docs/data/seasons', exist_ok=True)

print("Reading ratings...")
df = pd.read_csv('wnba_ratings_with_standings.csv')
df['date'] = pd.to_datetime(df['date']).dt.date

games = pd.read_csv('all_wnba_games.csv')
games['date_game'] = pd.to_datetime(games['date_game']).dt.date


def clean(val):
    if pd.isna(val):
        return ''
    return str(val)


def slug(name):
    return re.sub(r'[^\w]', '_', name).strip('_')


# Title-odds caches populated below by the LR block. Pre-defined here so
# _od_fields can reference the lookups safely even when called before the
# title odds block has run.
_title_odds_cache = {}        # (ranking_id, team) -> float
_title_odds_rank_cache = {}   # ranking_id -> {team: rank}

def _title_odds_val(ranking_id, team):
    if ranking_id is None or team is None:
        return None
    return _title_odds_cache.get((int(ranking_id), team))

def _title_odds_rk(ranking_id, team):
    if ranking_id is None or team is None:
        return None
    rm = _title_odds_rank_cache.get(int(ranking_id))
    return rm.get(team) if rm else None


def _od_fields(r):
    """Return rating_o/rating_d/rank_o/rank_d + title_odds/title_odds_rank
    safely from a row. Returns None for missing values so downstream
    consumers render '-' rather than '0'. title_odds is a probability (0-1)
    of winning the WNBA Finals at this snapshot; null for teams already
    eliminated."""
    rid = int(r['ranking_id']) if 'ranking_id' in r and not pd.isna(r['ranking_id']) else None
    team_name = r['name'] if 'name' in r else None
    odds = _title_odds_val(rid, team_name) if rid is not None and team_name else None
    odds_rank = _title_odds_rk(rid, team_name) if rid is not None and team_name else None
    return {
        'rating_o': round(float(r['rating_o']), 3) if 'rating_o' in r and not pd.isna(r['rating_o']) else None,
        'rating_d': round(float(r['rating_d']), 3) if 'rating_d' in r and not pd.isna(r['rating_d']) else None,
        'rank_o':   int(r['rank_o']) if 'rank_o' in r and not pd.isna(r['rank_o']) else None,
        'rank_d':   int(r['rank_d']) if 'rank_d' in r and not pd.isna(r['rank_d']) else None,
        'title_odds':      round(float(odds), 4) if odds is not None else None,
        'title_odds_rank': int(odds_rank) if odds_rank is not None else None,
    }


def _played(result):
    """True iff this row represents an actual game played. Upstream now
    writes empty strings for non-game-days (was 'No Game' previously) -
    both must be treated as "didn't play" or the forward-fill of last_match
    breaks for any snapshot date a team didn't play on."""
    if result is None or pd.isna(result):
        return False
    s = str(result).strip()
    return s not in ('', 'No Game')


# is_game_day: any row where the team actually played that snapshot date
df['is_game_day'] = df['last_game_result'].apply(_played).astype(int)
# is_end_of_season: collapse season_flag (1=last regular, 2=last postseason) to one boolean
df['is_end_of_season'] = df['season_flag'].isin([1, 2]).astype(int)

# Per-(team, season) forward-filled last game. Keying by season prevents
# cross-season carry-forward - at the start of a new season, teams that
# haven't played yet correctly show empty rather than their previous-season
# Finals result.
_last_game_history = {}
for (team, season), tdf in df[df['is_game_day'] == 1].sort_values('date').groupby(['name', 'season']):
    _last_game_history[(team, int(season))] = (
        [str(d) for d in tdf['date'].tolist()],
        tdf['last_game_result'].tolist(),
    )


def last_game_as_of(team, snap_date_str, season):
    """Most recent game result for team as of snap_date_str within `season`.
    Returns '' if the team hasn't played any games yet that season."""
    entry = _last_game_history.get((team, int(season)))
    if not entry:
        return ''
    dates, games_list = entry
    idx = bisect_right(dates, snap_date_str) - 1
    return games_list[idx] if idx >= 0 else ''


def last_game_date_as_of(team, snap_date_str, season):
    """Date string of the team's most recent game in `season` as of snap_date_str."""
    entry = _last_game_history.get((team, int(season)))
    if not entry:
        return ''
    dates, _ = entry
    idx = bisect_right(dates, snap_date_str) - 1
    return dates[idx] if idx >= 0 else ''


# Per-season last regular-season date - used to flag playoff vs regular-season entries
_rs_end_dates = (
    df[df['season_flag'] == 1]
    .groupby('season')['date']
    .max()
    .to_dict()
)


def is_playoff(season, date_val):
    rs_end = _rs_end_dates.get(season)
    if rs_end is None:
        return False
    return date_val > rs_end


# =========================================================
# TITLE ODDS (logistic regression, leave-one-season-out)
# =========================================================
# Continuous-progress model mirroring DUNCAN + GRIFFEY. WNBA-specific:
#   - RS game count varies a lot (28/30/32/34/36/40/44 depending on era +
#     2020 COVID 22).
#   - Playoff structure varies widely (1997 single-elim 2-round, 1998-2002
#     BO3 3-round, 2003-2015 BO3/BO5/BO5 expanding, 2016-2021 4-round with
#     double-bye for top-2 seeds, 2022+ BO3/BO5/BO5).
#   - No gametype column on the games CSV - PS games detected via date > rs_end.
# Structural round detection: walk back from Finals to assign per-series
# round numbers. Handles the 2016-21 double-bye era naturally via per-team
# entry depth.

print("\nComputing title odds (logistic regression, leave-one-season-out)...")
from scipy.optimize import minimize

TITLE_TRAIN_FROM_SEASON = 1998  # Skip 1997 unique single-elim 2-round format

PHASE_RS_MAX_TO        = 0.50
PHASE_POST_RS_TO       = 0.55
PHASE_FINALS_ENTRY_TO  = 0.95
PHASE_CHAMPION_TO      = 1.00

# RS game thresholds per season (mode of per-team games).
_TO_RS_GAMES = {
    1997: 28, 1998: 30,
    1999: 32, 2000: 32, 2001: 32, 2002: 32,
    2003: 34, 2004: 34, 2005: 34, 2006: 34, 2007: 34, 2008: 34,
    2009: 34, 2010: 34, 2011: 34, 2012: 34, 2013: 34, 2014: 34,
    2015: 34, 2016: 34, 2017: 34, 2018: 34, 2019: 34,
    2020: 22,  # COVID Wubble
    2021: 32, 2022: 36, 2023: 40, 2024: 40, 2025: 44,
}
def _to_rs_games(season):
    return _TO_RS_GAMES.get(int(season), 44)

# Era-aware clinch threshold for IN-PROGRESS series (for completed series we
# derive from data directly). round_pos is 1-indexed from earliest round in
# the team's bracket; round_total is the total rounds in this team's bracket.
def _to_clinch_threshold(season, round_pos, round_total):
    s = int(season)
    # 1997: single-elim 2-round format, every round BO1.
    if s == 1997:
        return 1
    is_finals = (round_pos == round_total)
    is_semis  = (round_pos == round_total - 1)
    # WNBA finals BO5 since 2005, briefly BO5 2000-01 and 2003-04, BO3 otherwise
    if is_finals:
        if s >= 2005: return 3
        if s in (2000, 2001, 2003, 2004): return 3
        return 2
    if is_semis:
        if s >= 2005: return 3  # BO5 semis 2005+
        return 2  # BO3 pre-2005
    # Earlier rounds
    if 2016 <= s <= 2021: return 1  # BO1 single-elim R1+R2
    return 2  # BO3 first round otherwise

def _to_pad_to_bo7(w, l, clinch):
    pad = 4 - (clinch if clinch is not None else 4)
    return w + pad, l + pad

def _to_progress(round_pos, round_total, is_champion=False):
    if is_champion:
        return PHASE_CHAMPION_TO
    if round_total <= 1:
        return PHASE_POST_RS_TO
    return PHASE_POST_RS_TO + (PHASE_FINALS_ENTRY_TO - PHASE_POST_RS_TO) * \
        (round_pos - 1) / (round_total - 1)


# Walk PS games per season, build per-team bracket path. games['date_game']
# is already a python date (LOBO loader uses .dt.date) - keep date type
# throughout the walker to match snap_date in _to_snap_state.
_to_team_path = {}            # (season, team) -> [series dict, ordered by start]
_to_champion = {}             # season -> champion team
_to_field = {}                # season -> set of teams in PS
_to_season_total_rounds = {}  # season -> int total rounds in this season's bracket

for season, rs_end in _rs_end_dates.items():
    s_int = int(season)
    # Build the bracket walker for EVERY season so 1997 (unique single-elim
    # 2-round format) also gets predicted - just won't contribute to LR
    # training. TITLE_TRAIN_FROM_SEASON only gates training-set inclusion.
    ps = games[(games['season'] == s_int) & (games['date_game'] > rs_end)].copy()
    if ps.empty:
        continue

    # Group games into series by sorted team pair.
    pair_bucket = {}
    for _, g in ps.iterrows():
        pair = tuple(sorted([g['home_team_name'], g['visitor_team_name']]))
        pair_bucket.setdefault(pair, []).append(g)

    series_list = []
    for pair, gs in pair_bucket.items():
        gs.sort(key=lambda x: x['date_game'])
        a, b = pair
        a_wins = sum(((g['home_team_name'] == a) and (g['home_pts'] > g['visitor_pts'])) or
                     ((g['visitor_team_name'] == a) and (g['visitor_pts'] > g['home_pts']))
                     for g in gs)
        b_wins = len(gs) - a_wins
        winner = a if a_wins > b_wins else b
        loser  = b if winner == a else a
        # Per-game state for both teams.
        state_a, state_b = [], []
        aw = bw = 0
        for g in gs:
            a_won = (g['home_team_name'] == a and g['home_pts'] > g['visitor_pts']) or \
                    (g['visitor_team_name'] == a and g['visitor_pts'] > g['home_pts'])
            if a_won: aw += 1
            else:     bw += 1
            state_a.append((g['date_game'], aw, bw))
            state_b.append((g['date_game'], bw, aw))
        series_list.append({
            'pair':      pair,
            'a':         a,
            'b':         b,
            'winner':    winner,
            'loser':     loser,
            'a_wins':    a_wins,
            'b_wins':    b_wins,
            'start':     gs[0]['date_game'],
            'end':       gs[-1]['date_game'],
            'games':     gs,
            'state_by_team': {a: state_a, b: state_b},
        })
    series_list.sort(key=lambda s: s['end'])

    # Structural bracket walk: starting from Finals (last series by end_date),
    # walk back via each series's participants' immediately-preceding series.
    # depth 0 = Finals, depth 1 = semis, etc. round_pos = total_rounds - depth.
    # Handles 2016-2021 double-bye era naturally - bye'd teams just don't have
    # earlier-round series in their chain.
    finals = series_list[-1]
    series_depth = {id(finals): 0}
    queue = [finals]
    while queue:
        s = queue.pop(0)
        d = series_depth[id(s)]
        for team in s['pair']:
            preceding = [x for x in series_list
                         if team in x['pair'] and x['end'] < s['start']]
            if not preceding:
                continue
            preceding.sort(key=lambda x: x['end'])
            prev = preceding[-1]
            if id(prev) not in series_depth:
                series_depth[id(prev)] = d + 1
                queue.append(prev)

    max_depth = max(series_depth.values()) if series_depth else 0
    total_rounds = max_depth + 1
    _to_season_total_rounds[s_int] = total_rounds

    # Champion = finals winner; field = all teams that played in PS.
    _to_champion[s_int] = finals['winner']
    field = set()
    for s in series_list:
        field.update(s['pair'])
        # Tag round_pos for each series (1 = first round in bracket, N = Finals).
        s['round_pos'] = total_rounds - series_depth[id(s)]
        # Clinch threshold is era-aware - derived from (round_pos, season).
        # Always era-driven (not max-of-winner_wins) so the in-progress logic
        # works the same way for closed and current seasons; for 1997 BO1
        # this correctly yields clinch=1 throughout.
        s['clinch'] = _to_clinch_threshold(s_int, s['round_pos'], total_rounds)
    _to_field[s_int] = field

    # Per-team paths sorted by start.
    for team in field:
        team_series = [s for s in series_list if team in s['pair']]
        team_series.sort(key=lambda x: x['start'])
        _to_team_path[(s_int, team)] = team_series


def _to_games_played(s_int, team, snap_date):
    """Count RS games team played by snap_date (date object)."""
    rs_end = _rs_end_dates.get(s_int)
    if rs_end is None:
        return 0
    sub = games[(games['season'] == s_int)
                & ((games['home_team_name'] == team) | (games['visitor_team_name'] == team))
                & (games['date_game'] <= snap_date)
                & (games['date_game'] <= rs_end)]
    return len(sub)


def _to_snap_state(s_int, team, snap_date):
    """Return (in_field, is_eliminated, progress, series_w_pad, series_l_pad).
    Uses dynamic decided-as-of detection (same as the GRIFFEY clinch-day fix)
    so on a series-clinching day the team is advanced to next-round state."""
    rs_end = _rs_end_dates.get(s_int)
    if rs_end is None or snap_date <= rs_end:
        gp = _to_games_played(s_int, team, snap_date)
        progress = PHASE_RS_MAX_TO * min(gp / _to_rs_games(s_int), 1.0)
        return (True, False, progress, 0, 0)

    in_field = (s_int in _to_field) and (team in _to_field[s_int])
    if not in_field:
        return (False, False, None, 0, 0)

    total_rounds = _to_season_total_rounds.get(s_int, 3)
    path = _to_team_path[(s_int, team)]
    current = None
    last_won = None
    for s in path:
        if s['start'] > snap_date:
            break
        team_state = s['state_by_team'][team]
        w_at = l_at = 0
        for d, sw, sl in team_state:
            if d <= snap_date:
                w_at, l_at = sw, sl
            else:
                break
        clinch = s['clinch']
        decided = max(w_at, l_at) >= clinch
        team_won = decided and w_at > l_at
        team_lost = decided and l_at > w_at
        if team_lost:
            return (True, True, None, 0, 0)
        if team_won:
            last_won = s
            continue
        current = s
        break

    if current is not None:
        team_state = current['state_by_team'][team]
        w = l = 0
        for d, sw, sl in team_state:
            if d <= snap_date:
                w, l = sw, sl
            else:
                break
        progress = _to_progress(current['round_pos'], total_rounds)
        w_pad, l_pad = _to_pad_to_bo7(w, l, current['clinch'])
        return (True, False, progress, w_pad, l_pad)

    if last_won is not None:
        next_round = last_won['round_pos'] + 1
        if next_round > total_rounds:
            return (True, False, PHASE_CHAMPION_TO, 0, 0)
        return (True, False, _to_progress(next_round, total_rounds), 0, 0)

    return (True, False, PHASE_POST_RS_TO, 0, 0)


# Build feature rows for EVERY season (including 1997). Training set will
# be restricted later to TITLE_TRAIN_FROM_SEASON+; 1997 will be scored via
# the full-trained model the same way in-progress current seasons are.
_to_rows = []
rated = df[df['rating_o'].notna() & df['rating_d'].notna()].copy()
for _, r in rated.iterrows():
    s_int = int(r['season'])
    team = r['name']
    sd = r['date']
    in_field, is_elim, progress, sw, sl = _to_snap_state(s_int, team, sd)
    if is_elim or progress is None:
        continue
    _to_rows.append({
        'season':      s_int,
        'team':        team,
        'ranking_id':  int(r['ranking_id']),
        'rating':      float(r['rating']),
        'rating_o':    float(r['rating_o']),
        'rating_d':    float(r['rating_d']),
        'progress':    float(progress),
        'series_w':    int(sw),
        'series_l':    int(sl),
        'is_champion': 1 if _to_champion.get(s_int) == team else 0,
    })

_to_train_df = pd.DataFrame(_to_rows)
_train_pool = _to_train_df[_to_train_df['season'] >= TITLE_TRAIN_FROM_SEASON]
print(f"  Title-odds rows: {len(_to_train_df):,} "
      f"({int(_to_train_df['is_champion'].sum())} champion-positive) - "
      f"training pool {len(_train_pool):,}")


def _to_features(d):
    p = d['progress'].values
    return np.column_stack([
        d['rating'].values, d['rating_o'].values, d['rating_d'].values, p,
        d['rating'].values * p,
        d['rating_o'].values * p,
        d['rating_d'].values * p,
        d['series_w'].values,
        d['series_l'].values,
    ])


def _to_fit_logistic(X, y, reg=1e-3):
    n, k = X.shape
    Xa = np.column_stack([np.ones(n), X])
    def nll(beta):
        z = Xa @ beta
        return float(np.sum(np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z))) - y * z)
                     + reg * np.sum(beta[1:] ** 2))
    def grad(beta):
        z = Xa @ beta
        p_hat = 1.0 / (1.0 + np.exp(-z))
        g = Xa.T @ (p_hat - y)
        g[1:] += 2 * reg * beta[1:]
        return g
    res = minimize(nll, np.zeros(k + 1), jac=grad, method='BFGS',
                   options={'maxiter': 200, 'gtol': 1e-6})
    return res.x


def _to_predict_logistic(X, beta):
    Xa = np.column_stack([np.ones(X.shape[0]), X])
    z = Xa @ beta
    return 1.0 / (1.0 + np.exp(-z))


# LOO across completed seasons in the training pool (1998+). The held-out
# season is scored using a model trained on every OTHER pool season.
_completed_seasons = {s for s in _train_pool['season'].unique() if s in _to_champion}
for s_int in _completed_seasons:
    train = _train_pool[_train_pool['season'] != s_int]
    held  = _train_pool[_train_pool['season'] == s_int]
    if train.empty or held.empty:
        continue
    beta = _to_fit_logistic(_to_features(train), train['is_champion'].values.astype(float))
    held = held.copy()
    held['p_raw'] = _to_predict_logistic(_to_features(held), beta)
    held['p_norm'] = held.groupby('ranking_id')['p_raw'].transform(
        lambda x: x / x.sum() if x.sum() > 0 else 0.0)
    for _, r in held.iterrows():
        _title_odds_cache[(int(r['ranking_id']), r['team'])] = float(r['p_norm'])

# Everything outside the LOO loop (in-progress current season + 1997, which
# is excluded from training due to its unique single-elim format) gets
# scored using a model trained on the full 1998+ pool.
_remaining = _to_train_df[~_to_train_df['season'].isin(_completed_seasons)]
if not _remaining.empty and not _train_pool.empty:
    beta_full = _to_fit_logistic(_to_features(_train_pool),
                                 _train_pool['is_champion'].values.astype(float))
    cur = _remaining.copy()
    cur['p_raw'] = _to_predict_logistic(_to_features(cur), beta_full)
    cur['p_norm'] = cur.groupby('ranking_id')['p_raw'].transform(
        lambda x: x / x.sum() if x.sum() > 0 else 0.0)
    for _, r in cur.iterrows():
        _title_odds_cache[(int(r['ranking_id']), r['team'])] = float(r['p_norm'])

# Per-snapshot rank cache.
_pairs_by_rid = {}
for (rid, team), odds in _title_odds_cache.items():
    if odds is None or odds <= 0:
        continue
    _pairs_by_rid.setdefault(rid, []).append((team, odds))
for rid, pairs in _pairs_by_rid.items():
    pairs.sort(key=lambda x: -x[1])
    rank_map = {}
    prev_odds = None
    prev_rank = 0
    for i, (team, odds) in enumerate(pairs, start=1):
        if odds != prev_odds:
            prev_rank = i
            prev_odds = odds
        rank_map[team] = prev_rank
    _title_odds_rank_cache[rid] = rank_map

print(f"  Title odds cached for {len(_title_odds_cache):,} (snapshot, team) pairs "
      f"across {len(_title_odds_rank_cache):,} snapshots.")


# Regular-season-end record per (team, season): from season_flag == 1 snapshot
_reg_record_lookup = {
    (row['name'], int(row['season'])): row['record']
    for _, row in df[df['season_flag'] == 1].iterrows()
}

# End-of-playoffs combined record per (team, season): from season_flag == 2.
# Lets GOAT rows show the team's eventual playoff record regardless of which
# snapshot the row itself comes from (RS-end snapshots wouldn't know it yet).
_full_record_lookup = {
    (row['name'], int(row['season'])): row['record']
    for _, row in df[df['season_flag'] == 2].iterrows()
}


def _parse_record(rec):
    """Parse a 'W - L' string into (wins, losses). Returns None if unparseable."""
    if not rec or pd.isna(rec):
        return None
    m = re.match(r'(\d+)\s*-\s*(\d+)', str(rec))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def playoff_record(full_record, regular_record):
    """Compute playoff record as (full - regular). Both inputs are 'W - L' strings."""
    f = _parse_record(full_record)
    r = _parse_record(regular_record)
    if not f or not r:
        return ''
    pw, pl = f[0] - r[0], f[1] - r[1]
    if pw < 0 or pl < 0:
        return ''
    return f"{pw}-{pl}"


# ── 1. Current standings (latest snapshot) ───────────────────────────────────
print("Writing current_standings.json...")
latest_id = int(df['ranking_id'].max())
latest = df[df['ranking_id'] == latest_id].sort_values('rank').copy()
latest_date = str(latest['date'].iloc[0])

standings_data = {
    'updated': latest_date,
    'teams': [
        {
            'rank':            int(r['rank']),
            'team':            r['name'],
            'conference':      conf(r['name']),
            'rating':          round(float(r['rating']), 3),
            **_od_fields(r),
            'record':          clean(r['record']),
            'last_match':      clean(r['last_game_result']) if _played(r['last_game_result']) else last_game_as_of(r['name'], str(r['date']), r['season']),
            'finals_status':   int(r['finals_status']) if not pd.isna(r['finals_status']) else 0,
            'cup_status':      int(r['cup_status']) if 'cup_status' in r and not pd.isna(r['cup_status']) else 0,
        }
        for _, r in latest.iterrows()
    ],
}
with open('docs/data/current_standings.json', 'w') as f:
    json.dump(standings_data, f, separators=(',', ':'))

# ── 2. GOAT tables (end-of-RS + end-of-playoffs) ─────────────────────────────
# Two lists, matching the DUNCAN/SAKIC/GRIFFEY fleet pattern:
#   goat_rs.json - top 50 single-season ratings at end of regular season, all teams.
#   goat_ps.json - top 50 single-season ratings at end of playoffs, WNBA Finals participants only.
# Both gated to fully-complete seasons (a season is "complete" once a
# season_flag == 2 row exists for that season - i.e. the Finals have ended).
print("Writing goat_rs.json + goat_ps.json...")

# Short / disrupted seasons - flagged on GOAT rows so the UI can tag them
# inline. WNBA regular-season length has grown over time (28g → 44g) as the
# league has expanded, so most early-era short totals are "normal for the
# era", not disrupted. Only seasons with abnormal mid-stream disruption
# are tagged here.
SHORT_SEASONS = {
    2020: {
        'tag': 'COVID Wubble',
        'category': 'covid',
        'note': "The 2020 season was shortened to 22 games and played in a single-site bubble at IMG Academy in Bradenton, FL.",
    },
}

completed_seasons = set(df.loc[df['season_flag'] == 2, 'season'].astype(int).unique())


def build_goat(flag, require_finalist, sort_col='rating'):
    rows = df[(df['season_flag'] == flag) &
              (df['season'].astype(int).isin(completed_seasons))].copy()
    if require_finalist:
        rows = rows[rows['finals_status'].fillna(0) >= 1]
    rows = rows[rows[sort_col].notna()]
    rows = rows.sort_values(sort_col, ascending=False).head(50).reset_index(drop=True)
    out = []
    for i, (_, r) in enumerate(rows.iterrows()):
        s = int(r['season'])
        reg = _reg_record_lookup.get((r['name'], s), '')
        full = _full_record_lookup.get((r['name'], s), '')
        out.append({
            'rank':             i + 1,
            'team':             r['name'],
            'conference':       conf(r['name']),
            'season':           s,
            'short_season':          s in SHORT_SEASONS,
            'short_season_tag':      SHORT_SEASONS.get(s, {}).get('tag', '')      if s in SHORT_SEASONS else '',
            'short_season_category': SHORT_SEASONS.get(s, {}).get('category', '') if s in SHORT_SEASONS else '',
            'short_season_note':     SHORT_SEASONS.get(s, {}).get('note', '')     if s in SHORT_SEASONS else '',
            'rating':           round(float(r['rating']), 3),
            **_od_fields(r),
            'record':           clean(full or r['record']),
            'regular_record':   reg,
            'playoff_record':   playoff_record(full, reg) if full else '',
            'finals_status':    int(r['finals_status']) if not pd.isna(r['finals_status']) else 0,
            'cup_status':       int(r['cup_status']) if 'cup_status' in r and not pd.isna(r['cup_status']) else 0,
        })
    return out


# Six GOAT files: {Rating, Offense, Defense} × {RS-end, PS-end}. Mirrors DUNCAN.
goat_files = [
    ('goat_rs.json',   1, False, 'rating'),
    ('goat_ps.json',   2, True,  'rating'),
    ('goat_rs_o.json', 1, False, 'rating_o'),
    ('goat_rs_d.json', 1, False, 'rating_d'),
    ('goat_ps_o.json', 2, True,  'rating_o'),
    ('goat_ps_d.json', 2, True,  'rating_d'),
]
for fname, flag, require_finalist, sort_col in goat_files:
    payload = build_goat(flag=flag, require_finalist=require_finalist, sort_col=sort_col)
    with open(f'docs/data/{fname}', 'w') as f:
        json.dump(payload, f, separators=(',', ':'))

# ── 3. Per-team JSON files ───────────────────────────────────────────────────
print("Writing per-team JSON files...")
team_data = df[(df['is_game_day'] == 1) | (df['is_end_of_season'] == 1)].copy()
team_data = team_data.sort_values(['name', 'season', 'date'])

all_teams = sorted(df['name'].unique())
teams_index = []

for team in all_teams:
    tdf = team_data[team_data['name'] == team]
    if len(tdf) == 0:
        continue

    team_slug = slug(team)
    teams_index.append({'name': team, 'conference': conf(team), 'slug': team_slug})

    seasons = {}
    for season, sdf in tdf.groupby('season'):
        rs_end = _rs_end_dates.get(season)
        final_reg = _reg_record_lookup.get((team, int(season)))
        entries = []
        for _, r in sdf.sort_values('date').iterrows():
            in_postseason = (rs_end is not None) and (r['date'] > rs_end) and (final_reg is not None)
            if in_postseason:
                reg = final_reg
                po  = playoff_record(r['record'], final_reg)
            else:
                reg = clean(r['record'])
                po  = ''
            entries.append({
                'date':              str(r['date']),
                'rating':            round(float(r['rating']), 3),
                'rank':              int(r['rank']),
                **_od_fields(r),
                'record':            clean(r['record']),
                'regular_record':    reg,
                'playoff_record':    po,
                'last_match':        clean(r['last_game_result']) if _played(r['last_game_result']) else last_game_as_of(team, str(r['date']), season),
                'is_end_of_season':  int(r['is_end_of_season']),
                'season_flag':       int(r['season_flag']),
                'is_playoff':        int(is_playoff(season, r['date'])),
                'finals_status':     int(r['finals_status']) if not pd.isna(r['finals_status']) else 0,
                'cup_status':        int(r['cup_status']) if 'cup_status' in r and not pd.isna(r['cup_status']) else 0,
            })
        seasons[int(season)] = entries

    with open(f'docs/data/teams/{team_slug}.json', 'w') as f:
        json.dump({'team': team, 'conference': conf(team), 'seasons': seasons}, f, separators=(',', ':'))

teams_index.sort(key=lambda x: x['name'])
with open('docs/data/teams_index.json', 'w') as f:
    json.dump(teams_index, f, separators=(',', ':'))

# ── 4. Season standings files ─────────────────────────────────────────────────
print("Writing season standings files...")
all_seasons = sorted(df['season'].unique())

for season in all_seasons:
    sdf = df[df['season'] == season]
    snapshots = []
    for ranking_id, rdf in sdf.groupby('ranking_id'):
        rdf = rdf.sort_values('rank')
        snap_date = str(rdf['date'].iloc[0])
        flag = int(rdf['season_flag'].iloc[0])
        label = None
        if flag == 1:
            label = 'End of regular season'
        elif flag == 2:
            label = 'End of playoffs'

        snap_date_obj = rdf['date'].iloc[0]
        rs_end = _rs_end_dates.get(season)
        in_postseason = (rs_end is not None) and (snap_date_obj > rs_end)

        teams_snap = []
        for _, r in rdf.iterrows():
            if in_postseason:
                reg = _reg_record_lookup.get((r['name'], int(season)), r['record'])
                po  = playoff_record(r['record'], reg)
            else:
                reg = clean(r['record'])
                po  = ''
            played_today = _played(r['last_game_result'])
            teams_snap.append({
                'rank':            int(r['rank']),
                'team':            r['name'],
                'conference':      conf(r['name']),
                'rating':          round(float(r['rating']), 3),
                **_od_fields(r),
                'record':          clean(r['record']),
                'regular_record':  reg,
                'playoff_record':  po,
                'last_match':      clean(r['last_game_result']) if played_today else last_game_as_of(r['name'], snap_date, season),
                'last_match_date': snap_date if played_today else last_game_date_as_of(r['name'], snap_date, season),
                'finals_status':   int(r['finals_status']) if not pd.isna(r['finals_status']) else 0,
                'cup_status':      int(r['cup_status']) if 'cup_status' in r and not pd.isna(r['cup_status']) else 0,
            })
        snapshots.append({'date': snap_date, 'label': label, 'teams': teams_snap})

    snapshots.sort(key=lambda x: x['date'])
    with open(f'docs/data/seasons/{int(season)}.json', 'w') as f:
        json.dump({'season': int(season), 'snapshots': snapshots}, f, separators=(',', ':'))

seasons_meta = {
    'seasons':    [int(s) for s in reversed(all_seasons)],
    'first_date': str(games['date_game'].min()),  # actual first game (not first rated date)
    'last_date':  str(games['date_game'].max()),
    'generated_at': datetime.now(timezone.utc).isoformat(),
    'disrupted_seasons': {
        str(year): {'tag': info['tag'], 'category': info['category'], 'note': info['note']}
        for year, info in SHORT_SEASONS.items()
    },
}
with open('docs/data/seasons_index.json', 'w') as f:
    json.dump(seasons_meta, f, separators=(',', ':'))

# ── 5. Champions table ────────────────────────────────────────────────────────
print("Writing champions.json...")

# Pre-Finals snapshot per season: the ranking_id of the last rating snapshot
# STRICTLY BEFORE the Finals series begins, for each season. Used in the
# Lists sub-view to evaluate matchup quality / closeness / upsets without
# the circularity of letting the Finals result colour the "going-in" rating.
# Mirrors DUNCAN's pattern.
def _build_pre_finals_lookup():
    out = {}
    for season in df['season'].unique():
        season_df = df[df['season'] == season]
        champ_names = season_df[season_df['champ'] == 1]['name'].unique()
        ru_names    = season_df[season_df['runnerup'] == 1]['name'].unique()
        if len(champ_names) == 0 or len(ru_names) == 0:
            continue
        champ, ru = champ_names[0], ru_names[0]
        rs_end = _rs_end_dates.get(season)
        season_games = games[games['season'] == season]
        if rs_end is not None:
            season_games = season_games[season_games['date_game'] > rs_end]
        # Finals = playoff games where these two specific teams meet. WNBA's
        # bracket structure means the champion and runner-up only ever meet
        # in the Finals (different bracket sides), so all post-RS head-to-
        # head games are the Finals.
        finals = season_games[
            ((season_games['home_team_name'] == champ) & (season_games['visitor_team_name'] == ru)) |
            ((season_games['home_team_name'] == ru) & (season_games['visitor_team_name'] == champ))
        ]
        if finals.empty:
            continue
        finals_g1_date = finals['date_game'].min()
        # Latest ranking_id with date strictly before Finals Game 1.
        pre = season_df[season_df['date'] < finals_g1_date]
        if pre.empty:
            continue
        pre_id = int(pre['ranking_id'].max())
        snap = season_df[season_df['ranking_id'] == pre_id]
        for _, r in snap.iterrows():
            out[(r['name'], int(season))] = {
                'rating': round(float(r['rating']), 3),
                'rank':   int(r['rank']),
                'record': clean(r['record']),
                **_od_fields(r),
            }
    return out

_pre_finals_lookup = _build_pre_finals_lookup()
print(f"  pre-Finals snapshots computed for {len(set(s for (_, s) in _pre_finals_lookup))} seasons")


def pre_finals_fields(name, season, reg_record):
    """Return the pre-Finals rating/rank/playoff_record block, or empty if missing."""
    p = _pre_finals_lookup.get((name, int(season)))
    if p is None:
        return {}
    return {
        'rating_pre':         p['rating'],
        'rank_pre':           p['rank'],
        'rating_o_pre':       p.get('rating_o'),
        'rating_d_pre':       p.get('rating_d'),
        'rank_o_pre':         p.get('rank_o'),
        'rank_d_pre':         p.get('rank_d'),
        'playoff_record_pre': playoff_record(p['record'], reg_record),
    }


champions = []
for season in sorted(df['season'].unique(), reverse=True):
    sdf = df[(df['season'] == season) & (df['season_flag'] == 2)]
    if sdf.empty:
        continue
    champ_row = sdf[sdf['champ'] == 1]
    ru_row = sdf[sdf['runnerup'] == 1]
    if champ_row.empty or ru_row.empty:
        continue

    cr = champ_row.iloc[0]
    rr = ru_row.iloc[0]

    # Final score and Finals series score from games CSV
    season_games = games[games['season'] == season]
    final_score = ''
    series_score = ''
    if not season_games.empty:
        last_game = season_games.sort_values('date_game').iloc[-1]
        if last_game['home_team_name'] == cr['name']:
            final_score = f"{int(last_game['home_pts'])}-{int(last_game['visitor_pts'])}"
        elif last_game['visitor_team_name'] == cr['name']:
            final_score = f"{int(last_game['visitor_pts'])}-{int(last_game['home_pts'])}"

        # Series score: count champion vs runner-up wins in the postseason.
        # In WNBA they only meet in the Finals (different bracket sides), so all
        # head-to-head postseason games ARE the Finals.
        rs_end = _rs_end_dates.get(season)
        playoff_games = season_games[season_games['date_game'] > rs_end] if rs_end is not None else season_games
        finals = playoff_games[
            ((playoff_games['home_team_name'] == cr['name']) & (playoff_games['visitor_team_name'] == rr['name'])) |
            ((playoff_games['home_team_name'] == rr['name']) & (playoff_games['visitor_team_name'] == cr['name']))
        ]
        cw, rw = 0, 0
        for _, g in finals.iterrows():
            home_won = g['home_pts'] > g['visitor_pts']
            champ_was_home = g['home_team_name'] == cr['name']
            if home_won == champ_was_home:
                cw += 1
            else:
                rw += 1
        if cw + rw > 0:
            series_score = f"{cw}-{rw}"

    champ_reg = _reg_record_lookup.get((cr['name'], int(season)), '')
    ru_reg    = _reg_record_lookup.get((rr['name'], int(season)), '')

    champions.append({
        'season':       int(season),
        'final_score':  final_score,
        'series_score': series_score,
        'champion': {
            'team':           cr['name'],
            'rating':         round(float(cr['rating']), 3),
            'rank':           int(cr['rank']),
            **_od_fields(cr),
            'record':         clean(cr['record']),
            'regular_record': champ_reg,
            'playoff_record': playoff_record(cr['record'], champ_reg),
            'cup_status':     int(cr['cup_status']) if 'cup_status' in cr and not pd.isna(cr['cup_status']) else 0,
            **pre_finals_fields(cr['name'], season, champ_reg),
        },
        'runner_up': {
            'team':           rr['name'],
            'rating':         round(float(rr['rating']), 3),
            'rank':           int(rr['rank']),
            **_od_fields(rr),
            'record':         clean(rr['record']),
            'regular_record': ru_reg,
            'playoff_record': playoff_record(rr['record'], ru_reg),
            'cup_status':     int(rr['cup_status']) if 'cup_status' in rr and not pd.isna(rr['cup_status']) else 0,
            **pre_finals_fields(rr['name'], season, ru_reg),
        },
    })

# Running counts: walk chronologically (oldest first) so each entry
# records "this is your Nth title / runner-up appearance".
_champ_count = {}
_ru_count = {}
for entry in reversed(champions):
    ct = entry['champion']['team']
    rt = entry['runner_up']['team']
    _champ_count[ct] = _champ_count.get(ct, 0) + 1
    _ru_count[rt]    = _ru_count.get(rt, 0) + 1
    entry['champion']['title_count']     = _champ_count[ct]
    entry['runner_up']['runner_up_count'] = _ru_count[rt]

with open('docs/data/champions.json', 'w') as f:
    json.dump({'WNBA': champions}, f, separators=(',', ':'))

print(f"Done. {len(teams_index)} teams, {len(standings_data['teams'])} in current standings.")
print(f"Wrote {len(all_seasons)} season files. Standings date: {latest_date}")

# Hygiene: flag any rated team missing from TEAM_CONFERENCE. Without this,
# expansion teams (or future renames) silently fall through to 'Other' and
# disappear from the conference filter pillbox.
_unknown = sorted({t for t in df['name'].unique() if t not in TEAM_CONFERENCE})
if _unknown:
    print()
    print('⚠️  WARNING: teams in rated data missing from TEAM_CONFERENCE:')
    for t in _unknown:
        print(f'    - {t!r}')
    print('    These teams will display as "Other" until added.')
    print()
