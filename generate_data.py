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
# Commissioner's Cup finals feed ratings only (see wnba.py). Drop them here so
# regular-season game counts, phase progress, bracket detection, and last-match
# display all stay cup-blind, matching the standings/records built in wnba.py.
if 'is_cup_final' in games.columns:
    games = games[games['is_cup_final'] != 1].reset_index(drop=True)


def clean(val):
    if pd.isna(val):
        return ''
    return str(val)


def slug(name):
    return re.sub(r'[^\w]', '_', name).strip('_')


# Title-odds caches populated below by the Monte Carlo block. Pre-defined here so
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
        tdf['last_game_is_cup'].tolist() if 'last_game_is_cup' in tdf.columns else [0] * len(tdf),
    )


def last_game_as_of(team, snap_date_str, season):
    """Most recent game result for team as of snap_date_str within `season`.
    Returns '' if the team hasn't played any games yet that season."""
    entry = _last_game_history.get((team, int(season)))
    if not entry:
        return ''
    dates, games_list, _ = entry
    idx = bisect_right(dates, snap_date_str) - 1
    return games_list[idx] if idx >= 0 else ''


def last_game_date_as_of(team, snap_date_str, season):
    """Date string of the team's most recent game in `season` as of snap_date_str."""
    entry = _last_game_history.get((team, int(season)))
    if not entry:
        return ''
    dates, _, _ = entry
    idx = bisect_right(dates, snap_date_str) - 1
    return dates[idx] if idx >= 0 else ''


def last_game_is_cup_as_of(team, snap_date_str, season):
    """1 iff the team's most recent game as of snap_date_str is a Cup final."""
    entry = _last_game_history.get((team, int(season)))
    if not entry:
        return 0
    dates, _, cup_list = entry
    idx = bisect_right(dates, snap_date_str) - 1
    return int(cup_list[idx]) if idx >= 0 else 0


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
# TITLE ODDS (season + playoff Monte Carlo, playoff_sim.py)
# =========================================================
# Replaced the leave-one-season-out logistic model 2026-09-24. Every
# snapshot simulates the rest of the regular season, seeds that season's
# playoff format, and plays the bracket; played games are fixed. Besides
# title odds it yields per-round reach odds for the playoff odds tab.
print("\nComputing title odds (season + playoff Monte Carlo)...")
import playoff_sim
from wnba import REGULAR_SEASON_GAMES


def _conf_of(team, season):
    if team == 'Houston Comets' and season == 1997:
        return 'East'  # moved West in 1998
    return conf(team)


_sim_games = games.rename(columns={'date_game': 'date', 'home_team_name': 'home',
                                   'visitor_team_name': 'away'})
_sim_games = _sim_games[['season', 'date', 'home', 'away', 'home_pts', 'visitor_pts']].copy()
_sim_games['date'] = pd.to_datetime(_sim_games['date'])
# Current season's remaining schedule: basketball-reference's unplayed rows.
# Unplayed rows dated before the latest played game are stale postponements
# (the rescheduled game is its own row) and are dropped.
_loaded = pd.read_csv('loaded_wnba_games.csv')
_cur_season = int(_sim_games['season'].max())
_sched = _loaded[(_loaded['Season'] == _cur_season) & _loaded['PTS.1'].isna()].copy()
_sched['date'] = pd.to_datetime(_sched['Date'], format='%a, %b %d, %Y')
_last_played = _sim_games.loc[_sim_games['season'] == _cur_season, 'date'].max()
_sched = _sched[_sched['date'] >= _last_played]
_sim_games = pd.concat([_sim_games, pd.DataFrame({
    'season': _cur_season, 'date': _sched['date'],
    'home': _sched['Home/Neutral'], 'away': _sched['Visitor/Neutral'],
    'home_pts': np.nan, 'visitor_pts': np.nan})], ignore_index=True)

_sim_ratings = df[['season', 'date', 'name', 'rating']].copy()
_sim_ratings['date'] = pd.to_datetime(_sim_ratings['date'])
_playoff_odds, _brackets = playoff_sim.compute(
    _sim_games, _sim_ratings, lambda s: REGULAR_SEASON_GAMES.get(s, 44), _conf_of, _cur_season)
_playoff_odds['date'] = _playoff_odds['date'].dt.date
_rid_by_date = df.drop_duplicates('date').set_index('date')['ranking_id'].to_dict()
_playoff_odds['ranking_id'] = _playoff_odds['date'].map(_rid_by_date)
# Only non-zero odds are cached: eliminated teams render '-' (a stored 0
# would show as "<0.1%").
for rid, team, p in _playoff_odds[['ranking_id', 'team', 'champ']].itertuples(index=False):
    if p > 0:
        _title_odds_cache[(int(rid), team)] = float(p)


# ── Playoff odds tab (docs/data/playoff_odds.json) ──
# One entry per season (newest first) with a snapshot for every date from
# the end of the regular season on: seeds, series so far, and the chance to
# get past each round (last column = title odds).
def _playoff_odds_json():
    rt = df.set_index(['date', 'name'])
    seasons_out = []
    for season in sorted(_brackets, reverse=True):
        names, short = playoff_sim.round_names(season)
        n_rounds = len(names)
        enter = playoff_sim.entry_rounds(season)
        po = _playoff_odds[_playoff_odds['season'] == season].set_index(['date', 'team'])
        sg = _sim_games[(_sim_games['season'] == season) & _sim_games['home_pts'].notna()]
        snaps = []
        for d, (seeds, matchups, n_sims) in sorted(_brackets[season].items()):
            d_date = d.date() if hasattr(d, 'date') else d
            series = {}
            for rnd, bo, ta, tb, wins in matchups:
                need = bo // 2 + 1
                for me, opp in ((ta, tb), (tb, ta)):
                    w = sum(1 for x in wins if x == me); l = len(wins) - w
                    series.setdefault(me, []).append({
                        'round': short[rnd - 1], 'opp': opp, 'w': w, 'l': l,
                        'best_of': bo, 'done': max(w, l) >= need, 'won': w >= need})
            teams = []
            for team, seed in seeds.items():
                if seed not in enter or enter[seed] is None:
                    continue
                pr = po.loc[(d_date, team)]
                adv = [float(pr[f'r{k}']) for k in range(2, n_rounds + 1)] + [float(pr['champ'])]
                r = rt.loc[(d_date, team)]
                ser = series.get(team, [])
                teams.append({
                    'team': team, 'seed': seed, 'enter': int(enter[seed]),
                    'rating': round(float(r['rating']), 2), 'rank': int(r['rank']),
                    'rating_o': round(float(r['rating_o']), 2), 'rank_o': int(r['rank_o']),
                    'rating_d': round(float(r['rating_d']), 2), 'rank_d': int(r['rank_d']),
                    'adv': [round(x, 4) for x in adv],
                    'eliminated': any(x['done'] and not x['won'] for x in ser),
                    'series': ser,
                })
            # Seeds beyond the bracket (e.g. 9th place) never enter.
            teams = [t for t in teams if t['seed'] in enter]
            day = sg[sg['date'] == pd.Timestamp(d_date)]
            played = [x for x in matchups if x[4]]
            champ = [t['team'] for t in teams if t['adv'][-1] >= 1.0]
            if champ:
                stage = 'Champion'
            elif not played:
                stage = 'Before playoffs'
            else:
                live = [x for x in matchups if max(sum(1 for y in x[4] if y == x[2]),
                                                   sum(1 for y in x[4] if y == x[3])) < x[1] // 2 + 1]
                stage = names[min(x[0] for x in live) - 1] if live else names[max(x[0] for x in played) - 1]
            snaps.append({
                'date': str(d_date), 'stage': stage, 'n_sims': int(n_sims),
                'results': [{'home': x.home, 'away': x.away, 'hp': int(x.home_pts), 'vp': int(x.visitor_pts)}
                            for x in day.itertuples(index=False)] if played else [],
                'teams': teams,
            })
        seasons_out.append({'season': int(season), 'rounds': names, 'rounds_short': short,
                            'snapshots': snaps})
    return {'n_sims': playoff_sim.N_SIMS, 'current_season': int(_cur_season),
            'seasons': seasons_out}


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

# End-of-regular-season rating per (team, season): the snapshot flagged
# season_flag == 1 (the same one labeled "End of regular season"). The champion
# entry's own `rating` is the end-of-playoffs (flag == 2) rating, so pairing the
# two gives the Lists sub-view its "biggest playoff leap" ranking (PO - RS).
_rs_rating_lookup = {
    (r['name'], int(r['season'])): round(float(r['rating']), 3)
    for _, r in df[df['season_flag'] == 1].iterrows()
}

# End-of-regular-season title odds per (team, season): the model's championship
# probability for that team at the End-of-regular-season snapshot (flag == 1),
# i.e. how likely the eventual champion looked the day the playoffs began. Powers
# the "Biggest favorites" / "Longest shots" lists. title_odds_rank is the team's
# rank in those league-wide (sum-to-100%) odds at that snapshot. Odds are produced
# for every season (the model trains on 1998+ and 1997 is scored with that model),
# so this covers the full champions list.
_rs_title_odds_lookup = {}
_rs_title_odds_rank_lookup = {}
for _, r in df[df['season_flag'] == 1].iterrows():
    key = (r['name'], int(r['season']))
    _rs_title_odds_lookup[key]      = _title_odds_val(r['ranking_id'], r['name'])
    _rs_title_odds_rank_lookup[key] = _title_odds_rk(r['ranking_id'], r['name'])

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
            'last_match_is_cup': int(r['last_game_is_cup']) if _played(r['last_game_result']) else last_game_is_cup_as_of(r['name'], str(r['date']), r['season']),
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
        # Champions only - the PS GOAT is a greatest-CHAMPIONS list; dominant
        # non-winning seasons live on the RS GOAT. Length rounds down to the
        # nearest 10 (capped at 50) so it grows cleanly as titles accrue.
        rows = rows[rows['finals_status'].fillna(0) == 2]
        n = min(50, (len(rows) // 10) * 10)
    else:
        n = 50
    rows = rows[rows[sort_col].notna()]
    rows = rows.sort_values(sort_col, ascending=False).head(n).reset_index(drop=True)
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
                'last_match_is_cup': int(r['last_game_is_cup']) if _played(r['last_game_result']) else last_game_is_cup_as_of(team, str(r['date']), season),
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
                'last_match_is_cup': int(r['last_game_is_cup']) if played_today else last_game_is_cup_as_of(r['name'], snap_date, season),
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
            'rating_rs':      _rs_rating_lookup.get((cr['name'], int(season))),
            'rs_title_odds':      _rs_title_odds_lookup.get((cr['name'], int(season))),
            'rs_title_odds_rank': _rs_title_odds_rank_lookup.get((cr['name'], int(season))),
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

# Split per season so the page loads only the season it shows.
print("Writing playoff_odds/...")
os.makedirs('docs/data/playoff_odds', exist_ok=True)
_po = _playoff_odds_json()
for _s in _po['seasons']:
    with open(f"docs/data/playoff_odds/{_s['season']}.json", 'w') as f:
        json.dump(_s, f, separators=(',', ':'))
with open('docs/data/playoff_odds/index.json', 'w') as f:
    json.dump({'n_sims': _po['n_sims'], 'current_season': _po['current_season'],
               'seasons': [_s['season'] for _s in _po['seasons']]}, f, separators=(',', ':'))

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
