"""
generate_data.py — reads wnba_ratings_with_standings.csv and writes JSON for the LOBO web frontend.
Run after wnba.py. Outputs to docs/data/.

Mirrors the ZIDANE site architecture, simplified for a single league.
"""

import pandas as pd
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


def _od_fields(r):
    """Return rating_o/rating_d/rank_o/rank_d safely from a row. Returns
    None for missing values so downstream consumers render '-' rather
    than '0'."""
    return {
        'rating_o': round(float(r['rating_o']), 3) if 'rating_o' in r and not pd.isna(r['rating_o']) else None,
        'rating_d': round(float(r['rating_d']), 3) if 'rating_d' in r and not pd.isna(r['rating_d']) else None,
        'rank_o':   int(r['rank_o']) if 'rank_o' in r and not pd.isna(r['rank_o']) else None,
        'rank_d':   int(r['rank_d']) if 'rank_d' in r and not pd.isna(r['rank_d']) else None,
    }


def _played(result):
    """True iff this row represents an actual game played. Upstream now
    writes empty strings for non-game-days (was 'No Game' previously) —
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
# cross-season carry-forward — at the start of a new season, teams that
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


# Per-season last regular-season date — used to flag playoff vs regular-season entries
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
#   goat_rs.json — top 50 single-season ratings at end of regular season, all teams.
#   goat_ps.json — top 50 single-season ratings at end of playoffs, WNBA Finals participants only.
# Both gated to fully-complete seasons (a season is "complete" once a
# season_flag == 2 row exists for that season — i.e. the Finals have ended).
print("Writing goat_rs.json + goat_ps.json...")

# Short / disrupted seasons — flagged on GOAT rows so the UI can tag them
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
