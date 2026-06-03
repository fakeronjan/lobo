# =========================================================
# WNBA POWER RATINGS
# =========================================================

import requests
import pandas as pd
import numpy as np
from datetime import datetime, date

# =========================================================
# CONFIGURATION
# =========================================================

MIN_SEASON = 1997             # first WNBA season — DO NOT CHANGE

# Season-aware rolling window: window (game-days) = WINDOW_MULTIPLIER * games-per-team-per-season.
# At 1.5, a 44-game modern season gets a 66-day window (~57% of season). Bubble/lockout
# seasons get proportionally smaller windows automatically via REGULAR_SEASON_GAMES.
WINDOW_MULTIPLIER = 1.5

HOME_COURT_ADJUSTMENT = 2.0   # raw-point home advantage, subtracted from home margin pre-transform

# Margin transform: raw | sqrt | cap | log | tanh. cap=25 places the
# threshold at ~p94 of WNBA margins — typical games count linearly (the
# rating reads as an expected point spread), the ~6% of garbage-time
# blowouts get clipped so they don't dominate the solve.
MARGIN_TRANSFORM = "cap"
MARGIN_CAP = 25  # only used by cap / tanh

# Weighting mode: "wls" applies recency weights to observation influence
# via proper weighted least squares. The legacy "margin_scale" mode is
# preserved for historical reproduction only — it inflates margins by
# weight, which produces extreme ratings for thin-data (expansion) teams.
WEIGHTING_MODE = "wls"

# Re-process the most recent N ranking_ids (game-days) on every run so late-
# arriving WNBA data is absorbed. Without this a mid-day cron caches that
# day's snapshot and never re-ranks it even when more games for the same
# day finish hours later.
RECOMPUTE_TAIL_DAYS = 7

# Regular season game count per season (Option B proxy for playoff start).
# WNBA schedule has changed frequently — full lookup is cleaner than a default + overrides.
REGULAR_SEASON_GAMES = {
    1997: 28,
    1998: 30,
    1999: 32,
    2000: 32,
    2001: 32,
    2002: 32,
    2003: 34,
    2004: 34,
    2005: 34,
    2006: 34,
    2007: 34,
    2008: 34,
    2009: 34,
    2010: 34,
    2011: 34,
    2012: 34,
    2013: 34,
    2014: 34,
    2015: 34,
    2016: 34,
    2017: 34,
    2018: 34,
    2019: 34,
    2020: 22,  # bubble season
    2021: 32,  # COVID-shortened follow-up season
    2022: 36,
    2023: 40,
    2024: 40,
    2025: 44,
    2026: 44,
}

# WNBA Commissioner's Cup champion / runner-up by season.
# basketball-reference does NOT include the Cup final game in our scraped
# data (it's treated as exhibition), so honors come from this hardcoded
# lookup. Add a new entry each year — cup final happens mid-summer.
WNBA_CUP_RESULTS = {
    2021: ('Seattle Storm',     'Connecticut Sun'),
    2022: ('Las Vegas Aces',    'Chicago Sky'),
    2023: ('New York Liberty',  'Las Vegas Aces'),
    2024: ('Minnesota Lynx',    'New York Liberty'),
    2025: ('Indiana Fever',     'Minnesota Lynx'),
}

# =========================================================
# SCRAPING
# =========================================================

def scrape_games(min_season, max_season, existing_df):
    """
    Scrape any seasons not already fully captured in existing_df.
    Returns a combined DataFrame of all games (old + new), saved to loaded_wnba_games.csv.
    """
    max_season_completed = max(existing_df['Season']) - 1  # latest season may be partial
    min_season_completed = min(existing_df['Season'])

    print(f"Already have complete data for seasons {min_season_completed}–{max_season_completed}")
    print(f"Checking for new data through season {max_season}")

    new_frames = []
    for year in range(min_season, max_season + 1):
        if min_season_completed <= year <= max_season_completed:
            continue
        url = f'https://www.basketball-reference.com/wnba/years/{year}_games.html'
        try:
            df = pd.read_html(url)[0]
        except Exception:
            print(f"{year} — not found, skipping.")
            continue
        df['Season'] = year
        new_frames.append(df)
        print(f"{year} — scraped!")

    print("Successfully scraped!")

    combined = pd.concat([existing_df] + new_frames, axis=0, sort=False).reset_index(drop=True)
    combined.sort_values('Season', inplace=True)
    combined.drop_duplicates(keep="first", inplace=True)
    combined.to_csv('loaded_wnba_games.csv', index=False)
    return combined


# =========================================================
# GAME DATA PREPARATION
# =========================================================

def prepare_game_data(raw_df):
    """
    Clean and enrich the raw games DataFrame with margins, win flags,
    adjusted scores, date IDs, and result strings.
    """
    df = raw_df.copy()

    # Remove playoff separator rows and standardize column names
    df = df[df['Date'] != 'Playoffs']
    df.rename(columns={
        'Season':          'season',
        'Date':            'date_game',
        'Visitor/Neutral': 'visitor_team_name',
        'PTS':             'visitor_pts',
        'Home/Neutral':    'home_team_name',
        'PTS.1':           'home_pts',
    }, inplace=True)

    df = df[['season', 'date_game', 'visitor_team_name', 'visitor_pts', 'home_team_name', 'home_pts']].copy()

    df['visitor_pts'] = pd.to_numeric(df['visitor_pts'])
    df['home_pts'] = pd.to_numeric(df['home_pts'])

    # Margin of victory (raw points). HCA and the margin transform are
    # applied inside the solver, not here, so downstream consumers see
    # the unmodified game record.
    df['visitor_margin'] = df['visitor_pts'] - df['home_pts']
    df['home_margin'] = -df['visitor_margin']

    # Win flags
    df['visitor_win'] = np.where(df['visitor_margin'] > 0, 1, 0)
    df['home_win'] = 1 - df['visitor_win']

    # Drop incomplete rows before date parsing
    df = df.dropna()

    # Date parsing and sorting
    df['date_game'] = pd.to_datetime(df['date_game'], format='%a, %b %d, %Y')
    df.sort_values('date_game', inplace=True)
    df.drop_duplicates(keep="first", inplace=True)

    # Date and game IDs
    df['grouped_date_id'] = df.groupby(['date_game']).ngroup() + 1
    df['unique_game_id'] = df.groupby(df.columns.tolist(), sort=False).ngroup() + 1

    # Result strings
    df['home_pts'] = df['home_pts'].astype(int)
    df['visitor_pts'] = df['visitor_pts'].astype(int)
    df['home_wl'] = np.where(df['home_win'] == 1, "W", "L")
    df['visitor_wl'] = np.where(df['visitor_win'] == 1, "W", "L")
    df['home_result'] = (
        df['home_wl'] + " " + df['home_pts'].map(str) + "-" + df['visitor_pts'].map(str)
        + " vs. " + df['visitor_team_name']
    )
    df['visitor_result'] = (
        df['visitor_wl'] + " " + df['visitor_pts'].map(str) + "-" + df['home_pts'].map(str)
        + " @ " + df['home_team_name']
    )

    df.to_csv('all_wnba_games.csv', index=False)
    print("CSV of WNBA games is ready!")
    return df


# =========================================================
# FAKERONJAN WLS RATINGS — homebrew weighted least squares solver
# =========================================================

def _apply_margin_transform(margin, transform, cap):
    """Sign-preserving transform applied to (raw_margin - hca)."""
    m = np.asarray(margin, dtype=float)
    if transform == "raw":
        return m
    if transform == "sqrt":
        return np.sign(m) * np.sqrt(np.abs(m))
    if transform == "cap":
        return np.clip(m, -cap, cap)
    if transform == "log":
        return np.sign(m) * np.log1p(np.abs(m))
    if transform == "tanh":
        return cap * np.tanh(m / cap)
    raise ValueError(f"Unknown MARGIN_TRANSFORM: {transform}")


def _solve_wls(window_df, hca, weighting_mode, margin_transform, margin_cap):
    """
    Solve for team fakeronjan WLS ratings on a single rolling window.

    Builds X (n_games × n_teams) with +1 for home, -1 for visitor, y from
    the transformed HCA-adjusted home margin, and W from the recency
    weights. Solves min sum_i w_i * (X_i r - y_i)^2 with a zero-sum
    constraint enforced as an extra high-weight row.

    WLS via row-scaling: multiplying both X and y by sqrt(w_i) turns the
    weighted problem into an ordinary lstsq.
    """
    teams = sorted(set(window_df["home_team_name"]) | set(window_df["visitor_team_name"]))
    team_idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)
    n_games = len(window_df)

    X = np.zeros((n_games + 1, n_teams))
    y = np.zeros(n_games + 1)
    w = np.zeros(n_games + 1)

    home_pts = window_df["home_pts"].to_numpy(dtype=float)
    visitor_pts = window_df["visitor_pts"].to_numpy(dtype=float)
    weights = window_df["date_weight"].to_numpy(dtype=float)
    home_names = window_df["home_team_name"].to_numpy()
    visitor_names = window_df["visitor_team_name"].to_numpy()

    raw_margin = home_pts - visitor_pts - hca
    transformed = _apply_margin_transform(raw_margin, margin_transform, margin_cap)

    for i in range(n_games):
        X[i, team_idx[home_names[i]]] = 1.0
        X[i, team_idx[visitor_names[i]]] = -1.0

    if weighting_mode == "wls":
        y[:n_games] = transformed
        w[:n_games] = weights
    elif weighting_mode == "margin_scale":
        y[:n_games] = transformed * weights
        w[:n_games] = 1.0
    else:
        raise ValueError(f"Unknown WEIGHTING_MODE: {weighting_mode}")

    # Zero-sum constraint via high-weight extra row.
    X[-1, :] = 1.0
    y[-1] = 0.0
    w[-1] = 1.0e8

    sqrt_w = np.sqrt(w)
    Xw = X * sqrt_w[:, None]
    yw = y * sqrt_w
    r, *_ = np.linalg.lstsq(Xw, yw, rcond=None)

    out = pd.DataFrame({"name": teams, "rating": r})
    out["rank"] = out["rating"].rank(ascending=False, method="min").astype(int)
    return out


def _window_for_season(season):
    """Season-aware window size: WINDOW_MULTIPLIER × regular-season games per team."""
    reg_games = REGULAR_SEASON_GAMES.get(int(season), 34)
    return int(round(reg_games * WINDOW_MULTIPLIER))


# Smallest window across any season - floor for the loop's starting ranking_id.
# Using the max here would skip early seasons whose total game-days never reach
# the modern (44-game) window size (1997 has only 62 game-days vs. a 66 floor).
# Each ranking_id is then gated by its OWN season's window_size inside the loop.
_MIN_WINDOW = min(_window_for_season(s) for s in REGULAR_SEASON_GAMES)


def compute_ratings(master_df, existing_ratings_df):
    """
    Compute daily fakeronjan WLS power ratings using a season-aware rolling window
    (WINDOW_MULTIPLIER × games-per-team-this-season). Skips dates already
    present in existing_ratings_df. Re-processes the most recent
    RECOMPUTE_TAIL_DAYS ranking_ids each run to absorb late-arriving data.
    """
    max_date_id = max(master_df['grouped_date_id'])
    min_date_id = _MIN_WINDOW
    all_ids = sorted(existing_ratings_df['ranking_id'].unique())
    if len(all_ids) > RECOMPUTE_TAIL_DAYS:
        tail_threshold = all_ids[-RECOMPUTE_TAIL_DAYS]
        n_dropped = int((existing_ratings_df['ranking_id'] >= tail_threshold).sum())
        existing_ratings_df = existing_ratings_df[existing_ratings_df['ranking_id'] < tail_threshold].copy()
        print(f"  Re-processing tail {RECOMPUTE_TAIL_DAYS} game-days "
              f"({n_dropped:,} rows dropped from ratings cache for late-arriving-data refresh)")
    max_ranked = int(max(existing_ratings_df['ranking_id'])) if len(existing_ratings_df) else -1
    min_ranked = int(min(existing_ratings_df['ranking_id'])) if len(existing_ratings_df) else -1

    print("Running WNBA ratings for new data...")
    new_frames = []

    # Determine each ranking_id's season once up front so window sizing is fast.
    rid_to_season = (
        master_df.sort_values('grouped_date_id')
                 .drop_duplicates('grouped_date_id', keep='last')
                 .set_index('grouped_date_id')['season']
                 .to_dict()
    )

    for i in range(min_date_id, max_date_id + 1):
        if min_ranked <= i <= max_ranked:
            continue

        # Season of this ranking_id determines window size. For game-days
        # without a game (rare), fall back to the most recent earlier season.
        season_for_window = rid_to_season.get(i)
        if season_for_window is None:
            prior_ids = [k for k in rid_to_season if k < i]
            season_for_window = rid_to_season[max(prior_ids)] if prior_ids else MIN_SEASON
        window_size = _window_for_season(season_for_window)

        # Don't publish until this season's window can be filled. Each season's
        # window is sized to its game count, so the earliest publishable game-day
        # is the one where the lookback reaches a full window of games.
        if i < window_size:
            continue

        window = master_df[
            (master_df['grouped_date_id'] >= i - (window_size - 1)) &
            (master_df['grouped_date_id'] <= i)
        ].copy()

        window['date_weight'] = (window['grouped_date_id'] - i + window_size) / window_size

        current_date = window['date_game'].max()
        season = window['season'].max()
        print(current_date)

        ranked = _solve_wls(
            window,
            hca=HOME_COURT_ADJUSTMENT,
            weighting_mode=WEIGHTING_MODE,
            margin_transform=MARGIN_TRANSFORM,
            margin_cap=MARGIN_CAP,
        )
        ranked['ranking_date'] = current_date
        ranked['ranking_id'] = i
        ranked['season'] = season
        new_frames.append(ranked)

    ratings_df = pd.concat([existing_ratings_df] + new_frames, axis=0, sort=False).reset_index(drop=True)
    ratings_df.sort_values(['ranking_id', 'name'], inplace=True)
    ratings_df.drop_duplicates(keep="first", inplace=True)
    ratings_df['ranking_date'] = pd.to_datetime(ratings_df['ranking_date']).dt.date

    ratings_df.to_csv('wnba_ratings.csv', index=False)
    print("CSV of power rankings is ready!")
    return ratings_df


# =========================================================
# STANDINGS
# =========================================================

def _make_pivot(df, value_col, index_col, new_value_name, aggfunc=np.sum):
    """Helper: pivot, fillna, reset index, and standardize column names."""
    pivot = pd.pivot_table(df, values=value_col, index=[index_col], aggfunc=aggfunc)
    return (
        pivot.fillna(0)
             .reset_index()
             .rename(columns={value_col: new_value_name, index_col: 'name'})
    )


def compute_standings(master_df, existing_standings_df):
    """
    Compute cumulative season standings for each day in master_df.
    Skips dates already present in existing_standings_df.
    """
    game_df = master_df[['season', 'date_game', 'grouped_date_id', 'visitor_team_name', 'visitor_win', 'home_team_name', 'home_win']]
    max_date_id = max(master_df['grouped_date_id'])
    # Standings are cumulative - no window to fill, so we start from the first game-day.
    min_date_id = int(master_df['grouped_date_id'].min())
    all_ids = sorted(existing_standings_df['ranking_id'].unique())
    if len(all_ids) > RECOMPUTE_TAIL_DAYS:
        tail_threshold = all_ids[-RECOMPUTE_TAIL_DAYS]
        n_dropped = int((existing_standings_df['ranking_id'] >= tail_threshold).sum())
        existing_standings_df = existing_standings_df[existing_standings_df['ranking_id'] < tail_threshold].copy()
        print(f"  Re-processing tail {RECOMPUTE_TAIL_DAYS} game-days "
              f"({n_dropped:,} rows dropped from standings cache for late-arriving-data refresh)")
    max_ranked = int(max(existing_standings_df['ranking_id'])) if len(existing_standings_df) else -1
    min_ranked = int(min(existing_standings_df['ranking_id'])) if len(existing_standings_df) else -1

    print("Producing standings...")
    new_frames = []

    for i in range(min_date_id, max_date_id + 1):
        if min_ranked <= i <= max_ranked:
            continue

        season_slice = game_df[game_df['grouped_date_id'] <= i]
        season = season_slice['season'].max()
        season_slice = season_slice[season_slice['season'] == season]
        ranking_date = season_slice['date_game'].max()
        print(ranking_date)

        vw = _make_pivot(season_slice, 'visitor_win', 'visitor_team_name', 'visitor_wins')
        vg = _make_pivot(season_slice, 'visitor_win', 'visitor_team_name', 'visitor_games', aggfunc='count')
        hw = _make_pivot(season_slice, 'home_win',    'home_team_name',    'home_wins')
        hg = _make_pivot(season_slice, 'home_win',    'home_team_name',    'home_games',    aggfunc='count')

        merged = (
            vw.merge(vg, on='name', how='outer')
              .merge(hw, on='name', how='outer')
              .merge(hg, on='name', how='outer')
              .fillna(0)
        )

        merged['wins']   = (merged['visitor_wins'] + merged['home_wins']).astype(int)
        merged['losses'] = (merged['visitor_games'] + merged['home_games'] - merged['wins']).astype(int)
        merged['record'] = merged['wins'].map(str) + "-" + merged['losses'].map(str)
        merged = merged[['name', 'wins', 'losses', 'record']]

        merged['ranking_id']   = i
        merged['ranking_date'] = ranking_date
        merged['season']       = season
        new_frames.append(merged)

    standings_df = pd.concat([existing_standings_df] + new_frames, axis=0, sort=False).reset_index(drop=True)
    standings_df.sort_values(['ranking_id', 'name'], inplace=True)
    standings_df.drop_duplicates(keep="first", inplace=True)
    standings_df['ranking_date'] = pd.to_datetime(standings_df['ranking_date']).dt.date

    standings_df.to_csv('daily_standings.csv', index=False)
    print("CSV of standings is ready!")
    return standings_df


# =========================================================
# FINAL ASSEMBLY
# =========================================================

def _get_regular_season_end_date(master_df, season):
    """
    Estimate the last date of the regular season for a given season using
    Option B: find the last date where every active team has played at or
    under the expected regular season game count.
    Falls back to SHORTENED_SEASON_OVERRIDES for bubble/shortened years.
    """
    threshold = REGULAR_SEASON_GAMES.get(season, 34)  # 34 as fallback for any unlisted season
    season_games = master_df[master_df['season'] == season].copy()

    # Build cumulative game count per team per date
    home = season_games[['date_game', 'home_team_name']].rename(columns={'home_team_name': 'team'})
    away = season_games[['date_game', 'visitor_team_name']].rename(columns={'visitor_team_name': 'team'})
    all_games = pd.concat([home, away]).sort_values('date_game')
    all_games['team_game_num'] = all_games.groupby('team').cumcount() + 1

    # Last date where no team has exceeded the threshold
    within_rs = all_games[all_games['team_game_num'] <= threshold]
    if within_rs.empty:
        return None
    return within_rs['date_game'].max()


def assemble_final(master_df, ratings_df, standings_df):
    """Merge ratings and standings, add flags and last-game context."""
    print("Final step — merging WNBA ratings and standings...")

    final_df = pd.merge(ratings_df, standings_df, how='left', on=['ranking_id', 'name'])
    final_df.rename(columns={'ranking_date_x': 'date', 'season_x': 'season'}, inplace=True)
    final_df['season'] = final_df['season'].astype(int)
    final_df['record'] = final_df['record'].fillna("0-0")

    # Flag the most recent date overall
    latest_date_id = final_df['ranking_id'].max()
    final_df['current_date'] = (final_df['ranking_id'] == latest_date_id).astype(int)

    final_df['name_season'] = final_df['name'] + " - " + final_df['season'].map(str)

    # -------------------------------------------------------------------------
    # season_flag: 0 = regular season, 1 = last day of regular season,
    #              2 = last day of postseason
    # -------------------------------------------------------------------------
    final_df['season_flag'] = 0

    # Detect Finals champion + runner-up via bracket walk over post-RS
    # games. For each playoff matchup (a sorted team pair), the H2H winner
    # is the team with more wins; in-progress matchups (tied H2H) are
    # skipped. A team is "still in" if their LATEST matchup was a win.
    # When exactly one team is still in, they're the champion, and their
    # latest opponent is the runner-up. Format-agnostic — handles WNBA's
    # 1997 BO1, BO3 era, BO5 era, BO7 era, single-elim play-in rounds, and
    # variable bracket depths uniformly. No date cushion: the structure of
    # the bracket distinguishes a CF/semis clinch (2+ teams still in) from
    # the Finals clinch (exactly 1 team still in).
    def detect_finals_champion(season_games, rs_end_date):
        if rs_end_date is None:
            return None, None
        pg = season_games[pd.to_datetime(season_games['date_game']) > pd.Timestamp(rs_end_date)].copy()
        if pg.empty:
            return None, None
        pg['date_game'] = pd.to_datetime(pg['date_game'])
        pg['_matchup'] = pg.apply(
            lambda r: tuple(sorted([r['home_team_name'], r['visitor_team_name']])),
            axis=1
        )
        # Split each matchup into consecutive series (gap > 10 days = new
        # series). Harmless for normal WNBA seasons; defensive against any
        # round-robin-style postseason format that could pair same teams
        # in separate stages.
        team_history = {}
        for matchup, mg in pg.groupby('_matchup'):
            a, b = matchup
            mg_sorted = mg.sort_values('date_game').reset_index(drop=True)
            current_idx = [0]
            for i in range(1, len(mg_sorted)):
                gap = (mg_sorted.loc[i, 'date_game'] - mg_sorted.loc[i-1, 'date_game']).days
                if gap > 10:
                    _process_series(mg_sorted.iloc[current_idx], a, b, team_history)
                    current_idx = [i]
                else:
                    current_idx.append(i)
            _process_series(mg_sorted.iloc[current_idx], a, b, team_history)
        still_in = []
        for team, hist in team_history.items():
            hist.sort(key=lambda x: x[0])
            if hist[-1][1]:
                still_in.append((team, hist))
        if len(still_in) != 1:
            return None, None
        champion, hist = still_in[0]
        _, _, runner_up = hist[-1]
        return champion, runner_up

    def _process_series(series_df, a, b, team_history):
        a_wins = (((series_df['home_team_name'] == a) & (series_df['home_win'] == 1)) |
                  ((series_df['visitor_team_name'] == a) & (series_df['home_win'] == 0))).sum()
        b_wins = len(series_df) - a_wins
        if a_wins > b_wins:
            winner, loser = a, b
        elif b_wins > a_wins:
            winner, loser = b, a
        else:
            return
        last_date = series_df['date_game'].max()
        team_history.setdefault(winner, []).append((last_date, True, loser))
        team_history.setdefault(loser, []).append((last_date, False, winner))

    _finals_results = {}  # season -> (champion, runner_up)
    for season in final_df['season'].unique():
        season_games = master_df[master_df['season'] == season]
        if season_games.empty:
            continue
        rs_end = _get_regular_season_end_date(master_df, season)
        champ, ru = detect_finals_champion(season_games, rs_end)
        if champ is not None:
            _finals_results[season] = (champ, ru)

    def season_is_fully_complete(season):
        return season in _finals_results

    # Regular season is "done" once any team has played the threshold count
    regular_season_complete = set()
    for season in final_df['season'].unique():
        sg = master_df[master_df['season'] == season]
        if sg.empty:
            continue
        home = sg[['home_team_name']].rename(columns={'home_team_name': 'team'})
        away = sg[['visitor_team_name']].rename(columns={'visitor_team_name': 'team'})
        all_g = pd.concat([home, away])
        threshold = REGULAR_SEASON_GAMES.get(season, 34)
        if all_g.groupby('team').size().max() >= threshold:
            regular_season_complete.add(season)

    # Last day of postseason — only for fully complete seasons
    season_max_id = final_df.groupby('season')['ranking_id'].transform('max')
    is_completed = final_df['season'].apply(season_is_fully_complete)
    final_df['season_flag'] = np.where(
        (final_df['ranking_id'] == season_max_id) & is_completed,
        2,
        0
    )

    # Last day of regular season — only for seasons where regular season has actually ended
    for season in final_df['season'].unique():
        if season not in regular_season_complete:
            continue
        rs_end_date = _get_regular_season_end_date(master_df, season)
        if rs_end_date is None:
            continue
        rs_end_str = str(rs_end_date.date()) if hasattr(rs_end_date, 'date') else str(rs_end_date)
        match = final_df[(final_df['season'] == season) & (final_df['date'].astype(str) == rs_end_str)]
        if match.empty:
            continue
        rs_end_ranking_id = match['ranking_id'].max()
        final_df['season_flag'] = np.where(
            (final_df['season'] == season) &
            (final_df['ranking_id'] == rs_end_ranking_id) &
            (final_df['season_flag'] != 2),
            1,
            final_df['season_flag']
        )

    # -------------------------------------------------------------------------
    # Champion & runner-up: assign from the _finals_results dict computed
    # earlier alongside season_is_fully_complete (same event-based detection).
    # -------------------------------------------------------------------------
    final_df['champ'] = 0
    final_df['runnerup'] = 0
    for season, (champion, runner_up) in _finals_results.items():
        champ_season = f"{champion} - {season}"
        runnerup_season = f"{runner_up} - {season}"
        final_df['champ'] = np.where(final_df['name_season'] == champ_season, 1, final_df['champ'])
        final_df['runnerup'] = np.where(final_df['name_season'] == runnerup_season, 1, final_df['runnerup'])

    # Combined status column: 0 = neither, 1 = runner-up, 2 = champion
    final_df['finals_status'] = final_df['runnerup'] + 2 * final_df['champ']

    # -------------------------------------------------------------------------
    # WNBA Commissioner's Cup champion & runner-up (since 2021)
    # -------------------------------------------------------------------------
    final_df['cup_champ']    = 0
    final_df['cup_runnerup'] = 0
    for season, (champion, runner_up) in WNBA_CUP_RESULTS.items():
        champ_ns    = f"{champion} - {season}"
        runnerup_ns = f"{runner_up} - {season}"
        final_df['cup_champ']    = np.where(final_df['name_season'] == champ_ns,    1, final_df['cup_champ'])
        final_df['cup_runnerup'] = np.where(final_df['name_season'] == runnerup_ns, 1, final_df['cup_runnerup'])

    # 0 = neither, 1 = cup runner-up, 2 = cup champion
    final_df['cup_status'] = final_df['cup_runnerup'] + 2 * final_df['cup_champ']

    # -------------------------------------------------------------------------
    # Last game result
    # -------------------------------------------------------------------------
    final_df['date_str'] = final_df['date'].astype(str)

    lastgameh = (
        master_df[['date_game', 'home_team_name', 'home_result', 'visitor_team_name']]
        .rename(columns={'home_team_name': 'name', 'date_game': 'date_str'})
        .assign(date_str=lambda d: d['date_str'].astype(str))
    )
    lastgamev = (
        master_df[['date_game', 'visitor_team_name', 'visitor_result', 'home_team_name']]
        .rename(columns={'visitor_team_name': 'name', 'date_game': 'date_str'})
        .assign(date_str=lambda d: d['date_str'].astype(str))
    )

    final_df = final_df.merge(lastgameh, how='left', on=['date_str', 'name'])
    final_df = final_df.merge(lastgamev, how='left', on=['date_str', 'name'])

    for col in ['home_result', 'visitor_result', 'home_team_name', 'visitor_team_name']:
        final_df[col] = final_df[col].fillna("")

    final_df['last_game_result'] = (final_df['home_result'] + final_df['visitor_result'])
    final_df['opponent'] = final_df['home_team_name'] + final_df['visitor_team_name']

    final_df = final_df[[
        'ranking_id', 'date', 'season', 'name', 'rating', 'rank',
        'record', 'current_date', 'season_flag', 'name_season',
        'champ', 'runnerup', 'finals_status',
        'cup_champ', 'cup_runnerup', 'cup_status',
        'last_game_result', 'opponent'
    ]]

    final_df.to_csv('wnba_ratings_with_standings.csv', index=False)
    print("CSV of everything is ready!")
    return final_df


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    max_season = datetime.now().year + 1

    # 1. Scrape
    existing_games = pd.read_csv("loaded_wnba_games.csv")
    raw_df = scrape_games(MIN_SEASON, max_season, existing_games)

    # 2. Prepare game data
    master_df = prepare_game_data(raw_df)

    # 3. Ratings
    existing_ratings = pd.read_csv("wnba_ratings.csv")
    ratings_df = compute_ratings(master_df, existing_ratings)

    # 4. Standings
    existing_standings = pd.read_csv("daily_standings.csv")
    standings_df = compute_standings(master_df, existing_standings)

    # 5. Final merge
    assemble_final(master_df, ratings_df, standings_df)
