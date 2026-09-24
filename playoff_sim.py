"""LOBO title odds: Monte Carlo of the rest of the WNBA season + playoffs.

For every rating snapshot, simulate the remaining regular-season games,
seed that season's playoff format from the simulated standings, then play
the bracket. Anything already played as of the snapshot date is fixed.
Replaces the old leave-one-season-out logistic model (which needed ~25
seasons of champion labels per phase and couldn't produce per-round odds).

Game model (fit on 6,813 WNBA games 1997-2025, pre-game snapshot ratings,
probit by log loss):
    P(home win) = Phi(A * (rating_home - rating_away + home_pts))
A = 0.0654 (ratings are overconfident: realized margin ~ 0.71 x rating gap,
residual SD ~12 pts). Home court shrank over time, so home_pts is by era:
4.9 through 2011, 2.8 from 2012, 0 for the 2020 bubble.

Output per (snapshot, team): probability of making the playoffs, reaching
each later round, and winning the title (the Title odds column).
"""
import numpy as np
import pandas as pd
from scipy.special import ndtr

N_SIMS = 10000
A = 0.0654


# Standings ties the WNBA broke differently from our head-to-head + point
# differential rule (older tiebreak rules aren't reconstructable from game
# data). Each team listed wins its tie. Found by the bracket-consumption
# check: the real first-round pairings only match with these.
TIEBREAK_WINNERS = {
    1997: ['Charlotte Sting'],   # over Cleveland (15-13) for the 4th spot
    2003: ['Minnesota Lynx'],    # over Seattle (18-16) for West 4th
}


def home_pts(season):
    if season == 2020:
        return 0.0
    return 4.9 if season <= 2011 else 2.8


# ── Formats ──────────────────────────────────────────────────────────────────
# Seeds: ('conf', conference, k) = k-th in conference; ('lg', k) = k-th
# league-wide; ('W', match_id) = that match's winner; ('RS', match_ids, k) =
# k-th best seed among those matches' winners (re-seeded rounds). Matches:
# (id, round, best_of, slot_a, slot_b). Round 1 = first round played; teams
# with byes enter later.
def _conf_bracket(n_per_conf, first, cf_bo, finals_bo):
    """1998-2015 style: seeded within each conference."""
    m = []
    for c in ('East', 'West'):
        if n_per_conf == 4:
            m += [(f'{c}1', 1, first, ('conf', c, 1), ('conf', c, 4)),
                  (f'{c}2', 1, first, ('conf', c, 2), ('conf', c, 3)),
                  (f'{c}F', 2, cf_bo, ('W', f'{c}1'), ('W', f'{c}2'))]
        elif n_per_conf == 3:  # 1999: 2v3 single game, winner at the 1 seed
            m += [(f'{c}1', 1, 1, ('conf', c, 2), ('conf', c, 3)),
                  (f'{c}F', 2, cf_bo, ('conf', c, 1), ('W', f'{c}1'))]
    last = 3
    m.append(('F', last, finals_bo, ('W', 'EastF'), ('W', 'WestF')))
    return m


FORMAT_1997 = [  # conference winners seeded 1-2, next two best records 3-4
    ('S1', 1, 1, ('lg', 1), ('lg', 4)),
    ('S2', 1, 1, ('lg', 2), ('lg', 3)),
    ('F', 2, 1, ('W', 'S1'), ('W', 'S2')),
]
FORMAT_1998 = [  # top two per conference, crossed
    ('S1', 1, 3, ('conf', 'West', 1), ('conf', 'East', 2)),
    ('S2', 1, 3, ('conf', 'East', 1), ('conf', 'West', 2)),
    ('F', 2, 3, ('W', 'S1'), ('W', 'S2')),
]
FORMAT_2016 = [  # league-wide 8, single-game R1/R2, re-seeded, byes for 1-4
    ('A', 1, 1, ('lg', 5), ('lg', 8)),
    ('B', 1, 1, ('lg', 6), ('lg', 7)),
    ('C', 2, 1, ('lg', 3), ('RS', ('A', 'B'), 2)),
    ('D', 2, 1, ('lg', 4), ('RS', ('A', 'B'), 1)),
    ('S1', 3, 5, ('lg', 1), ('RS', ('C', 'D'), 2)),
    ('S2', 3, 5, ('lg', 2), ('RS', ('C', 'D'), 1)),
    ('F', 4, 5, ('W', 'S1'), ('W', 'S2')),
]


def _league8(finals_bo):  # 2022+: fixed bracket, best-of-3 first round
    return [
        ('A', 1, 3, ('lg', 1), ('lg', 8)),
        ('B', 1, 3, ('lg', 4), ('lg', 5)),
        ('C', 1, 3, ('lg', 2), ('lg', 7)),
        ('D', 1, 3, ('lg', 3), ('lg', 6)),
        ('S1', 2, 5, ('W', 'A'), ('W', 'B')),
        ('S2', 2, 5, ('W', 'C'), ('W', 'D')),
        ('F', 3, finals_bo, ('W', 'S1'), ('W', 'S2')),
    ]


def bracket_for(season):
    if season == 1997:
        return FORMAT_1997
    if season == 1998:
        return FORMAT_1998
    if season == 1999:
        return _conf_bracket(3, 1, 3, 3)
    if season <= 2004:
        return _conf_bracket(4, 3, 3, 3)
    if season <= 2015:
        return _conf_bracket(4, 3, 3, 5)
    if season <= 2021:
        return FORMAT_2016
    if season <= 2024:
        return _league8(5)
    return _league8(7)


def host_pattern(season, best_of):
    """Per game: True = better seed hosts."""
    if best_of == 1:
        return [True]
    if best_of == 3:
        if season <= 2015:
            return [False, True, True]   # lower seed hosted game 1
        if 2022 <= season <= 2024:
            return [True, True, False]
        return [True, False, True]       # 2025+: 1-1-1
    if best_of == 5:
        return [True, True, False, False, True]
    return [True, True, False, False, True, False, True]


def round_names(season):
    """(full, short) names for every round, first to last."""
    n = max(r for _, r, *_ in bracket_for(season))
    if n == 2:
        return ['Semifinals', 'Finals'], ['Semi', 'Final']
    if n == 4:
        return (['First Round', 'Second Round', 'Semifinals', 'Finals'],
                ['R1', 'R2', 'Semi', 'Final'])
    if season <= 2015:
        return ['First Round', 'Conference Finals', 'Finals'], ['R1', 'Conf Final', 'Final']
    return ['First Round', 'Semifinals', 'Finals'], ['R1', 'Semi', 'Final']


def entry_rounds(season):
    """Seed slot -> round that seed enters (later than 1 = bye)."""
    out = {}
    for _, rnd, _, *slots in bracket_for(season):
        for sl in slots:
            if sl[0] == 'conf':
                out.setdefault(f"{sl[1][0]}{sl[2]}", rnd)
            elif sl[0] == 'lg':
                out.setdefault(str(sl[1]), rnd)
    return out


class SeasonSim:
    def __init__(self, season, games, rs_games_total, conf_of, ratings):
        """games: season's games (date, home, away, home_pts, visitor_pts; NaN
        pts = scheduled). Regular season = each team's first rs_games_total
        games in date order; everything after is postseason."""
        self.season = season
        g = games.sort_values('date', kind='stable').reset_index(drop=True)
        counts, is_rs = {}, []
        for h, a in zip(g['home'], g['away']):
            ch, ca = counts.get(h, 0) + 1, counts.get(a, 0) + 1
            counts[h], counts[a] = ch, ca
            is_rs.append(ch <= rs_games_total and ca <= rs_games_total)
        g['is_rs'] = is_rs
        rs = g[g['is_rs']]
        self.teams = sorted(set(rs['home']) | set(rs['away']))
        self.idx = {t: i for i, t in enumerate(self.teams)}
        self.conf = np.array([conf_of(t, season) for t in self.teams])
        rs = rs.assign(h=rs['home'].map(self.idx), a=rs['away'].map(self.idx))
        self.rs = rs
        self.ps = g[~g['is_rs'] & g['home_pts'].notna()].copy()
        self.ps['winner'] = np.where(self.ps['home_pts'] > self.ps['visitor_pts'],
                                     self.ps['home'], self.ps['away'])
        self.ratings = ratings
        self.bracket = bracket_for(season)
        self.n_rounds = max(r for _, r, *_ in self.bracket)

    def _static_tiebreak(self, done):
        """Deterministic tiebreak once the regular season is complete:
        head-to-head win% within each group tied on win%, then point
        differential. Returns a per-team score (higher = better)."""
        T = len(self.teams)
        w = np.zeros(T); gp = np.zeros(T); pd_ = np.zeros(T)
        for h, a, hp, vp in done[['h', 'a', 'home_pts', 'visitor_pts']].itertuples(index=False):
            gp[h] += 1; gp[a] += 1; pd_[h] += hp - vp; pd_[a] += vp - hp
            w[h if hp > vp else a] += 1
        pct = w / np.maximum(gp, 1)
        score = np.zeros(T)
        for p in np.unique(pct):
            grp = np.where(pct == p)[0]
            if len(grp) < 2:
                continue
            gs = set(grp)
            sub = done[done['h'].isin(gs) & done['a'].isin(gs)]
            hw = np.zeros(T); hg = np.zeros(T)
            for h, a, hp, vp in sub[['h', 'a', 'home_pts', 'visitor_pts']].itertuples(index=False):
                hg[h] += 1; hg[a] += 1; hw[h if hp > vp else a] += 1
            h2h = np.where(hg > 0, hw / np.maximum(hg, 1), 0.5)
            score[grp] = h2h[grp] * 1000 + pd_[grp] / 1000
        for t in TIEBREAK_WINNERS.get(self.season, []):
            score[self.idx[t]] += 1e6
        return score

    def odds_at(self, d, n_sims=N_SIMS):
        T = len(self.teams)
        rng = np.random.default_rng(int(pd.Timestamp(d).strftime('%Y%m%d')))
        rt = self.ratings.get(d, {})
        R = np.array([rt.get(t, 0.0) for t in self.teams])
        hp = home_pts(self.season)

        def p_home(h, a, edge):
            return ndtr(A * (R[h] - R[a] + edge))

        played = self.rs['home_pts'].notna() & (self.rs['date'] <= d)
        done, rest = self.rs[played], self.rs[~played]
        w0 = np.zeros(T); g0 = np.zeros(T)
        for h, a, hpt, vpt in done[['h', 'a', 'home_pts', 'visitor_pts']].itertuples(index=False):
            g0[h] += 1; g0[a] += 1; w0[h if hpt > vpt else a] += 1
        W = np.tile(w0, (n_sims, 1)); G = np.tile(g0, (n_sims, 1))
        if len(rest):
            h = rest['h'].to_numpy(); a = rest['a'].to_numpy()
            hw = (rng.random((n_sims, len(rest))) < p_home(h, a, hp)).astype(float)
            Hm = np.zeros((len(rest), T)); Hm[np.arange(len(rest)), h] = 1
            Am = np.zeros((len(rest), T)); Am[np.arange(len(rest)), a] = 1
            W += hw @ Hm + (1 - hw) @ Am
            G += np.ones_like(hw) @ (Hm + Am)
        pct = W / np.maximum(G, 1)
        static = self._static_tiebreak(done) if rest.empty else np.zeros(T)
        tieb = np.tile(static, (n_sims, 1)) + rng.random((n_sims, T)) * 1e-6
        sim_ix = np.arange(n_sims)

        # Rank within a team subset: returns (n_sims, k) team idx, best first.
        def ranked(members):
            m = np.array(members)
            k = len(m)
            order = np.lexsort([(-tieb[:, m]).ravel(), (-pct[:, m]).ravel(), np.repeat(sim_ix, k)])
            return m[order.reshape(n_sims, k) % k]

        lg_all = ranked(range(T))
        by_conf = {c: ranked(np.where(self.conf == c)[0]) for c in ('East', 'West')}
        if self.season == 1997:  # conference winners take seeds 1-2
            winners = np.stack([by_conf['East'][:, 0], by_conf['West'][:, 0]], 1)
            wp = pct[sim_ix[:, None], winners] + tieb[sim_ix[:, None], winners]
            o = np.argsort(-wp, axis=1)
            top2 = np.take_along_axis(winners, o, 1)
            rest_ = np.array([[t for t in row if t not in set(tp)] for row, tp in zip(lg_all, top2)])
            lg_all = np.concatenate([top2, rest_], 1)
        # overall rank number per team (for home court + re-seeding)
        lg_rank = np.empty((n_sims, T), dtype=int)
        lg_rank[sim_ix[:, None], lg_all] = np.arange(T)[None, :]

        ps_by_pair = {}
        for r in self.ps[self.ps['date'] <= d].itertuples(index=False):
            ps_by_pair.setdefault(frozenset((r.home, r.away)), []).append(r.winner)

        self.used_actual = 0  # validation: real PS games the bracket consumed
        # Once the regular season is over the seeds are fixed; record them and
        # every matchup whose teams are settled (for the playoff odds tab).
        self.rs_complete = rest.empty
        self.seeds = {}
        if self.rs_complete:
            conf_fmt = any(sl[0] == 'conf' for m in self.bracket for sl in m[3:])
            if conf_fmt:
                for c, arr in by_conf.items():
                    for k, t in enumerate(arr[0]):
                        self.seeds[self.teams[t]] = f"{c[0]}{k + 1}"
            else:
                for k, t in enumerate(lg_all[0]):
                    self.seeds[self.teams[t]] = str(k + 1)
        self.matchups = []  # (round, best_of, team_a, team_b, [winners so far])
        reach = np.zeros((self.n_rounds + 2, T))  # [0]=playoffs, [k]=reach round k, [-1]=champ
        entered = np.zeros((n_sims, T), dtype=bool)
        res = {}

        def slot(s):
            if s[0] == 'conf':
                return by_conf[s[1]][:, s[2] - 1]
            if s[0] == 'lg':
                return lg_all[:, s[1] - 1]
            if s[0] == 'W':
                return res[s[1]]
            pool = np.stack([res[m] for m in s[1]], 1)            # 're-seed'
            o = np.argsort(lg_rank[sim_ix[:, None], pool], axis=1)
            return np.take_along_axis(pool, o, 1)[:, s[2] - 1]

        for mid, rnd, bo, sa, sb in self.bracket:
            a, b = slot(sa), slot(sb)
            for t, arr in ((a, 'a'), (b, 'b')):
                new = ~entered[sim_ix, t]
                entered[sim_ix, t] = True
                np.add.at(reach[0], t[new], 1)
                # A bye counts as getting through the rounds it skipped.
                for k in range(2, rnd):
                    np.add.at(reach[k], t[new], 1)
                np.add.at(reach[rnd], t, 1)
            a_better = lg_rank[sim_ix, a] < lg_rank[sim_ix, b]
            fixed = np.all(a == a[0]) and np.all(b == b[0])
            actual = ps_by_pair.get(frozenset((self.teams[a[0]], self.teams[b[0]])), []) if fixed else []
            if fixed and self.rs_complete:
                self.matchups.append((rnd, bo, self.teams[a[0]], self.teams[b[0]], list(actual[:bo])))
            need = bo // 2 + 1
            wa = np.zeros(n_sims, dtype=int); wb = np.zeros(n_sims, dtype=int)
            for gi, better_hosts in enumerate(host_pattern(self.season, bo)):
                if gi < len(actual):
                    won = np.full(n_sims, actual[gi] == self.teams[a[0]])
                    self.used_actual += 1
                else:
                    a_home = a_better if better_hosts else ~a_better
                    edge = np.where(a_home, hp, -hp)
                    won = rng.random(n_sims) < ndtr(A * (R[a] - R[b] + edge))
                live = (wa < need) & (wb < need)
                wa += won & live; wb += ~won & live
            res[mid] = np.where(wa >= need, a, b)
        np.add.at(reach[-1], res['F'], 1)
        reach /= n_sims
        cols = ['playoffs'] + [f'r{k}' for k in range(2, self.n_rounds + 1)] + ['champ']
        rows = np.vstack([reach[0]] + [reach[k] for k in range(2, self.n_rounds + 1)] + [reach[-1]])
        return pd.DataFrame(rows.T, index=self.teams, columns=cols)


def compute(games, ratings_df, rs_games_by_season, conf_of, log=print):
    """games: all WNBA games (season, date, home, away, home_pts, visitor_pts),
    Commissioner's Cup final excluded, scheduled games with NaN points.
    ratings_df: (season, date, name, rating). Returns (odds, brackets):
    odds = long DataFrame (season, date, team, playoffs, r2.., champ);
    brackets = {season: {date: (seeds, matchups)}} for every snapshot on or
    after the end of the regular season."""
    out = []
    brackets = {}
    for season, g in games.groupby('season'):
        season = int(season)
        rsub = ratings_df[ratings_df['season'] == season]
        ratings = {d: dict(zip(x['name'], x['rating'])) for d, x in rsub.groupby('date')}
        if not ratings:
            continue
        sim = SeasonSim(season, g, rs_games_by_season(season), conf_of, ratings)
        for d in sorted(ratings):
            o = sim.odds_at(d)
            if sim.rs_complete:
                brackets.setdefault(season, {})[d] = (dict(sim.seeds), list(sim.matchups))
            o.index.name = 'team'
            o = o.reset_index()
            o['season'] = season
            o['date'] = d
            out.append(o)
        log(f"  {season}: {len(ratings)} snapshots")
    return pd.concat(out, ignore_index=True), brackets
