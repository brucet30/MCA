"""
Minimum Correlation Algorithm - Dataset 2 Replication + Momentum Filter
Paper: "The Minimum Correlation Algorithm: A Practical Diversification Tool"
Varadi, Kapler, Bee, Rittenhouse (2012)

Dataset 2: SPY, QQQ, EEM, IWM, EFA, TLT, IYR, GLD
Out of Sample: June 2012 - March 2026

Momentum filter (MCA Mom / MCA2 Mom):
  Composite score = equal-weighted average of 12m, 6m, 3m, 1m total returns
  Select top 50% of assets (top 4 of 8) at each rebalance period
  Apply MCA / MCA2 to the selected subset only
  Assets not selected receive zero weight

--- SPYDER SETUP (do this once before running) ---
Plots appear in Spyder's Plots pane. If nothing shows up:
  1. Tools > Preferences > IPython console > Graphics
  2. Set Backend to: Inline
  3. Click OK, then: Consoles > Restart kernel
  4. Re-run the script

Do NOT manually call matplotlib.use() -- Spyder manages the backend.

Plots produced:
  Fig 1 - Cumulative equity curves (log scale)
  Fig 2 - Performance bar charts (Sharpe, CAGR, MaxDD, Volatility)
  Fig 3 - CDI component time series (D, RD, CDI) with EMA smoothing
  Fig 4 - Transition maps (stacked area allocation per strategy)
  Fig 5 - Average allocations grouped bar chart
  Fig 6 - Annual returns heatmap
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Spyder inline display
# --- Spyder display settings ---
# Plots appear in Spyder's Plots pane automatically when backend = Inline
# (the default). If plots are not showing, go to:
#   Tools > Preferences > IPython console > Graphics > Backend > Inline
# Then restart the kernel (Consoles > Restart kernel).
# Do NOT call matplotlib.use() in script code -- Spyder manages the backend.
plt.rcParams.update({
    'figure.dpi':       110,
    'figure.facecolor': 'white',
    'axes.facecolor':   '#f8f8f8',
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'grid.color':       '#cccccc',
    'font.size':        9,
})

# ==============================================================================
# CONFIG
# ==============================================================================

TICKERS    = ['SPY', 'QQQ', 'EEM', 'IWM', 'EFA', 'TLT', 'IYR', 'GLD']
#START_DATE = '2002-08-01'
START_DATE = '2012-06-01'
END_DATE   = '2026-03-31'
LOOKBACK   = 60        # daily returns ending on each Friday rebalance date
REBAL_FREQ = 'W-FRI'
RISK_FREE  = 0.0

# Strategy colour palette (roughly matching paper colours)
COLORS = {
    'Equal Weight': '#CC79A7',   # mauve/pink
    'Risk Parity':  '#56B4E9',   # sky blue
    'Min Variance': '#0072B2',   # blue
    'Max Div':      '#009E73',   # green
    'MCA':          '#D55E00',   # red-orange
    'MCA2':         '#000000',   # black
    'MCA Mom':      '#E69F00',   # amber
    'MCA2 Mom':     '#56B4E9',   # light blue -- overrides Risk Parity for clarity
}

TICKER_COLORS = [
    '#1F77B4','#FF7F0E','#2CA02C','#D62728',
    '#9467BD','#8C564B','#E377C2','#BCBD22',
]

# ==============================================================================
# DATA
# ==============================================================================

def get_prices(tickers, start, end):
    print(f"Downloading price data for {tickers}...")
    raw    = yf.download(tickers, start=start, end=end,
                         auto_adjust=True, progress=False)
    prices = raw['Close'][tickers]
    prices = prices.dropna(how='all')
    prices = prices.ffill()
    print(f"  Loaded {len(prices)} rows  "
          f"({prices.index[0].date()} to {prices.index[-1].date()})")
    print("  First valid date per ticker:")
    for t in tickers:
        fv = prices[t].first_valid_index()
        print(f"    {t:5s}: {fv.date() if fv is not None else 'n/a'}")
    return prices[tickers]


def _synthetic_prices(tickers, start, end):
    """Fallback synthetic data if yfinance unavailable."""
    np.random.seed(42)
    dates = pd.bdate_range(start=start, end=end)
    n, k  = len(dates), len(tickers)
    ann_rets  = [0.07, 0.09, 0.12, 0.08, 0.08, 0.08, 0.07, 0.12]
    ann_vols  = [0.18, 0.22, 0.25, 0.20, 0.20, 0.10, 0.25, 0.18]
    daily_mu  = [r/252 for r in ann_rets]
    daily_sig = [v/np.sqrt(252) for v in ann_vols]
    C = np.array([
        [1.00, 0.88, 0.82, 0.92, 0.88,-0.35, 0.78, 0.05],
        [0.88, 1.00, 0.78, 0.88, 0.82,-0.28, 0.70, 0.02],
        [0.82, 0.78, 1.00, 0.82, 0.88,-0.25, 0.72, 0.10],
        [0.92, 0.88, 0.82, 1.00, 0.85,-0.33, 0.80, 0.04],
        [0.88, 0.82, 0.88, 0.85, 1.00,-0.30, 0.75, 0.08],
        [-0.35,-0.28,-0.25,-0.33,-0.30, 1.00,-0.18, 0.15],
        [0.78, 0.70, 0.72, 0.80, 0.75,-0.18, 1.00, 0.10],
        [0.05, 0.02, 0.10, 0.04, 0.08, 0.15, 0.10, 1.00],
    ])
    L  = np.linalg.cholesky(C)
    dr = np.random.randn(n, k) @ L.T * daily_sig + daily_mu
    mask = (dates >= '2008-09-15') & (dates <= '2009-03-01')
    for col in [0,1,2,3,4,6]: dr[mask, col] -= 0.005
    dr[mask, 5] += 0.003
    prices = pd.DataFrame((1+dr).cumprod(axis=0)*100, index=dates, columns=tickers)
    print("  Using synthetic data (replace with real yfinance locally)")
    return prices


# ==============================================================================
# PORTFOLIO WEIGHT FUNCTIONS
# ==============================================================================

def _valid_hist(returns_df, lookback):
    hist       = returns_df.iloc[-lookback:].copy()
    valid_cols = hist.columns[hist.notna().all()].tolist()
    return hist[valid_cols].values, valid_cols


def _expand_weights(w_partial, valid_cols, all_cols):
    w_full = np.zeros(len(all_cols))
    for j, col in enumerate(valid_cols):
        w_full[list(all_cols).index(col)] = w_partial[j]
    return w_full


def equal_weight(returns, lookback):
    _, valid = _valid_hist(returns, lookback)
    w = np.zeros(len(returns.columns))
    for col in valid:
        w[list(returns.columns).index(col)] = 1.0 / len(valid)
    return w


def risk_parity(returns, lookback):
    hist, valid = _valid_hist(returns, lookback)
    vol = hist.std(axis=0)
    w_p = (1.0/vol) / (1.0/vol).sum()
    return _expand_weights(w_p, valid, returns.columns.tolist())


def min_variance(returns, lookback):
    hist, valid = _valid_hist(returns, lookback)
    cov = np.cov(hist.T)
    n   = len(valid)
    if n == 1:
        return _expand_weights(np.array([1.0]), valid, returns.columns.tolist())
    res = minimize(lambda w: w @ cov @ w, np.ones(n)/n, method='SLSQP',
                   bounds=[(0,1)]*n,
                   constraints=[{'type':'eq','fun':lambda w: w.sum()-1}],
                   options={'ftol':1e-12,'maxiter':1000})
    w_p = res.x if res.success else np.ones(n)/n
    return _expand_weights(w_p, valid, returns.columns.tolist())


def max_diversification(returns, lookback):
    hist, valid = _valid_hist(returns, lookback)
    vols = hist.std(axis=0)
    corr = np.corrcoef(hist.T)
    n    = len(valid)
    res  = minimize(lambda w: np.sqrt(w @ corr @ w), np.ones(n)/n,
                    method='SLSQP', bounds=[(0,1)]*n,
                    constraints=[{'type':'eq','fun':lambda w: w.sum()-1}],
                    options={'ftol':1e-12,'maxiter':1000})
    w   = res.x if res.success else np.ones(n)/n
    w   = w / vols
    w_p = w / w.sum()
    return _expand_weights(w_p, valid, returns.columns.tolist())


def min_corr(returns, lookback):
    """MCA (mincorr) - Varadi 2012."""
    hist, valid = _valid_hist(returns, lookback)
    vols = hist.std(axis=0)
    corr = np.corrcoef(hist.T)
    n    = len(valid)

    upper_idx = np.triu_indices(n, k=1)
    cor_vals  = corr[upper_idx]
    cor_mu    = cor_vals.mean()
    cor_sd    = cor_vals.std(ddof=1)

    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                adj[i, j] = 1.0 - norm.cdf(corr[i, j], cor_mu, cor_sd)

    row_means = adj.sum(axis=1) / (n-1)
    ranks     = pd.Series(row_means).rank(ascending=True)
    rank_w    = (ranks / ranks.sum()).values

    base_w = rank_w @ adj
    base_w = base_w / base_w.sum()
    w_p    = (base_w / vols)
    w_p    = w_p / w_p.sum()
    return _expand_weights(w_p, valid, returns.columns.tolist())


def min_corr2(returns, lookback):
    """MCA2 (mincorr2) - Varadi 2012."""
    hist, valid = _valid_hist(returns, lookback)
    vols = hist.std(axis=0)
    corr = np.corrcoef(hist.T)
    n    = len(valid)

    corr_nd   = corr.copy()
    np.fill_diagonal(corr_nd, 0.0)
    row_means = corr_nd.sum(axis=1) / (n-1)

    mu0, sd0 = row_means.mean(), row_means.std(ddof=1)
    wT       = 1.0 - norm.cdf(row_means, mu0, sd0)

    ranks  = pd.Series(wT).rank(ascending=True)
    rank_w = (ranks / ranks.sum()).values

    base_w = rank_w @ (1.0 - corr_nd)
    base_w = base_w / base_w.sum()
    w_p    = base_w / vols
    w_p    = w_p / w_p.sum()
    return _expand_weights(w_p, valid, returns.columns.tolist())



# ==============================================================================
# MOMENTUM FILTER
# ==============================================================================

MOM_PERIODS = [252, 126, 63, 21]   # 12m, 6m, 3m, 1m in trading days
MOM_TOP_N   = 4                     # top 50% of 8 assets

def momentum_score(prices, dt):
    """
    Composite momentum score at date dt.
    Equal-weighted average of 12m, 6m, 3m, 1m total returns.
    Returns a Series indexed by ticker, NaN if insufficient history.
    """
    pos = prices.index.searchsorted(dt)
    scores = {}
    for t in prices.columns:
        period_returns = []
        for lookback_days in MOM_PERIODS:
            start_pos = pos - lookback_days
            if start_pos < 0:
                period_returns.append(np.nan)
                continue
            p_end   = prices[t].iloc[pos]
            p_start = prices[t].iloc[start_pos]
            if pd.isna(p_end) or pd.isna(p_start) or p_start == 0:
                period_returns.append(np.nan)
            else:
                period_returns.append(p_end / p_start - 1)
        valid = [r for r in period_returns if not np.isnan(r)]
        scores[t] = np.mean(valid) if len(valid) >= 2 else np.nan
    return pd.Series(scores)


def momentum_filter(prices, dt, top_n=MOM_TOP_N):
    """
    Return list of top_n tickers by composite momentum at date dt.
    Falls back to all tickers if fewer than top_n have valid scores.
    """
    scores = momentum_score(prices, dt)
    valid  = scores.dropna()
    if len(valid) < top_n:
        return list(valid.index)
    return list(valid.nlargest(top_n).index)


def min_corr_mom(returns, lookback, prices=None, dt=None):
    """MCA with momentum pre-filter: top 50% of assets by composite momentum.
    Selects top MOM_TOP_N assets, runs MCA on that subset, then re-expands
    weights back to the full universe (unselected assets get zero weight).
    """
    all_cols = returns.columns.tolist()
    if prices is None or dt is None:
        return min_corr(returns, lookback)
    selected = momentum_filter(prices, dt)
    selected = [c for c in selected if c in returns.columns]
    if len(selected) < 2:
        return min_corr(returns, lookback)
    # run MCA on the filtered subset -- returns weights over selected cols only
    returns_filtered = returns[selected]
    w_partial = min_corr(returns_filtered, lookback)
    # re-expand to full universe: map selected col weights back to all_cols
    w_full = np.zeros(len(all_cols))
    for col, wt in zip(selected, w_partial[:len(selected)]):
        idx = all_cols.index(col)
        w_full[idx] = wt
    return w_full


def min_corr2_mom(returns, lookback, prices=None, dt=None):
    """MCA2 with momentum pre-filter: top 50% of assets by composite momentum.
    Selects top MOM_TOP_N assets, runs MCA2 on that subset, then re-expands
    weights back to the full universe (unselected assets get zero weight).
    """
    all_cols = returns.columns.tolist()
    if prices is None or dt is None:
        return min_corr2(returns, lookback)
    selected = momentum_filter(prices, dt)
    selected = [c for c in selected if c in returns.columns]
    if len(selected) < 2:
        return min_corr2(returns, lookback)
    returns_filtered = returns[selected]
    w_partial = min_corr2(returns_filtered, lookback)
    w_full = np.zeros(len(all_cols))
    for col, wt in zip(selected, w_partial[:len(selected)]):
        idx = all_cols.index(col)
        w_full[idx] = wt
    return w_full


# ==============================================================================
# BACKTEST ENGINE
# ==============================================================================

STRATEGIES = {
    'Equal Weight': equal_weight,
    'Risk Parity':  risk_parity,
    'Min Variance': min_variance,
    'Max Div':      max_diversification,
    'MCA':          min_corr,
    'MCA2':         min_corr2,
    'MCA Mom':      min_corr_mom,
    'MCA2 Mom':     min_corr2_mom,
}


def run_backtest(prices, lookback=LOOKBACK, rebal_freq=REBAL_FREQ):
    daily_returns = prices.pct_change().dropna(how='all')

    rebal_dates = daily_returns.resample(rebal_freq).last().index
    rebal_dates = rebal_dates[rebal_dates >= daily_returns.index[lookback]]

    all_weights = {
        name: pd.DataFrame(index=rebal_dates,
                            columns=prices.columns, dtype=float)
        for name in STRATEGIES
    }

    print(f"Running backtest: {len(rebal_dates)} rebalance periods "
          f"({rebal_dates[0].date()} to {rebal_dates[-1].date()})...")

    for dt in rebal_dates:
        pos     = daily_returns.index.searchsorted(dt)
        hist_df = daily_returns.iloc[max(0, pos - lookback + 1): pos + 1]
        if len(hist_df) < lookback // 2:
            continue
        for name, fn in STRATEGIES.items():
            try:
                # momentum strategies accept prices + dt for their filter
                import inspect
                sig = inspect.signature(fn)
                if 'prices' in sig.parameters:
                    w = fn(hist_df, lookback, prices=prices, dt=dt)
                else:
                    w = fn(hist_df, lookback)
                all_weights[name].loc[dt] = w
            except Exception:
                all_weights[name].loc[dt] = (
                    np.ones(len(prices.columns)) / len(prices.columns))

    equity_curves = {}
    for name, wdf in all_weights.items():
        wdf       = wdf.dropna(how='all').ffill()
        port_rets = []
        dates_out = []

        for i in range(len(rebal_dates) - 1):
            d0 = rebal_dates[i]
            d1 = rebal_dates[i + 1]
            w  = wdf.loc[d0].values.astype(float)
            if np.any(np.isnan(w)):
                continue

            pos0   = daily_returns.index.searchsorted(d0)
            pos1   = daily_returns.index.searchsorted(d1)
            period = daily_returns.iloc[pos0 + 1 : pos1 + 1]
            if len(period) == 0:
                continue

            valid_mask = period.notna().all()
            if not valid_mask.all():
                w_adj = w * valid_mask.values.astype(float)
                w_adj = w_adj / w_adj.sum() if w_adj.sum() > 0 else w
            else:
                w_adj = w

            daily = (period.fillna(0) * w_adj).sum(axis=1)
            port_rets.extend(daily.values)
            dates_out.extend(daily.index)

        eq = pd.Series(port_rets, index=dates_out)
        equity_curves[name] = (1 + eq).cumprod()

    return equity_curves, all_weights


# ==============================================================================
# METRICS
# ==============================================================================

def compute_metrics(equity_curves):
    results = {}
    for name, eq in equity_curves.items():
        rets   = eq.pct_change().dropna()
        n_yrs  = len(rets) / 252
        cagr   = (eq.iloc[-1] / eq.iloc[0]) ** (1/n_yrs) - 1
        vol    = rets.std() * np.sqrt(252)
        sharpe = (rets.mean()*252 - RISK_FREE) / vol
        dd     = (eq - eq.cummax()) / eq.cummax()
        results[name] = {
            'CAGR':   round(cagr*100, 2),
            'Sharpe': round(sharpe, 2),
            'Vol':    round(vol*100, 2),
            'MaxDD':  round(dd.min()*100, 2),
            'AvgDD':  round(dd[dd<0].mean()*100, 2),
        }
    return pd.DataFrame(results).T


def gini(values):
    v = np.sort(np.abs(values))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    idx = np.arange(1, n+1)
    return (2*(idx*v).sum() / (n*v.sum())) - (n+1)/n


def compute_weight_stats(all_weights):
    stats = {}
    for name, wdf in all_weights.items():
        wdf = wdf.dropna(how='all')
        stats[name] = {
            'Avg Holdings': round((wdf > 0.001).sum(axis=1).mean(), 1),
            'W Gini':       round(wdf.apply(lambda r: gini(r.values), axis=1).mean()*100, 1),
        }
    return pd.DataFrame(stats).T


def annual_returns_table(equity_curves):
    annual = {}
    for name, eq in equity_curves.items():
        yr = eq.resample('YE').last().pct_change().dropna() * 100
        yr.index = [pd.Timestamp(x).year for x in yr.index]
        annual[name] = yr.round(1)
    return pd.DataFrame(annual)


def avg_weights_table(all_weights):
    rows = {}
    for name, wdf in all_weights.items():
        rows[name] = wdf.dropna(how='all').mean().round(3)
    return pd.DataFrame(rows).T * 100


# ==============================================================================
# CDI COMPUTATION
# ==============================================================================

def compute_cdi(all_weights, prices, rebal_dates, ema_span=10):
    """
    D   = 1 - portfolio_vol / weighted_avg_asset_vol
    RD  = 1 - Gini(risk contributions)
    CDI = 0.5*D + 0.5*RD
    EMA-smoothed with span=10 periods (matching paper's '10 period EMA').
    """
    daily_returns = prices.pct_change().dropna(how='all')
    results       = {}

    for name, wdf in all_weights.items():
        wdf = wdf.dropna(how='all').ffill()
        D_vals, RD_vals, dates_out = [], [], []

        for dt in rebal_dates:
            if dt not in wdf.index:
                continue
            w = wdf.loc[dt].values.astype(float)
            if np.any(np.isnan(w)):
                continue

            pos   = daily_returns.index.searchsorted(dt)
            hist  = daily_returns.iloc[max(0, pos - LOOKBACK + 1): pos + 1]
            valid = hist.columns[hist.notna().all()].tolist()
            if len(valid) < 2:
                continue

            h       = hist[valid].values
            w_valid = np.array([w[list(daily_returns.columns).index(c)] for c in valid])
            if w_valid.sum() == 0:
                continue
            w_valid = w_valid / w_valid.sum()

            ann     = np.sqrt(252)
            vols    = h.std(axis=0) * ann
            corr    = np.corrcoef(h.T)
            cov     = corr * np.outer(vols, vols)

            port_vol    = np.sqrt(w_valid @ cov @ w_valid)
            wtd_avg_vol = (w_valid * vols).sum()
            D           = max(0, 1 - port_vol / wtd_avg_vol) if wtd_avg_vol > 0 else 0

            # marginal risk contributions
            if port_vol > 0:
                rc = w_valid * (cov @ w_valid) / port_vol
            else:
                rc = w_valid
            RD = max(0, 1 - gini(np.abs(rc)))

            D_vals.append(D)
            RD_vals.append(RD)
            dates_out.append(dt)

        if not dates_out:
            continue

        df          = pd.DataFrame({'D': D_vals, 'RD': RD_vals},
                                    index=pd.DatetimeIndex(dates_out))
        df['CDI']   = 0.5*df['D'] + 0.5*df['RD']
        results[name] = df.ewm(span=ema_span, adjust=False).mean()

    return results


# ==============================================================================
# PLOTS
# ==============================================================================

PLOT_ORDER = ['MCA2 Mom', 'MCA Mom', 'MCA2', 'MCA', 'Max Div', 'Min Variance', 'Risk Parity', 'Equal Weight']


def plot_equity_curves(equity_curves, perf):
    """Fig 1: Cumulative performance, log scale, matching paper Fig p.32."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle('Fig 1 — Cumulative Performance  (Dataset 2: 8 ETFs, Aug 2002–Jun 2012)',
                 fontsize=10, fontweight='bold')

    for name in PLOT_ORDER:
        if name not in equity_curves:
            continue
        eq  = equity_curves[name]
        lbl = f"{name}  {eq.iloc[-1]:.2f}"
        ax.semilogy(eq.index, eq.values, label=lbl,
                    color=COLORS.get(name, 'grey'), linewidth=1.5)

    ax.set_ylabel('Cumulative Performance (log scale)')
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.legend(fontsize=8.5, loc='upper left', framealpha=0.8)
    ax.set_xlim(list(equity_curves.values())[0].index[0],
                list(equity_curves.values())[0].index[-1])
    plt.tight_layout()
    plt.show()
    plt.pause(0.1)


def plot_performance_bars(perf):
    """Fig 2: Horizontal bar charts for Sharpe, CAGR, MaxDD, Volatility."""
    metrics = [
        ('Sharpe', True,  'Sharpe Ratio',    'higher is better →'),
        ('CAGR',   True,  'CAGR (%)',         'higher is better →'),
        ('MaxDD',  False, 'Max Drawdown (%)', '← lower is better'),
        ('Vol',    False, 'Volatility (%)',   '← lower is better'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle('Fig 2 — Performance Summary  (Dataset 2: 8 ETFs)',
                 fontsize=10, fontweight='bold')

    for ax, (metric, higher_better, title, xlabel) in zip(axes.flat, metrics):
        data   = perf[metric].reindex(
            [s for s in PLOT_ORDER if s in perf.index]
        ).sort_values(ascending=not higher_better)
        colors = [COLORS.get(n, 'grey') for n in data.index]
        bars   = ax.barh(range(len(data)), data.values, color=colors,
                         edgecolor='white', height=0.6)
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(data.index, fontsize=8.5)
        ax.set_title(title, fontsize=9.5, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=8)
        ax.axvline(0, color='black', linewidth=0.8)

        rng = abs(data.values).max()
        for bar, val in zip(bars, data.values):
            ax.text(bar.get_width() + rng * 0.02,
                    bar.get_y() + bar.get_height()/2,
                    f'{val:.2f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.show()
    plt.pause(0.1)


def plot_cdi(cdi_results):
    """Fig 3: Three-panel CDI time series matching paper p.34."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle('Fig 3 — Composite Diversification Indicator  (10-period EMA)',
                 fontsize=10, fontweight='bold')

    panels = [
        ('D',   'D = 1 – Portfolio Vol / Weighted Avg Asset Vol'),
        ('RD',  '1 – Gini(Risk Contributions)'),
        ('CDI', 'CDI = 0.5·D + 0.5·RD'),
    ]

    for ax, (col, title) in zip(axes, panels):
        for name in PLOT_ORDER:
            if name not in cdi_results:
                continue
            s   = cdi_results[name][col]
            avg = s.mean()
            ax.plot(s.index, s.values,
                    label=f'{name}  {avg:.2f}',
                    color=COLORS.get(name, 'grey'), linewidth=1.3)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8, loc='upper right', ncol=2, framealpha=0.8)

    idx0 = list(cdi_results.values())[0].index
    axes[-1].set_xlim(idx0[0], idx0[-1])
    plt.tight_layout()
    plt.show()
    plt.pause(0.1)


def plot_transition_maps(all_weights):
    """Fig 4: Stacked area allocation maps, one panel per strategy (paper p.45)."""
    n_strat = len(PLOT_ORDER)
    fig, axes = plt.subplots(n_strat, 1, figsize=(11, 11), sharex=True)
    fig.suptitle('Fig 4 — Transition Map: Portfolio Allocations',
                 fontsize=10, fontweight='bold')

    valid_tickers = [t for t in TICKERS
                     if any(t in aw.columns for aw in all_weights.values())]

    for ax, name in zip(axes, PLOT_ORDER):
        if name not in all_weights:
            ax.set_visible(False)
            continue
        wdf = all_weights[name].dropna(how='all').ffill().clip(lower=0) * 100
        ax.stackplot(
            wdf.index,
            [wdf[t].fillna(0).values for t in valid_tickers if t in wdf.columns],
            labels=valid_tickers,
            colors=TICKER_COLORS[:len(valid_tickers)],
            alpha=0.88,
        )
        ax.set_ylim(0, 100)
        ax.set_ylabel(name, fontsize=8, rotation=0, labelpad=65, va='center')
        ax.yaxis.set_label_position('right')
        ax.set_yticks([0, 50, 100])
        ax.set_yticklabels(['0', '50', '100%'], fontsize=7.5)
        ax.set_xlim(wdf.index[0], wdf.index[-1])

    # single legend above all panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=8,
               fontsize=8, bbox_to_anchor=(0.5, 1.0), framealpha=0.8)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
    plt.pause(0.1)


def plot_avg_allocations(all_weights):
    """Fig 5: Grouped bar chart of average allocations per strategy."""
    avg   = avg_weights_table(all_weights)
    order = [s for s in PLOT_ORDER if s in avg.index]
    avg   = avg.loc[order]

    fig, ax = plt.subplots(figsize=(11, 4))
    fig.suptitle('Fig 5 — Average Allocations by Strategy (%)',
                 fontsize=10, fontweight='bold')

    x     = np.arange(len(TICKERS))
    n_s   = len(order)
    width = 0.13
    offs  = np.linspace(-(n_s-1)*width/2, (n_s-1)*width/2, n_s)

    for i, name in enumerate(order):
        vals = [avg.loc[name, t] if t in avg.columns else 0 for t in TICKERS]
        ax.bar(x + offs[i], vals, width=width*0.9,
               label=name, color=COLORS.get(name, 'grey'),
               edgecolor='white', linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(TICKERS, fontsize=9)
    ax.set_ylabel('Average Weight (%)')
    ax.legend(fontsize=8.5, ncol=3, framealpha=0.8)
    ax.axhline(100/len(TICKERS), color='grey', linestyle='--',
               linewidth=0.8, label='Equal weight')
    plt.tight_layout()
    plt.show()
    plt.pause(0.1)


def plot_annual_returns_heatmap(equity_curves):
    """Fig 6: Annual returns heatmap (strategies × years)."""
    try:
        import seaborn as sns
        USE_SNS = True
    except ImportError:
        USE_SNS = False

    ann   = annual_returns_table(equity_curves)
    order = [s for s in PLOT_ORDER if s in ann.columns]
    ann   = ann[order].T      # strategies as rows, years as columns

    fig, ax = plt.subplots(figsize=(13, 4))
    fig.suptitle('Fig 6 — Annual Returns (%) by Strategy',
                 fontsize=10, fontweight='bold')

    vmax = max(abs(ann.values[~np.isnan(ann.values)]).max(), 1)

    if USE_SNS:
        sns.heatmap(
            ann, annot=True, fmt='.1f', center=0,
            cmap='RdYlGn', vmin=-vmax, vmax=vmax,
            linewidths=0.5, linecolor='white',
            ax=ax, annot_kws={'size': 8.5},
            cbar_kws={'shrink': 0.6, 'label': 'Return (%)'},
        )
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=9)
    else:
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im   = ax.imshow(ann.values, cmap='RdYlGn', norm=norm, aspect='auto')
        ax.set_xticks(range(ann.shape[1]))
        ax.set_xticklabels(ann.columns, fontsize=8.5)
        ax.set_yticks(range(ann.shape[0]))
        ax.set_yticklabels(ann.index, fontsize=8.5)
        for i in range(ann.shape[0]):
            for j in range(ann.shape[1]):
                val = ann.iloc[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                            fontsize=8, color='black')
        plt.colorbar(im, ax=ax, shrink=0.6, label='Return (%)')

    plt.tight_layout()
    plt.show()
    plt.pause(0.1)


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == '__main__':

    # --- data ---
    try:
        prices = get_prices(TICKERS, START_DATE, END_DATE)
    except Exception as e:
        print(f"yfinance failed ({e}), using synthetic data")
        prices = _synthetic_prices(TICKERS, START_DATE, END_DATE)

    # --- backtest ---
    equity_curves, all_weights = run_backtest(prices)

    # --- metrics ---
    perf = compute_metrics(equity_curves)
    perf_sorted = perf.sort_values('Sharpe', ascending=False)

    # --- console ---
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY  (Dataset 2: 8 ETFs, Aug 2002 - Jun 2012)")
    print("Paper targets: MCA Sharpe=1.34  MCA2=1.26  MinVar=1.41  MaxDiv=1.35")
    print("="*70)
    print(perf_sorted.to_string())

    print("\n" + "="*70)
    print("WEIGHT / DIVERSIFICATION STATS")
    print("="*70)
    print(compute_weight_stats(all_weights).to_string())

    print("\n" + "="*70)
    print("AVERAGE ALLOCATIONS (%)")
    print("="*70)
    print(avg_weights_table(all_weights).round(1).to_string())

    print("\n" + "="*70)
    print("ANNUAL RETURNS (%)")
    print("="*70)
    print(annual_returns_table(equity_curves).to_string())

    print("\n" + "="*70)
    print("TERMINAL EQUITY  -- paper: MCA=3.23, MCA2=3.10")
    print("="*70)
    for name, eq in sorted(equity_curves.items(),
                            key=lambda x: x[1].iloc[-1], reverse=True):
        print(f"  {name:<20} {eq.iloc[-1]:.3f}")

    # --- momentum filter stats ---
    print("\n" + "="*70)
    print("MOMENTUM FILTER STATS (MCA Mom / MCA2 Mom)")
    print("="*70)
    for mname in ['MCA Mom', 'MCA2 Mom']:
        if mname in all_weights:
            wdf = all_weights[mname].dropna(how='all')
            avg_held = (wdf > 0.001).sum(axis=1).mean()
            print(f"  {mname:<20}  Avg assets held: {avg_held:.1f}")
            # average allocation per ticker
            avg_alloc = (wdf > 0.001).mean() * 100
            top = avg_alloc.sort_values(ascending=False)
            print(f"  Selection frequency (% of periods):")
            for t, v in top.items():
                print(f"    {t:5s}: {v:.1f}%")

    # --- CDI ---
    rebal_dates = all_weights['MCA'].dropna(how='all').index
    cdi_results = compute_cdi(all_weights, prices, rebal_dates)

    print("\n" + "="*70)
    print("CDI SUMMARY (mean over full backtest)")
    print("="*70)
    for name in PLOT_ORDER:
        if name in cdi_results:
            df = cdi_results[name]
            print(f"  {name:<20}  D={df['D'].mean():.3f}  "
                  f"RD={df['RD'].mean():.3f}  CDI={df['CDI'].mean():.3f}")

    # --- plots ---
    print("\nGenerating plots (6 figures)...")
    plot_equity_curves(equity_curves, perf_sorted)
    plot_performance_bars(perf_sorted)
    plot_cdi(cdi_results)
    plot_transition_maps(all_weights)
    plot_avg_allocations(all_weights)
    plot_annual_returns_heatmap(equity_curves)
    print("All plots complete.")