import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import Holt
from arch import arch_model
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from backtester import BackTester

# ---------------------------------------------------------------------------
# Adjustable Parameters
# ---------------------------------------------------------------------------
BURN_IN = 250          # days of history required before we trade at all
REFIT_FREQ = 5         # refit ARIMA+GARCH every N days (expanding window)
ARIMA_ORDER = (0, 1, 0)  # random walk with drift. AIC on an early training
                       
THRESH_K = 0.1         # signal threshold = THRESH_K * GARCH forecast sigma
                       # chosen empirically: at k=0.1, threshold (~0.003-0.004 in
                       # log-return terms) lines up with the ~0.3% round-trip fee floor
ATR_STOP_MULT = 2.0    # trailing stop distance in multiples of ATR


def compute_atr(data, period=14):
    """Causal ATR: each value only uses rows <= i."""
    high, low, close = data['high'], data['low'], data['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def process_data(data):
    """
    Compute indicators needed for the strategy.
    Everything here is causal: value at row i only uses rows <= i.
    Box-Cox transform and Holt's linear trend are walk-forward fit models,
    so they are computed inside strat()'s loop, not here.
    """
    data = data.copy()
    close = data['close']

    data['ATR'] = compute_atr(data, period=14)
    data['log_close'] = np.log(close)
    data['log_ret'] = data['log_close'].diff()

    # Trend indicators
    data['SMA_50'] = close.rolling(50).mean()
    data['SMA_100'] = close.rolling(100).mean()
    data['SMA_200'] = close.rolling(200).mean()
    data['EMA_15'] = close.ewm(span=15, adjust=False).mean()

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26
    data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # RSI (14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    return data


def boxcox_scalar(x, lam):
    """Apply a known Box-Cox lambda to a single value (avoids retransforming
    the whole expanding history just to get the latest point)."""
    return (x ** lam - 1) / lam if lam != 0 else np.log(x)


class WalkForwardForecaster:
    """
    Maintains an expanding-window ARIMA(mean) + GARCH(1,1)(variance) model,
    refit every REFIT_FREQ days, and produces one-step-ahead forecasts.

    Uses a walk-forward Box-Cox transform (lambda re-estimated at every refit,
    using only data up to day i) instead of a fixed log transform. This
    generalizes log (lambda=0 is the special case) and lets the data pick the
    variance-stabilizing power rather than assuming it.

    Critically: fit(history) only ever sees data up to and including the
    current day i. It is called from strat() with data.loc[:i], never later.
    """
    def __init__(self, order=ARIMA_ORDER, refit_freq=REFIT_FREQ):
        self.order = order
        self.refit_freq = refit_freq
        self.days_since_refit = 0
        self.arima_res = None
        self.garch_res = None
        self.lam = None
        self.last_bc_level = None
        self.resid_scale = 1.0

    def _refit(self, close_history):
        # Box-Cox transform (requires strictly positive values -> true for price)
        bc_vals, lam = stats.boxcox(close_history.values)

        # ARIMA on the Box-Cox level series; order's d=1 handles differencing
        arima_model = ARIMA(bc_vals, order=self.order)
        new_arima_res = arima_model.fit()

        resid = new_arima_res.resid
        std = resid.std()
        new_resid_scale = 1.0 / std if std > 0 else 1.0
        scaled_resid = resid * new_resid_scale

        garch_model = arch_model(scaled_resid, vol='GARCH', p=1, q=1, mean='Zero', dist='normal')
        new_garch_res = garch_model.fit(disp='off', show_warning=False)

        # Only commit the refit if GARCH actually converged; otherwise keep the
        # previous model rather than trading on a volatility estimate we can't
        # trust. A stale-but-valid model is safer than a fresh-but-broken one.
        if new_garch_res.convergence_flag == 0 or self.garch_res is None:
            self.lam = lam
            self.last_bc_level = bc_vals[-1]
            self.arima_res = new_arima_res
            self.resid_scale = new_resid_scale
            self.garch_res = new_garch_res
        else:
            # keep old model, but still refresh last_bc_level under the old lambda
            self.last_bc_level = boxcox_scalar(close_history.values[-1], self.lam)

    def update_and_forecast(self, close_history):
        """
        close_history: pd.Series of raw close prices up to and including day i.
        Returns (mean_forecast, sigma_forecast) for day i+1, in Box-Cox-diff units.
        """
        needs_refit = (self.arima_res is None) or (self.days_since_refit >= self.refit_freq)

        if needs_refit:
            self._refit(close_history)
            self.days_since_refit = 0
        else:
            self.days_since_refit += 1
            # last_bc_level must track today's actual close under the current
            # lambda even between refits, so the "change" is measured correctly
            self.last_bc_level = boxcox_scalar(close_history.values[-1], self.lam)

        forecast_level = self.arima_res.forecast(steps=1)[0]
        mean_fc = forecast_level - self.last_bc_level  # implied 1-step change

        garch_fc = self.garch_res.forecast(horizon=1, reindex=False)
        sigma_fc = np.sqrt(garch_fc.variance.values[-1, 0]) / self.resid_scale

        return mean_fc, sigma_fc


class HoltForecaster:
    """
    Walk-forward Holt's linear trend (double exponential smoothing) model.
    Refit at the same cadence as the ARIMA-GARCH model. Used purely as a
    technical trend-confirmation vote, not as the primary forecast.
    """
    def __init__(self, refit_freq=REFIT_FREQ):
        self.refit_freq = refit_freq
        self.days_since_refit = 0
        self.res = None

    def update_and_forecast(self, close_history):
        needs_refit = (self.res is None) or (self.days_since_refit >= self.refit_freq)
        if needs_refit:
            model = Holt(close_history.values)
            self.res = model.fit(optimized=True)
            self.days_since_refit = 0
        else:
            self.days_since_refit += 1
        return self.res.forecast(1)[0]


def get_technical_view(row, holt_fc):
    """
    Returns a net vote in {-1, 0, +1} from 5 technical indicators, each voting
    +1 (bullish) or -1 (bearish). Acts as a confirmation filter for the
    statistical (ARIMA-GARCH) view, not an independent entry trigger.
    """
    votes = []

    # SMA alignment: bullish if price sits above a properly stacked rising
    # SMA_50 > SMA_100 > SMA_200; bearish if fully inverted; else neutral (0)
    if row['close'] > row['SMA_50'] > row['SMA_100'] > row['SMA_200']:
        votes.append(1)
    elif row['close'] < row['SMA_50'] < row['SMA_100'] < row['SMA_200']:
        votes.append(-1)
    else:
        votes.append(0)

    votes.append(1 if row['close'] > row['EMA_15'] else -1)
    votes.append(1 if row['MACD'] > row['MACD_signal'] else -1)
    votes.append(1 if row['RSI'] > 50 else -1)
    votes.append(1 if holt_fc > row['close'] else -1)

    score = sum(votes)
    if score > 0:
        return 1
    elif score < 0:
        return -1
    return 0


def strat(data):
    """
    Combined strategy:
      - Statistical view: Box-Cox ARIMA(mean) + GARCH(1,1)(variance), walk-forward.
        Signal fires only if the forecast exceeds a volatility-scaled threshold
        (fee-aware), giving direction + confidence.
      - Technical view: net vote from SMA(50/100/200) alignment, EMA-15,
        MACD, RSI, and Holt's linear trend -- a trend/momentum confirmation
        filter, not an independent trigger.
      - A trade is only taken when both views agree on direction.
      - An ATR-based trailing stop manages exits, tracked purely through
        emitted signals (see note below on why we avoid the SL column).
    """
    data['trade_type'] = "HOLD"
    data['signals'] = 0
    # NOTE: we deliberately do NOT use the TP/SL columns here. BackTester checks
    # those intrabar (via master_data high/low) *before* processing that day's
    # signal, which can close a position on a different day than our own
    # close-based stop check below would. Running both at once desyncs our
    # local `position` tracker from BackTester's actual position -> invalid
    # signal errors. So stops are managed purely through emitted signals,
    # matching the starter code's approach.

    position = 0          # 0 = flat, 1 = long, -1 = short
    trailing_stop = None

    forecaster = WalkForwardForecaster()
    holt = HoltForecaster()

    for i in range(len(data)):
        if i < BURN_IN:
            continue  # not enough history yet; stay flat

        close_history = data['close'].iloc[:i+1]  # up to & including day i, no lookahead
        mean_fc, sigma_fc = forecaster.update_and_forecast(close_history)
        holt_fc = holt.update_and_forecast(close_history)

        threshold = THRESH_K * sigma_fc
        if mean_fc > threshold:
            stat_view = 1
        elif mean_fc < -threshold:
            stat_view = -1
        else:
            stat_view = 0

        tech_view = get_technical_view(data.loc[i], holt_fc)

        # Only act when the statistical and technical views agree
        view = stat_view if (stat_view != 0 and stat_view == tech_view) else 0

        atr_i = data.loc[i, 'ATR']
        close_i = data.loc[i, 'close']

        # --- position transition logic, following the signal table ---
        if position == 0:
            if view == 1:
                data.loc[i, 'signals'] = 1
                data.loc[i, 'trade_type'] = "LONG"
                position = 1
                trailing_stop = close_i - ATR_STOP_MULT * atr_i
            elif view == -1:
                data.loc[i, 'signals'] = -1
                data.loc[i, 'trade_type'] = "SHORT"
                position = -1
                trailing_stop = close_i + ATR_STOP_MULT * atr_i

        elif position == 1:  # currently long
            stopped_out = close_i < trailing_stop
            if view == -1:
                data.loc[i, 'signals'] = -2
                data.loc[i, 'trade_type'] = "REVERSE_LONG_TO_SHORT"
                position = -1
                trailing_stop = close_i + ATR_STOP_MULT * atr_i
            elif stopped_out:
                data.loc[i, 'signals'] = -1
                data.loc[i, 'trade_type'] = "CLOSE_STOP"
                position = 0
                trailing_stop = None
            else:
                trailing_stop = max(trailing_stop, close_i - ATR_STOP_MULT * atr_i)

        elif position == -1:  # currently short
            stopped_out = close_i > trailing_stop
            if view == 1:
                data.loc[i, 'signals'] = 2
                data.loc[i, 'trade_type'] = "REVERSE_SHORT_TO_LONG"
                position = 1
                trailing_stop = close_i - ATR_STOP_MULT * atr_i
            elif stopped_out:
                data.loc[i, 'signals'] = 1
                data.loc[i, 'trade_type'] = "CLOSE_STOP"
                position = 0
                trailing_stop = None
            else:
                trailing_stop = min(trailing_stop, close_i + ATR_STOP_MULT * atr_i)

    return data


def make_trade_graph(bt, save_path="trade_graph.png"):
    """
    Candlestick-style price chart with shaded backgrounds showing when the
    strategy was long (green) or short (red). Reimplemented here because
    backtester.py's make_pnl_graph() (despite its name) only builds this
    chart, and main.py cannot modify backtester.py -- doing it here keeps
    the plotting logic fully under our control and dependency-light
    (matplotlib only, no plotly required).
    """
    data = bt.data
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(data.index, data['close'], color='black', linewidth=0.8, label='Close')

    for trade in bt.trades:
        color = 'green' if trade.qty > 0 else 'red'
        ax.axvspan(trade.init_timestamp, trade.final_timestamp, color=color, alpha=0.15)

    # currently open position (if any) extends shading to the end of the data
    if bt.position.qty != 0:
        color = 'green' if bt.position.qty > 0 else 'red'
        ax.axvspan(bt.position.timestamp, data.index[-1], color=color, alpha=0.15)

    ax.set_title("Trade Graph — BTC/USD Price with Strategy Positioning")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price (USD)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def make_pnl_graph_custom(bt, save_path="pnl_graph.png"):
    """
    Capital ($) vs close price over time. backtester.py's calc_capital()
    computes bt.data['capital'] for us (public method); we just plot it,
    since the actual plotting code for this chart is commented out in the
    provided backtester.py and we can't add it there.
    """
    bt.calc_capital()
    data = bt.data

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(data.index, data['capital'], color='blue', linewidth=1.2, label='Capital ($)')
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Capital ($)", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.plot(data.index, data['close'], color='gray', linewidth=0.8, alpha=0.7, label='BTC Close')
    ax2.set_ylabel("BTC/USD Close Price", color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    fig.suptitle("Capital vs. BTC/USD Close Price Over Time")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    data = pd.read_csv("impdata.csv")
    processed_data = process_data(data)
    result_data = strat(processed_data)
    result_data.to_csv("final_data.csv", index=False)

    bt = BackTester("BTC", signal_data_path="final_data.csv",
                     master_file_path="final_data.csv", compound_flag=1)
    bt.get_trades(1000)

    stats = bt.get_statistics()
    print("=== Performance ===")
    for key, val in stats.items():
        print(key, ":", val)

    # Lookahead bias spot-check: re-run the pipeline on a truncated prefix and
    # confirm the signal at that cutoff day doesn't change. Checking every
    # single row would re-run the full walk-forward loop O(n) times (very
    # slow); a spread of sample points is enough to catch a real leak.
    print("\nChecking for lookahead bias (sampled)...")
    lookahead_bias = False
    sample_points = range(BURN_IN + 10, len(data), max(1, (len(data) - BURN_IN) // 15))
    for i in sample_points:
        truncated = data.iloc[:i + 1].copy()
        truncated = process_data(truncated)
        truncated = strat(truncated)
        if truncated.loc[i, 'signals'] != result_data.loc[i, 'signals']:
            print(f"Lookahead bias detected at index {i}")
            lookahead_bias = True

    if not lookahead_bias:
        print("No lookahead bias detected in sampled indices.")

    print("\nGenerating graphs...")
    make_trade_graph(bt, save_path="trade_graph.png")
    make_pnl_graph_custom(bt, save_path="pnl_graph.png")
    print("Saved trade_graph.png and pnl_graph.png")


if __name__ == "__main__":
    main()

#arigato