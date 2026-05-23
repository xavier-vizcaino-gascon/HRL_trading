"""CryptoMarketEnvConsolidatedReward: entorn amb recompensa esparsa consolidada.

Variant de CryptoMarketEnv on la recompensa només es dona al tancar posicions:
  reward = log(equity_at_close / equity_at_entry)

Això obliga l'agent a avaluar trades holísticament en lloc d'optimitzar
per rewards densos pas a pas.

Ús:
    from src.rl.envs.crypto_market_env_consolidated_reward import (
        CryptoMarketEnvConsolidatedReward, make_env_consolidated
    )

    env = make_env_consolidated(df_train, feature_cols)
    obs, info = env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(action)
"""
import csv
from datetime import datetime
from enum import IntEnum
from pathlib import Path

import gymnasium as gym
import numpy as np
import polars as pl
from gymnasium import spaces

# ── Comissions Kraken Tipus B (tier 0 < $10k/30d) ────────────────────────────
TAKER_FEE            = 0.0040   # 0.40%
MAKER_FEE            = 0.0025   # 0.25%
MARGIN_OPENING_LONG  = 0.00025  # 0.025%
MARGIN_OPENING_SHORT = 0.00010  # 0.010%
ROLLOVER_LONG_4H     = 0.00025  # 0.025% / 4h
ROLLOVER_SHORT_4H    = 0.00010  # 0.010% / 4h
MAX_LEVERAGE         = 1
STEPS_PER_4H         = 4

# ── Defaults entorn ───────────────────────────────────────────────────────────
INITIAL_BALANCE    = 10_000.0
MAX_EPISODE_STEPS  = 2_016
MIN_EPISODE_STEPS  =   336

# ── Columnes del log CSV ──────────────────────────────────────────────────────
LOG_COLS = [
    "step", "episode_start", "action",
    "open", "high", "low", "close",
    "pos_side", "pos_qty", "entry_price", "cash", "equity",
    "unrealized_pnl", "max_equity", "cum_fees", "steps_in_pos",
    "n_trades", "realized_pnl",
    "r_pnl", "r_sortino", "r_cost", "r_dd", "r_hold", "r_realize", "reward",
    "c_pnl", "c_sortino", "c_cost", "c_dd",
]
_LOG_FLUSH_EVERY = 1_000

# ── Penalitzacions i thresholds ──────────────────────────────────────────────
HOLD_PENALTY          = -0.00002  # per step quan FLAT
DD_THRESHOLD          = 0.20      # drawdown màxim permès (20%)
DD_PENALTY            = -50.0     # penalització terminal per drawdown excessiu
OPEN_POSITION_PENALTY = -5.0      # penalització per posició oberta al truncar


class PositionSide(IntEnum):
    """Costat de la posició oberta."""
    SHORT = -1
    FLAT  =  0
    LONG  =  1


class CryptoMarketEnvConsolidatedReward(gym.Env):
    """
    Entorn Gymnasium amb recompensa esparsa consolidada al tancar posicions.

    Diferències respecte CryptoMarketEnv:
      - Reward = 0 a cada pas (+ hold_penalty si FLAT)
      - Reward = log(equity_close / equity_entry) només al tancar posició
      - Paràmetres eliminats: w_pnl, w_sortino, w_cost, w_dd, sortino_window,
        realize_bonus, realize_bonus_symmetric, realize_bonus_scale

    Args:
        df: DataFrame amb columnes Norm_* + open/high/low/close
        feature_cols: Llista de columnes Norm_* per a l'observació de l'agent
        initial_balance: Capital inicial en USD
        max_episode_steps: Màxim de passos per episodi
        variable_length: Si True, longitud variable entre min i max
        min_episode_steps: Longitud mínima (només actiu si variable_length=True)
        taker_fee: Fee taker (obertura + tancament)
        maker_fee: Fee maker (referència)
        margin_opening_long: Fee d'obertura de marge LONG
        margin_opening_short: Fee d'obertura de marge SHORT
        rollover_long_4h: Fee de rollover LONG cada 4h
        rollover_short_4h: Fee de rollover SHORT cada 4h
        max_leverage: Palanquejament màxim (1 = sense palanquejament)
        steps_per_4h: Passos per interval de 4h
        dd_threshold: Drawdown màxim permès (termination anticipada)
        dd_penalty: Penalització quan l'episodi termina per drawdown excessiu
        hold_penalty: Penalització per pas quan l'agent és FLAT
        open_position_penalty: Penalització quan l'episodi es trunca amb posició oberta
        seed: Seed per a reproducibilitat (None = no fix)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pl.DataFrame,
        feature_cols: list[str],
        initial_balance: float = INITIAL_BALANCE,
        max_episode_steps: int = MAX_EPISODE_STEPS,
        variable_length: bool = False,
        min_episode_steps: int = MIN_EPISODE_STEPS,
        taker_fee: float = TAKER_FEE,
        maker_fee: float = MAKER_FEE,
        margin_opening_long: float = MARGIN_OPENING_LONG,
        margin_opening_short: float = MARGIN_OPENING_SHORT,
        rollover_long_4h: float = ROLLOVER_LONG_4H,
        rollover_short_4h: float = ROLLOVER_SHORT_4H,
        max_leverage: int = MAX_LEVERAGE,
        steps_per_4h: int = STEPS_PER_4H,
        dd_threshold: float = DD_THRESHOLD,
        dd_penalty: float = DD_PENALTY,
        hold_penalty: float = HOLD_PENALTY,
        open_position_penalty: float = OPEN_POSITION_PENALTY,
        seed: int | None = None,
    ):
        super().__init__()

        # ── Dades ──────────────────────────────────────────────────────────
        self._feature_cols  = list(feature_cols)
        self._n_features    = len(feature_cols)
        self._n_rows        = len(df)

        arr = df.select(feature_cols).to_numpy().astype(np.float32)
        self._feats_arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)

        self._close_arr = df["close"].to_numpy().astype(np.float64)
        self._open_arr  = (
            df["open"].to_numpy().astype(np.float64)
            if "open" in df.columns
            else self._close_arr.copy()
        )
        self._high_arr  = (
            df["high"].to_numpy().astype(np.float64)
            if "high" in df.columns
            else self._close_arr.copy()
        )
        self._low_arr   = (
            df["low"].to_numpy().astype(np.float64)
            if "low" in df.columns
            else self._close_arr.copy()
        )

        # ── Configuració ───────────────────────────────────────────────────
        self._initial_balance      = float(initial_balance)
        self._max_episode_steps    = int(max_episode_steps)
        self._variable_length      = bool(variable_length)
        self._min_episode_steps    = int(min_episode_steps)
        self._current_episode_steps = int(max_episode_steps)
        self._taker_fee            = float(taker_fee)
        self._maker_fee            = float(maker_fee)
        self._margin_opening_long  = float(margin_opening_long)
        self._margin_opening_short = float(margin_opening_short)
        self._rollover_long_4h     = float(rollover_long_4h)
        self._rollover_short_4h    = float(rollover_short_4h)
        self._max_leverage         = float(max_leverage)
        self._steps_per_4h         = int(steps_per_4h)
        self._dd_threshold          = float(dd_threshold)
        self._dd_penalty            = float(dd_penalty)
        self._hold_penalty          = float(hold_penalty)
        self._open_position_penalty = float(open_position_penalty)
        self.fee_factor             = 1.0

        # ── Espais ─────────────────────────────────────────────────────────
        obs_dim = self._n_features + 6
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(4)

        if seed is not None:
            self.observation_space.seed(seed)
            self.action_space.seed(seed)

        # ── Logging CSV ────────────────────────────────────────────────────
        self._log_enabled   = False
        self._log_file      = None
        self._log_writer    = None
        self._log_row_count = 0

        self._initialized = False
        self._reset_state(episode_start=0)

    # ── Helpers de preu i features ─────────────────────────────────────────

    def _idx(self) -> int:
        return self._episode_start + self._step

    def _get_price(self) -> float:
        return float(self._close_arr[self._idx()])

    def _get_feats(self) -> np.ndarray:
        return self._feats_arr[self._idx()]

    def _equity(self, price: float | None = None) -> float:
        p = price if price is not None else self._get_price()
        return self._cash + self._pos_qty * p

    def _unrealized_pnl(self, price: float | None = None) -> float:
        if self._pos_side == PositionSide.FLAT:
            return 0.0
        p = price if price is not None else self._get_price()
        return self._pos_qty * (p - self._entry_price)

    # ── Reset de l'estat ──────────────────────────────────────────────────

    def _reset_state(self, episode_start: int = 0):
        self._step          = 0
        self._episode_start = episode_start
        self._cash          = self._initial_balance
        self._pos_qty       = 0.0
        self._pos_side      = PositionSide.FLAT
        self._entry_price   = 0.0
        self._steps_in_pos  = 0
        self._max_equity    = self._initial_balance
        self._cum_fees         = 0.0
        self._cum_trading_fees = 0.0
        self._cum_margin_fees  = 0.0
        self._n_trades      = 0
        self._n_wins        = 0
        self._realized_pnl  = 0.0
        self._returns_buf: list[float] = []
        self._equity_at_entry = 0.0  # CANVI: tracking per consolidated reward
        self._initialized   = True

    # ── Execució de trades ─────────────────────────────────────────────────

    def _open_long(self, price: float) -> float:
        """Compra màxim possible. Retorna fee pagat."""
        notional    = self._cash * self._max_leverage
        rate_total  = (self._taker_fee + self._margin_opening_long) * self.fee_factor
        qty         = notional / (price * (1.0 + rate_total))
        fee_trading = qty * price * self._taker_fee * self.fee_factor
        fee_margin  = qty * price * self._margin_opening_long * self.fee_factor
        fee         = fee_trading + fee_margin
        self._cash         -= qty * price + fee
        self._pos_qty       = qty
        self._pos_side      = PositionSide.LONG
        self._entry_price   = price
        self._steps_in_pos  = 0
        self._n_trades     += 1
        self._cum_fees         += fee
        self._cum_trading_fees += fee_trading
        self._cum_margin_fees  += fee_margin
        return fee

    def _close_long(self, price: float) -> tuple[float, float]:
        """Ven tota la posició llarga. Retorna (fee, pnl)."""
        qty      = self._pos_qty
        proceeds = qty * price
        fee      = proceeds * self._taker_fee * self.fee_factor
        pnl      = qty * (price - self._entry_price) - fee
        self._cash         += proceeds - fee
        self._pos_qty       = 0.0
        self._pos_side      = PositionSide.FLAT
        self._entry_price   = 0.0
        self._steps_in_pos  = 0
        self._cum_fees         += fee
        self._cum_trading_fees += fee
        self._realized_pnl += pnl
        if pnl > 0:
            self._n_wins += 1
        return fee, pnl

    def _open_short(self, price: float) -> float:
        """Obre posició curta. Retorna fee pagat."""
        notional    = self._cash * self._max_leverage
        qty         = notional / price
        fee_trading = notional * self._taker_fee * self.fee_factor
        fee_margin  = notional * self._margin_opening_short * self.fee_factor
        fee         = fee_trading + fee_margin
        self._cash         += notional - fee
        self._pos_qty       = -qty
        self._pos_side      = PositionSide.SHORT
        self._entry_price   = price
        self._steps_in_pos  = 0
        self._n_trades     += 1
        self._cum_fees         += fee
        self._cum_trading_fees += fee_trading
        self._cum_margin_fees  += fee_margin
        return fee

    def _close_short(self, price: float) -> tuple[float, float]:
        """Tanca posició curta. Retorna (fee, pnl)."""
        qty  = abs(self._pos_qty)
        cost = qty * price
        fee  = cost * self._taker_fee * self.fee_factor
        pnl  = qty * (self._entry_price - price) - fee
        self._cash         -= cost + fee
        self._pos_qty       = 0.0
        self._pos_side      = PositionSide.FLAT
        self._entry_price   = 0.0
        self._steps_in_pos  = 0
        self._cum_fees         += fee
        self._cum_trading_fees += fee
        self._realized_pnl += pnl
        if pnl > 0:
            self._n_wins += 1
        return fee, pnl

    def _apply_rollover(self) -> float:
        """Aplica rollover fee cada steps_per_4h passos. Retorna fee pagat."""
        if self._pos_side == PositionSide.FLAT:
            return 0.0
        if self._steps_in_pos > 0 and self._steps_in_pos % self._steps_per_4h == 0:
            notional = abs(self._pos_qty) * self._get_price()
            rate = (self._rollover_long_4h if self._pos_side == PositionSide.LONG
                    else self._rollover_short_4h)
            fee = notional * rate * self.fee_factor
            self._cash             -= fee
            self._cum_fees         += fee
            self._cum_margin_fees  += fee
            return fee
        return 0.0

    # ── Observació i recompensa ────────────────────────────────────────────

    def _internal_state(self, price: float) -> np.ndarray:
        """Calcula l'estat intern de 6 dimensions."""
        equity = self._equity(price)
        upnl   = self._unrealized_pnl(price)

        pos_side    = float(self._pos_side)
        upnl_norm   = upnl / (abs(equity) + 1e-8)
        eq_norm     = equity / self._initial_balance
        dd          = (equity - self._max_equity) / (self._max_equity + 1e-8)
        dd          = min(0.0, dd)
        tip_norm    = self._steps_in_pos / (self._current_episode_steps + 1e-8)
        fees_norm   = self._cum_fees / (self._initial_balance + 1e-8)

        return np.array(
            [pos_side, upnl_norm, eq_norm, dd, tip_norm, fees_norm],
            dtype=np.float32,
        )

    def _get_observation(self, price: float) -> np.ndarray:
        feats    = self._get_feats()
        intstate = self._internal_state(price)
        return np.concatenate([feats, intstate]).astype(np.float32)

    def _compute_reward(
        self, price: float, fee_step: float, prev_equity: float,
        closed_position: bool = False,
    ) -> tuple[float, dict]:
        """Recompensa esparsa consolidada: 0 a cada pas excepte CLOSE.

        r_pnl es calcula per logging però NO contribueix al reward.
        closed_position=True evita aplicar hold_penalty al pas de CLOSE,
        ja que la transició a FLAT és acció activa, no inacció.
        """
        equity = self._equity(price)

        # r_pnl per logging (compatibilitat amb CSV)
        r_pnl = float(np.log(max(equity, 1e-8) / max(prev_equity, 1e-8)))
        self._returns_buf.append(r_pnl)
        if len(self._returns_buf) > 50:
            self._returns_buf.pop(0)

        # Hold penalty: única recompensa no-esparsa (no s'aplica al pas de CLOSE)
        r_hold = (
            self._hold_penalty
            if self._pos_side == PositionSide.FLAT and not closed_position
            else 0.0
        )
        reward = r_hold

        reward_info = {
            "r_pnl": r_pnl, "r_sortino": 0.0, "r_cost": 0.0, "r_dd": 0.0,
            "r_hold": r_hold, "r_realize": 0.0,
            "c_pnl": 0.0, "c_sortino": 0.0, "c_cost": 0.0, "c_dd": 0.0,
        }
        return reward, reward_info

    # ── Logging CSV ────────────────────────────────────────────────────────

    def enable_log(self, project: str, log_dir: str | Path = "results_sparse") -> Path:
        """Activa el logging CSV per-step."""
        self.disable_log()
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        ts   = datetime.now().strftime("%H%M-%y%m%d")
        path = log_dir / f"{ts}_{project}_log.csv"
        self._log_file      = open(path, "w", newline="", encoding="utf-8")  # noqa: SIM115
        self._log_writer    = csv.writer(self._log_file)
        self._log_writer.writerow(LOG_COLS)
        self._log_enabled   = True
        self._log_row_count = 0
        return path

    def disable_log(self) -> None:
        """Desactiva el logging i tanca el fitxer CSV."""
        self._log_enabled = False
        if self._log_file is not None:
            try:
                self._log_file.flush()
                self._log_file.close()
            except Exception:
                pass
            self._log_file   = None
            self._log_writer = None

    def _log_step(self, action: int, price: float, reward: float, ri: dict) -> None:
        """Escriu una fila al log CSV amb l'estat complet del pas actual."""
        if not self._log_enabled or self._log_writer is None:
            return
        idx = self._idx()
        self._log_writer.writerow([
            self._step,
            self._episode_start,
            action,
            float(self._open_arr[idx]),
            float(self._high_arr[idx]),
            float(self._low_arr[idx]),
            float(self._close_arr[idx]),
            int(self._pos_side),
            self._pos_qty,
            self._entry_price,
            self._cash,
            self._equity(price),
            self._unrealized_pnl(price),
            self._max_equity,
            self._cum_fees,
            self._steps_in_pos,
            self._n_trades,
            self._realized_pnl,
            ri["r_pnl"],
            ri["r_sortino"],
            ri["r_cost"],
            ri["r_dd"],
            ri["r_hold"],
            ri["r_realize"],
            reward,
            ri["c_pnl"],
            ri["c_sortino"],
            ri["c_cost"],
            ri["c_dd"],
        ])
        self._log_row_count += 1
        if self._log_row_count % _LOG_FLUSH_EVERY == 0:
            self._log_file.flush()

    # ── API Gymnasium ──────────────────────────────────────────────────────

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset de l'entorn."""
        super().reset(seed=seed)

        rng = np.random.default_rng(seed)

        if self._variable_length:
            self._current_episode_steps = int(rng.integers(
                self._min_episode_steps,
                self._max_episode_steps + 1,
            ))
        else:
            self._current_episode_steps = self._max_episode_steps

        if options is not None and "episode_start" in options:
            start = int(options["episode_start"])
        else:
            max_start = max(1, self._n_rows - self._current_episode_steps - 1)
            start     = int(rng.integers(0, max_start))

        self._current_episode_steps = min(
            self._current_episode_steps,
            self._n_rows - start - 1,
        )

        self._reset_state(episode_start=start)

        if options is not None and "initial_equity" in options:
            self._cash     = float(options["initial_equity"])
            self._max_equity = self._cash

        price = self._get_price()
        obs   = self._get_observation(price)
        info  = {
            "equity":           self._equity(price),
            "cash":             self._cash,
            "pos_qty":          self._pos_qty,
            "pos_side":         int(self._pos_side),
            "fee_step":         0.0,
            "cum_fees":         self._cum_fees,
            "cum_trading_fees": self._cum_trading_fees,
            "cum_margin_fees":  self._cum_margin_fees,
            "n_trades":         self._n_trades,
            "realized_pnl":     self._realized_pnl,
            "step":             self._step,
            "current_price":    price,
        }
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Executa un pas.

        Args:
            action: 0=HOLD, 1=GO_LONG, 2=GO_SHORT, 3=CLOSE

        Returns:
            Tuple (obs, reward, terminated, truncated, info)
        """
        assert self._initialized, "Cal cridar reset() primer"
        assert 0 <= action <= 3, f"Acció invàlida: {action}"

        price       = self._get_price()
        prev_equity = self._equity(price)
        fee_step    = 0.0

        # ── Executa acció ────────────────────────────────────────────────
        _opened_position  = False
        _closed_position  = False

        if action == 0:
            pass  # HOLD

        elif action == 1:  # GO_LONG
            if self._pos_side == PositionSide.FLAT:
                fee_step += self._open_long(price)
                _opened_position = True

        elif action == 2:  # GO_SHORT
            if self._pos_side == PositionSide.FLAT:
                fee_step += self._open_short(price)
                _opened_position = True

        elif action == 3:  # CLOSE
            if self._pos_side == PositionSide.LONG:
                fee, _ = self._close_long(price)
                fee_step += fee
                _closed_position = True
            elif self._pos_side == PositionSide.SHORT:
                fee, _ = self._close_short(price)
                fee_step += fee
                _closed_position = True

        # ── Rollover ─────────────────────────────────────────────────────
        if self._pos_side != PositionSide.FLAT:
            self._steps_in_pos += 1
        fee_step += self._apply_rollover()

        # ── Avança pas i actualitza màxim d'equity ───────────────────────
        self._step += 1
        price  = self._get_price()
        equity = self._equity(price)

        # Snapshot equity_at_entry DESPRÉS d'avançar el pas, consistent amb
        # el valor que info['equity'] retorna al agent en el pas GO_LONG/SHORT.
        if _opened_position:
            self._equity_at_entry = equity
        if equity > self._max_equity:
            self._max_equity = equity

        # ── Recompensa ───────────────────────────────────────────────────
        reward, reward_info = self._compute_reward(price, fee_step, prev_equity, _closed_position)

        # CANVI: Recompensa consolidada al tancar posició
        if action == 3 and self._equity_at_entry > 0:
            equity_after_close = self._equity(price)
            r_trade = float(np.log(
                max(equity_after_close, 1e-8) / max(self._equity_at_entry, 1e-8)
            ))
            reward += r_trade
            reward_info["r_realize"] = r_trade

        self._log_step(action, price, reward, reward_info)

        # ── Terminació ───────────────────────────────────────────────────
        truncated  = self._step >= self._current_episode_steps
        dd_pct     = (self._max_equity - equity) / (self._max_equity + 1e-8)
        terminated = bool(dd_pct > self._dd_threshold)
        if terminated:
            reward += self._dd_penalty
        if truncated and self._pos_side != PositionSide.FLAT:
            reward += self._open_position_penalty

        obs  = self._get_observation(price)
        info = {
            "equity":           equity,
            "cash":             self._cash,
            "pos_qty":          self._pos_qty,
            "pos_side":         int(self._pos_side),
            "unrealized_pnl":   self._unrealized_pnl(price),
            "fee_step":         fee_step,
            "cum_fees":         self._cum_fees,
            "cum_trading_fees": self._cum_trading_fees,
            "cum_margin_fees":  self._cum_margin_fees,
            "n_trades":         self._n_trades,
            "realized_pnl":     self._realized_pnl,
            "step":             self._step,
            "current_price":    price,
            **reward_info,
        }
        return obs, reward, terminated, truncated, info

    def render(self, mode: str = "human"):
        price  = self._get_price()
        equity = self._equity(price)
        print(
            f"Step {self._step:5d} | ${price:>10.2f} | "
            f"Equity ${equity:>10.2f} | {self._pos_side.name:5s} "
            f"({self._pos_qty:+.4f}) | Cash ${self._cash:>10.2f}"
        )

    def close(self):
        self.disable_log()

    def __del__(self):
        if hasattr(self, "_log_file"):
            self.disable_log()


def steps_per_4h_for(timeframe: str) -> int:
    """Retorna el nombre de passos per interval de 4h per al timeframe donat."""
    minutes = int(timeframe.replace('m', ''))
    return max(1, 4 * 60 // minutes)


def make_env_consolidated(
    df: pl.DataFrame,
    feature_cols: list[str],
    initial_balance: float = INITIAL_BALANCE,
    max_episode_steps: int = MAX_EPISODE_STEPS,
    variable_length: bool = False,
    min_episode_steps: int = MIN_EPISODE_STEPS,
    timeframe: str | None = None,
    **kwargs,
) -> CryptoMarketEnvConsolidatedReward:
    """Factoria per crear CryptoMarketEnvConsolidatedReward.

    Args:
        df: DataFrame amb columnes Norm_* + open/high/low/close
        feature_cols: Llista de columnes Norm_* (observació de l'agent)
        initial_balance: Capital inicial en USD
        max_episode_steps: Màxim de passos per episodi
        variable_length: Si True, longitud variable entre min i max
        min_episode_steps: Longitud mínima (només actiu si variable_length=True)
        timeframe: Timeframe de les dades (p.ex. '60m')
        **kwargs: Paràmetres addicionals

    Returns:
        Instància de CryptoMarketEnvConsolidatedReward
    """
    if timeframe is not None and 'steps_per_4h' not in kwargs:
        kwargs['steps_per_4h'] = steps_per_4h_for(timeframe)
    return CryptoMarketEnvConsolidatedReward(
        df=df,
        feature_cols=feature_cols,
        initial_balance=initial_balance,
        max_episode_steps=max_episode_steps,
        variable_length=variable_length,
        min_episode_steps=min_episode_steps,
        **kwargs,
    )
