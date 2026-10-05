"""Ассоциация свистов (verdict из предсказаний) с разрядами WWLLN на секундах, минутах и часах.

Используется в `wwlln_reader.ipynb`, выводы сведены в `stat_assoc_summary.ipynb`. Параметры
прогона — `AssocConfig`, расчёт — `run_association`, рисунки —
`plot_statistical_suite` / `plot_causal_suite`, отдельные числа — `lift_at`.

Поиск разрядов: окно по времени через searchsorted, расстояние — haversine.
Геометрия: 'satellite' — подспутниковая точка; 'conjugate' — грубый диполь (−lat, lon).
Ночь и список суток берутся из `wwlln_common`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from wwlln_common import (  # noqa: F401  (реэкспорт для ноутбука)
    NIGHT_SZA_DEG,
    PRED_MSK_OFFSET_HOURS,
    ae_path_for_day,
    analysis_days,
    list_pred_days,
    night_mask,
    pred_day_key_from_name,
    solar_zenith_deg,
)

R_EARTH_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Расстояние по большому кругу, км. lat1/lon1 — скаляры или массивы, lat2/lon2 — массивы, градусы."""
    lat1 = np.deg2rad(np.asarray(lat1, dtype=np.float64))
    lon1 = np.deg2rad(np.asarray(lon1, dtype=np.float64))
    lat2 = np.deg2rad(np.asarray(lat2, dtype=np.float64))
    lon2 = np.deg2rad(np.asarray(lon2, dtype=np.float64))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon * 0.5) ** 2
    return 2.0 * R_EARTH_KM * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def conjugate_latlon(lat, lon):
    """Грубая сопряжённая точка дипольного поля: (−lat, lon)."""
    return -np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64)


def apply_geometry(lat, lon, geometry: str):
    """Координаты, вокруг которых ищутся разряды: 'satellite' — как есть,
    'conjugate' — `conjugate_latlon`. Возвращает (lat, lon) float64."""
    g = geometry.lower().strip()
    if g in ("satellite", "sat", "footprint_proxy", "local"):
        return np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64)
    if g in ("conjugate", "conj", "dipole"):
        return conjugate_latlon(lat, lon)
    raise ValueError(f"Unknown geometry: {geometry}")


def read_pred_day(path: Path, sc: str, msk_offset_hours: float = PRED_MSK_OFFSET_HOURS) -> pd.DataFrame:
    """Файл предсказаний одних суток: timestamp, geo_lat, geo_lon, is_night, verdict, sc.
    Время МСК → UTC (минус msk_offset_hours)."""
    d = pd.read_csv(
        path,
        usecols=["timestamp", "geo_lat", "geo_lon", "is_night", "verdict"],
        dtype={
            "geo_lat": "float32",
            "geo_lon": "float32",
            "is_night": "int8",
            "verdict": "int8",
        },
    )
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=False) - pd.Timedelta(hours=msk_offset_hours)
    d["sc"] = sc
    return d


@lru_cache(maxsize=4)
def read_ae_day(path: Path, energy_min: Optional[float] = None) -> pd.DataFrame:
    """AE-файл одних суток: datetime (UTC), lat, lon, energy.

    energy_min — None: все разряды, включая energy == 0; число — только energy ≥ energy_min, Дж.
    Результат кэшируется на 4 файла (окно ±1 сутки едет по датам); возвращаемый
    кадр не изменять.
    """
    if not path.exists():
        return pd.DataFrame(columns=["datetime", "lat", "lon", "energy"])
    d = pd.read_csv(
        path,
        header=None,
        compression="gzip",
        usecols=[0, 1, 2, 3, 6],
        names=["date", "time", "lat", "lon", "energy"],
        dtype={
            "date": "string",
            "time": "string",
            "lat": "float32",
            "lon": "float32",
            "energy": "float32",
        },
    )
    d["datetime"] = pd.to_datetime(
        d["date"].astype(str).str.strip() + " " + d["time"].astype(str).str.strip(),
        errors="coerce",
    )
    d = d.drop(columns=["date", "time"]).dropna(subset=["datetime", "lat", "lon"])
    if energy_min is not None:
        d = d.loc[d["energy"].to_numpy() >= float(energy_min)]
    return d.reset_index(drop=True)


def load_ae_window(
    ae_dir: Path,
    day: str,
    pad_days: int = 1,
    energy_min: Optional[float] = None,
) -> pd.DataFrame:
    """Разряды за сутки day ('YYYYMMDD') ± pad_days, отсортированные по времени."""
    center = pd.Timestamp(day)
    parts = []
    for off in range(-pad_days, pad_days + 1):
        d = (center + pd.Timedelta(days=off)).strftime("%Y%m%d")
        parts.append(read_ae_day(ae_path_for_day(ae_dir, d), energy_min=energy_min))
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values("datetime").reset_index(drop=True)


@dataclass
class AssocConfig:
    """Параметры `run_association`.

    Данные и сутки:
      pred_dir, ae_dir — каталоги 'pred' и 'ae' из `wwlln_common.resolve_paths()`.
      sc_list — аппараты.
      day_list — явный список 'YYYYMMDD' (держит ka1 и ka2 на одних датах);
        None — все сутки с AE и предсказаниями.
      max_days — взять первые N суток после фильтра day_list; None — все.
      geometry — 'satellite' или 'conjugate' (см. `apply_geometry`).
      energy_min — только разряды с энергией ≥ energy_min, Дж; None — все.
      msk_offset_hours — сдвиг времени предсказаний к UTC.
      night_sza_deg — порог ночи для night_filter в `run_association`.
      lat_range — (lat_min, lat_max) подспутниковой точки, например полушарие.

    Секундный тест «этот разряд → этот свист» (каузальная часть):
      causal_r_km — радиус пар разряд–свист для гистограммы Δt.
      causal_tmax_s, causal_bin_s — окно ±T и шаг гистограммы Δt, с.
      radius_grid_km — радиусы для кривой excess(R): доля свистов с разрядом
        в (0, T] до свиста минус доля с разрядом в [−T, 0) после.
      max_whistlers_per_day — свистов на сутки, случайная выборка; None — все.

    Статистическая ассоциация (минуты–часы):
      stat_r_km, stat_dt_min — сетка R × Δt для lift: сэмпл «экспонирован»,
        если был разряд ближе R за Δt минут до него.
      max_points_per_day — сэмплов на сутки, случайная выборка; None — все.
      time_near_r_km — радиус второй кривой «P от времени с последней молнии»
        (первая — на max(stat_r_km)).

    Event study (молнии вокруг начала свиста):
      event_r_km — радиус; event_pre_h, event_post_h, event_bin_h — окно до и
        после начала свиста и шаг, ч.
      max_onsets_per_sc_day — начал свиста на аппарат и сутки.
      onset_max_gap_s — разрыв записи, после которого «0 → 1» не считается началом.

    Режимы: causal_only — только секундный тест; skip_causal — без него.
    rng_seed — зерно случайных выборок.
    """
    pred_dir: Path
    ae_dir: Path
    sc_list: tuple[str, ...] = ("ka1", "ka2")
    geometry: str = "satellite"
    energy_min: Optional[float] = None
    causal_r_km: float = 1500.0
    causal_tmax_s: float = 30.0
    causal_bin_s: float = 0.25
    # 10 км убран: на такой радиус экспонированных сэмплов единицы, оценивать нечем
    radius_grid_km: tuple[float, ...] = (50, 100, 250, 500, 1000, 1500, 2000)
    stat_r_km: tuple[float, ...] = (50, 100, 250, 500, 1000, 1500, 2000)
    stat_dt_min: tuple[float, ...] = (1, 5, 15, 30, 60, 120, 180)
    max_points_per_day: Optional[int] = 12000
    max_whistlers_per_day: Optional[int] = 4000
    rng_seed: int = 42
    event_r_km: float = 1500.0
    event_pre_h: float = 12.0
    event_post_h: float = 12.0
    event_bin_h: float = 1.0
    max_onsets_per_sc_day: int = 80
    max_days: Optional[int] = 14
    day_list: Optional[tuple[str, ...]] = None
    causal_only: bool = False
    skip_causal: bool = False
    msk_offset_hours: float = PRED_MSK_OFFSET_HOURS
    night_sza_deg: float = NIGHT_SZA_DEG
    onset_max_gap_s: float = 5.0
    lat_range: Optional[tuple[float, float]] = None
    time_near_r_km: float = 250.0


@dataclass
class AssocResult:
    """Результат `run_association`.

    config, days, meta — параметры, использованные сутки, фильтры.
    Секундный тест: causal_dt_edges_s, causal_hist_pairs — гистограмма Δt пар;
      n_whistlers; excess_r_km, matches_forward, matches_chance — свистов с
      разрядом до и после по радиусам.
    Lift: lift_R, lift_dt — сетка; n_exp_v1 / n_exp_v0 / n_no_v1 / n_no_v0 —
      сэмплов с verdict 1 / 0 среди экспонированных и неэкспонированных (R × Δt).
      Для чисел — `lift_matrix`, `risk_ratio_matrix`, `lift_at`.
    P от расстояния и времени: dist_edges_km, dist_n, dist_v1; time_edges_min,
      time_n, time_v1 (и *_near для time_near_r_km); n_far, v1_far — сэмплы
      без молний в окне.
    Event study: event_lag_h, event_lightning_mean, event_lightning_null, n_events.
    """
    config: AssocConfig
    days: list[str] = field(default_factory=list)
    causal_dt_edges_s: np.ndarray = field(default_factory=lambda: np.array([]))
    causal_hist_pairs: np.ndarray = field(default_factory=lambda: np.array([]))
    n_whistlers: int = 0
    excess_r_km: np.ndarray = field(default_factory=lambda: np.array([]))
    matches_forward: np.ndarray = field(default_factory=lambda: np.array([]))
    matches_chance: np.ndarray = field(default_factory=lambda: np.array([]))
    lift_R: np.ndarray = field(default_factory=lambda: np.array([]))
    lift_dt: np.ndarray = field(default_factory=lambda: np.array([]))
    n_exp_v1: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    n_exp_v0: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    n_no_v1: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    n_no_v0: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    dist_edges_km: np.ndarray = field(default_factory=lambda: np.array([]))
    dist_n: np.ndarray = field(default_factory=lambda: np.array([]))
    dist_v1: np.ndarray = field(default_factory=lambda: np.array([]))
    time_edges_min: np.ndarray = field(default_factory=lambda: np.array([]))
    time_n: np.ndarray = field(default_factory=lambda: np.array([]))
    time_v1: np.ndarray = field(default_factory=lambda: np.array([]))
    # «молний в R_max за окно не было вообще» — отдельная группа, а не крайний бин:
    # окно назад длится max(stat_dt_min) минут, поэтому бина «больше окна» не существует
    n_far: float = 0.0
    v1_far: float = 0.0
    # то же для time_near_r_km (обычно 250 км)
    time_near_r_km: float = 250.0
    time_n_near: np.ndarray = field(default_factory=lambda: np.array([]))
    time_v1_near: np.ndarray = field(default_factory=lambda: np.array([]))
    n_far_near: float = 0.0
    v1_far_near: float = 0.0
    event_lag_h: np.ndarray = field(default_factory=lambda: np.array([]))
    event_lightning_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    event_lightning_null: np.ndarray = field(default_factory=lambda: np.array([]))
    n_events: int = 0
    meta: dict = field(default_factory=dict)


def _sample_indices(n: int, k: Optional[int], rng: np.random.Generator) -> np.ndarray:
    if k is None or k >= n:
        return np.arange(n)
    return rng.choice(n, size=k, replace=False)


def _rate(v1: np.ndarray, v0: np.ndarray, min_n: float = 1.0) -> np.ndarray:
    """Доля verdict=1; NaN там, где наблюдений нет (иначе пустая ячейка выглядит нулём)."""
    n = v1 + v0
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(n >= min_n, v1 / np.where(n > 0, n, 1.0), np.nan)
    return out


def lift_matrix(res: AssocResult, min_n: float = 200.0) -> np.ndarray:
    """P(v=1|молния) / P(v=1|нет молнии) по сетке R×Δt.

    min_n: ячейки, где в какой-то из групп меньше min_n наблюдений, дают NaN.
    В углу сетки (max R, max Δt) «неэкспонированных» почти не остаётся,
    и без такого порога максимум по сетке ловит именно этот шум.
    """
    p_exp = _rate(res.n_exp_v1, res.n_exp_v0, min_n)
    p_no = _rate(res.n_no_v1, res.n_no_v0, min_n)
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = p_exp / p_no
    return np.where(np.isfinite(lift), lift, np.nan)


def risk_ratio_matrix(res: AssocResult, min_n: float = 200.0):
    """(P_exp, P_no, lift) по сетке R × Δt; min_n — как в `lift_matrix`."""
    return (_rate(res.n_exp_v1, res.n_exp_v0, min_n),
            _rate(res.n_no_v1, res.n_no_v0, min_n),
            lift_matrix(res, min_n))


def lift_at(res: AssocResult, r_km: float, dt_min: float) -> dict:
    """Lift в ЗАРАНЕЕ выбранной ячейке сетки.

    Максимум по сетке брать нельзя: это максимум из 49 коррелированных оценок,
    он смещён вверх и садится туда, где знаменатель меньше всего.
    """
    i = int(np.argmin(np.abs(res.lift_R - r_km)))
    j = int(np.argmin(np.abs(res.lift_dt - dt_min)))
    n_exp = res.n_exp_v1[i, j] + res.n_exp_v0[i, j]
    n_no = res.n_no_v1[i, j] + res.n_no_v0[i, j]
    p_exp = res.n_exp_v1[i, j] / n_exp if n_exp else np.nan
    p_no = res.n_no_v1[i, j] / n_no if n_no else np.nan
    return {
        "R_km": float(res.lift_R[i]), "dt_min": float(res.lift_dt[j]),
        "P_exp": float(p_exp), "P_no": float(p_no),
        "lift": float(p_exp / p_no) if p_no else np.nan,
        "n_exp": float(n_exp), "n_no": float(n_no),
    }


def _window_slice(ae_t: np.ndarray, t0: int, t_lo_s: float, t_hi_s: float):
    lo = t0 + int(t_lo_s * 1e9)
    hi = t0 + int(t_hi_s * 1e9)
    i0 = int(np.searchsorted(ae_t, lo, side="left"))
    i1 = int(np.searchsorted(ae_t, hi, side="right"))
    return i0, i1


def run_association(
    cfg: AssocConfig,
    night_filter: Optional[int] = None,
    sc_filter: Optional[str] = None,
    verbose: bool = True,
) -> AssocResult:
    """Ассоциация свистов с разрядами по параметрам cfg.

    night_filter — None: все сэмплы; 1 — только ночь, 0 — только день (зенитный
    угол на поверхности > cfg.night_sza_deg).
    sc_filter — один аппарат из cfg.sc_list и только его сутки; None — все из sc_list.
    Сутки: с AE и предсказаниями, затем cfg.day_list, затем первые cfg.max_days.
    Возвращает `AssocResult`.
    """
    rng = np.random.default_rng(cfg.rng_seed)
    day_map = list_pred_days(cfg.pred_dir, cfg.sc_list)
    days = sorted(d for d in day_map if ae_path_for_day(cfg.ae_dir, d).exists())
    if sc_filter:
        days = [d for d in days if sc_filter in day_map[d]]
    if cfg.day_list is not None:
        allowed = set(cfg.day_list)
        days = [d for d in days if d in allowed]
    if cfg.max_days is not None:
        days = days[: cfg.max_days]

    T = float(cfg.causal_tmax_s)
    edges = np.arange(-T, T + 1e-9, cfg.causal_bin_s)
    hist_pairs = np.zeros(len(edges) - 1, dtype=np.float64)
    n_wh = 0

    r_grid = np.asarray(cfg.radius_grid_km, dtype=np.float64)
    wh_with_fwd = np.zeros(len(r_grid), dtype=np.float64)
    wh_with_chance = np.zeros(len(r_grid), dtype=np.float64)

    R_list = np.asarray(cfg.stat_r_km, dtype=np.float64)
    dt_list = np.asarray(cfg.stat_dt_min, dtype=np.float64)
    nR, nDt = len(R_list), len(dt_list)
    n_exp_v1 = np.zeros((nR, nDt), dtype=np.float64)
    n_exp_v0 = np.zeros((nR, nDt), dtype=np.float64)
    n_no_v1 = np.zeros((nR, nDt), dtype=np.float64)
    n_no_v0 = np.zeros((nR, nDt), dtype=np.float64)

    # Бины «до ближайшей молнии» строятся ИЗ сетки R, а не задаются отдельно:
    # предфильтр по широте отбрасывает разряды дальше R_max, поэтому бин с
    # границей больше R_max наполнялся бы неверно.
    dist_edges = np.concatenate([[0.0], R_list, [np.inf]])
    dist_n = np.zeros(len(dist_edges) - 1, dtype=np.float64)
    dist_v1 = np.zeros(len(dist_edges) - 1, dtype=np.float64)
    # тоже из сетки: верхняя граница = длина окна поиска max(stat_dt_min),
    # бинов «дольше окна» не бывает
    time_edges = np.concatenate([[0.0], dt_list])
    time_n = np.zeros(len(time_edges) - 1, dtype=np.float64)
    time_v1 = np.zeros(len(time_edges) - 1, dtype=np.float64)
    n_far = 0.0   # молний в R_max за окно не было
    v1_far = 0.0
    R_near = float(cfg.time_near_r_km)
    time_n_near = np.zeros(len(time_edges) - 1, dtype=np.float64)
    time_v1_near = np.zeros(len(time_edges) - 1, dtype=np.float64)
    n_far_near = 0.0
    v1_far_near = 0.0

    lag_edges = np.arange(-cfg.event_pre_h, cfg.event_post_h + 1e-9, cfg.event_bin_h)
    lag_centers = 0.5 * (lag_edges[:-1] + lag_edges[1:])
    event_sum = np.zeros(len(lag_centers), dtype=np.float64)
    event_null_sum = np.zeros(len(lag_centers), dtype=np.float64)
    n_events = 0

    dt_sec_list = dt_list * 60.0
    dt_max_s = float(dt_list.max()) * 60.0
    R_max = float(R_list.max())
    sc_use = (sc_filter,) if sc_filter else cfg.sc_list

    for di, day in enumerate(days):
        frames = []
        for sc in sc_use:
            p = day_map[day].get(sc)
            if p is not None:
                frames.append(read_pred_day(p, sc, msk_offset_hours=cfg.msk_offset_hours))
        if not frames:
            continue
        pred = pd.concat(frames, ignore_index=True)
        if night_filter is not None:
            # ночь = SZA > порога на поверхности (единое определение по ноутбуку),
            # а не is_night из файла — там тень Земли на высоте орбиты, SZA > 117.6°
            ngt = night_mask(
                pred["timestamp"].to_numpy(dtype="datetime64[ns]"),
                pred["geo_lat"].to_numpy(), pred["geo_lon"].to_numpy(),
                sza_deg=cfg.night_sza_deg,
            )
            pred = pred.loc[ngt == night_filter].reset_index(drop=True)
        if cfg.lat_range is not None:
            lo_lat, hi_lat = cfg.lat_range
            gl = pred["geo_lat"].to_numpy()
            pred = pred.loc[(gl >= lo_lat) & (gl <= hi_lat)].reset_index(drop=True)
        if pred.empty:
            continue

        ae = load_ae_window(cfg.ae_dir, day, pad_days=1, energy_min=cfg.energy_min)
        if ae.empty:
            continue

        plat, plon = apply_geometry(
            pred["geo_lat"].to_numpy(), pred["geo_lon"].to_numpy(), cfg.geometry
        )
        alat = ae["lat"].to_numpy(dtype=np.float64)
        alon = ae["lon"].to_numpy(dtype=np.float64)
        ae_t = ae["datetime"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        pr_t = pred["timestamp"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        verdict = pred["verdict"].to_numpy(dtype=np.int8)

        # --- causal ---
        wh_idx = np.empty(0, dtype=np.int64)
        if not cfg.skip_causal:
            wh_idx = np.flatnonzero(verdict == 1)
            if cfg.max_whistlers_per_day is not None and wh_idx.size > cfg.max_whistlers_per_day:
                wh_idx = rng.choice(wh_idx, size=cfg.max_whistlers_per_day, replace=False)
            n_wh += int(wh_idx.size)

        for wi in wh_idx:
            t0 = int(pr_t[wi])
            i0, i1 = _window_slice(ae_t, t0, -T, T)
            if i1 <= i0:
                continue
            dkm = haversine_km(plat[wi], plon[wi], alat[i0:i1], alon[i0:i1])
            dt_s = (t0 - ae_t[i0:i1]).astype(np.float64) / 1e9

            in_r = dkm <= cfg.causal_r_km
            if in_r.any():
                hist_pairs += np.histogram(dt_s[in_r], bins=edges)[0]

            for ir, rmax in enumerate(r_grid):
                m = dkm <= rmax
                if not m.any():
                    continue
                dts = dt_s[m]
                if np.any((dts > 0.0) & (dts <= T)):
                    wh_with_fwd[ir] += 1.0
                if np.any((dts < 0.0) & (dts >= -T)):
                    wh_with_chance[ir] += 1.0

        if cfg.causal_only:
            if verbose:
                print(
                    f"  [{di + 1}/{len(days)}] {day}: pred={len(pred):,} ae={len(ae):,} "
                    f"wh_used={wh_idx.size:,} [causal_only]",
                    flush=True,
                )
            continue

        # --- statistical sample ---
        # выборка строго случайная: добавление лишних verdict=1 завышало бы P(v=1)
        idx = _sample_indices(len(pred), cfg.max_points_per_day, rng)

        for ii in idx:
            ii = int(ii)
            v = int(verdict[ii])
            t0 = int(pr_t[ii])
            # lookback only for stats; also allow tiny future for completeness not needed
            i0, i1 = _window_slice(ae_t, t0, -dt_max_s, 0.0)
            if i1 <= i0:
                dist_n[-1] += 1
                dist_v1[-1] += v
                n_far += 1
                v1_far += v
                n_far_near += 1
                v1_far_near += v
                if v == 1:
                    n_no_v1 += 1.0
                else:
                    n_no_v0 += 1.0
                continue

            # Предфильтр по широте: дуга большого круга >= 111.19 км × |Δlat|,
            # поэтому всё, что дальше R_max по широте, заведомо дальше R_max.
            # Отбрасывает ~2/3 разрядов до вызова haversine.
            lat_w = alat[i0:i1]
            keep = np.abs(lat_w - plat[ii]) <= (R_max / 111.19)
            if not keep.any():
                dist_n[-1] += 1
                dist_v1[-1] += v
                n_far += 1
                v1_far += v
                n_far_near += 1
                v1_far_near += v
                if v == 1:
                    n_no_v1 += 1.0
                else:
                    n_no_v0 += 1.0
                continue

            dkm = haversine_km(plat[ii], plon[ii], lat_w[keep], alon[i0:i1][keep])
            dt_s = (t0 - ae_t[i0:i1][keep]).astype(np.float64) / 1e9  # >= 0 in this window

            # «ближайшая молния» — среди тех, что были за окно lookback.
            # Отброшенные предфильтром лежат за R_max, т.е. в том же крайнем бине.
            nearest = float(dkm.min())
            db = int(np.searchsorted(dist_edges, nearest, side="right") - 1)
            db = min(max(db, 0), len(dist_n) - 1)
            dist_n[db] += 1
            dist_v1[db] += v

            # P vs время: две кривые — последняя молния в R_max и в R_near (250 км)
            near_max = dkm <= R_max
            if near_max.any():
                tb = int(np.searchsorted(time_edges, float(dt_s[near_max].min()) / 60.0,
                                         side="right") - 1)
                tb = min(max(tb, 0), len(time_n) - 1)
                time_n[tb] += 1
                time_v1[tb] += v
            else:
                n_far += 1
                v1_far += v
            near_250 = dkm <= R_near
            if near_250.any():
                tb = int(np.searchsorted(time_edges, float(dt_s[near_250].min()) / 60.0,
                                         side="right") - 1)
                tb = min(max(tb, 0), len(time_n_near) - 1)
                time_n_near[tb] += 1
                time_v1_near[tb] += v
            else:
                n_far_near += 1
                v1_far_near += v

            # Экспозиция на сетке R×Δt: ячейка (i, j) «экспонирована», если есть
            # хоть один разряд ближе R_i и свежее Δt_j. Считается двумерной
            # гистограммой с накопительной суммой — вместо 49 проходов по всем
            # разрядам окна получается два searchsorted и один bincount.
            ir_f = np.searchsorted(R_list, dkm, side="left")          # первый R >= dkm
            jd_f = np.searchsorted(dt_sec_list, dt_s, side="left")    # первый Δt >= dt_s
            hist = np.bincount(ir_f * (nDt + 1) + jd_f,
                               minlength=(nR + 1) * (nDt + 1)).reshape(nR + 1, nDt + 1)
            cnt = hist.cumsum(axis=0).cumsum(axis=1)[:nR, :nDt]
            exposed = cnt > 0
            if v == 1:
                n_exp_v1 += exposed
                n_no_v1 += ~exposed
            else:
                n_exp_v0 += exposed
                n_no_v0 += ~exposed

        # --- event study ---
        sc_arr = pred["sc"].astype(str).to_numpy()
        for sc in np.unique(sc_arr):
            msc = np.flatnonzero(sc_arr == sc)
            vv = verdict[msc]
            tt = pr_t[msc]
            la = plat[msc]
            lo = plon[msc]
            if vv.size < 2:
                continue
            # «начало свиста» — только если предыдущий сэмпл действительно рядом
            # по времени; иначе за onset принимался стык орбитального пропуска
            gap_ok = (tt[1:] - tt[:-1]) <= int(cfg.onset_max_gap_s * 1e9)
            onsets = np.flatnonzero((vv[1:] == 1) & (vv[:-1] == 0) & gap_ok) + 1
            if onsets.size > cfg.max_onsets_per_sc_day:
                onsets = rng.choice(onsets, size=cfg.max_onsets_per_sc_day, replace=False)
            pre_s = -cfg.event_pre_h * 3600.0
            post_s = cfg.event_post_h * 3600.0
            for oi in onsets:
                t_on = int(tt[oi])
                n_events += 1
                # реальное окно
                i0, i1 = _window_slice(ae_t, t_on, pre_s, post_s)
                if i1 > i0:
                    dkm = haversine_km(la[oi], lo[oi], alat[i0:i1], alon[i0:i1])
                    m = dkm <= cfg.event_r_km
                    if m.any():
                        lag_h = (ae_t[i0:i1][m] - t_on).astype(np.float64) / 1e9 / 3600.0
                        event_sum += np.histogram(lag_h, bins=lag_edges)[0]
                # null считается ДЛЯ КАЖДОГО события, даже если рядом не было молний:
                # иначе такие события попадали в знаменатель, но не в нулевую сумму,
                # и null оказывался занижен
                t_null = int(rng.integers(int(tt.min()), int(tt.max()) + 1))
                i0n, i1n = _window_slice(ae_t, t_null, pre_s, post_s)
                if i1n <= i0n:
                    continue
                dkm_n = haversine_km(la[oi], lo[oi], alat[i0n:i1n], alon[i0n:i1n])
                mn = dkm_n <= cfg.event_r_km
                if mn.any():
                    lag_null = (ae_t[i0n:i1n][mn] - t_null).astype(np.float64) / 1e9 / 3600.0
                    event_null_sum += np.histogram(lag_null, bins=lag_edges)[0]

        if verbose:
            print(
                f"  [{di + 1}/{len(days)}] {day}: pred={len(pred):,} ae={len(ae):,} "
                f"wh_used={wh_idx.size:,}",
                flush=True,
            )

    return AssocResult(
        config=cfg,
        days=days,
        causal_dt_edges_s=edges,
        causal_hist_pairs=hist_pairs,
        n_whistlers=n_wh,
        excess_r_km=r_grid,
        matches_forward=wh_with_fwd,
        matches_chance=wh_with_chance,
        lift_R=R_list,
        lift_dt=dt_list,
        n_exp_v1=n_exp_v1,
        n_exp_v0=n_exp_v0,
        n_no_v1=n_no_v1,
        n_no_v0=n_no_v0,
        dist_edges_km=dist_edges,
        dist_n=dist_n,
        dist_v1=dist_v1,
        time_edges_min=time_edges,
        time_n=time_n,
        time_v1=time_v1,
        n_far=n_far,
        v1_far=v1_far,
        time_near_r_km=R_near,
        time_n_near=time_n_near,
        time_v1_near=time_v1_near,
        n_far_near=n_far_near,
        v1_far_near=v1_far_near,
        event_lag_h=lag_centers,
        event_lightning_mean=event_sum / max(n_events, 1),
        event_lightning_null=event_null_sum / max(n_events, 1),
        n_events=n_events,
        meta={"night_filter": night_filter, "sc_filter": sc_filter, "geometry": cfg.geometry},
    )


def plot_causal_suite(res: AssocResult, title_prefix: str = "", show: bool = True):
    """Рисунки секундного теста: гистограмма Δt и excess(R).
    title_prefix — префикс заголовков; show — показать и закрыть. Возвращает список фигур."""
    import matplotlib.pyplot as plt

    figs = []
    pref = (title_prefix + " ").strip()
    if res.n_whistlers == 0:
        print(f"{pref}каузальная часть пропущена (skip_causal)")
        return figs

    edges = res.causal_dt_edges_s
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(centers, res.causal_hist_pairs, width=np.diff(edges), align="center", color="steelblue", alpha=0.85)
    ax.axvline(0.0, color="k", lw=1)
    ax.set_xlabel("Δt = t_whistler − t_lightning, s (UTC)")
    ax.set_ylabel("Число пар")
    ax.set_title(f"{pref}Каузальное окно: Δt (R≤{res.config.causal_r_km:.0f} км)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    figs.append(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    denom = max(res.n_whistlers, 1)
    excess = res.matches_forward - res.matches_chance
    ax.plot(res.excess_r_km, res.matches_forward / denom, "o-", label="молния до (0,T]")
    ax.plot(res.excess_r_km, res.matches_chance / denom, "s--", label="молния после [−T,0)")
    ax.plot(res.excess_r_km, excess / denom, "d-", color="crimson", label="excess")
    ax.set_xlabel("R, км")
    ax.set_ylabel("Доля whistler с ≥1 match")
    ax.set_title(f"{pref}Excess vs R (T={res.config.causal_tmax_s:.0f} с), N_wh={res.n_whistlers:,}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    figs.append(fig)

    if show:
        plt.show()
        for f in figs:
            plt.close(f)
    return figs


def _edge_labels(edges: np.ndarray, unit: str = "") -> list[str]:
    """Подписи бинов по границам; открытый последний бин — «≥ x»."""
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        if not np.isfinite(b):
            out.append(f"≥{a:g}")
        else:
            out.append(f"{a:g}–{b:g}")
    return out


def plot_statistical_suite(res: AssocResult, title_prefix: str = "", show: bool = True,
                           mark_cells: tuple = ()):
    """Рисунки статистической ассоциации: карты P_exp, P_no, lift по R × Δt (общая
    и свои шкалы), P от расстояния и от времени с последней молнии, event study.

    title_prefix — префикс заголовков; show — показать и закрыть;
    mark_cells — ячейки (R_км, Δt_мин), обводимые на картах. Возвращает список фигур.
    """
    import matplotlib.pyplot as plt

    figs = []
    pref = (title_prefix + " ").strip()

    p_exp, p_no, lift = risk_ratio_matrix(res)

    def _map(ax, mat, lab, lim=(None, None)):
        im = ax.imshow(mat, origin="lower", aspect="auto", cmap="viridis",
                       vmin=lim[0], vmax=lim[1])
        ax.set_xticks(range(len(res.lift_dt)))
        ax.set_xticklabels([str(int(x)) for x in res.lift_dt], rotation=45)
        ax.set_yticks(range(len(res.lift_R)))
        ax.set_yticklabels([str(int(x)) for x in res.lift_R])
        ax.set_xlabel("Δt lookback, мин")
        ax.set_ylabel("R, км")
        ax.set_title(lab)
        # origin='lower' → R растёт снизу вверх, малые радиусы в НИЖНЕМ левом углу
        for r_km, dt_min in mark_cells:
            i = int(np.argmin(np.abs(res.lift_R - r_km)))
            j = int(np.argmin(np.abs(res.lift_dt - dt_min)))
            ax.plot(j, i, marker="s", mfc="none", mec="white", mew=2, ms=14)
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Шкала общая для двух карт вероятностей: иначе одинаковый цвет означает
    # разные P и панели нельзя сравнивать глазом.
    pv = np.concatenate([p_exp[np.isfinite(p_exp)].ravel(), p_no[np.isfinite(p_no)].ravel()])
    p_lim = (float(pv.min()), float(pv.max())) if pv.size else (None, None)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    _map(axes[0], p_exp, "P(v=1 | lightning)", p_lim)
    _map(axes[1], p_no, "P(v=1 | no lightning)", p_lim)
    _map(axes[2], lift, "lift = P_exp / P_no")
    fig.suptitle(f"{pref}Статистическая ассоциация — общая шкала вероятности", y=1.02)
    fig.tight_layout()
    figs.append(fig)

    # То же с индивидуальными шкалами: общая шкала «сплющивает» контрольную карту,
    # и по ней не видно, как P(v=1|нет молнии) само меняется по сетке. Сравнивать
    # уровни между панелями здесь нельзя — только форму внутри каждой.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    _map(axes[0], p_exp, "P(v=1 | lightning), своя шкала")
    _map(axes[1], p_no, "P(v=1 | no lightning), своя шкала")
    fig.suptitle(f"{pref}То же с индивидуальными шкалами (сравнима только форма)", y=1.02)
    fig.tight_layout()
    figs.append(fig)

    look_h = float(np.max(res.lift_dt)) / 60.0
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # оси категориальные: бины неравной ширины, а последний по расстоянию — открытый
    d_lab = _edge_labels(res.dist_edges_km, "км")
    xd = np.arange(len(d_lab))
    axes[0].plot(xd, res.dist_v1 / np.maximum(res.dist_n, 1.0), "o-", color="darkorange")
    axes[0].set_xticks(xd)
    axes[0].set_xticklabels(d_lab, rotation=45, ha="right")
    axes[0].set_xlabel(f"Расстояние до ближайшей молнии за последние {look_h:.0f} ч, км")
    axes[0].set_title(f"{pref}P(whistler) vs расстояние")

    t_lab = _edge_labels(res.time_edges_min, "мин")
    xt = np.arange(len(t_lab))
    r_max = float(np.max(res.lift_R))
    r_near = float(res.time_near_r_km)
    axes[1].plot(xt, res.time_v1 / np.maximum(res.time_n, 1.0), "o-",
                 color="seagreen", label=f"R ≤ {r_max:.0f} км")
    if res.time_n_near.size:
        axes[1].plot(xt, res.time_v1_near / np.maximum(res.time_n_near, 1.0), "s--",
                     color="darkorange", label=f"R ≤ {r_near:.0f} км")
    if res.n_far > 0:
        p_far = res.v1_far / res.n_far
        axes[1].axhline(p_far, color="gray", ls=":", lw=1.2,
                        label=f"нет в {r_max:.0f} км: P={p_far:.3f}")
    if res.n_far_near > 0:
        p_far_n = res.v1_far_near / res.n_far_near
        axes[1].axhline(p_far_n, color="peru", ls=":", lw=1.2,
                        label=f"нет в {r_near:.0f} км: P={p_far_n:.3f}")
    axes[1].set_xticks(xt)
    axes[1].set_xticklabels(t_lab, rotation=45, ha="right")
    axes[1].set_xlabel(f"Минут с последней молнии (окно {look_h:.0f} ч)")
    axes[1].set_title(f"{pref}P(whistler) vs время с молнии")
    axes[1].legend(fontsize=8)

    for ax in axes:
        ax.set_ylabel("P(verdict=1)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    figs.append(fig)

    # Event study — это контроль, СОГЛАСОВАННЫЙ ПО МЕСТУ: null берётся в той же
    # точке, меняется только время. Поэтому география из сравнения выпадает, и
    # отношение real/null здесь — куда более честная величина, чем lift.
    span_h = res.config.event_pre_h + res.config.event_post_h
    to_min = span_h <= 2.0
    x = res.event_lag_h * (60.0 if to_min else 1.0)
    tot, tot_n = res.event_lightning_mean.sum(), res.event_lightning_null.sum()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, res.event_lightning_mean, "o-", label="вокруг onset verdict=1")
    ax.plot(x, res.event_lightning_null, "s--", label="null (то же место, случайное время)")
    ax.axvline(0.0, color="k", lw=1)
    ax.set_xlabel(f"{'Минуты' if to_min else 'Часы'} относительно onset verdict=1")
    ax.set_ylabel(f"Молний в R≤{res.config.event_r_km:.0f} км / событие")
    ax.set_title(f"{pref}Event study (N={res.n_events:,}, "
                 f"сумма real/null = {tot / max(tot_n, 1e-9):.2f})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    figs.append(fig)

    if show:
        plt.show()
        for f in figs:
            plt.close(f)
    return figs


def plot_association_suite(res: AssocResult, title_prefix: str = "", show: bool = True):
    """`plot_causal_suite` и `plot_statistical_suite` подряд."""
    figs = plot_causal_suite(res, title_prefix=title_prefix, show=False)
    figs.extend(plot_statistical_suite(res, title_prefix=title_prefix, show=False))
    if show:
        import matplotlib.pyplot as plt
        plt.show()
        for f in figs:
            plt.close(f)
    return figs
