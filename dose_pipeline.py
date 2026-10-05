"""Пайплайн «доза–эффект»: сетка → доза → страты → накопители → анализ Мантеля–Хензеля.

Порядок работы: `run_accumulation` (или `sweep_r_dt`, `sweep_night_sza`,
`run_median_energy_lon`) копит экспозицию и число свистов по стратам и бинам
дозы; `analyze` / `analyze_energy_lon` строят кривую с блочным бутстрапом по
суткам; `report_bin` выбирает отчётный бин.

Общие аргументы накопителей:
  df, df_pred — молнии и предсказания из `wwlln_common.ensure_data`; в df_pred
    только аппараты из sc_list, иначе ValueError.
  sc_list — аппараты; позиция в кортеже — индекс `sc` / `si` в анализе.
  acc_days — номера суток (`wwlln_common.day_codes`), уже покрытые и df, и df_pred.
  dose_r_km, dose_dt_min — радиус (км) и окно (мин) дозы: молнии в окрестности
    ±R по широте и долготе (в ячейках сетки) за Δt минут до наблюдения,
    включая текущую минуту.
  glat, glon, gdt_min — шаг сетки, градусы и минуты; Δt должно делиться на gdt_min.
  null_lags — сутки назад для null-модели (доза из чужих суток); () — null выключен,
    тогда `*_null_d` нулевые и `rr_n` в анализе NaN.
  null_symmetric — лаги ещё и вперёд (day + L); null_season_safe — пропускать лаг
    из другого сезона.
  night_sza_deg — порог ночи (зенитный угол на поверхности); None — `wc.NIGHT_SZA_DEG`.
  strata_scheme — 'lst': широта × местное время × сезон × ночь;
    'lon': широта × долгота × сезон × ночь.
  n_latb, n_hourb, n_lonb — число полос широты, местного времени ('lst'), долготы ('lon').
  geometry — где ищется доза: 'satellite' — под спутником; 'footprint' — след
    силовой линии на 100 км в том же полушарии; 'conjugate' — в противоположном
    (AACGM-v2, `mag_coords.footprint_conjugate_arr`).
  sat_height_km — высота спутника для трассировки.
  strata_by_geometry — True: страты и ночь считаются в той же точке, что доза;
    False: страты у подспутниковой точки.
  dose_edges — границы бинов дозы; None — `DEFAULT_DOSE_EDGES`.
  verbose — печатать ход расчёта.

Накопители — массивы (сутки × КА × широта × [местное время | долгота] × сезон ×
ночь × бин дозы): `exp_*` — экспозиция (число секундных сэмплов), `pos_*` — число
сэмплов с verdict = 1, `prb_*` — сумма prob_whistle; `*_true_d` — реальная доза,
`*_null_d` — null.
"""
from __future__ import annotations

import gc
import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd

import wwlln_common as wc

DEFAULT_DOSE_EDGES = np.array([0, 1, 3, 10, 30, 100, 300, 1000, 3000, np.inf], dtype=np.float64)
DEFAULT_DOSE_LABELS = ['0', '1–2', '3–9', '10–29', '30–99', '100–299', '300–999',
                        '1000–2999', '≥3000']
DEFAULT_E_EDGES = np.array([0.0, 2000.0, 6000.0, np.inf], dtype=np.float64)
DEFAULT_E_LABELS = ['<2 кДж', '2–6 кДж', '≥6 кДж']


def run_accumulation(
    df: pd.DataFrame,
    df_pred: pd.DataFrame,
    *,
    sc_list: Sequence[str],
    acc_days: Sequence[int],
    dose_r_km: float = 1000.0,
    dose_dt_min: int = 30,
    glat: float = 2.0,
    glon: float = 2.0,
    gdt_min: int = 5,
    null_lags: Sequence[int] = (3, 7, 14),
    null_symmetric: bool = False,
    null_season_safe: bool = True,
    do_energy: bool = True,
    night_sza_deg: Optional[float] = None,
    strata_scheme: str = 'lst',   # 'lst' (широта×LST×сезон×ночь) или 'lon' (широта×долгота×сезон×ночь)
    n_latb: int = 12,
    n_hourb: int = 8,             # используется при strata_scheme='lst'
    n_lonb: int = 8,              # используется при strata_scheme='lon'
    geometry: str = 'satellite',  # 'satellite' | 'footprint' | 'conjugate' — откуда искать дозу
    sat_height_km: float = 827.0,
    strata_by_geometry: bool = False,
    strata_coords: str = 'geo',   # 'geo' (гео. широта/долгота footprint/conjugate) | 'mag' (AACGM)
    dose_edges: Optional[np.ndarray] = None,
    e_edges: Optional[np.ndarray] = None,
    dose_r_km_inner: Optional[float] = None,
    dose_inner_edges: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> dict:
    """Накопители «доза–эффект» для одного радиуса и окна.

    Общие аргументы — в докстринге модуля. Свои:
      do_energy — добавить ось средней энергии молний в окрестности по границам
        `e_edges` (None — `DEFAULT_E_EDGES`): ключи `exp_en_d`, `pos_en_d`.
      strata_coords — 'geo': страты по географическим координатам footprint/conjugate;
        'mag': по геомагнитным AACGM той же точки. Имеет смысл при
        strata_by_geometry=True и geometry ≠ 'satellite'. Обратный пересчёт AACGM →
        география у геомагнитного экватора разрывен: соседние подспутниковые широты
        дают footprint-широты, различающиеся на 30° и больше. В геомагнитных
        координатах разрыва нет. Ночь всегда считается по географическим.
      dose_r_km_inner, dose_inner_edges — если задан внутренний радиус, доза — кольцо
        dose_r_km_inner < r ≤ dose_r_km, а сплошной внутренний круг (бины
        dose_inner_edges) становится ещё одной осью страт перед осью дозы. MH тогда
        сравнивает бины кольца внутри одинаковой внутренней дозы. Не сочетается с
        null_lags и do_energy.

    При null_symmetric=True сетки молний строятся с опережением на max(null_lags)
    суток, чтобы к накоплению суток day сетка day + L уже была готова.
    acc_days функция не проверяет: сутки без данных дадут пустые накопители.

    Возвращает dict: `exp_true_d`, `pos_true_d`, `prb_true_d`, `exp_null_d`,
    `pos_null_d`, `prb_null_d`, `exp_en_d` / `pos_en_d` (при do_energy, иначе None),
    `n_null_used_d` (использованных лагов по суткам) и параметры прогона
    (`sc_list`, `dose_edges`, `n_dose`, `dose_r_km`, `dose_dt_min`, `geometry`, ...).
    """
    if strata_scheme not in ('lst', 'lon'):
        raise ValueError(f"strata_scheme должен быть 'lst' или 'lon', получено {strata_scheme!r}")
    if geometry not in ('satellite', 'footprint', 'conjugate'):
        raise ValueError(f"geometry должен быть 'satellite'/'footprint'/'conjugate', получено {geometry!r}")
    if strata_coords not in ('geo', 'mag'):
        raise ValueError(f"strata_coords должен быть 'geo' или 'mag', получено {strata_coords!r}")
    _mc = None
    if geometry != 'satellite':
        import mag_coords as _mc  # ленивый импорт — aacgmv2 нужен только для этой геометрии
    if night_sza_deg is None:
        night_sza_deg = wc.NIGHT_SZA_DEG
    if dose_edges is None:
        dose_edges = DEFAULT_DOSE_EDGES
    if e_edges is None:
        e_edges = DEFAULT_E_EDGES
    dose_edges = np.asarray(dose_edges, dtype=np.float64)
    e_edges = np.asarray(e_edges, dtype=np.float64)

    has_inner = dose_r_km_inner is not None
    if has_inner:
        if not (0 < dose_r_km_inner < dose_r_km):
            raise ValueError('dose_r_km_inner должен быть в (0, dose_r_km) — '
                             'dose_r_km становится внешним радиусом кольца')
        if dose_inner_edges is None:
            raise ValueError('dose_inner_edges обязателен, если задан dose_r_km_inner')
        if null_lags:
            raise NotImplementedError(
                'dose_r_km_inner пока не поддержан вместе с null_lags (null для внутренней '
                'дозы не считается) — используйте null_lags=()')
        if do_energy:
            raise NotImplementedError('dose_r_km_inner пока не поддержан вместе с do_energy=True')
        dose_inner_edges = np.asarray(dose_inner_edges, dtype=np.float64)
        n_dose_inner = len(dose_inner_edges) - 1

    max_lag = max(null_lags) if null_lags else 0
    lookahead = max_lag if null_symmetric else 0
    lag_offsets = tuple(-L for L in null_lags) + (tuple(null_lags) if null_symmetric else ())

    # ================= сетка =================
    lat_edges = np.arange(-90.0, 90.0 + 1e-9, glat)
    lon_edges = np.arange(-180.0, 180.0 + 1e-9, glon)
    nlat, nlon = len(lat_edges) - 1, len(lon_edges) - 1
    nt = int(round(24 * 60 / gdt_min))
    pad = int(round(dose_dt_min / gdt_min))
    lat_c = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_c = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    ncell = nt * nlat * nlon

    hlat = int(np.ceil(dose_r_km / 111.0 / glat))
    hlon = np.ceil(
        dose_r_km / (111.0 * np.maximum(np.cos(np.radians(lat_c)), 0.05)) / glon
    ).astype(int)
    hlon = np.minimum(hlon, nlon // 2)

    if has_inner:
        hlat_in = int(np.ceil(dose_r_km_inner / 111.0 / glat))
        hlon_in = np.ceil(
            dose_r_km_inner / (111.0 * np.maximum(np.cos(np.radians(lat_c)), 0.05)) / glon
        ).astype(int)
        hlon_in = np.minimum(hlon_in, nlon // 2)

    def _grid(lat, lon, tmin, weights=None):
        ti = np.clip((tmin / gdt_min).astype(np.int64), 0, nt - 1)
        ii = np.clip(((lat + 90.0) / glat).astype(np.int64), 0, nlat - 1)
        jj = np.clip(((lon + 180.0) / glon).astype(np.int64), 0, nlon - 1)
        idx = (ti * nlat + ii) * nlon + jj
        g = np.bincount(idx, weights=weights, minlength=ncell)
        return g.reshape(nt, nlat, nlon).astype(np.float32)

    def _dose_grid(cur, prev, hlat_use=hlat, hlon_use=hlon):
        G = np.concatenate([prev[nt - pad:], cur], axis=0)
        S = np.cumsum(G, axis=0, dtype=np.float32)
        W = S[pad:pad + nt] - S[0:nt]
        del G, S
        A = np.cumsum(np.concatenate([W, W, W], axis=2), axis=2, dtype=np.float32)
        L = np.empty_like(W)
        jj = np.arange(nlon)
        for i in range(nlat):
            h = int(hlon_use[i])
            if h >= nlon // 2:
                L[:, i, :] = (A[:, i, 2 * nlon - 1] - A[:, i, nlon - 1])[:, None]
            else:
                L[:, i, :] = A[:, i, nlon + jj + h] - A[:, i, nlon + jj - h - 1]
        del A, W
        C = np.cumsum(L, axis=1, dtype=np.float32)
        ii = np.arange(nlat)
        out = C[:, np.clip(ii + hlat_use, 0, nlat - 1), :].copy()
        lo = ii - hlat_use - 1
        m = lo >= 0
        out[:, m, :] -= C[:, lo[m], :]
        return out

    # ================= индексы предсказаний =================
    n_dose = len(dose_edges) - 1
    n_eb = len(e_edges) - 1
    n_axis4 = n_hourb if strata_scheme == 'lst' else n_lonb
    n_sea, n_night = 4, 2
    sc_list = tuple(sc_list)
    n_sc = len(sc_list)
    sc_base = n_sc  # множитель для ключа (сутки*sc_base + код_КА) — безопасен для любого n_sc

    _pts = df_pred['timestamp'].values
    _plat, _plon = df_pred['geo_lat'].values, df_pred['geo_lon'].values
    _pver = df_pred['verdict'].values
    _pprob = df_pred['prob_whistle'].values.astype(np.float64)
    sc_to_code = {sc: i for i, sc in enumerate(sc_list)}
    pred_sccode = df_pred['sc'].astype(str).map(sc_to_code)
    if pred_sccode.isna().any():
        bad = sorted(set(df_pred['sc'].astype(str)) - set(sc_list))
        raise ValueError(f'В df_pred есть КА вне sc_list: {bad}')
    pred_sccode = pred_sccode.to_numpy(dtype=np.int64)

    pred_daycode = _pts.astype('datetime64[D]').astype(np.int64)
    _key = pred_daycode * sc_base + pred_sccode
    pred_order = np.argsort(_key, kind='stable')
    _ks = _key[pred_order]
    _uk, _us = np.unique(_ks, return_index=True)
    _ue = np.append(_us[1:], len(_ks))
    pred_slices = {int(k): (int(s), int(e)) for k, s, e in zip(_uk, _us, _ue)}
    del _key, _ks

    ae_t_all = df['datetime'].values
    l_lat_all = df['lat'].values.astype(np.float64)
    l_lon_all = df['lon'].values.astype(np.float64)
    l_energy_all = df['energy'].values.astype(np.float64) if do_energy else None

    acc_days = sorted(int(d) for d in acc_days)
    acc_set = set(acc_days)
    if not acc_days:
        raise ValueError('acc_days пуст')
    day_pos = {d: i for i, d in enumerate(acc_days)}

    grid_day_lo = acc_days[0] - max_lag - 1
    grid_day_hi = acc_days[-1] + lookahead
    grid_days = list(range(grid_day_lo, grid_day_hi + 1))

    # ================= накопители =================
    shape6 = ((n_sc, n_latb, n_axis4, n_sea, n_night, n_dose_inner, n_dose) if has_inner
              else (n_sc, n_latb, n_axis4, n_sea, n_night, n_dose))
    shape7 = (len(acc_days),) + shape6
    exp_true_d = np.zeros(shape7, dtype=np.float32)
    pos_true_d = np.zeros(shape7, dtype=np.float32)
    prb_true_d = np.zeros(shape7, dtype=np.float32)
    exp_null_d = np.zeros(shape7, dtype=np.float32)
    pos_null_d = np.zeros(shape7, dtype=np.float32)
    prb_null_d = np.zeros(shape7, dtype=np.float32)
    exp_en_d = np.zeros(shape7 + (n_eb,), dtype=np.float32) if do_energy else None
    pos_en_d = np.zeros(shape7 + (n_eb,), dtype=np.float32) if do_energy else None
    n_null_used_d = np.zeros(len(acc_days), dtype=np.int64)

    if verbose:
        mb = (6 * np.prod(shape7) + (2 * np.prod(shape7) * n_eb if do_energy else 0)) * 4 / 2 ** 20
        print(f'Накопители: {mb:.0f} МБ; сетки строим для {len(grid_days)} дней '
              f'(схема страт={strata_scheme!r}, лаги={null_lags}'
              f'{"±" if null_symmetric else " назад"}, '
              f'сезонный фильтр={"вкл" if null_season_safe else "выкл"})')

    cnt_grids, en_grids = {}, {}
    zero_grid = np.zeros((nt, nlat, nlon), dtype=np.float32)

    for n_done, grid_day in enumerate(grid_days, start=1):
        d0 = np.datetime64(grid_day, 'D').astype('datetime64[ns]')
        d1 = np.datetime64(grid_day + 1, 'D').astype('datetime64[ns]')
        s0, s1 = np.searchsorted(ae_t_all, [d0, d1])
        if s1 > s0:
            l_tmin = (ae_t_all[s0:s1] - d0) / np.timedelta64(1, 'm')
            cnt_grids[grid_day] = _grid(l_lat_all[s0:s1], l_lon_all[s0:s1], l_tmin)
            if do_energy:
                en_grids[grid_day] = _grid(l_lat_all[s0:s1], l_lon_all[s0:s1], l_tmin,
                                           weights=l_energy_all[s0:s1])
        else:
            cnt_grids[grid_day] = zero_grid.copy()
            if do_energy:
                en_grids[grid_day] = zero_grid.copy()

        day = grid_day - lookahead
        if day in acc_set:
            di = day_pos[day]
            dose_t = _dose_grid(cnt_grids[day], cnt_grids.get(day - 1, zero_grid)).ravel()
            if has_inner:
                # dose_t остаётся полным кругом до dose_r_km (используется ниже как
                # «внешний» при вычитании); dose_t_ring — само кольцо (dose_r_km_inner,
                # dose_r_km] — неотрицательно поточечно, т.к. внутренний прямоугольник
                # вложен во внешний (hlat_in<=hlat и hlon_in[i]<=hlon[i] при R_inner<R).
                dose_t_inner_full = _dose_grid(cnt_grids[day], cnt_grids.get(day - 1, zero_grid),
                                               hlat_in, hlon_in).ravel()
                dose_t_ring = dose_t - dose_t_inner_full

            month_day = int(str(np.datetime64(day, 'D'))[5:7])
            lag_days = []
            for off in lag_offsets:
                nd = day + off
                if nd not in cnt_grids:
                    continue
                if null_season_safe:
                    month_nd = int(str(np.datetime64(nd, 'D'))[5:7])
                    if not wc.same_season_quarter(month_day, month_nd):
                        continue
                lag_days.append(nd)
            n_null_used_d[di] = len(lag_days)
            dose_nulls = [_dose_grid(cnt_grids[nd], cnt_grids.get(nd - 1, zero_grid)).ravel()
                          for nd in lag_days]
            dose_e = (_dose_grid(en_grids[day], en_grids.get(day - 1, zero_grid)).ravel()
                      if do_energy else None)

            for si, sc in enumerate(sc_list):
                sl = pred_slices.get(day * sc_base + si)
                if sl is None:
                    continue
                idx = pred_order[sl[0]:sl[1]]
                lat = _plat[idx].astype(np.float64)
                lon = _plon[idx].astype(np.float64)
                ver = _pver[idx]
                prb = _pprob[idx]
                tt = _pts[idx]
                tmin = (tt - tt.astype('datetime64[D]')) / np.timedelta64(1, 'm')
                night_sat = (wc.solar_zenith_deg(tt, lat, lon) > night_sza_deg).astype(np.int8)

                exp_g = _grid(lat, lon, tmin).ravel()
                pos_g = _grid(lat[ver == 1], lon[ver == 1], tmin[ver == 1]).ravel()
                prb_g = _grid(lat, lon, tmin, weights=prb).ravel()
                cells = np.flatnonzero(exp_g > 0)
                if cells.size == 0:
                    continue

                ti = cells // (nlat * nlon)
                rem = cells % (nlat * nlon)
                lat_cc = lat_c[rem // nlon]
                lon_cc = lon_c[rem % nlon]

                # Доза ищется по geometry: 'satellite' — та же ячейка (cells);
                # 'footprint'/'conjugate' — footprint/conjugate ОДНОЙ представительной
                # точки ячейки (lat_cc/lon_cc), а не каждого сырого наблюдения — иначе
                # наблюдения из одной и той же спутниковой ячейки 2°×2° расходятся по
                # РАЗНЫМ footprint-ячейкам (проверено: разброс footprint внутри одной
                # ячейки ~1-2°, сопоставимо с самой ячейкой — фрагментирует страты).
                if geometry == 'satellite':
                    dose_idx = cells
                    g_lat_cc, g_lon_cc = lat_cc, lon_cc
                    m_lat_cc, m_lon_cc = lat_cc, lon_cc
                else:
                    day_date = np.datetime64(day, 'D').astype('datetime64[s]').astype(object)
                    fp_lat_c, fp_lon_c, cj_lat_c, cj_lon_c = _mc.footprint_conjugate_arr(
                        lat_cc, lon_cc, sat_height_km, day_date)
                    gg_lat, gg_lon = (fp_lat_c, fp_lon_c) if geometry == 'footprint' else (cj_lat_c, cj_lon_c)
                    valid_cc = np.isfinite(gg_lat) & np.isfinite(gg_lon)
                    g_lat_cc = np.where(valid_cc, gg_lat, lat_cc)  # запасной вариант — подспутниковая
                    g_lon_cc = np.where(valid_cc, gg_lon, lon_cc)
                    g_ii = np.clip(((g_lat_cc + 90.0) / glat).astype(np.int64), 0, nlat - 1)
                    g_jj = np.clip(((g_lon_cc + 180.0) / glon).astype(np.int64), 0, nlon - 1)
                    dose_idx = (ti * nlat + g_ii) * nlon + g_jj
                    if strata_coords == 'mag':
                        # геомагнитные координаты той же представительной точки ячейки —
                        # без обратного A2G-пересчёта в географию (там и есть разрыв
                        # вблизи магнитного экватора, см. докстринг strata_coords выше)
                        mlat_c, mlon_c = _mc.geomagnetic_arr(lat_cc, lon_cc, sat_height_km, day_date)
                        mm_lat_c = mlat_c if geometry == 'footprint' else -mlat_c
                        valid_mag = np.isfinite(mm_lat_c) & np.isfinite(mlon_c)
                        m_lat_cc = np.where(valid_mag, mm_lat_c, lat_cc)
                        m_lon_cc = np.where(valid_mag, mlon_c, lon_cc)
                    else:
                        m_lat_cc, m_lon_cc = g_lat_cc, g_lon_cc  # не используется при strata_coords='geo'

                # Страты (широта/[LST|долгота]/сезон) — по подспутниковой, если не просили
                # иначе; при strata_by_geometry=True — по той же представительной точке,
                # что и доза (geo: g_lat_cc/g_lon_cc) либо по её геомагнитным координатам
                # (strata_coords='mag': m_lat_cc/m_lon_cc), уже посчитанным один раз на
                # ячейку выше. Ветка 'geo' бит-в-бит совпадает со старым поведением.
                if not strata_by_geometry:
                    strata_lat_cc, strata_lon_cc = lat_cc, lon_cc
                elif strata_coords == 'mag':
                    strata_lat_cc, strata_lon_cc = m_lat_cc, m_lon_cc
                else:
                    strata_lat_cc, strata_lon_cc = g_lat_cc, g_lon_cc

                latb = np.clip(((strata_lat_cc + 90.0) / (180.0 / n_latb)).astype(np.int64),
                               0, n_latb - 1)
                if strata_scheme == 'lst':
                    lst = np.mod(ti * gdt_min / 60.0 + gdt_min / 120.0 + strata_lon_cc / 15.0, 24.0)
                    axis4 = np.clip((lst / (24.0 / n_hourb)).astype(np.int64), 0, n_hourb - 1)
                else:
                    axis4 = np.clip(((strata_lon_cc + 180.0) / (360.0 / n_lonb)).astype(np.int64),
                                    0, n_lonb - 1)
                _, seab = wc.season_index(strata_lat_cc, month_day)
                seab = seab.astype(np.int64)

                # Ночь: при strata_by_geometry=True — тоже по представительной точке ячейки
                # (затенение D-области над footprint/conjugate, не под спутником), но со
                # своим временем каждого наблюдения. Ячейка каждого сырого наблюдения уже
                # входит в `cells` (иначе exp_g там было бы 0) — ищем её позицию через
                # searchsorted (cells отсортирован), не пересчитывая footprint заново на
                # каждое наблюдение (та же причина фрагментации, что и с dose_idx выше).
                if strata_by_geometry and geometry != 'satellite':
                    ti_obs = np.clip((tmin / gdt_min).astype(np.int64), 0, nt - 1)
                    ii_obs = np.clip(((lat + 90.0) / glat).astype(np.int64), 0, nlat - 1)
                    jj_obs = np.clip(((lon + 180.0) / glon).astype(np.int64), 0, nlon - 1)
                    obs_cell = (ti_obs * nlat + ii_obs) * nlon + jj_obs
                    pos_in_cells = np.searchsorted(cells, obs_cell)
                    night = (wc.solar_zenith_deg(tt, g_lat_cc[pos_in_cells], g_lon_cc[pos_in_cells])
                             > night_sza_deg).astype(np.int8)
                else:
                    night = night_sat
                ngt_g = _grid(lat, lon, tmin, weights=night.astype(np.float64)).ravel()
                nightb = (ngt_g[cells] / exp_g[cells] > 0.5).astype(np.int64)

                e_vals, p_vals, b_vals = exp_g[cells], pos_g[cells], prb_g[cells]
                if has_inner:
                    dt_cells = dose_t_ring[dose_idx]
                    db_in = np.clip(
                        np.searchsorted(dose_inner_edges, dose_t_inner_full[dose_idx], side='right') - 1,
                        0, n_dose_inner - 1)
                    key_t = (di, si, latb, axis4, seab, nightb, db_in)
                else:
                    dt_cells = dose_t[dose_idx]
                    key_t = (di, si, latb, axis4, seab, nightb)
                db_t = np.clip(np.searchsorted(dose_edges, dt_cells, side='right') - 1,
                               0, n_dose - 1)
                key_t = key_t + (db_t,)
                np.add.at(exp_true_d, key_t, e_vals)
                np.add.at(pos_true_d, key_t, p_vals)
                np.add.at(prb_true_d, key_t, b_vals)

                for dose_n in dose_nulls:
                    db_n = np.clip(np.searchsorted(dose_edges, dose_n[dose_idx], side='right') - 1,
                                   0, n_dose - 1)
                    key_n = (di, si, latb, axis4, seab, nightb, db_n)
                    np.add.at(exp_null_d, key_n, e_vals)
                    np.add.at(pos_null_d, key_n, p_vals)
                    np.add.at(prb_null_d, key_n, b_vals)

                if do_energy and dose_e is not None:
                    mean_e = np.divide(dose_e[dose_idx], dt_cells, out=np.zeros(cells.size),
                                       where=dt_cells > 0)
                    eb = np.clip(np.searchsorted(e_edges, mean_e, side='right') - 1, 0, n_eb - 1)
                    key_e = (di, si, latb, axis4, seab, nightb, db_t, eb)
                    np.add.at(exp_en_d, key_e, e_vals)
                    np.add.at(pos_en_d, key_e, p_vals)

            del dose_t, dose_nulls, dose_e
            if has_inner:
                del dose_t_inner_full, dose_t_ring

        for old in [k for k in cnt_grids if k < day - max_lag - 1]:
            cnt_grids.pop(old, None)
        for old in [k for k in en_grids if k < day - 1]:
            en_grids.pop(old, None)
        if verbose and n_done % 20 == 0:
            print(f'  ... {n_done}/{len(grid_days)} дней', flush=True)
            gc.collect()

    del cnt_grids, en_grids
    gc.collect()

    if verbose:
        exp_true = exp_true_d.sum(dtype=np.float64)
        pos_true = pos_true_d.sum(dtype=np.float64)
        exp_null = exp_null_d.sum(dtype=np.float64)
        avg_lags_used = n_null_used_d.mean() if len(n_null_used_d) else 0.0
        print(f'Готово. Экспозиция: {exp_true:,.0f} сэмплов, verdict=1: {pos_true:,.0f} '
              f'({100 * pos_true / max(exp_true, 1):.2f}%); '
              f'экспозиция null: {exp_null:,.0f} '
              f'(в среднем {avg_lags_used:.2f} лага/сутки использовано из '
              f'{len(lag_offsets)} возможных)')

    return {
        'exp_true_d': exp_true_d, 'pos_true_d': pos_true_d, 'prb_true_d': prb_true_d,
        'exp_null_d': exp_null_d, 'pos_null_d': pos_null_d, 'prb_null_d': prb_null_d,
        'exp_en_d': exp_en_d, 'pos_en_d': pos_en_d,
        'acc_days': acc_days, 'day_pos': day_pos, 'n_null_used_d': n_null_used_d,
        'sc_list': sc_list, 'n_sc': n_sc, 'n_latb': n_latb, 'n_axis4': n_axis4,
        'n_hourb': n_hourb, 'n_lonb': n_lonb,
        'n_sea': n_sea, 'n_night': n_night, 'n_dose': n_dose, 'n_eb': n_eb,
        'strata_scheme': strata_scheme, 'dose_edges': dose_edges, 'e_edges': e_edges,
        'dose_r_km': dose_r_km, 'dose_dt_min': dose_dt_min, 'gdt_min': gdt_min,
        'null_lags': tuple(null_lags), 'null_symmetric': null_symmetric,
        'null_season_safe': null_season_safe, 'night_sza_deg': night_sza_deg,
        'geometry': geometry, 'sat_height_km': sat_height_km,
        'strata_by_geometry': strata_by_geometry,
        'dose_r_km_inner': dose_r_km_inner,
        'dose_inner_edges': dose_inner_edges if has_inner else None,
        'n_dose_inner': n_dose_inner if has_inner else None,
    }


def run_median_energy_lon(
    df: pd.DataFrame,
    df_pred: pd.DataFrame,
    *,
    sc_list: Sequence[str],
    acc_days: Sequence[int],
    dose_r_km: float = 1000.0,
    dose_dt_min: int = 30,
    glat: float = 2.0,
    glon: float = 2.0,
    gdt_min: int = 5,
    night_sza_deg: Optional[float] = None,
    strata_scheme: str = 'lon',
    n_latb: int = 12,
    n_hourb: int = 8,
    n_lonb: int = 8,
    dose_edges: Optional[np.ndarray] = None,
    e_edges: Optional[np.ndarray] = None,
    n_hist_bins: int = 16,
    e_hist_max: float = 8000.0,
    geometry: str = 'satellite',
    sat_height_km: float = 827.0,
    strata_by_geometry: bool = False,
    verbose: bool = True,
) -> dict:
    """Накопители для эффекта медианной энергии молний при фиксированной дозе.

    Медиана не складывается из кумулятивных сумм, поэтому на каждую ячейку сетки
    копится гистограмма энергии из n_hist_bins тонких бинов на [0, e_hist_max] Дж
    (плюс переполнение). Медиана в окне — линейная интерполяция по кумулятивной
    гистограмме; точность — e_hist_max / n_hist_bins. Null-модели нет.

    Общие аргументы — в докстринге модуля (без null_* и strata_coords). Свои:
      e_edges — классы медианной энергии, Дж; None — `DEFAULT_E_EDGES`.
        В footprint-ноутбуке — терцили [0, 600, 1900, inf].
      n_hist_bins, e_hist_max — гистограмма, по которой ищется медиана.

    Кэш на 2 суток держит n_hist_bins сеток: при gdt_min=5 и n_hist_bins=16 это
    около 570 МБ, на минутной сетке примерно впятеро больше.

    Возвращает dict: `exp_en_d`, `pos_en_d` формы (сутки × КА × широта × [местное
    время | долгота] × сезон × ночь × бин дозы × класс энергии) и параметры прогона.
    Для анализа — `analyze_energy_lon`.
    """
    if strata_scheme not in ('lst', 'lon'):
        raise ValueError(f"strata_scheme должен быть 'lst' или 'lon', получено {strata_scheme!r}")
    if geometry not in ('satellite', 'footprint', 'conjugate'):
        raise ValueError(f"geometry должен быть 'satellite'/'footprint'/'conjugate', получено {geometry!r}")
    _mc = None
    if geometry != 'satellite':
        import mag_coords as _mc  # ленивый импорт — aacgmv2 нужен только для этой геометрии
    if night_sza_deg is None:
        night_sza_deg = wc.NIGHT_SZA_DEG
    if dose_edges is None:
        dose_edges = DEFAULT_DOSE_EDGES
    if e_edges is None:
        e_edges = DEFAULT_E_EDGES
    dose_edges = np.asarray(dose_edges, dtype=np.float64)
    e_edges = np.asarray(e_edges, dtype=np.float64)

    e_hist_edges = np.concatenate([np.linspace(0.0, e_hist_max, n_hist_bins), [np.inf]])
    n_hb = len(e_hist_edges) - 1

    # ================= сетка (те же формулы, что в run_accumulation) =================
    lat_edges = np.arange(-90.0, 90.0 + 1e-9, glat)
    lon_edges = np.arange(-180.0, 180.0 + 1e-9, glon)
    nlat, nlon = len(lat_edges) - 1, len(lon_edges) - 1
    nt = int(round(24 * 60 / gdt_min))
    pad = int(round(dose_dt_min / gdt_min))
    lat_c = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_c = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    ncell = nt * nlat * nlon

    hlat = int(np.ceil(dose_r_km / 111.0 / glat))
    hlon = np.ceil(
        dose_r_km / (111.0 * np.maximum(np.cos(np.radians(lat_c)), 0.05)) / glon
    ).astype(int)
    hlon = np.minimum(hlon, nlon // 2)

    def _grid(lat, lon, tmin, weights=None):
        ti = np.clip((tmin / gdt_min).astype(np.int64), 0, nt - 1)
        ii = np.clip(((lat + 90.0) / glat).astype(np.int64), 0, nlat - 1)
        jj = np.clip(((lon + 180.0) / glon).astype(np.int64), 0, nlon - 1)
        idx = (ti * nlat + ii) * nlon + jj
        g = np.bincount(idx, weights=weights, minlength=ncell)
        return g.reshape(nt, nlat, nlon).astype(np.float32)

    def _grid_hist(lat, lon, tmin, ebin):
        ti = np.clip((tmin / gdt_min).astype(np.int64), 0, nt - 1)
        ii = np.clip(((lat + 90.0) / glat).astype(np.int64), 0, nlat - 1)
        jj = np.clip(((lon + 180.0) / glon).astype(np.int64), 0, nlon - 1)
        idx = ((ti * nlat + ii) * nlon + jj) * n_hb + ebin
        g = np.bincount(idx, minlength=ncell * n_hb)
        return g.reshape(nt, nlat, nlon, n_hb).astype(np.float32)

    def _dose_grid(cur, prev):
        # Не завязана на размерность последней оси — работает и на 3D (t,lat,lon)
        # дозе, и на 4D (t,lat,lon,energy_bin) гистограмме без изменений: все
        # операции идут по axis=0 (время) и axis=1/2 (широта/долгота), любая
        # дополнительная последняя ось просто едет пассажиром.
        G = np.concatenate([prev[nt - pad:], cur], axis=0)
        S = np.cumsum(G, axis=0, dtype=np.float32)
        W = S[pad:pad + nt] - S[0:nt]
        del G, S
        A = np.cumsum(np.concatenate([W, W, W], axis=2), axis=2, dtype=np.float32)
        L = np.empty_like(W)
        jj = np.arange(nlon)
        for i in range(nlat):
            h = int(hlon[i])
            if h >= nlon // 2:
                L[:, i, :] = (A[:, i, 2 * nlon - 1] - A[:, i, nlon - 1])[:, None]
            else:
                L[:, i, :] = A[:, i, nlon + jj + h] - A[:, i, nlon + jj - h - 1]
        del A, W
        C = np.cumsum(L, axis=1, dtype=np.float32)
        ii = np.arange(nlat)
        out = C[:, np.clip(ii + hlat, 0, nlat - 1), :].copy()
        lo = ii - hlat - 1
        m = lo >= 0
        out[:, m, :] -= C[:, lo[m], :]
        return out

    # ================= индексы предсказаний =================
    n_dose = len(dose_edges) - 1
    n_eb = len(e_edges) - 1
    n_axis4 = n_hourb if strata_scheme == 'lst' else n_lonb
    n_sea, n_night = 4, 2
    sc_list = tuple(sc_list)
    n_sc = len(sc_list)
    sc_base = n_sc

    _pts = df_pred['timestamp'].values
    _plat, _plon = df_pred['geo_lat'].values, df_pred['geo_lon'].values
    _pver = df_pred['verdict'].values
    sc_to_code = {sc: i for i, sc in enumerate(sc_list)}
    pred_sccode = df_pred['sc'].astype(str).map(sc_to_code)
    if pred_sccode.isna().any():
        bad = sorted(set(df_pred['sc'].astype(str)) - set(sc_list))
        raise ValueError(f'В df_pred есть КА вне sc_list: {bad}')
    pred_sccode = pred_sccode.to_numpy(dtype=np.int64)

    pred_daycode = _pts.astype('datetime64[D]').astype(np.int64)
    _key = pred_daycode * sc_base + pred_sccode
    pred_order = np.argsort(_key, kind='stable')
    _ks = _key[pred_order]
    _uk, _us = np.unique(_ks, return_index=True)
    _ue = np.append(_us[1:], len(_ks))
    pred_slices = {int(k): (int(s), int(e)) for k, s, e in zip(_uk, _us, _ue)}
    del _key, _ks

    ae_t_all = df['datetime'].values
    l_lat_all = df['lat'].values.astype(np.float64)
    l_lon_all = df['lon'].values.astype(np.float64)
    l_energy_all = df['energy'].values.astype(np.float64)

    acc_days = sorted(int(d) for d in acc_days)
    acc_set = set(acc_days)
    if not acc_days:
        raise ValueError('acc_days пуст')
    day_pos = {d: i for i, d in enumerate(acc_days)}

    grid_days = list(range(acc_days[0] - 1, acc_days[-1] + 1))

    shape6 = (n_sc, n_latb, n_axis4, n_sea, n_night, n_dose)
    shape7 = (len(acc_days),) + shape6
    exp_en_d = np.zeros(shape7 + (n_eb,), dtype=np.float32)
    pos_en_d = np.zeros(shape7 + (n_eb,), dtype=np.float32)

    if verbose:
        mb_hist = (2 * nt * nlat * nlon * n_hb) * 4 / 2 ** 20
        print(f'Гистограмма энергии (медиана): {n_hb} тонких бинов (0…{e_hist_max:.0f}, '
              f'+overflow), ~{mb_hist:.0f} МБ на кэш из 2 суток; сетки строим для '
              f'{len(grid_days)} дней (схема страт={strata_scheme!r})')

    zero_grid = np.zeros((nt, nlat, nlon), dtype=np.float32)
    zero_hist = np.zeros((nt, nlat, nlon, n_hb), dtype=np.float32)
    cnt_grids, hist_grids = {}, {}

    for n_done, day in enumerate(grid_days, start=1):
        d0 = np.datetime64(day, 'D').astype('datetime64[ns]')
        d1 = np.datetime64(day + 1, 'D').astype('datetime64[ns]')
        s0, s1 = np.searchsorted(ae_t_all, [d0, d1])
        if s1 > s0:
            l_tmin = (ae_t_all[s0:s1] - d0) / np.timedelta64(1, 'm')
            cnt_grids[day] = _grid(l_lat_all[s0:s1], l_lon_all[s0:s1], l_tmin)
            ebin = np.clip(
                np.searchsorted(e_hist_edges, l_energy_all[s0:s1], side='right') - 1,
                0, n_hb - 1)
            hist_grids[day] = _grid_hist(l_lat_all[s0:s1], l_lon_all[s0:s1], l_tmin, ebin)
        else:
            cnt_grids[day] = zero_grid.copy()
            hist_grids[day] = zero_hist.copy()

        if day in acc_set:
            di = day_pos[day]
            dose_t = _dose_grid(cnt_grids[day], cnt_grids.get(day - 1, zero_grid)).ravel()
            dose_hist = _dose_grid(
                hist_grids[day], hist_grids.get(day - 1, zero_hist)).reshape(-1, n_hb)

            month_day = int(str(np.datetime64(day, 'D'))[5:7])

            for si, sc in enumerate(sc_list):
                sl = pred_slices.get(day * sc_base + si)
                if sl is None:
                    continue
                idx = pred_order[sl[0]:sl[1]]
                lat = _plat[idx].astype(np.float64)
                lon = _plon[idx].astype(np.float64)
                ver = _pver[idx]
                tt = _pts[idx]
                tmin = (tt - tt.astype('datetime64[D]')) / np.timedelta64(1, 'm')
                night_sat = (wc.solar_zenith_deg(tt, lat, lon) > night_sza_deg).astype(np.int8)

                exp_g = _grid(lat, lon, tmin).ravel()
                pos_g = _grid(lat[ver == 1], lon[ver == 1], tmin[ver == 1]).ravel()
                cells = np.flatnonzero(exp_g > 0)
                if cells.size == 0:
                    continue

                ti = cells // (nlat * nlon)
                rem = cells % (nlat * nlon)
                lat_cc = lat_c[rem // nlon]
                lon_cc = lon_c[rem % nlon]

                # Доза/страты по geometry — та же логика, что в run_accumulation: footprint/
                # conjugate считается ОДИН РАЗ на ячейку сетки (lat_cc/lon_cc), а не на каждое
                # сырое наблюдение — иначе нелинейность AACGM дробит одну спутниковую ячейку
                # на несколько footprint-ячеек и разваливает общий носитель.
                if geometry == 'satellite':
                    dose_idx = cells
                    g_lat_cc, g_lon_cc = lat_cc, lon_cc
                else:
                    day_date = np.datetime64(day, 'D').astype('datetime64[s]').astype(object)
                    fp_lat_c, fp_lon_c, cj_lat_c, cj_lon_c = _mc.footprint_conjugate_arr(
                        lat_cc, lon_cc, sat_height_km, day_date)
                    gg_lat, gg_lon = (fp_lat_c, fp_lon_c) if geometry == 'footprint' else (cj_lat_c, cj_lon_c)
                    valid_cc = np.isfinite(gg_lat) & np.isfinite(gg_lon)
                    g_lat_cc = np.where(valid_cc, gg_lat, lat_cc)
                    g_lon_cc = np.where(valid_cc, gg_lon, lon_cc)
                    g_ii = np.clip(((g_lat_cc + 90.0) / glat).astype(np.int64), 0, nlat - 1)
                    g_jj = np.clip(((g_lon_cc + 180.0) / glon).astype(np.int64), 0, nlon - 1)
                    dose_idx = (ti * nlat + g_ii) * nlon + g_jj

                strata_lat_cc = g_lat_cc if strata_by_geometry else lat_cc
                strata_lon_cc = g_lon_cc if strata_by_geometry else lon_cc

                latb = np.clip(((strata_lat_cc + 90.0) / (180.0 / n_latb)).astype(np.int64),
                               0, n_latb - 1)
                if strata_scheme == 'lst':
                    lst = np.mod(ti * gdt_min / 60.0 + gdt_min / 120.0 + strata_lon_cc / 15.0, 24.0)
                    axis4 = np.clip((lst / (24.0 / n_hourb)).astype(np.int64), 0, n_hourb - 1)
                else:
                    axis4 = np.clip(((strata_lon_cc + 180.0) / (360.0 / n_lonb)).astype(np.int64),
                                    0, n_lonb - 1)
                _, seab = wc.season_index(strata_lat_cc, month_day)
                seab = seab.astype(np.int64)

                # Ночь: при strata_by_geometry=True — тоже по представительной точке ячейки
                # (см. run_accumulation), иначе по подспутниковой (night_sat).
                if strata_by_geometry and geometry != 'satellite':
                    ti_obs = np.clip((tmin / gdt_min).astype(np.int64), 0, nt - 1)
                    ii_obs = np.clip(((lat + 90.0) / glat).astype(np.int64), 0, nlat - 1)
                    jj_obs = np.clip(((lon + 180.0) / glon).astype(np.int64), 0, nlon - 1)
                    obs_cell = (ti_obs * nlat + ii_obs) * nlon + jj_obs
                    pos_in_cells = np.searchsorted(cells, obs_cell)
                    night = (wc.solar_zenith_deg(tt, g_lat_cc[pos_in_cells], g_lon_cc[pos_in_cells])
                             > night_sza_deg).astype(np.int8)
                else:
                    night = night_sat
                ngt_g = _grid(lat, lon, tmin, weights=night.astype(np.float64)).ravel()
                nightb = (ngt_g[cells] / exp_g[cells] > 0.5).astype(np.int64)

                e_vals, p_vals = exp_g[cells], pos_g[cells]
                dt_cells = dose_t[dose_idx]
                db_t = np.clip(np.searchsorted(dose_edges, dt_cells, side='right') - 1,
                               0, n_dose - 1)

                # --- приближённая медиана энергии через кумулятивную гистограмму ---
                # Стандартная формула медианы сгруппированных данных
                # (L + (n/2 − CF)/f × h) — интерполяция внутри бина, где
                # кумулятивная гистограмма впервые достигает n/2. Проверено на
                # синтетике: на гладких (унимодальных) распределениях энергий
                # ошибка бакетирования ~0.2-0.3%, но если ровно на границе бина
                # исчерпывается весь его счёт (frac=1), а следом идёт пустой
                # промежуток гистограммы — формула "залипает" на этой границе,
                # хотя реальных данных там уже нет (на синтетическом двумодальном
                # тесте — до 17% неверных бакетов). Поправка ниже: в этом
                # пограничном случае берём середину промежутка до следующего
                # непустого бина, а не жмёмся к краю — снижает долю неверных
                # бакетов на том же двумодальном тесте до ~2%.
                h = dose_hist[dose_idx]                       # (n_cells, n_hb)
                cum = np.cumsum(h, axis=1)
                total = cum[:, -1]
                target = total / 2.0
                bin_idx = np.clip(np.sum(cum < target[:, None], axis=1), 0, n_hb - 1)
                cum_before = np.where(
                    bin_idx > 0,
                    np.take_along_axis(cum, (bin_idx - 1)[:, None], axis=1)[:, 0],
                    0.0)
                count_in_bin = np.take_along_axis(h, bin_idx[:, None], axis=1)[:, 0]
                lo_edge = e_hist_edges[bin_idx]
                hi_edge = np.minimum(e_hist_edges[bin_idx + 1], e_hist_max)
                frac = np.divide(target - cum_before, count_in_bin,
                                 out=np.zeros_like(target), where=count_in_bin > 0)
                median_e = lo_edge + frac * (hi_edge - lo_edge)

                at_edge = (frac >= 1.0 - 1e-9) & (total > 0)
                col_idx = np.arange(n_hb)
                after_mask = (col_idx[None, :] > bin_idx[:, None]) & (h > 0)
                has_next = after_mask.any(axis=1)
                next_j = np.where(has_next, after_mask.argmax(axis=1), n_hb)
                real_gap = at_edge & has_next & (next_j > bin_idx + 1)
                right_full = np.where(next_j < n_hb, e_hist_edges[np.minimum(next_j, n_hb - 1)],
                                      e_hist_max)
                right_full = np.minimum(right_full, e_hist_max)
                median_e = np.where(real_gap, 0.5 * (hi_edge + right_full), median_e)

                median_e = np.where(dt_cells > 0, median_e, 0.0)

                eb = np.clip(np.searchsorted(e_edges, median_e, side='right') - 1, 0, n_eb - 1)
                key_e = (di, si, latb, axis4, seab, nightb, db_t, eb)
                np.add.at(exp_en_d, key_e, e_vals)
                np.add.at(pos_en_d, key_e, p_vals)

        for old in [k for k in cnt_grids if k < day - 1]:
            cnt_grids.pop(old, None)
        for old in [k for k in hist_grids if k < day - 1]:
            hist_grids.pop(old, None)
        if verbose and n_done % 20 == 0:
            print(f'  ... {n_done}/{len(grid_days)} дней', flush=True)
            gc.collect()

    del cnt_grids, hist_grids
    gc.collect()

    if verbose:
        exp_total = exp_en_d.sum(dtype=np.float64)
        pos_total = pos_en_d.sum(dtype=np.float64)
        print(f'Готово. Экспозиция: {exp_total:,.0f} сэмплов, verdict=1: {pos_total:,.0f} '
              f'({100 * pos_total / max(exp_total, 1):.2f}%)')

    return {
        'exp_en_d': exp_en_d, 'pos_en_d': pos_en_d,
        'sc_list': sc_list, 'n_sc': n_sc, 'n_latb': n_latb, 'n_axis4': n_axis4,
        'n_hourb': n_hourb, 'n_lonb': n_lonb,
        'n_sea': n_sea, 'n_night': n_night, 'n_dose': n_dose, 'n_eb': n_eb,
        'strata_scheme': strata_scheme, 'dose_edges': dose_edges, 'e_edges': e_edges,
        'e_hist_edges': e_hist_edges, 'n_hist_bins': n_hist_bins,
        'dose_r_km': dose_r_km, 'dose_dt_min': dose_dt_min, 'gdt_min': gdt_min,
        'night_sza_deg': night_sza_deg,
        'geometry': geometry, 'sat_height_km': sat_height_km,
        'strata_by_geometry': strata_by_geometry,
    }


# ============================== анализ ==============================

def select_strata(a, sc=None, night=None, day_axis=False):
    """Срез по аппарату и ночи и склейка осей условий в один индекс страты.

    a — накопитель с осью суток первой; sc — индекс аппарата или None (все);
    night — 0 / 1 или None (день и ночь вместе). Возвращает (страты × бин) или,
    при day_axis=True, (сутки × страты × бин).
    """
    o = 1 if day_axis else 0
    x = a[:, [sc]] if (day_axis and sc is not None) else (a[[sc]] if sc is not None else a)
    if night is not None:
        x = x[(slice(None),) * (o + 4) + ([night],)]
    return x.reshape((x.shape[0], -1, x.shape[-1]) if day_axis else (-1, x.shape[-1]))


def mh(E, P, ref=0, min_exp=500):
    """Мантель–Хензель: отношение уровней «бин k / опорный бин» внутри страт.

    E, P — экспозиция и число свистов (страты × бин). В сравнение бина k с ref
    входят страты, где в обоих бинах экспозиция не меньше min_exp.
    Возвращает (rr, p_ref, p_k, nst, expo) по бинам: отношение MH, уровни опорного
    бина и бина k на общих стратах, число страт, экспозиция бина k.
    """
    nb = E.shape[1]
    rr = np.full(nb, np.nan)
    p_ref = np.full(nb, np.nan)
    p_k = np.full(nb, np.nan)
    nst = np.zeros(nb, dtype=np.int64)
    expo = np.zeros(nb)
    ok = E >= min_exp
    for k in range(nb):
        m = ok[:, ref] & ok[:, k]
        if not m.any():
            continue
        e0, e1 = E[m, ref].astype(np.float64), E[m, k].astype(np.float64)
        p0, p1 = P[m, ref].astype(np.float64), P[m, k].astype(np.float64)
        tot = e0 + e1
        w = e0 * e1 / tot
        den = (p0 * e1 / tot).sum()
        rr[k] = (p1 * e0 / tot).sum() / den if den > 0 else np.nan
        p_ref[k] = (p0 / e0 * w).sum() / w.sum()
        p_k[k] = (p1 / e1 * w).sum() / w.sum()
        nst[k] = int(m.sum())
        expo[k] = e1.sum()
    return rr, p_ref, p_k, nst, expo


def pooled(E, P):
    """Сырая доля P / E по бинам без контроля страт (ключ `raw` в `analyze`)."""
    s_e, s_p = E.sum(axis=0), P.sum(axis=0)
    return np.where(s_e > 0, s_p / np.where(s_e > 0, s_e, 1), np.nan)


def analyze(accum, sc=None, night=None, outcome='verdict', n_boot=200, seed=0,
            ref_bin=0, min_strat_exp=500):
    """Кривая «доза–эффект» с контролем страт и блочным бутстрапом по суткам.

    accum — накопитель `run_accumulation` / `sweep_r_dt` / `sweep_night_sza`.
    sc — индекс аппарата в sc_list; None — все вместе.
    night — None: день и ночь вместе; 0 — день; 1 — ночь.
    outcome — 'verdict': доля сэмплов со свистом; иное — средняя prob_whistle.
    n_boot, seed — повторы бутстрапа (сутки с возвращением) и зерно.
    ref_bin — опорный бин дозы.
    min_strat_exp — минимальная экспозиция страты в обоих сравниваемых бинах.

    Возвращает dict массивов по бинам дозы:
      raw — доля без страт;
      rr_t, ci_t — рост MH относительно ref_bin и 95%-интервал (ci_t[0], ci_t[1]);
      pk_t, p0_t, ci_pk_t, ci_p0_t — уровень в бине k и в опорном на общих для пары
        стратах (p0_t свой для каждого k);
      bpk_t, bp0_t, bpk_n — те же уровни по повторам бутстрапа (n_boot × бин), для
        интервалов разностей;
      nst, expo — число страт в сравнении и экспозиция бина;
      rr_n, ci_n, pk_n, p0_n, rrr (= rr_t / rr_n), ci_rrr — null-ветка; при
        null_lags=() NaN.
    """
    Pt_d, Pn_d = ((accum['pos_true_d'], accum['pos_null_d']) if outcome == 'verdict'
                  else (accum['prb_true_d'], accum['prb_null_d']))
    Et = select_strata(accum['exp_true_d'], sc, night, day_axis=True)
    Pt = select_strata(Pt_d, sc, night, day_axis=True)
    En = select_strata(accum['exp_null_d'], sc, night, day_axis=True)
    Pn = select_strata(Pn_d, sc, night, day_axis=True)

    rr_t, p0_t, pk_t, nst, expo = mh(Et.sum(axis=0), Pt.sum(axis=0), ref=ref_bin,
                                     min_exp=min_strat_exp)
    rr_n, p0_n, pk_n, nst_n, _ = mh(En.sum(axis=0), Pn.sum(axis=0), ref=ref_bin,
                                    min_exp=min_strat_exp)

    rng = np.random.default_rng(seed)
    nd, nb = Et.shape[0], Et.shape[2]
    bt = np.full((n_boot, nb), np.nan)
    bn = np.full((n_boot, nb), np.nan)
    bpk_t = np.full((n_boot, nb), np.nan)   # уровень (не отношение) — для CI на графике "уровень vs доза"
    bp0_t = np.full((n_boot, nb), np.nan)
    bpk_n = np.full((n_boot, nb), np.nan)
    for b in range(n_boot):
        c = np.bincount(rng.integers(0, nd, nd), minlength=nd).astype(np.float32)
        mh_t = mh(np.tensordot(c, Et, axes=1), np.tensordot(c, Pt, axes=1),
                  ref=ref_bin, min_exp=min_strat_exp)
        mh_n = mh(np.tensordot(c, En, axes=1), np.tensordot(c, Pn, axes=1),
                  ref=ref_bin, min_exp=min_strat_exp)
        bt[b], bp0_t[b], bpk_t[b] = mh_t[0], mh_t[1], mh_t[2]
        bn[b], bpk_n[b] = mh_n[0], mh_n[2]

    with np.errstate(invalid='ignore', divide='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        brrr = bt / bn

        def q(a):
            return np.nanpercentile(a, [2.5, 97.5], axis=0)

        ci_t, ci_n, ci_rrr = q(bt), q(bn), q(brrr)
        ci_pk_t, ci_p0_t, ci_pk_n = q(bpk_t), q(bp0_t), q(bpk_n)
    return {
        'rr_t': rr_t, 'rr_n': rr_n, 'rrr': rr_t / rr_n,
        'ci_t': ci_t, 'ci_n': ci_n, 'ci_rrr': ci_rrr,
        'p0_t': p0_t, 'pk_t': pk_t, 'p0_n': p0_n, 'pk_n': pk_n,
        'ci_pk_t': ci_pk_t, 'ci_p0_t': ci_p0_t, 'ci_pk_n': ci_pk_n,
        'nst': nst, 'expo': expo,
        # bpk_t/bpk_n — сырые бутстрап-реплики уровня pk_t/pk_n по (бутстрап × бин),
        # а не только их 2.5/97.5 перцентили (ci_pk_t/ci_pk_n) — нужны, когда CI
        # считается для ПАРНОЙ разности соседних бинов (одна и та же реплика дня
        # для обоих бинов), а не для каждого бина по отдельности. bp0_t — то же самое
        # для p0_t (опорный уровень пересчитывается на СВОЁМ общем носителе для
        # каждого k, то есть не константа по бинам) — нужен, чтобы вычесть из pk_t[k]
        # именно ЕГО собственную базу (excess[k] = pk_t[k]-p0_t[k]), а не считать
        # p0_t одним и тем же числом для всех бинов.
        'bpk_t': bpk_t, 'bpk_n': bpk_n, 'bp0_t': bp0_t,
        'raw': pooled(select_strata(accum['exp_true_d'], sc, night, True).sum(axis=0),
                      select_strata(Pt_d, sc, night, True).sum(axis=0)),
    }


def report_bin(res, min_exp=20000, min_strata=5, ref_bin=0, require_null=True):
    """Верхний бин дозы, на который можно ссылаться.

    res — результат `analyze`. Бин проходит, если rr_t конечен, экспозиция не
    меньше min_exp, страт не меньше min_strata; ref_bin не рассматривается.
    require_null=True дополнительно требует конечный rr_n — при null_lags=() он
    весь NaN, тогда нужен require_null=False.
    Возвращает индекс бина или None.
    """
    ok = (np.isfinite(res['rr_t']) & (res['expo'] >= min_exp) & (res['nst'] >= min_strata))
    if require_null:
        ok &= np.isfinite(res['rr_n'])
    ok[ref_bin] = False
    idx = np.flatnonzero(ok)
    return int(idx[-1]) if idx.size else None


def analyze_energy_lon(accum, si, db, ref_bin=0, n_boot=200, seed=0, min_strat_exp=500):
    """Эффект энергии молний при фиксированном их числе.

    accum — накопитель с осями `exp_en_d` / `pos_en_d`: `run_median_energy_lon`
    (медиана) или `run_accumulation(..., do_energy=True)` (среднее).
    si — индекс аппарата; db — бин дозы, внутри которого сравниваются классы энергии;
    ref_bin — опорный класс энергии; n_boot, seed, min_strat_exp — как в `analyze`.
    Null-модели нет.

    Возвращает (rr, ci, p_k, nst, expo) по классам энергии: рост MH относительно
    ref_bin, 95%-интервал (2 × классы), уровень verdict = 1, число страт, экспозиция.
    """
    Et_full = accum['exp_en_d'][:, si, ..., db, :]
    Pt_full = accum['pos_en_d'][:, si, ..., db, :]
    nd = Et_full.shape[0]
    E2 = Et_full.reshape(nd, -1, Et_full.shape[-1])
    P2 = Pt_full.reshape(nd, -1, Pt_full.shape[-1])
    rr, p_ref, p_k, nst, expo = mh(E2.sum(axis=0), P2.sum(axis=0), ref=ref_bin,
                                   min_exp=min_strat_exp)
    rng = np.random.default_rng(seed)
    bs = np.full((n_boot, E2.shape[2]), np.nan)
    for b in range(n_boot):
        c = np.bincount(rng.integers(0, nd, nd), minlength=nd).astype(np.float32)
        bs[b] = mh(np.tensordot(c, E2, axes=1), np.tensordot(c, P2, axes=1),
                  ref=ref_bin, min_exp=min_strat_exp)[0]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        ci = np.nanpercentile(bs, [2.5, 97.5], axis=0)
    return rr, ci, p_k, nst, expo


# ============================== sweep: R/Δt ==============================

def sweep_r_dt(
    df: pd.DataFrame,
    df_pred: pd.DataFrame,
    *,
    sc_list: Sequence[str],
    acc_days: Sequence[int],
    r_dt_pairs: Sequence[tuple],   # [(dose_r_km, dose_dt_min), ...]
    glat: float = 2.0,
    glon: float = 2.0,
    gdt_min: int = 5,
    null_lags: Sequence[int] = (3, 7, 14),
    null_symmetric: bool = False,
    null_season_safe: bool = True,
    night_sza_deg: Optional[float] = None,
    strata_scheme: str = 'lst',
    n_latb: int = 12,
    n_hourb: int = 8,
    n_lonb: int = 8,
    geometry: str = 'satellite',
    sat_height_km: float = 827.0,
    strata_by_geometry: bool = False,
    dose_edges: Optional[np.ndarray] = None,
    dose_edges_base_dt: Optional[float] = None,
    dose_edges_base_r: Optional[float] = None,
    verbose: bool = True,
) -> dict:
    """Накопители для набора пар (радиус, окно) за один проход по данным.

    Сетки молний и спутника от R и Δt не зависят и строятся один раз; для каждой
    пары пересчитываются только доза и накопление. Срез по энергии не считается.

    Общие аргументы — в докстринге модуля. Свои:
      r_dt_pairs — список (dose_r_km, dose_dt_min).
      dose_edges_base_dt — если задан, границы дозы для пары умножаются на
        dt_min / dose_edges_base_dt: сравнивается частота молний вместо их числа
        («≥3000 за 30 мин» ≈ «≥200 за 2 мин»).
      dose_edges_base_r — если задан, границы дополнительно умножаются на
        (r_km / dose_edges_base_r) ** 2 — поправка на площадь круга.
      Подписи бинов при масштабировании не меняются; фактические границы пары —
      в её `accum['dose_edges']`.

    Возвращает dict {(r_km, dt_min): накопитель} того же вида, что у
    `run_accumulation` (без `exp_en_d` / `pos_en_d`); к каждому применим `analyze`.
    """
    if strata_scheme not in ('lst', 'lon'):
        raise ValueError(f"strata_scheme должен быть 'lst' или 'lon', получено {strata_scheme!r}")
    if geometry not in ('satellite', 'footprint', 'conjugate'):
        raise ValueError(f"geometry должен быть 'satellite'/'footprint'/'conjugate', получено {geometry!r}")
    _mc = None
    if geometry != 'satellite':
        import mag_coords as _mc  # ленивый импорт — aacgmv2 нужен только для этой геометрии
    if night_sza_deg is None:
        night_sza_deg = wc.NIGHT_SZA_DEG
    if dose_edges is None:
        dose_edges = DEFAULT_DOSE_EDGES
    dose_edges = np.asarray(dose_edges, dtype=np.float64)
    r_dt_pairs = list(r_dt_pairs)

    max_lag = max(null_lags) if null_lags else 0
    lookahead = max_lag if null_symmetric else 0
    lag_offsets = tuple(-L for L in null_lags) + (tuple(null_lags) if null_symmetric else ())

    lat_edges = np.arange(-90.0, 90.0 + 1e-9, glat)
    lon_edges = np.arange(-180.0, 180.0 + 1e-9, glon)
    nlat, nlon = len(lat_edges) - 1, len(lon_edges) - 1
    nt = int(round(24 * 60 / gdt_min))
    lat_c = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_c = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    ncell = nt * nlat * nlon

    def _grid(lat, lon, tmin, weights=None):
        ti = np.clip((tmin / gdt_min).astype(np.int64), 0, nt - 1)
        ii = np.clip(((lat + 90.0) / glat).astype(np.int64), 0, nlat - 1)
        jj = np.clip(((lon + 180.0) / glon).astype(np.int64), 0, nlon - 1)
        idx = (ti * nlat + ii) * nlon + jj
        g = np.bincount(idx, weights=weights, minlength=ncell)
        return g.reshape(nt, nlat, nlon).astype(np.float32)

    # для каждой пары (R,Δt) — своя геометрия окрестности/окна и своя _dose_grid
    dose_geoms = {}
    dose_edges_by_pair = {}
    for r_km, dt_min in r_dt_pairs:
        pad = int(round(dt_min / gdt_min))
        hlat = int(np.ceil(r_km / 111.0 / glat))
        hlon = np.ceil(
            r_km / (111.0 * np.maximum(np.cos(np.radians(lat_c)), 0.05)) / glon
        ).astype(int)
        hlon = np.minimum(hlon, nlon // 2)
        dose_geoms[(r_km, dt_min)] = (pad, hlat, hlon)
        scale = 1.0
        if dose_edges_base_dt:
            scale *= dt_min / dose_edges_base_dt
        if dose_edges_base_r:
            scale *= (r_km / dose_edges_base_r) ** 2
        dose_edges_by_pair[(r_km, dt_min)] = dose_edges * scale if scale != 1.0 else dose_edges

    def _dose_grid(cur, prev, pad, hlat, hlon):
        G = np.concatenate([prev[nt - pad:], cur], axis=0)
        S = np.cumsum(G, axis=0, dtype=np.float32)
        W = S[pad:pad + nt] - S[0:nt]
        del G, S
        A = np.cumsum(np.concatenate([W, W, W], axis=2), axis=2, dtype=np.float32)
        L = np.empty_like(W)
        jj = np.arange(nlon)
        for i in range(nlat):
            h = int(hlon[i])
            if h >= nlon // 2:
                L[:, i, :] = (A[:, i, 2 * nlon - 1] - A[:, i, nlon - 1])[:, None]
            else:
                L[:, i, :] = A[:, i, nlon + jj + h] - A[:, i, nlon + jj - h - 1]
        del A, W
        C = np.cumsum(L, axis=1, dtype=np.float32)
        ii = np.arange(nlat)
        out = C[:, np.clip(ii + hlat, 0, nlat - 1), :].copy()
        lo = ii - hlat - 1
        m = lo >= 0
        out[:, m, :] -= C[:, lo[m], :]
        return out

    n_dose = len(dose_edges) - 1
    n_axis4 = n_hourb if strata_scheme == 'lst' else n_lonb
    n_sea, n_night = 4, 2
    sc_list = tuple(sc_list)
    n_sc = len(sc_list)
    sc_base = n_sc

    _pts = df_pred['timestamp'].values
    _plat, _plon = df_pred['geo_lat'].values, df_pred['geo_lon'].values
    _pver = df_pred['verdict'].values
    _pprob = df_pred['prob_whistle'].values.astype(np.float64)
    sc_to_code = {sc: i for i, sc in enumerate(sc_list)}
    pred_sccode = df_pred['sc'].astype(str).map(sc_to_code)
    if pred_sccode.isna().any():
        bad = sorted(set(df_pred['sc'].astype(str)) - set(sc_list))
        raise ValueError(f'В df_pred есть КА вне sc_list: {bad}')
    pred_sccode = pred_sccode.to_numpy(dtype=np.int64)

    pred_daycode = _pts.astype('datetime64[D]').astype(np.int64)
    _key = pred_daycode * sc_base + pred_sccode
    pred_order = np.argsort(_key, kind='stable')
    _ks = _key[pred_order]
    _uk, _us = np.unique(_ks, return_index=True)
    _ue = np.append(_us[1:], len(_ks))
    pred_slices = {int(k): (int(s), int(e)) for k, s, e in zip(_uk, _us, _ue)}
    del _key, _ks

    ae_t_all = df['datetime'].values
    l_lat_all = df['lat'].values.astype(np.float64)
    l_lon_all = df['lon'].values.astype(np.float64)

    acc_days = sorted(int(d) for d in acc_days)
    acc_set = set(acc_days)
    if not acc_days:
        raise ValueError('acc_days пуст')
    day_pos = {d: i for i, d in enumerate(acc_days)}

    grid_day_lo = acc_days[0] - max_lag - 1
    grid_day_hi = acc_days[-1] + lookahead
    grid_days = list(range(grid_day_lo, grid_day_hi + 1))

    shape7 = (len(acc_days), n_sc, n_latb, n_axis4, n_sea, n_night, n_dose)
    accums = {}
    for pair in r_dt_pairs:
        accums[pair] = {
            'exp_true_d': np.zeros(shape7, dtype=np.float32),
            'pos_true_d': np.zeros(shape7, dtype=np.float32),
            'prb_true_d': np.zeros(shape7, dtype=np.float32),
            'exp_null_d': np.zeros(shape7, dtype=np.float32),
            'pos_null_d': np.zeros(shape7, dtype=np.float32),
            'prb_null_d': np.zeros(shape7, dtype=np.float32),
        }

    if verbose:
        mb = 6 * np.prod(shape7) * 4 / 2 ** 20 * len(r_dt_pairs)
        print(f'Sweep R/Δt: {len(r_dt_pairs)} пар, накопители ~{mb:.0f} МБ суммарно; '
              f'сетки молний/спутника — общие, строятся один раз')

    cnt_grids = {}
    zero_grid = np.zeros((nt, nlat, nlon), dtype=np.float32)

    for n_done, grid_day in enumerate(grid_days, start=1):
        d0 = np.datetime64(grid_day, 'D').astype('datetime64[ns]')
        d1 = np.datetime64(grid_day + 1, 'D').astype('datetime64[ns]')
        s0, s1 = np.searchsorted(ae_t_all, [d0, d1])
        if s1 > s0:
            l_tmin = (ae_t_all[s0:s1] - d0) / np.timedelta64(1, 'm')
            cnt_grids[grid_day] = _grid(l_lat_all[s0:s1], l_lon_all[s0:s1], l_tmin)
        else:
            cnt_grids[grid_day] = zero_grid.copy()

        day = grid_day - lookahead
        if day in acc_set:
            di = day_pos[day]
            month_day = int(str(np.datetime64(day, 'D'))[5:7])

            lag_days = []
            for off in lag_offsets:
                nd = day + off
                if nd not in cnt_grids:
                    continue
                if null_season_safe:
                    month_nd = int(str(np.datetime64(nd, 'D'))[5:7])
                    if not wc.same_season_quarter(month_day, month_nd):
                        continue
                lag_days.append(nd)

            # --- общая (не зависящая от R/Δt) часть: сетки спутника по каждому (day,sc) ---
            sat_cache = {}
            for si, sc in enumerate(sc_list):
                sl = pred_slices.get(day * sc_base + si)
                if sl is None:
                    continue
                idx = pred_order[sl[0]:sl[1]]
                lat = _plat[idx].astype(np.float64)
                lon = _plon[idx].astype(np.float64)
                ver = _pver[idx]
                prb = _pprob[idx]
                tt = _pts[idx]
                tmin = (tt - tt.astype('datetime64[D]')) / np.timedelta64(1, 'm')
                night_sat = (wc.solar_zenith_deg(tt, lat, lon) > night_sza_deg).astype(np.int8)

                exp_g = _grid(lat, lon, tmin).ravel()
                pos_g = _grid(lat[ver == 1], lon[ver == 1], tmin[ver == 1]).ravel()
                prb_g = _grid(lat, lon, tmin, weights=prb).ravel()
                cells = np.flatnonzero(exp_g > 0)
                if cells.size == 0:
                    continue

                ti = cells // (nlat * nlon)
                rem = cells % (nlat * nlon)
                lat_cc = lat_c[rem // nlon]
                lon_cc = lon_c[rem % nlon]

                # geometry/strata_by_geometry — см. докстринг и run_accumulation(): footprint/
                # conjugate считается один раз на ячейку (lat_cc/lon_cc), не на наблюдение.
                if geometry == 'satellite':
                    dose_idx = cells
                    g_lat_cc, g_lon_cc = lat_cc, lon_cc
                else:
                    day_date = np.datetime64(day, 'D').astype('datetime64[s]').astype(object)
                    fp_lat_c, fp_lon_c, cj_lat_c, cj_lon_c = _mc.footprint_conjugate_arr(
                        lat_cc, lon_cc, sat_height_km, day_date)
                    gg_lat, gg_lon = (fp_lat_c, fp_lon_c) if geometry == 'footprint' else (cj_lat_c, cj_lon_c)
                    valid_cc = np.isfinite(gg_lat) & np.isfinite(gg_lon)
                    g_lat_cc = np.where(valid_cc, gg_lat, lat_cc)
                    g_lon_cc = np.where(valid_cc, gg_lon, lon_cc)
                    g_ii = np.clip(((g_lat_cc + 90.0) / glat).astype(np.int64), 0, nlat - 1)
                    g_jj = np.clip(((g_lon_cc + 180.0) / glon).astype(np.int64), 0, nlon - 1)
                    dose_idx = (ti * nlat + g_ii) * nlon + g_jj

                strata_lat_cc = g_lat_cc if strata_by_geometry else lat_cc
                strata_lon_cc = g_lon_cc if strata_by_geometry else lon_cc

                latb = np.clip(((strata_lat_cc + 90.0) / (180.0 / n_latb)).astype(np.int64),
                               0, n_latb - 1)
                if strata_scheme == 'lst':
                    lst = np.mod(ti * gdt_min / 60.0 + gdt_min / 120.0 + strata_lon_cc / 15.0, 24.0)
                    axis4 = np.clip((lst / (24.0 / n_hourb)).astype(np.int64), 0, n_hourb - 1)
                else:
                    axis4 = np.clip(((strata_lon_cc + 180.0) / (360.0 / n_lonb)).astype(np.int64),
                                    0, n_lonb - 1)
                _, seab = wc.season_index(strata_lat_cc, month_day)
                seab = seab.astype(np.int64)

                if strata_by_geometry and geometry != 'satellite':
                    ti_obs = np.clip((tmin / gdt_min).astype(np.int64), 0, nt - 1)
                    ii_obs = np.clip(((lat + 90.0) / glat).astype(np.int64), 0, nlat - 1)
                    jj_obs = np.clip(((lon + 180.0) / glon).astype(np.int64), 0, nlon - 1)
                    obs_cell = (ti_obs * nlat + ii_obs) * nlon + jj_obs
                    pos_in_cells = np.searchsorted(cells, obs_cell)
                    night = (wc.solar_zenith_deg(tt, g_lat_cc[pos_in_cells], g_lon_cc[pos_in_cells])
                             > night_sza_deg).astype(np.int8)
                else:
                    night = night_sat
                ngt_g = _grid(lat, lon, tmin, weights=night.astype(np.float64)).ravel()
                nightb = (ngt_g[cells] / exp_g[cells] > 0.5).astype(np.int64)

                sat_cache[si] = (dose_idx, exp_g[cells], pos_g[cells], prb_g[cells],
                                 latb, axis4, seab, nightb)

            # --- часть, зависящая от R/Δt: доза + накопление, для каждой пары своя ---
            for pair in r_dt_pairs:
                pad, hlat, hlon = dose_geoms[pair]
                pair_edges = dose_edges_by_pair[pair]
                dose_t = _dose_grid(cnt_grids[day], cnt_grids.get(day - 1, zero_grid),
                                    pad, hlat, hlon).ravel()
                dose_nulls = [_dose_grid(cnt_grids[nd], cnt_grids.get(nd - 1, zero_grid),
                                         pad, hlat, hlon).ravel()
                             for nd in lag_days]
                acc = accums[pair]
                for si, (dose_idx, e_vals, p_vals, b_vals, latb, axis4, seab, nightb) in sat_cache.items():
                    dt_cells = dose_t[dose_idx]
                    db_t = np.clip(np.searchsorted(pair_edges, dt_cells, side='right') - 1,
                                   0, n_dose - 1)
                    key_t = (di, si, latb, axis4, seab, nightb, db_t)
                    np.add.at(acc['exp_true_d'], key_t, e_vals)
                    np.add.at(acc['pos_true_d'], key_t, p_vals)
                    np.add.at(acc['prb_true_d'], key_t, b_vals)
                    for dose_n in dose_nulls:
                        db_n = np.clip(np.searchsorted(pair_edges, dose_n[dose_idx], side='right') - 1,
                                       0, n_dose - 1)
                        key_n = (di, si, latb, axis4, seab, nightb, db_n)
                        np.add.at(acc['exp_null_d'], key_n, e_vals)
                        np.add.at(acc['pos_null_d'], key_n, p_vals)
                        np.add.at(acc['prb_null_d'], key_n, b_vals)

        for old in [k for k in cnt_grids if k < day - max_lag - 1]:
            cnt_grids.pop(old, None)
        if verbose and n_done % 40 == 0:
            print(f'  ... {n_done}/{len(grid_days)} дней', flush=True)
            gc.collect()

    del cnt_grids
    gc.collect()

    out = {}
    for pair in r_dt_pairs:
        acc = accums[pair]
        acc.update({
            'exp_en_d': None, 'pos_en_d': None,
            'acc_days': acc_days, 'day_pos': day_pos,
            'sc_list': sc_list, 'n_sc': n_sc, 'n_latb': n_latb, 'n_axis4': n_axis4,
            'n_hourb': n_hourb, 'n_lonb': n_lonb, 'n_sea': n_sea, 'n_night': n_night,
            'n_dose': n_dose, 'n_eb': 0,
            'strata_scheme': strata_scheme, 'dose_edges': dose_edges_by_pair[pair],
            'dose_edges_base_dt': dose_edges_base_dt,
            'dose_r_km': pair[0], 'dose_dt_min': pair[1], 'gdt_min': gdt_min,
            'null_lags': tuple(null_lags), 'null_symmetric': null_symmetric,
            'null_season_safe': null_season_safe, 'night_sza_deg': night_sza_deg,
            'geometry': geometry, 'sat_height_km': sat_height_km,
            'strata_by_geometry': strata_by_geometry,
        })
        out[pair] = acc
        if verbose:
            e = acc['exp_true_d'].sum(dtype=np.float64)
            p = acc['pos_true_d'].sum(dtype=np.float64)
            print(f'  R={pair[0]:.0f}км Δt={pair[1]}мин: экспозиция={e:,.0f}, '
                  f'verdict=1={p:,.0f} ({100*p/max(e,1):.2f}%)')
    return out


# ============================== sweep: порог ночи ==============================

def sweep_night_sza(
    df: pd.DataFrame,
    df_pred: pd.DataFrame,
    *,
    sc_list: Sequence[str],
    acc_days: Sequence[int],
    sza_list: Sequence[float],
    dose_r_km: float = 1000.0,
    dose_dt_min: int = 30,
    glat: float = 2.0,
    glon: float = 2.0,
    gdt_min: int = 5,
    null_lags: Sequence[int] = (3, 7, 14),
    null_symmetric: bool = False,
    null_season_safe: bool = True,
    strata_scheme: str = 'lst',
    n_latb: int = 12,
    n_hourb: int = 8,
    n_lonb: int = 8,
    dose_edges: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> dict:
    """Накопители для набора порогов ночи за один проход по данным.

    Доза от порога не зависит и считается один раз; для каждого порога
    пересчитывается только деление на день и ночь. Геометрия — только 'satellite'.

    Общие аргументы — в докстринге модуля. Свой:
      sza_list — пороги зенитного угла, градусы.

    Возвращает dict {sza_deg: накопитель}; к каждому применим `analyze`.
    """
    if strata_scheme not in ('lst', 'lon'):
        raise ValueError(f"strata_scheme должен быть 'lst' или 'lon', получено {strata_scheme!r}")
    if dose_edges is None:
        dose_edges = DEFAULT_DOSE_EDGES
    dose_edges = np.asarray(dose_edges, dtype=np.float64)
    sza_list = list(sza_list)

    max_lag = max(null_lags) if null_lags else 0
    lookahead = max_lag if null_symmetric else 0
    lag_offsets = tuple(-L for L in null_lags) + (tuple(null_lags) if null_symmetric else ())

    lat_edges = np.arange(-90.0, 90.0 + 1e-9, glat)
    lon_edges = np.arange(-180.0, 180.0 + 1e-9, glon)
    nlat, nlon = len(lat_edges) - 1, len(lon_edges) - 1
    nt = int(round(24 * 60 / gdt_min))
    pad = int(round(dose_dt_min / gdt_min))
    lat_c = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_c = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    ncell = nt * nlat * nlon

    hlat = int(np.ceil(dose_r_km / 111.0 / glat))
    hlon = np.ceil(
        dose_r_km / (111.0 * np.maximum(np.cos(np.radians(lat_c)), 0.05)) / glon
    ).astype(int)
    hlon = np.minimum(hlon, nlon // 2)

    def _grid(lat, lon, tmin, weights=None):
        ti = np.clip((tmin / gdt_min).astype(np.int64), 0, nt - 1)
        ii = np.clip(((lat + 90.0) / glat).astype(np.int64), 0, nlat - 1)
        jj = np.clip(((lon + 180.0) / glon).astype(np.int64), 0, nlon - 1)
        idx = (ti * nlat + ii) * nlon + jj
        g = np.bincount(idx, weights=weights, minlength=ncell)
        return g.reshape(nt, nlat, nlon).astype(np.float32)

    def _dose_grid(cur, prev):
        G = np.concatenate([prev[nt - pad:], cur], axis=0)
        S = np.cumsum(G, axis=0, dtype=np.float32)
        W = S[pad:pad + nt] - S[0:nt]
        del G, S
        A = np.cumsum(np.concatenate([W, W, W], axis=2), axis=2, dtype=np.float32)
        L = np.empty_like(W)
        jj = np.arange(nlon)
        for i in range(nlat):
            h = int(hlon[i])
            if h >= nlon // 2:
                L[:, i, :] = (A[:, i, 2 * nlon - 1] - A[:, i, nlon - 1])[:, None]
            else:
                L[:, i, :] = A[:, i, nlon + jj + h] - A[:, i, nlon + jj - h - 1]
        del A, W
        C = np.cumsum(L, axis=1, dtype=np.float32)
        ii = np.arange(nlat)
        out = C[:, np.clip(ii + hlat, 0, nlat - 1), :].copy()
        lo = ii - hlat - 1
        m = lo >= 0
        out[:, m, :] -= C[:, lo[m], :]
        return out

    n_dose = len(dose_edges) - 1
    n_axis4 = n_hourb if strata_scheme == 'lst' else n_lonb
    n_sea, n_night = 4, 2
    sc_list = tuple(sc_list)
    n_sc = len(sc_list)
    sc_base = n_sc

    _pts = df_pred['timestamp'].values
    _plat, _plon = df_pred['geo_lat'].values, df_pred['geo_lon'].values
    _pver = df_pred['verdict'].values
    _pprob = df_pred['prob_whistle'].values.astype(np.float64)
    sc_to_code = {sc: i for i, sc in enumerate(sc_list)}
    pred_sccode = df_pred['sc'].astype(str).map(sc_to_code)
    if pred_sccode.isna().any():
        bad = sorted(set(df_pred['sc'].astype(str)) - set(sc_list))
        raise ValueError(f'В df_pred есть КА вне sc_list: {bad}')
    pred_sccode = pred_sccode.to_numpy(dtype=np.int64)

    pred_daycode = _pts.astype('datetime64[D]').astype(np.int64)
    _key = pred_daycode * sc_base + pred_sccode
    pred_order = np.argsort(_key, kind='stable')
    _ks = _key[pred_order]
    _uk, _us = np.unique(_ks, return_index=True)
    _ue = np.append(_us[1:], len(_ks))
    pred_slices = {int(k): (int(s), int(e)) for k, s, e in zip(_uk, _us, _ue)}
    del _key, _ks

    ae_t_all = df['datetime'].values
    l_lat_all = df['lat'].values.astype(np.float64)
    l_lon_all = df['lon'].values.astype(np.float64)

    acc_days = sorted(int(d) for d in acc_days)
    acc_set = set(acc_days)
    if not acc_days:
        raise ValueError('acc_days пуст')
    day_pos = {d: i for i, d in enumerate(acc_days)}

    grid_day_lo = acc_days[0] - max_lag - 1
    grid_day_hi = acc_days[-1] + lookahead
    grid_days = list(range(grid_day_lo, grid_day_hi + 1))

    shape7 = (len(acc_days), n_sc, n_latb, n_axis4, n_sea, n_night, n_dose)
    accums = {}
    for sza in sza_list:
        accums[sza] = {
            'exp_true_d': np.zeros(shape7, dtype=np.float32),
            'pos_true_d': np.zeros(shape7, dtype=np.float32),
            'prb_true_d': np.zeros(shape7, dtype=np.float32),
            'exp_null_d': np.zeros(shape7, dtype=np.float32),
            'pos_null_d': np.zeros(shape7, dtype=np.float32),
            'prb_null_d': np.zeros(shape7, dtype=np.float32),
        }

    if verbose:
        mb = 6 * np.prod(shape7) * 4 / 2 ** 20 * len(sza_list)
        print(f'Sweep NIGHT_SZA_DEG: {len(sza_list)} порогов {sza_list}, '
              f'накопители ~{mb:.0f} МБ суммарно; доза общая, считается один раз')

    cnt_grids = {}
    zero_grid = np.zeros((nt, nlat, nlon), dtype=np.float32)

    for n_done, grid_day in enumerate(grid_days, start=1):
        d0 = np.datetime64(grid_day, 'D').astype('datetime64[ns]')
        d1 = np.datetime64(grid_day + 1, 'D').astype('datetime64[ns]')
        s0, s1 = np.searchsorted(ae_t_all, [d0, d1])
        if s1 > s0:
            l_tmin = (ae_t_all[s0:s1] - d0) / np.timedelta64(1, 'm')
            cnt_grids[grid_day] = _grid(l_lat_all[s0:s1], l_lon_all[s0:s1], l_tmin)
        else:
            cnt_grids[grid_day] = zero_grid.copy()

        day = grid_day - lookahead
        if day in acc_set:
            di = day_pos[day]
            month_day = int(str(np.datetime64(day, 'D'))[5:7])
            dose_t = _dose_grid(cnt_grids[day], cnt_grids.get(day - 1, zero_grid)).ravel()

            lag_days = []
            for off in lag_offsets:
                nd = day + off
                if nd not in cnt_grids:
                    continue
                if null_season_safe:
                    month_nd = int(str(np.datetime64(nd, 'D'))[5:7])
                    if not wc.same_season_quarter(month_day, month_nd):
                        continue
                lag_days.append(nd)
            dose_nulls = [_dose_grid(cnt_grids[nd], cnt_grids.get(nd - 1, zero_grid)).ravel()
                         for nd in lag_days]

            for si, sc in enumerate(sc_list):
                sl = pred_slices.get(day * sc_base + si)
                if sl is None:
                    continue
                idx = pred_order[sl[0]:sl[1]]
                lat = _plat[idx].astype(np.float64)
                lon = _plon[idx].astype(np.float64)
                ver = _pver[idx]
                prb = _pprob[idx]
                tt = _pts[idx]
                tmin = (tt - tt.astype('datetime64[D]')) / np.timedelta64(1, 'm')

                # SZA-независимая часть: считаем один раз
                exp_g = _grid(lat, lon, tmin).ravel()
                pos_g = _grid(lat[ver == 1], lon[ver == 1], tmin[ver == 1]).ravel()
                prb_g = _grid(lat, lon, tmin, weights=prb).ravel()
                cells = np.flatnonzero(exp_g > 0)
                if cells.size == 0:
                    continue

                ti = cells // (nlat * nlon)
                rem = cells % (nlat * nlon)
                lat_cc = lat_c[rem // nlon]
                lon_cc = lon_c[rem % nlon]
                latb = np.clip(((lat_cc + 90.0) / (180.0 / n_latb)).astype(np.int64),
                               0, n_latb - 1)
                if strata_scheme == 'lst':
                    lst = np.mod(ti * gdt_min / 60.0 + gdt_min / 120.0 + lon_cc / 15.0, 24.0)
                    axis4 = np.clip((lst / (24.0 / n_hourb)).astype(np.int64), 0, n_hourb - 1)
                else:
                    axis4 = np.clip(((lon_cc + 180.0) / (360.0 / n_lonb)).astype(np.int64),
                                    0, n_lonb - 1)
                _, seab = wc.season_index(lat_cc, month_day)
                seab = seab.astype(np.int64)

                e_vals, p_vals, b_vals = exp_g[cells], pos_g[cells], prb_g[cells]
                dt_cells = dose_t[cells]
                db_t = np.clip(np.searchsorted(dose_edges, dt_cells, side='right') - 1,
                               0, n_dose - 1)
                db_n_list = [np.clip(np.searchsorted(dose_edges, dose_n[cells], side='right') - 1,
                                     0, n_dose - 1) for dose_n in dose_nulls]

                # SZA-зависимая часть: только ночь/день, для каждого порога отдельно
                solzen = wc.solar_zenith_deg(tt, lat, lon)
                for sza in sza_list:
                    night = (solzen > sza).astype(np.float64)
                    ngt_g = _grid(lat, lon, tmin, weights=night).ravel()
                    nightb = (ngt_g[cells] / exp_g[cells] > 0.5).astype(np.int64)

                    acc = accums[sza]
                    key_t = (di, si, latb, axis4, seab, nightb, db_t)
                    np.add.at(acc['exp_true_d'], key_t, e_vals)
                    np.add.at(acc['pos_true_d'], key_t, p_vals)
                    np.add.at(acc['prb_true_d'], key_t, b_vals)
                    for db_n in db_n_list:
                        key_n = (di, si, latb, axis4, seab, nightb, db_n)
                        np.add.at(acc['exp_null_d'], key_n, e_vals)
                        np.add.at(acc['pos_null_d'], key_n, p_vals)
                        np.add.at(acc['prb_null_d'], key_n, b_vals)

        for old in [k for k in cnt_grids if k < day - max_lag - 1]:
            cnt_grids.pop(old, None)
        if verbose and n_done % 40 == 0:
            print(f'  ... {n_done}/{len(grid_days)} дней', flush=True)
            gc.collect()

    del cnt_grids
    gc.collect()

    out = {}
    for sza in sza_list:
        acc = accums[sza]
        acc.update({
            'exp_en_d': None, 'pos_en_d': None,
            'acc_days': acc_days, 'day_pos': day_pos,
            'sc_list': sc_list, 'n_sc': n_sc, 'n_latb': n_latb, 'n_axis4': n_axis4,
            'n_hourb': n_hourb, 'n_lonb': n_lonb, 'n_sea': n_sea, 'n_night': n_night,
            'n_dose': n_dose, 'n_eb': 0,
            'strata_scheme': strata_scheme, 'dose_edges': dose_edges,
            'dose_r_km': dose_r_km, 'dose_dt_min': dose_dt_min, 'gdt_min': gdt_min,
            'null_lags': tuple(null_lags), 'null_symmetric': null_symmetric,
            'null_season_safe': null_season_safe, 'night_sza_deg': sza,
        })
        out[sza] = acc
        if verbose:
            e = acc['exp_true_d'].sum(dtype=np.float64)
            p = acc['pos_true_d'].sum(dtype=np.float64)
            print(f'  SZA>{sza:.0f}°: экспозиция={e:,.0f}, verdict=1={p:,.0f} '
                  f'({100*p/max(e,1):.2f}%)')
    return out
