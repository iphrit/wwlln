"""Магнитные координаты: геомагнитный диполь, трассировка силовой линии IGRF,
footprint/conjugate через aacgmv2.

Для расчётов — `footprint_conjugate_arr` (AACGM-v2, готовые коэффициенты,
~1.4 мкс на точку). `trace_field_line` — собственная трассировка по IGRF
(ppigrf): footprint совпадает с aacgmv2 до нескольких км, но считает секунды
на точку и для conjugate на высоких широтах иногда не сходится; оставлена для
проверки. apexpy не используется: требует Fortran-компилятор.
"""
from __future__ import annotations

import aacgmv2
import numpy as np
import ppigrf

RE_KM = 6371.2  # средний радиус Земли — стандартный для работы с IGRF


# ============================== центрированный диполь ==============================
# Быстрая, не требующая трассировки оценка геомагнитных координат — использована
# для валидации подхода против готовых mag_lat/mag_lon в файлах (см. диалог:
# медиана расхождения 0.06° по широте, 0.11° по долготе на выборке ka1).

def dipole_pole(date):
    """Геомагнитный северный полюс (lat, lon), градусы, на дату date — из g10/g11/h11 IGRF
    (берётся ближайшая эпоха модели)."""
    g, h = ppigrf.ppigrf.read_shc(ppigrf.ppigrf.shc_fn)
    idx = g.index.get_indexer([np.datetime64(date, 'Y').astype('datetime64[D]')], method='nearest')[0]
    g10, g11 = g.iloc[idx][(1, 0)], g.iloc[idx][(1, 1)]
    h11 = h.iloc[idx][(1, 1)]
    B0 = np.sqrt(g10 ** 2 + g11 ** 2 + h11 ** 2)
    theta0 = np.arccos(-g10 / B0)
    lat_p = np.degrees(np.pi / 2 - theta0)
    lon_p = np.degrees(np.arctan2(-h11, -g11))
    return lat_p, lon_p


def dipole_maglatlon(lat, lon, pole):
    """Геомагнитная широта/долгота точки (lat,lon) в системе центрированного
    диполя с полюсом `pole=(lat_p,lon_p)`. Векторизовано (lat/lon — массивы)."""
    lat_p, lon_p = pole
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    latp_r, lonp_r = np.radians(lat_p), np.radians(lon_p)
    theta, theta0 = np.pi / 2 - lat_r, np.pi / 2 - latp_r
    dlon = lon_r - lonp_r
    cos_tm = np.clip(np.cos(theta) * np.cos(theta0) + np.sin(theta) * np.sin(theta0) * np.cos(dlon),
                     -1, 1)
    theta_m = np.arccos(cos_tm)
    mlat = np.degrees(np.pi / 2 - theta_m)
    sin_tm = np.sin(theta_m)
    safe = np.where(sin_tm == 0, 1, sin_tm)
    sin_pm = np.sin(theta) * np.sin(dlon) / safe
    cos_pm = (np.cos(theta) * np.sin(theta0) - np.sin(theta) * np.cos(theta0) * np.cos(dlon)) / safe
    mlon = np.degrees(np.arctan2(sin_pm, cos_pm))
    return mlat, mlon


# ============================== трассировка силовой линии ==============================

def _geod_to_cart(lat_deg, lon_deg, r_km):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return np.array([r_km * np.cos(lat) * np.cos(lon),
                     r_km * np.cos(lat) * np.sin(lon),
                     r_km * np.sin(lat)])


def _cart_to_geod(pos):
    x, y, z = pos
    r = np.sqrt(x * x + y * y + z * z)
    lat = np.degrees(np.arcsin(np.clip(z / r, -1, 1)))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon, r


def _bfield_cart(pos, date, coeff_fn=None):
    """Вектор B в декартовых координатах (нТл) в точке `pos` (км, ECI-подобная
    геоцентрическая декартова система)."""
    lat, lon, r = _cart_to_geod(pos)
    theta = 90.0 - lat
    kw = {} if coeff_fn is None else {'coeff_fn': coeff_fn}
    Br, Bt, Bp = ppigrf.igrf_gc(r, theta, lon, date, **kw)
    Br, Bt, Bp = float(np.asarray(Br).ravel()[0]), float(np.asarray(Bt).ravel()[0]), float(np.asarray(Bp).ravel()[0])
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    r_hat = np.array([np.cos(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.sin(lon_r), np.sin(lat_r)])
    theta_hat = np.array([np.sin(lat_r) * np.cos(lon_r), np.sin(lat_r) * np.sin(lon_r), -np.cos(lat_r)])
    phi_hat = np.array([-np.sin(lon_r), np.cos(lon_r), 0.0])
    return Br * r_hat + Bt * theta_hat + Bp * phi_hat


def trace_field_line(lat_deg, lon_deg, h_km, date, target_h_km=100.0, step_km=10.0,
                     max_steps=3000, conjugate=False, coeff_fn=None):
    """Трассирует силовую линию IGRF от (lat, lon, h_km) до высоты `target_h_km`.

    conjugate=False — footprint: ближайшее пересечение той же силовой линии на
    целевой высоте (то же полушарие, короткий путь вниз).
    conjugate=True — сопряжённая точка: тот же старт, но шаг в противоположном
    направлении вдоль линии — сначала вверх к вершине дуги, затем вниз в другом
    полушарии, до пересечения целевой высоты там.

    Возвращает (lat, lon) в градусах, либо (nan, nan), если не сошлось
    (например, силовая линия открытая/уходит далеко — редко на широтах,
    актуальных для гроз/свистов).
    """
    r0 = RE_KM + h_km
    r_ref = RE_KM + target_h_km
    pos = _geod_to_cart(lat_deg, lon_deg, r0)

    B0 = _bfield_cart(pos, date, coeff_fn)
    r_hat0 = pos / np.linalg.norm(pos)
    br_sign = np.sign(np.dot(B0, r_hat0))
    if br_sign == 0:
        return np.nan, np.nan
    # footprint: шаг так, чтобы r убывал сразу; conjugate: чтобы сперва рос
    step_sign = float(br_sign) if conjugate else -float(br_sign)

    r_max_allowed = r0 + 20000.0  # защита от расходимости/открытых линий
    for _ in range(max_steps):
        Bvec = _bfield_cart(pos, date, coeff_fn)
        Bnorm = np.linalg.norm(Bvec)
        if not np.isfinite(Bnorm) or Bnorm == 0:
            return np.nan, np.nan
        pos = pos + step_sign * step_km * Bvec / Bnorm
        r_now = np.linalg.norm(pos)
        if r_now <= r_ref:
            lat, lon, _ = _cart_to_geod(pos)
            return lat, lon
        if r_now > r_max_allowed:
            return np.nan, np.nan
    return np.nan, np.nan


# ============================== AACGM-v2 (aacgmv2) — footprint/conjugate ==============================
# Быстрее и авторитетнее собственной трассировки: используются готовые
# коэффициенты AACGM-v2 (научный стандарт), а не итеративное интегрирование
# силовой линии. Проверено на реальных точках орбиты — footprint совпадает с
# результатом ручного трассировщика (trace_field_line) с точностью до
# нескольких км; на широте, где ручной трассировщик не сходился за разумное
# число шагов (высокие широты, длинная дуга до conjugate), aacgmv2 даёт ответ
# мгновенно, т.к. не интегрирует линию, а использует аппроксимацию по
# сферическим гармоникам. Скорость: ~1.4 мкс/точка (векторизовано) — весь
# датасет (43 млн строк) обрабатывается за единицы минут, без кэша по
# грубой сетке.

def footprint_conjugate_arr(lat, lon, h_km, dtime, target_h_km=100.0):
    """Footprint и conjugate для массива точек (lat, lon) на высоте h_km, на
    одну дату `dtime` (aacgmv2 берёт один datetime на вызов — вызывать
    отдельно на каждые сутки, а не на весь период сразу, для точности
    (вековой ход поля) это не обязательно, но так проще присоединить к
    циклу по суткам, который и так уже есть в run_accumulation).

    Возвращает (fp_lat, fp_lon, cj_lat, cj_lon) — массивы той же длины.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    h = np.full(lat.shape, h_km, dtype=np.float64)
    mlat, mlon, _ = aacgmv2.convert_latlon_arr(lat, lon, h, dtime, method_code='G2A')
    h_ref = np.full(lat.shape, target_h_km, dtype=np.float64)
    fp_lat, fp_lon, _ = aacgmv2.convert_latlon_arr(mlat, mlon, h_ref, dtime, method_code='A2G')
    cj_lat, cj_lon, _ = aacgmv2.convert_latlon_arr(-mlat, mlon, h_ref, dtime, method_code='A2G')
    return fp_lat, fp_lon, cj_lat, cj_lon


def geomagnetic_arr(lat, lon, h_km, dtime):
    """Геомагнитные (AACGM) широта и долгота точек.

    lat, lon — градусы (массивы); h_km — высота точек, км; dtime — datetime
    (одна дата на вызов). Возвращает (mlat, mlon). Тот же шаг G2A, что внутри
    `footprint_conjugate_arr`.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    h = np.full(lat.shape, h_km, dtype=np.float64)
    mlat, mlon, _ = aacgmv2.convert_latlon_arr(lat, lon, h, dtime, method_code='G2A')
    return mlat, mlon
