"""Общие определения для всех ноутбуков: ночь, сезоны, пути, сутки, загрузка данных.

«Ночь», «сезон» и «период анализа» везде считаются этими функциями. Ночь —
солнечный зенитный угол на поверхности больше `NIGHT_SZA_DEG`; колонка
`is_night` из файлов предсказаний (тень Земли на высоте орбиты, SZA > 117.6°)
в анализе не используется.

Константы, которые меняют поведение всех блоков сразу: `NIGHT_SZA_DEG`,
`PRED_MSK_OFFSET_HOURS`, таблицы сезонов `_SEA_N` / `_SEA_S`.

Кэш молний `cache/AE_CACHE_NAME` собирает `ensure_data` при первой загрузке
или `build_ae_cache`.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

# ============================== ночь ==============================
# Порог по солнечному зенитному углу НА ПОВЕРХНОСТИ. 98° — солнце на 8° ниже
# горизонта: к этому моменту D-область уже перестала подсвечиваться и ОНЧ-волна
# проходит вверх без сильного поглощения. Именно это важно для свиста, а не
# освещён ли сам аппарат на высоте 827 км.
NIGHT_SZA_DEG = 98.0


def solar_zenith_deg(dt64_utc, lat, lon) -> np.ndarray:
    """Солнечный зенитный угол (°) на поверхности. NOAA-приближение, точность ~0.2°.

    dt64_utc — моменты UTC (массив datetime64); lat, lon — градусы, массивы той же
    длины или скаляры.
    """
    dti = pd.DatetimeIndex(dt64_utc)
    doy = dti.dayofyear.to_numpy(dtype=np.float64)
    hh = (dti.hour.to_numpy(dtype=np.float64)
          + dti.minute.to_numpy(dtype=np.float64) / 60.0
          + dti.second.to_numpy(dtype=np.float64) / 3600.0)
    g = 2.0 * np.pi / 365.0 * (doy - 1.0 + (hh - 12.0) / 24.0)
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                    - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    dec = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g)
           - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
           - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    tst = np.mod(hh * 60.0 + eqt + 4.0 * np.asarray(lon, dtype=np.float64), 1440.0)
    ha = np.deg2rad(tst / 4.0 - 180.0)
    la = np.deg2rad(np.asarray(lat, dtype=np.float64))
    cz = np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha)
    return np.degrees(np.arccos(np.clip(cz, -1.0, 1.0)))


def night_mask(dt64_utc, lat, lon, sza_deg: float = NIGHT_SZA_DEG) -> np.ndarray:
    """1 — под точкой (lat, lon) ночь, 0 — день. Единое определение для всех блоков.

    Аргументы как у `solar_zenith_deg`; sza_deg — порог ночи, по умолчанию
    `NIGHT_SZA_DEG`. Возвращает int8-массив.
    """
    return (solar_zenith_deg(dt64_utc, lat, lon) > sza_deg).astype(np.int8)


# ============================== сезоны ==============================
# Сезон локальный: в южном полушарии зеркальный (JJA = «зима»).
SEASON_ORDER = ('зима', 'весна', 'лето', 'осень')
_SEA_N = np.array([-1, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0], dtype=np.int8)  # индекс по месяцу
_SEA_S = np.array([-1, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1, 2], dtype=np.int8)
SEASON_TO_I = {s: i for i, s in enumerate(SEASON_ORDER)}


def season_index(lat, month):
    """Локальный сезон точки.

    lat — широта (°), month — номер месяца 1–12 (массивы одной длины или скаляры).
    Возвращает (hem, sea): полушарие 0 = N / 1 = S и индекс сезона в
    `SEASON_ORDER`. В южном полушарии сезоны сдвинуты на полгода.
    """
    hem = (np.asarray(lat) < 0).astype(np.int8)
    sea = np.where(hem == 0, _SEA_N[month], _SEA_S[month]).astype(np.int8)
    return hem, sea


def same_season_quarter(month_a: int, month_b: int) -> bool:
    """Месяцы в одном календарном 3-месячном квартале — независимо от полушария.

    Границы кварталов у `_SEA_N`/`_SEA_S` одинаковые (различаются только
    названия сезонов), поэтому сравнение достаточно делать по одной таблице.
    Используется для null-лагов: пропускать лаг, если день и день±L
    оказались в разных сезонах.
    """
    return bool(_SEA_N[month_a] == _SEA_N[month_b])


def require_globals(env: dict, names, hint: str = 'выполните ячейку загрузки (основной блок) в этом же ядре перед этой ячейкой') -> None:
    """Явная проверка, что нужные переменные уже есть в окружении ноутбука.

    Превращает голый NameError/KeyError (например, при запуске ячеек не по
    порядку или после рестарта ядра без повторной загрузки) в понятную
    ошибку с указанием, чего не хватает и что сделать.
    """
    missing = [n for n in names if n not in env]
    if missing:
        raise RuntimeError(f'Не определено: {missing}. {hint}')


# ============================== пути ==============================
def resolve_paths(base: Optional[Path] = None) -> dict:
    """Каталоги проекта — единственное место, где задаются пути к данным.

    base — корень проекта; None — текущая папка. Если запуск из `WWLLN/` или
    из соседней подпапки, корень поднимается на уровень выше.
    Возвращает dict: 'root', 'a' (WWLLN/Afiles), 'ae' (WWLLN/AEfiles),
    'pred' (predictions/whistler_binary), 'cache' (cache/).
    """
    root = Path(base).resolve() if base is not None else Path('.').resolve()
    if root.name == 'WWLLN':
        root = root.parent
    if not (root / 'WWLLN').exists() and (root.parent / 'WWLLN').exists():
        root = root.parent
    wwlln = root / 'WWLLN'
    return {
        'root': root,
        'a': wwlln / 'Afiles',
        'ae': wwlln / 'AEfiles',
        'pred': root / 'predictions' / 'whistler_binary',
        'cache': root / 'cache',
    }


# ============================== дни ==============================
_PRED_RE = re.compile(r"predictions_(ka\d+|sc\d+)_fl_(\d{6})000000_(\d{6})000000\.csv$")


def pred_day_key_from_name(name: str) -> Optional[str]:
    """Имя файла предсказаний → сутки 'YYYYMMDD' ('20' + шесть цифр после `_fl_`);
    None, если имя не по шаблону (например, маркеры `SUCCESS_*`)."""
    m = _PRED_RE.search(name)
    return '20' + m.group(2) if m else None


def list_pred_days(pred_dir: Path, sc_list: Iterable[str]) -> dict[str, dict[str, Path]]:
    """{'YYYYMMDD': {'ka1': path, 'ka2': path, ...}} — какие дни есть у какого КА."""
    out: dict[str, dict[str, Path]] = defaultdict(dict)
    for sc in sc_list:
        for f in sorted((Path(pred_dir) / sc).glob(f"predictions_{sc}_fl_*.csv")):
            key = pred_day_key_from_name(f.name)
            if key is not None:
                out[key][sc] = f
    return dict(out)


def ae_path_for_day(ae_dir: Path, yyyymmdd: str) -> Path:
    """Путь к AE-файлу суток: `ae_dir/AEYYYYMMDD.loc.gz` (существование не проверяется)."""
    return Path(ae_dir) / f"AE{yyyymmdd}.loc.gz"


def day_codes(days: Iterable[str]) -> np.ndarray:
    """'YYYYMMDD' → номер суток от 1970-01-01 (datetime64[D] в int64).

    В таком виде сутки ждут функции `dose_pipeline` (`acc_days`). Обратно:
    `np.datetime64(code, 'D')`.
    """
    return np.array(
        [np.datetime64(f'{d[:4]}-{d[4:6]}-{d[6:8]}', 'D').astype(np.int64) for d in days],
        dtype=np.int64,
    )


def analysis_days(
    pred_dir: Path,
    ae_dir: Path,
    sc_list: Sequence[str] = ('ka1', 'ka2'),
    require_all_sc: bool = True,
) -> tuple[str, ...]:
    """Сутки, для которых есть AE-файл и файлы предсказаний.

    pred_dir, ae_dir — каталоги из `resolve_paths()` ('pred', 'ae').
    sc_list — аппараты.
    require_all_sc=True — пересечение покрытий всех КА из sc_list
    (нужно, когда аппараты сравнивают на одних датах).
    require_all_sc=False — объединение: день входит, если есть AE и хотя бы
    один из перечисленных КА.
    Возвращает отсортированный кортеж 'YYYYMMDD'. Период анализа задают
    обрезкой этого кортежа.
    """
    day_map = list_pred_days(pred_dir, sc_list)
    days = []
    for d, m in sorted(day_map.items()):
        if not ae_path_for_day(ae_dir, d).exists():
            continue
        if require_all_sc and not all(sc in m for sc in sc_list):
            continue
        if not require_all_sc and not any(sc in m for sc in sc_list):
            continue
        days.append(d)
    return tuple(days)


def sc_days(pred_dir: Path, ae_dir: Path, sc: str) -> tuple[str, ...]:
    """Сутки одного аппарата, для которых есть файл AE."""
    return analysis_days(pred_dir, ae_dir, (sc,), require_all_sc=True)


def spread_days(days: Sequence[str], n: Optional[int]) -> tuple[str, ...]:
    """Равномерно по индексу: n суток из доступных (или все, если их меньше n)."""
    days = tuple(days)
    if not days:
        return ()
    if n is None or n >= len(days):
        return days
    if n <= 1:
        return (days[0],)
    idx = np.unique(np.round(np.linspace(0, len(days) - 1, n)).astype(int))
    # linspace при округлении иногда даёт меньше n — добираем ближайшими
    if idx.size < n:
        extra = [i for i in range(len(days)) if i not in set(idx)]
        need = n - idx.size
        idx = np.sort(np.concatenate([idx, np.asarray(extra[:need], dtype=int)]))
    return tuple(days[int(i)] for i in idx)


def days_by_sc(
    pred_dir: Path,
    ae_dir: Path,
    sc_list: Sequence[str],
    n_per_sc: Optional[int] = None,
    mode: str = 'spread',
) -> dict[str, tuple[str, ...]]:
    """Свои сутки у каждого КА — без требования пересечения покрытий.

    mode:
      'spread' — n_per_sc суток равномерно по покрытию этого КА;
      'pilot'  — первые n_per_sc суток;
      'full'   — все доступные сутки КА.
    """
    out: dict[str, tuple[str, ...]] = {}
    for sc in sc_list:
        all_d = sc_days(pred_dir, ae_dir, sc)
        if mode == 'full' or n_per_sc is None:
            out[sc] = all_d
        elif mode == 'pilot':
            out[sc] = all_d[:n_per_sc]
        else:
            out[sc] = spread_days(all_d, n_per_sc)
    return out


# ============================== загрузка ==============================
_AE_NAMES = ['date', 'time', 'lat', 'lon', 'resid', 'nsta', 'energy', 'energy_err', 'esta']
_AE_DTYPES = {'date': 'string', 'time': 'string', 'lat': 'float32', 'lon': 'float32',
              'resid': 'float32', 'nsta': 'int16', 'energy': 'float32',
              'energy_err': 'float32', 'esta': 'int16'}


def _empty_ae_frame(columns: Sequence[str] = ('datetime', 'lat', 'lon', 'energy')) -> pd.DataFrame:
    """Пустой AE-фрейм с ПРАВИЛЬНЫМИ dtype (а не `object` по умолчанию у
    pd.DataFrame(columns=...)) — иначе pd.concat() с настоящими данными молча
    приводит всю колонку 'datetime' к object (numpy.datetime64 → boxed
    pandas.Timestamp), и .values.astype(...) ниже по стеку падает."""
    dtypes = {'datetime': 'datetime64[us]'}
    return pd.DataFrame({c: pd.Series([], dtype=dtypes.get(c, 'float32')) for c in columns})


AE_CACHE_NAME = 'AE_20250101_20251231.parquet'
AE_CACHE_COLUMNS = ('datetime', 'lat', 'lon', 'resid', 'nsta', 'energy', 'energy_err', 'esta')


def read_ae_file(filepath: Path,
                 columns: Sequence[str] = ('datetime', 'lat', 'lon', 'energy')) -> pd.DataFrame:
    """Один AE-файл (.loc.gz) → DataFrame с колонками columns (из `AE_CACHE_COLUMNS`)."""
    filepath = Path(filepath)
    if not filepath.exists():
        return _empty_ae_frame(columns)
    d = pd.read_csv(filepath, header=None, names=_AE_NAMES, dtype=_AE_DTYPES,
                    skipinitialspace=True, engine='c')
    d['datetime'] = pd.to_datetime(d['date'] + ' ' + d['time'], format='%Y/%m/%d %H:%M:%S.%f',
                                   errors='coerce').astype('datetime64[us]')
    return d.loc[d['datetime'].notna(), list(columns)]


def build_ae_cache(
    ae_dir: Path,
    cache_path: Path,
    start: str = '20250101',
    end: str = '20251231',
    n_jobs: int = 8,
    verbose: bool = True,
) -> Path:
    """Пишет parquet-кэш молний из .loc.gz за сутки start..end ('YYYYMMDD').

    Формат тот же, что у кэша из `wwlln_reader.ipynb`: колонки `AE_CACHE_COLUMNS`.
    Пишет помесячно (пик около 3.5 ГБ памяти), сначала во временный файл, поэтому
    прерванная сборка не оставляет неполного кэша. Сутки без файла пропускаются.
    Год — около 4 мин и 5.7 ГБ на диске. Возвращает cache_path.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from concurrent.futures import ThreadPoolExecutor

    cache_path = Path(cache_path)
    all_days = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq='D')
    files = [ae_path_for_day(ae_dir, d.strftime('%Y%m%d')) for d in all_days]
    months = defaultdict(list)
    for d, f in zip(all_days, files):
        if f.exists():
            months[d.strftime('%Y%m')].append(f)
    n_files = sum(len(v) for v in months.values())
    if not n_files:
        raise FileNotFoundError(f'Нет AE-файлов за {start}–{end} в {ae_dir}')
    if verbose:
        print(f'Кэш молний: {n_files} файлов за {start}–{end} → {cache_path}')

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_name(cache_path.name + '.tmp')
    writer = None
    n_rows = 0
    read = lambda f: read_ae_file(f, AE_CACHE_COLUMNS)
    try:
        with ThreadPoolExecutor(max_workers=min(n_jobs, os.cpu_count() or 1)) as ex:
            for month in sorted(months):
                parts = [p for p in ex.map(read, months[month]) if not p.empty]
                if not parts:
                    continue
                part = pd.concat(parts, ignore_index=True).sort_values('datetime')
                table = pa.Table.from_pandas(part, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(tmp, table.schema)
                writer.write_table(table.cast(writer.schema), row_group_size=1 << 20)
                n_rows += len(part)
                if verbose:
                    print(f'  {month}: {len(part):,} разрядов')
        writer.close()
        tmp.replace(cache_path)
    except BaseException:
        if writer is not None:
            writer.close()
        tmp.unlink(missing_ok=True)
        raise
    if verbose:
        print(f'Кэш записан: {n_rows:,} разрядов')
    return cache_path


def load_ae(
    ae_dir: Optional[Path] = None,
    cache_path: Optional[Path] = None,
    days: Optional[Sequence[str]] = None,
    columns: Sequence[str] = ('datetime', 'lat', 'lon', 'energy'),
    n_jobs: int = 8,
    verbose: bool = True,
) -> pd.DataFrame:
    """Молнии AE: сначала parquet-кэш, затем .loc.gz за сутки, которых в кэше нет.

    ae_dir — каталог AE-файлов; cache_path — parquet-кэш (None или несуществующий
    файл — читать только .loc.gz). Кэш эта функция не создаёт: его пишут
    `build_ae_cache` (его вызывает `ensure_data`, если кэша нет) и ячейка загрузки
    `wwlln_reader.ipynb`.
    days — сутки 'YYYYMMDD'; None — весь кэш или все файлы каталога.
    columns — колонки из кэша; из .loc.gz всегда читаются datetime, lat, lon, energy.
    n_jobs — потоков на чтение .loc.gz.
    Возвращает DataFrame, отсортированный по datetime (UTC).
    """
    from concurrent.futures import ThreadPoolExecutor

    def _read_files(files):
        files = [f for f in files if f.exists()]
        if verbose:
            print(f'AE: читаю {len(files)} файлов из {ae_dir}')
        with ThreadPoolExecutor(max_workers=min(n_jobs, os.cpu_count() or 1)) as ex:
            parts = [p for p in ex.map(read_ae_file, files) if not p.empty]
        return (pd.concat(parts, ignore_index=True) if parts
                else _empty_ae_frame(columns))

    if cache_path is not None and Path(cache_path).exists():
        if verbose:
            print(f'AE из кэша: {cache_path}')
        out = pd.read_parquet(cache_path, columns=list(columns))
        if days is not None:
            # day_codes()/astype('datetime64[D]') — векторизовано в numpy, в отличие
            # от .dt.strftime, который на кэше в сотни млн строк невыносимо медленный
            req_codes = day_codes(days)
            day_code_col = out['datetime'].values.astype('datetime64[D]').astype(np.int64)
            mask = np.isin(day_code_col, req_codes)
            have = set(np.unique(day_code_col[mask]).tolist())
            missing = [d for d, c in zip(days, req_codes) if int(c) not in have]
            out = out[mask]
            if missing:
                if ae_dir is None:
                    raise ValueError(
                        f'В кэше нет {len(missing)} нужных дней, а ae_dir не задан'
                    )
                if verbose:
                    print(f'AE: в кэше нет {len(missing)} из запрошенных дней '
                          f'({missing[0]}…{missing[-1]}) — дочитываю из .loc.gz')
                extra = _read_files(ae_path_for_day(ae_dir, d) for d in missing)
                if not extra.empty:
                    out = pd.concat([out, extra], ignore_index=True)
        return out.sort_values('datetime').reset_index(drop=True)

    if ae_dir is None:
        raise ValueError('Нет ни кэша, ни каталога AE')
    files = ([ae_path_for_day(ae_dir, d) for d in days] if days
             else sorted(Path(ae_dir).glob('AE*.loc.gz')))
    out = _read_files(files)
    return out.sort_values('datetime').reset_index(drop=True)


PRED_MSK_OFFSET_HOURS = 3  # время в файлах предсказаний — МСК
PRED_COLS = ['timestamp', 'geo_lat', 'geo_lon', 'is_night', 'solar_zenith_angle',
             'prob_whistle', 'verdict']
_PRED_DTYPES = {'geo_lat': 'float32', 'geo_lon': 'float32', 'is_night': 'int8',
                'solar_zenith_angle': 'float32', 'prob_whistle': 'float32', 'verdict': 'int8'}


def read_pred_file(filepath: Path, sc: str, msk_offset_hours: float = PRED_MSK_OFFSET_HOURS,
                   usecols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Один файл предсказаний. Время МСК → UTC (минус msk_offset_hours).

    sc — имя аппарата, пишется в колонку 'sc'; usecols — подмножество `PRED_COLS`.
    """
    cols = list(usecols) if usecols else PRED_COLS
    d = pd.read_csv(filepath, usecols=cols,
                    dtype={k: v for k, v in _PRED_DTYPES.items() if k in cols})
    d['timestamp'] = pd.to_datetime(d['timestamp']) - pd.Timedelta(hours=msk_offset_hours)
    d['sc'] = sc
    return d


def load_predictions(
    pred_dir: Path,
    sc_list: Sequence[str] = ('ka1', 'ka2'),
    days: Optional[Sequence[str]] = None,
    usecols: Optional[Sequence[str]] = None,
    n_jobs: int = 8,
    verbose: bool = True,
) -> pd.DataFrame:
    """Предсказания по аппаратам в одной таблице.

    pred_dir — каталог 'pred' из `resolve_paths()`; sc_list — аппараты;
    days — сутки 'YYYYMMDD', None — все доступные; usecols — подмножество
    `PRED_COLS`; n_jobs — потоков на чтение.
    Возвращает DataFrame: колонки файла, timestamp в UTC, 'sc' (category).
    Пустой DataFrame, если файлов нет.
    """
    from concurrent.futures import ThreadPoolExecutor

    day_map = list_pred_days(pred_dir, sc_list)
    keys = sorted(day_map) if days is None else [d for d in days if d in day_map]
    parts = []
    for sc in sc_list:
        files = [day_map[d][sc] for d in keys if sc in day_map[d]]
        if verbose:
            print(f'{sc}: {len(files)} файлов из {Path(pred_dir) / sc}')
        if not files:
            continue
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            frames = list(ex.map(lambda f, s=sc: read_pred_file(f, s, usecols=usecols), files))
        parts.append(pd.concat(frames, ignore_index=True))
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out['sc'] = out['sc'].astype('category')
    return out


def ensure_data(env: dict, days: Optional[Sequence[str]] = None,
                ae_days: Optional[Sequence[str]] = None, need_ae: bool = True,
                need_pred: bool = True, build_cache: bool = True, verbose: bool = True):
    """Догружает `df` (молнии) и `df_pred` (предсказания), если их нет в окружении.

    env — словарь переменных ноутбука, обычно `globals()`; туда же кладутся
    результаты. Уже загруженные `df` / `df_pred` не перечитываются: для другого
    набора суток их сначала удаляют (`del df, df_pred`).
    days — сутки предсказаний 'YYYYMMDD'.
    ae_days — сутки молний; по умолчанию = days. Начинают на сутки раньше периода
    (окно дозы у полуночи), при null-модели — ещё на max(null_lags) раньше.
    need_ae, need_pred — что загружать.
    Молнии — из `cache/AE_20250101_20251231.parquet` (`AE_CACHE_NAME`), недостающие
    сутки из .loc.gz. build_cache — если кэша нет, сначала собрать его за 2025 год
    (`build_ae_cache`, около 4 мин один раз); False — читать только нужные .loc.gz.
    Предсказания — только ka1 и ka2; ka4 загружают `load_predictions(..., sc_list=('ka4',))`.
    Возвращает (df, df_pred).
    """
    paths = resolve_paths()
    cache = paths['cache'] / AE_CACHE_NAME

    df = env.get('df')
    if need_ae and (df is None or getattr(df, 'empty', True)):
        if verbose:
            print('df не найден — загружаю молнии')
        if not cache.exists():
            if build_cache:
                build_ae_cache(paths['ae'], cache, verbose=verbose)
            elif verbose:
                print(f'Кэша {cache} нет — читаю .loc.gz (build_cache=False)')
        df = load_ae(ae_dir=paths['ae'], cache_path=cache,
                     days=(ae_days if ae_days is not None else days), verbose=verbose)
        env['df'] = df
    elif need_ae and 'energy' not in df.columns:
        raise ValueError(
            'df уже определён в ядре, но без energy (похоже, остался от другой ячейки) — '
            'перезапустите ядро и выполните ячейки заново'
        )

    dfp = env.get('df_pred')
    if need_pred and (dfp is None or getattr(dfp, 'empty', True)):
        if verbose:
            print('df_pred не найден — загружаю предсказания')
        dfp = load_predictions(paths['pred'], days=days, verbose=verbose)
        env['df_pred'] = dfp
    elif need_pred and 'prob_whistle' not in dfp.columns:
        raise ValueError('df_pred загружен без prob_whistle — перезапустите ячейку загрузки')

    return df, dfp
