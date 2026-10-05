"""Графики и таблицы полуорбит для ``wwlln_orbit_profiles.ipynb``.

Профиль полуорбиты (`p`, `profile`) — dict массивов по шагам вдоль трассы от
полюса к полюсу: 't' — время шага (unix, с), 'lat', 'lon' — подспутниковая
точка, 'verdict' — доля verdict = 1, 'count' — число разрядов в радиусе.
Профили строит ноутбук.

Списки профилей, шаг вдоль орбиты и имя спутника ноутбук передаёт при вызове:
их пересобирают отдельные ячейки, и значение на момент импорта было бы устаревшим.
Общие keyword-аргументы: steps — номера шагов (ось X); profiles — список
профилей одного аппарата; maxn — нормировка к максимуму 1; sc — имя аппарата.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import HTML, display

_LAND_COLORS = {1.0: '#7fb069', 0.0: '#8ec5e8', 'nan': '#d9d9d9'}
_LAND_PREP = None

def _land_geom():
    """Подготовленная геометрия суши Natural Earth 50m (кэш cartopy, сеть не нужна)."""
    global _LAND_PREP
    if _LAND_PREP is None:
        try:
            from cartopy.io import shapereader as shpreader
            from shapely.ops import unary_union
            from shapely.prepared import prep
            path = shpreader.natural_earth(resolution='50m', category='physical', name='land')
            _LAND_PREP = prep(unary_union(list(shpreader.Reader(path).geometries())))
        except Exception as exc:
            print(f'маска суши недоступна ({exc}) - полоса суша/вода не рисуется')
            _LAND_PREP = False
    return _LAND_PREP if _LAND_PREP is not False else None

def steps_over_land(p):
    """Для каждого шага профиля p: 1 — над сушей, 0 — над водой, NaN — нет координат
    (дыра в телеметрии). Без маски суши Natural Earth — все NaN."""
    land = _land_geom()
    lat = np.asarray(p['lat'], dtype=np.float64)
    lon = np.asarray(p['lon'], dtype=np.float64)
    out = np.full(lat.size, np.nan)
    if land is None:
        return out
    from shapely.geometry import Point
    for k in range(lat.size):
        if np.isfinite(lat[k]) and np.isfinite(lon[k]):
            out[k] = float(land.contains(Point(((lon[k] + 180.0) % 360.0) - 180.0, lat[k])))
    return out

def add_land_strip(ax, profiles, row_labels=None, row_frac=0.055, *, steps):
    """Полоса под осью X: где шаг проходил над сушей, а где над водой.

    ax — ось графика профиля; profiles — профили, по строке полосы на каждый;
    row_labels — подписи строк справа; row_frac — высота строки в долях оси Y.
    Возвращает handles для легенды (пусто, если маска суши недоступна).
    """
    if _land_geom() is None:
        return []
    top = ax.get_ylim()[1]
    row_h = row_frac * top
    for i, prof in enumerate(profiles):
        flags = steps_over_land(prof)
        colors = [_LAND_COLORS['nan'] if not np.isfinite(f) else _LAND_COLORS[f] for f in flags]
        bottom = -(i + 1) * row_h
        ax.bar(steps, np.full(len(steps), row_h), bottom=bottom, width=1.0,
               color=colors, linewidth=0, clip_on=False, zorder=5)
        if row_labels is not None:
            ax.text(1.004, bottom + row_h / 2, row_labels[i], transform=ax.get_yaxis_transform(),
                    ha='left', va='center', fontsize=6, clip_on=False)
    ax.set_ylim(-len(profiles) * row_h, top)
    ax.set_yticks(np.arange(0, top + 1e-9, 0.25))
    return [
        plt.Rectangle((0, 0), 1, 1, color=_LAND_COLORS[1.0], label='суша'),
        plt.Rectangle((0, 0), 1, 1, color=_LAND_COLORS[0.0], label='вода'),
    ]

def plot_orbit_map(ax, p, color='#c0392b', title='трасса на карте'):
    """Мировая карта: трасса полуорбиты по lat/lon шагов (полюс → полюс).

    ax — ось с проекцией cartopy (PlateCarree); p — профиль или None (пустая карта);
    color, title — цвет трассы и заголовок. Отмечены старт, финиш и пересечение экватора.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    ax.set_global()
    ax.set_facecolor('#d9eaf5')
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                               facecolor='#e8f0d8', edgecolor='none'))
    ax.coastlines(resolution='50m', linewidth=0.45, color='0.35')
    ax.gridlines(draw_labels=False, linewidth=0.25, color='0.7', alpha=0.6)
    if p is None:
        ax.set_title('нет трассы', fontsize=9)
        return
    lat = np.asarray(p['lat'], dtype=np.float64)
    lon = np.asarray(p['lon'], dtype=np.float64)
    ok = np.isfinite(lat) & np.isfinite(lon)
    if not ok.any():
        ax.set_title('нет координат', fontsize=8)
        return
    lon_w = ((lon + 180.0) % 360.0) - 180.0
    ax.plot(lon_w[ok], lat[ok], color=color, lw=1.8, transform=ccrs.Geodetic(), zorder=4)
    ax.scatter(lon_w[ok], lat[ok], s=10, c=color, transform=ccrs.PlateCarree(),
               zorder=5, linewidths=0)
    i0 = int(np.flatnonzero(ok)[0])
    i1 = int(np.flatnonzero(ok)[-1])
    ax.plot(lon_w[i0], lat[i0], marker='^', color='#1a1a1a', ms=7, transform=ccrs.PlateCarree(),
            zorder=6, label='старт')
    ax.plot(lon_w[i1], lat[i1], marker='s', color='#1a1a1a', ms=5, transform=ccrs.PlateCarree(),
            zorder=6, label='финиш')
    eq = int(np.nanargmin(np.abs(lat)))
    if np.isfinite(lat[eq]) and np.isfinite(lon_w[eq]):
        ax.plot(lon_w[eq], lat[eq], marker='o', color='#f4c430', ms=6, mec='#333333', mew=0.6,
                transform=ccrs.PlateCarree(), zorder=6)
    ax.set_title(title, fontsize=9)

def _adjacent_orbit_table(adjacent, start_idx, direction_label='восходящая', *, sc):
    """Таблица в том же HTML-виде, что _orbit_table в п.7: время пролёта и диапазон долгот."""
    rows = []
    for col, p in enumerate(adjacent):
        t0, t1 = p['t'][0], p['t'][-1]
        dt0 = pd.Timestamp(int(t0), unit='s', tz='UTC').strftime('%Y-%m-%d %H:%M:%S')
        dt1 = pd.Timestamp(int(t1), unit='s', tz='UTC').strftime('%H:%M:%S')
        lon0, lon1 = p['lon'][0], p['lon'][-1]
        cell = (f'{dt0} .. {dt1}<br>'
                f'долгота {lon0:.1f}° → {lon1:.1f}°')
        rows.append(f'<tr><td>{direction_label} #{start_idx + col}</td><td>{cell}</td></tr>')
    table = ('<table style="border-collapse:collapse;font-size:13px" border="1" cellpadding="6">'
             f'<tr><th></th><th>{sc}</th></tr>' + ''.join(rows) + '</table>')
    display(HTML(table))

def plot_adjacent_orbits(START_IDX, N_ADJACENT=3, want_ascending=True, *, profiles, steps, maxn, sc):
    """Графики соседних полуорбит одного направления + таблица времени/долгот.

    START_IDX — номер первой орбиты среди орбит выбранного направления;
    N_ADJACENT — сколько орбит подряд.
    want_ascending=True — восходящие (широта растёт), False — нисходящие (широта падает).
    Если орбиты окна не идут подряд по времени (пропущен виток), — ValueError.
    Соседние по индексу в all_profiles орбиты чередуют направление, поэтому сначала
    фильтруем, потом берём подряд: шаг в один полный виток, а не пол-витка.
    """
    all_profiles = profiles
    _maxn = maxn
    SC = sc
    direction_label = 'восходящая' if want_ascending else 'нисходящая'
    direction_title = 'ВОСХОДЯЩИХ' if want_ascending else 'НИСХОДЯЩИХ'
    pool = [p for p in all_profiles if (p['lat'][0] < p['lat'][-1]) == want_ascending]
    adjacent = pool[START_IDX:START_IDX + N_ADJACENT]

    # проверка на скрытые разрывы: простой срез списка не гарантирует, что соседние
    # элементы реально идут подряд по времени (один полный виток друг от друга) - если между
    # ними пропущена орбита (не прошла ORBIT_DUR_RANGE_S из-за дыры в данных), START_IDX может
    # молча указывать на орбиты, разделённые 2-3 периодами. Проверяем явно.
    # эталон - полный виток (~101 мин), то есть ДВЕ полуорбиты по ~50 мин: между двумя
    # орбитами одного направления всегда лежит одна противоположная. Разрыв в данных,
    # съевший один виток, даёт ~202 мин - формально "соседние" по индексу, физически - нет.
    _all_dts = np.diff([p['t'][0] for p in pool])
    _med_dt = np.median(_all_dts)
    _win_dts = np.diff([p['t'][0] for p in adjacent])
    _bad = [(k, d) for k, d in enumerate(_win_dts) if abs(d - _med_dt) > 0.2 * _med_dt]
    if _bad:
        msg = [f'START_IDX={START_IDX}: {("восходящие" if want_ascending else "нисходящие")} орбиты не соседние по времени ({len(_bad)} разрывов), '
               f'ожидался шаг ~{_med_dt/60:.0f} мин (полный виток = две полуорбиты):']
        for k, d in _bad:
            t_k = pd.Timestamp(int(adjacent[k]['t'][0]), unit='s', tz='UTC').strftime('%Y-%m-%d %H:%M:%S')
            t_k1 = pd.Timestamp(int(adjacent[k + 1]['t'][0]), unit='s', tz='UTC').strftime('%Y-%m-%d %H:%M:%S')
            msg.append(f'  #{START_IDX + k} ({t_k}) -> #{START_IDX + k + 1} ({t_k1}): '
                       f'{d/60:.0f} мин (x{d/_med_dt:.2f} от витка) - пропущено витков: {d/_med_dt - 1:.0f}')
        raise ValueError('\n'.join(msg))
    print(f'проверка соседства: чисто, все {len(adjacent)} {direction_title.lower()} орбит идут подряд '
          f'(шаг ~{_med_dt:.0f}с = {_med_dt/60:.0f} мин, полный виток)')

    _orbit_colors = ['#e05c2a', '#2a7ae0', '#2aa02a', '#c02a8a', '#8a5a2a']
    _orbit_labels = []
    for col, p in enumerate(adjacent):
        eq_idx = np.nanargmin(np.abs(p['lat']))
        lon_eq = p['lon'][eq_idx]
        _orbit_labels.append(f'#{START_IDX + col} (долг.экв.~{lon_eq:.0f}°)')

    # --- график 1: каждая орбита на своей панели (доля + количество) и карта трассы ---
    import cartopy.crs as ccrs
    n_adj = len(adjacent)
    fig = plt.figure(figsize=(6 * n_adj, 7.4))
    gs = fig.add_gridspec(2, n_adj, height_ratios=[1.45, 1.0], hspace=0.28, wspace=0.18)
    for col, p in enumerate(adjacent):
        ax = fig.add_subplot(gs[0, col])
        # zorder, а не порядок вызовов: жёлтый (доля) рисуется поверх синего (количество)
        ax.bar(steps, _maxn(p['verdict']), width=0.8, color='#fcfb00', alpha=0.6,
               label='доля verdict=1', zorder=3)
        ax.bar(steps, _maxn(p['count']), width=0.8, color='steelblue', alpha=0.6,
               label='количество молний', zorder=2)
        lat0, lat1 = p['lat'][0], p['lat'][-1]
        ax.set_title(f'{direction_label} орбита #{START_IDX + col}\nшир. {lat0:.0f}°→{lat1:.0f}°, {_orbit_labels[col]}')
        ax.set_xlabel('Шаг вдоль орбиты (полюс → полюс)')
        if col == 0:
            ax.set_ylabel('Форма (max=1)')
        _land_h = add_land_strip(ax, [p], steps=steps)
        ax.legend(handles=ax.get_legend_handles_labels()[0] + _land_h, fontsize=8)
        ax.grid(alpha=0.3, axis='y', zorder=0)

        axm = fig.add_subplot(gs[1, col], projection=ccrs.PlateCarree())
        plot_orbit_map(axm, p)
    fig.suptitle(f'{SC}: {n_adj} соседних {direction_title} орбиты подряд, без привязки к региону', y=0.98)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.04, wspace=0.18, hspace=0.28)
    plt.show()
    plt.close(fig)

    # --- график 2: все орбиты вместе - отдельно доля, отдельно количество ---
    fig, (ax_v, ax_d) = plt.subplots(1, 2, figsize=(13, 4.5))
    for col, p in enumerate(adjacent):
        color = _orbit_colors[col % len(_orbit_colors)]
        ax_v.bar(steps, _maxn(p['verdict']), width=0.8, color=color, alpha=0.5, label=_orbit_labels[col])
        ax_d.bar(steps, _maxn(p['count']), width=0.8, color=color, alpha=0.5, label=_orbit_labels[col])

    ax_v.set_title('Доля verdict=1')
    ax_d.set_title('Число разрядов в радиусе 1000 км')
    for ax in (ax_v, ax_d):
        ax.set_xlabel('Шаг вдоль орбиты (полюс → полюс)')
        ax.set_ylabel('Форма (max=1)')
        ax.legend(fontsize=8, title='орбита')
        ax.grid(alpha=0.3, axis='y')

    fig.suptitle(f'{SC}: те же {len(adjacent)} орбиты, доля и количество - каждое на своём общем графике', y=1.03)
    fig.tight_layout()
    plt.show()
    plt.close(fig)

    _adjacent_orbit_table(adjacent, START_IDX, direction_label, sc=sc)

def _find_by_direction(all_profs, idx0, step, want_ascending):
    """Ищет ближайший полувиток нужного направления, двигаясь от idx0 на step (-1 назад,
    +1 вперёд) за раз - НЕ по фиксированному смещению ±2, т.к. чередование
    восходящая/нисходящая ломается, если полуорбита нестандартной длины отброшена
    при нарезке и в списке её нет.
    None, если не нашлось до края окна."""
    j = idx0 + step
    while 0 <= j < len(all_profs):
        is_asc = all_profs[j]['lat'][0] < all_profs[j]['lat'][-1]
        if is_asc == want_ascending:
            return j
        j += step
    return None

def _orbit_sequence(p_current, all_profs):
    """Текущий полувиток + предыдущий/следующий восходящий и нисходящий, найденные по
    фактическому направлению (устойчиво к пропускам в данных). Ищем по времени начала
    (p_current['t'][0]), а не по identity объекта - если ячейку загрузки/сегментации
    перезапустить ПОСЛЕ того, как pairs/pairs_desc уже построены, all_profiles(_ka2)
    станет новым списком с новыми объектами (те же орбиты, но другие dict), и identity-
    сравнение молча упало бы StopIteration, хотя орбита физически та же самая."""
    idx0 = next(i for i, x in enumerate(all_profs) if x['t'][0] == p_current['t'][0])
    seq = {'текущая': p_current}
    for step, want_asc, label in [(-1, True, 'предыдущий восходящий'),
                                   (-1, False, 'предыдущий нисходящий'),
                                   (1, False, 'следующий нисходящий'),
                                   (1, True, 'следующий восходящий')]:
        j = _find_by_direction(all_profs, idx0, step, want_asc)
        seq[label] = all_profs[j] if j is not None else None
    return seq

def _cell_html(p):
    """Время + широта/долгота начала-конца пролёта - одна ячейка таблицы."""
    if p is None:
        return 'нет — край окна'
    t0, t1 = p['t'][0], p['t'][-1]
    dt0 = pd.Timestamp(int(t0), unit='s', tz='UTC').strftime('%Y-%m-%d %H:%M:%S')
    dt1 = pd.Timestamp(int(t1), unit='s', tz='UTC').strftime('%H:%M:%S')
    lat0, lat1 = p['lat'][0], p['lat'][-1]
    lon0, lon1 = p['lon'][0], p['lon'][-1]
    return (f'{dt0} .. {dt1}<br>'
            f'широта {lat0:.1f}° → {lat1:.1f}°, долгота {lon0:.1f}° → {lon1:.1f}°')

_ROW_ORDER = ['предыдущий восходящий', 'предыдущий нисходящий', 'текущая',
              'следующий нисходящий', 'следующий восходящий']

# для нисходящей текущей орбиты ближайший по времени сосед - ВОСХОДЯЩИЙ (см.
# _order_warnings ниже), поэтому хронологический порядок строк зеркальный - пары
# предыдущий/следующий восходящий-нисходящий меняются местами (см. п.8)
_ROW_ORDER_DESC = ['предыдущий нисходящий', 'предыдущий восходящий', 'текущая',
                   'следующий восходящий', 'следующий нисходящий']



def _order_warnings(sc_label, seq, want_ascending=True):
    """При отсутствии разрывов, если текущая - восходящая: "предыдущий восходящий" по
    времени раньше "предыдущего нисходящего", а "следующий нисходящий" раньше "следующего
    восходящего" (обычное чередование). Если текущая - нисходящая, порядок соседей по
    построению обратный (см. п.8) - want_ascending переключает ожидаемое направление
    сравнения. Если разрыв в данных "съел" непосредственного соседа, порядок
    переворачивается - это не ошибка данных, а сигнал, что между показанными витками
    пропущен ещё один."""
    warnings = []
    pa, pd_ = seq['предыдущий восходящий'], seq['предыдущий нисходящий']
    prev_same, prev_opp = (pa, pd_) if want_ascending else (pd_, pa)
    prev_same_label = 'предыдущий восходящий' if want_ascending else 'предыдущий нисходящий'
    prev_opp_label = 'предыдущий нисходящий' if want_ascending else 'предыдущий восходящий'
    if prev_same is not None and prev_opp is not None and prev_same['t'][0] > prev_opp['t'][0]:
        warnings.append(f'⚠ {sc_label}: "{prev_same_label}" по времени ПОЗЖЕ "{prev_opp_label}" - '
                         f'между текущей орбитой и "{prev_opp_label}" пропущен виток (разрыв в данных)')
    nd, na = seq['следующий нисходящий'], seq['следующий восходящий']
    next_opp, next_same = (nd, na) if want_ascending else (na, nd)
    next_opp_label = 'следующий нисходящий' if want_ascending else 'следующий восходящий'
    next_same_label = 'следующий восходящий' if want_ascending else 'следующий нисходящий'
    if next_opp is not None and next_same is not None and next_opp['t'][0] > next_same['t'][0]:
        warnings.append(f'⚠ {sc_label}: "{next_opp_label}" по времени ПОЗЖЕ "{next_same_label}" - '
                         f'между текущей орбитой и "{next_same_label}" пропущен виток (разрыв в данных)')
    return warnings

def _second_best(seq2, t_eq1, want_ascending=True):
    """Среди кандидатов ТОЛЬКО того же направления, что и текущая orbита -
    "предыдущий"/"следующий" в этом направлении (want_ascending переключает
    восходящие/нисходящие) - находит того, чьё пересечение экватора ближе всего по
    времени к t_eq1. Кандидаты противоположного направления исключены намеренно: их
    профиль зеркален по развёртке широты вдоль шагов (см. п.4/п.5) и не сравним с текущей,
    хотя по чистому времени они почти всегда ближе (сдвинуты на пол-периода, а не на
    период) - сравнивать их с текущей было бы физически бессмысленно."""
    labels = ('предыдущий восходящий', 'следующий восходящий') if want_ascending else \
             ('предыдущий нисходящий', 'следующий нисходящий')
    best_label, best_p, best_dt = None, None, None
    for label in labels:
        p = seq2[label]
        if p is None:
            continue
        t_eq = p['t'][np.nanargmin(np.abs(p['lat']))]
        dt = abs(t_eq - t_eq1)
        if best_dt is None or dt < best_dt:
            best_label, best_p, best_dt = label, p, dt
    return best_label, best_p

def _orbit_table(p_primary, p_secondary, primary_profs=None, secondary_profs=None,
                  want_ascending=True, primary_name='ka1', secondary_name='ka2',
                  *, profiles_by_name):
    """Таблица: строки - предыдущий/текущий/следующий полувиток, столбцы - primary_name/
    secondary_name (по умолчанию ka1/ka2, но роли переставимы - см. п.9). "Текущая" и
    "вторая подходящая" (см. _second_best) в столбце secondary выделены жирным - без
    цветной заливки. Порядок строк берётся из _ROW_ORDER (восходящая текущая) или
    _ROW_ORDER_DESC (нисходящая) - хронологический порядок соседей зеркальный в
    зависимости от направления текущей орбиты (см. п.8). См. _order_warnings для явного
    предупреждения, если разрыв в данных нарушает обычное чередование. Возвращает профиль
    "второй подходящей" орбиты secondary (для второго графика). primary_profs/
    secondary_profs разрешаются на момент ВЫЗОВА, а не через default-значение аргумента
    (которое иначе фиксируется один раз при первом выполнении этой ячейки) - иначе после
    повторного запуска ячейки загрузки/сегментации (новый all_profiles_ka2 с новыми
    объектами) функция молча сравнивала бы со старым списком по identity и падала бы с
    StopIteration в _orbit_sequence."""
    # дефолт зависит от ИМЕНИ (primary_name/secondary_name), а не от роли primary/
    # secondary - иначе при перестановке ролей (см. п.9: primary_name='ka2') None
    # молча резолвился бы в неправильный спутник
    if primary_profs is None:
        primary_profs = profiles_by_name[primary_name]
    if secondary_profs is None:
        secondary_profs = profiles_by_name[secondary_name]
    seq1 = _orbit_sequence(p_primary, primary_profs)
    seq2 = _orbit_sequence(p_secondary, secondary_profs)
    for w in _order_warnings(primary_name, seq1, want_ascending) + _order_warnings(secondary_name, seq2, want_ascending):
        print(w)
    t_eq1 = p_primary['t'][np.nanargmin(np.abs(p_primary['lat']))]
    second_label, second_p = _second_best(seq2, t_eq1, want_ascending)
    row_order = _ROW_ORDER if want_ascending else _ROW_ORDER_DESC
    rows = []
    for label in row_order:
        tr_style = ' style="font-weight:bold;"' if label == 'текущая' else ''
        sec_style = ' style="font-weight:bold;"' if label == second_label else ''
        rows.append(f'<tr{tr_style}><td>{label}</td>'
                    f'<td>{_cell_html(seq1[label])}</td>'
                    f'<td{sec_style}>{_cell_html(seq2[label])}</td></tr>')
    table = ('<table style="border-collapse:collapse;font-size:13px" border="1" cellpadding="6">'
             f'<tr><th></th><th>{primary_name}</th><th>{secondary_name}</th></tr>' + ''.join(rows) + '</table>')
    display(HTML(table))
    return second_p

_legend_ka = [
    plt.Line2D([], [], color='#e05c2a', lw=2, label='ka1'),
    plt.Line2D([], [], color='#2a7ae0', lw=2, label='ka2'),
]


def plot_verdict_pair_with_maps(tracks, title, legend_handles=None, *, steps):
    """Доля verdict=1 слева; справа — по карте на каждую трассу с этого графика.

    tracks: [(profile, color, map_title), ...] — None-профиль даёт пустую карту-заглушку.
    title — заголовок левого графика; legend_handles — легенда, по умолчанию ka1 / ka2.
    """
    import cartopy.crs as ccrs
    if legend_handles is None:
        legend_handles = _legend_ka
    n_maps = max(len(tracks), 1)
    fig = plt.figure(figsize=(6.4 + 4.3 * n_maps, 5.0))
    gs = fig.add_gridspec(1, 1 + n_maps, width_ratios=[1.45] + [1.0] * n_maps)
    ax = fig.add_subplot(gs[0, 0])
    for prof, color, _lab in tracks:
        if prof is not None:
            ax.plot(steps, prof['verdict'], color=color, alpha=0.6, lw=1.5)
    ax.set_title(title)
    ax.set_xlabel('Шаг вдоль орбиты (полюс → полюс)')
    ax.set_ylabel('Доля verdict=1')
    ax.legend(handles=legend_handles, fontsize=8)
    ax.grid(alpha=0.3)
    for i, (prof, color, map_title) in enumerate(tracks):
        axm = fig.add_subplot(gs[0, 1 + i], projection=ccrs.PlateCarree())
        plot_orbit_map(axm, prof, color=color, title=map_title)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.12, wspace=0.16)
    plt.show()
    plt.close(fig)
