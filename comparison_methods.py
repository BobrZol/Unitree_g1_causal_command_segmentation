from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from constants import (RANDOM_STATE, WINDOW_STEP_SECONDS, N_ACTION_PROTOTYPES, N_SUBACTION_PROTOTYPES, DISTRACTOR_ID, NEUTRAL_ID, ACTIVE_COMMAND_IDS, COMMANDS, MAJORITY_WINDOWS, CONFIRMATION_WINDOWS, CONFIDENCE_THRESHOLDS, EVENT_IOU_THRESHOLD, EVENT_IOU_THRESHOLDS)
from matplotlib.patches import Patch
import argparse
from constants import SPLIT_DATA_DIR, RESULTS_DIR, RECORD_IDS, M4_F1_TOLERANCE, M4_SELECTION_MODE


DATA_DIR = SPLIT_DATA_DIR
FIG_DIR = RESULTS_DIR / 'figures'
LABEL_NAMES = {0: 'DISTRACTOR', 1: 'NEUTRAL', 2: 'FORWARD', 3: 'LEFT', 4: 'RIGHT'}
ALL_LABELS = np.array(sorted(LABEL_NAMES))

np.set_printoptions(precision=3, suppress=True); sns.set_theme(style='whitegrid', context='notebook')

def choose_m4_params(table, *, baseline_f1, f1_tolerance=M4_F1_TOLERANCE,
                     mode=M4_SELECTION_MODE):
    """Два правила подбора M4 на conf: f1 и reliability."""
    if mode not in ("f1", "reliability"):
        raise ValueError("Режим подбора должен быть f1 или reliability.")
    if not np.isfinite(baseline_f1) or not 0 <= baseline_f1 <= 1:
        raise ValueError("Нужен лучший F1 среди M1–M3 на conf от 0 до 1.")
    if not np.isfinite(f1_tolerance) or not 0 <= f1_tolerance <= 1:
        raise ValueError("Относительный допуск F1 должен быть от 0 до 1.")
    valid = table.loc[np.isfinite(table["event_f1@50"])]
    if valid.empty:
        raise ValueError("Нет вариантов M4 с конечным Event-F1@50 на conf.")
    best_f1 = valid["event_f1@50"].max()
    if mode == "f1":
        cutoff = best_f1
        selected = valid.sort_values(
            ["event_f1@50", "false_activations_per_min", "p95_delay_s", "median_delay_s"],
            ascending=[False, True, True, True], kind="stable", na_position="last",
        ).iloc[0].copy()
    else:
        cutoff = baseline_f1 if best_f1 >= baseline_f1 else best_f1 * (1 - f1_tolerance)
        eligible = valid.loc[valid["event_f1@50"] >= cutoff - 1e-12]
        selected = eligible.sort_values(
            ["false_activations_per_min", "double_commands_per_transition",
             "p95_delay_s", "median_delay_s", "event_f1@50"],
            ascending=[True, True, True, True, False],
            kind="stable", na_position="last",
        ).iloc[0].copy()
    selected["selection_mode"] = mode
    selected["conf_baseline_f1@50"] = baseline_f1
    selected["conf_best_m4_f1@50"] = best_f1
    selected["conf_min_m4_f1@50"] = cutoff
    return selected


def load_split(name, data_dir=DATA_DIR):
    df = pd.read_csv(Path(data_dir) / f'{name}.csv')
    meta_columns = ['record_id', 'window_start_frame', 'window_end_frame']
    pc_columns = [c for c in df.columns if c.startswith('pc')]

    return df[pc_columns].to_numpy(float), df['target'].to_numpy(int), df[meta_columns].copy()

def record_slices(record_ids, meta):
    ids = meta['record_id'].to_numpy()
    if len(ids) == 0:
        raise ValueError('Выборка не содержит окон')
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    stops = np.r_[starts[1:], len(ids)]
    actual_ids = tuple(ids[starts])

    assert actual_ids == tuple(record_ids), (actual_ids, record_ids)
    slices = [slice(start, stop) for start, stop in zip(starts, stops)]
    return slices, [sl.stop - sl.start for sl in slices]

# Обучение VQ и сопоставление прототипов
def fit_cluster_mapping(cluster_ids, labels, n_clusters, *, commands=COMMANDS):
    """Сопоставление с переданными классами; по умолчанию — команды G1."""
    score = np.zeros((n_clusters, len(commands)), dtype=int)
    for k in range(n_clusters):
        for j, c in enumerate(commands):
            score[k, j] = np.sum((cluster_ids == k) & (labels == c))

    rows, cols = linear_sum_assignment(-score)
    mapping = {int(k): int(commands[j]) for k, j in zip(rows, cols)}

    if len(mapping) != n_clusters or any(score[k].sum() == 0 for k in mapping):
        raise RuntimeError('Не удалось однозначно сопоставить VQ-прототипы')
    return mapping, score

def predict_one_level(X, one_mapping, one_vq):
    return np.array([one_mapping[int(k)] for k in one_vq.predict(X)], dtype=int)

def predict_two_level(X, sub_vq, z_to_a, two_mapping, with_confidence=False,
                      *, n_action_prototypes=N_ACTION_PROTOTYPES):
    dz = sub_vq.transform(X)
    da = np.column_stack([dz[:, z_to_a == a].min(axis=1) for a in range(n_action_prototypes)])
    order = np.argsort(da, axis=1)
    best = order[:, 0]

    pred = np.array([two_mapping[int(a)] for a in best], dtype=int)
    d1, d2 = da[np.arange(len(X)), best], da[np.arange(len(X)), order[:, 1]]
    confidence = np.clip(1 - d1 / (d2 + 1e-12), 0, 1)

    return (pred, confidence) if with_confidence else pred

# Каузальные фильтры и метрики
# Event-F1 показывается для IoU >= 0.10, 0.25 и 0.50. Остальные event-метрики
# используют основной строгий порог IoU >= 0.50.
# Ложная активация - непарное предсказанное активное событие, двойная команда - каждый повторный старт той же команды внутри одного эталонного активного сегмента.
# Задержка измеряется для найденных событий. Кроме того двойственность возможна в понятии двойных команд мы учитываем просто bool мол состоялась ли двойная команда
# в интервале, или количество колибаний, я посчитал количество кажется это логичнее. 

def causal_majority(labels, window):
    """
        Cкользящее большинство по последним window окнам
        Вход:
            labels - метки классов
            window - размер окна
        Выход:
            out - большинство по последним window окнам
    """
    out = np.empty_like(labels)
    for t in range(len(labels)):
        v, n = np.unique(labels[max(0, t - window +  1): t + 1], return_counts=True)
        out[t] = v[np.argmax(n)]
    return out

def transition_automaton(labels, confidence, confirmations, threshold,
                         *, neutral_id=NEUTRAL_ID, label_names=LABEL_NAMES,
                         distractor_id=DISTRACTOR_ID):
    """
        Каузальный автомат с отдельными состояниями NEUTRAL и DISTRACTOR.
        Новая команда активируется только после confirmations подряд идущих
        одинаковых решений с уверенностью не ниже threshold.

        Вход:
            labels - метки классов от квантования (N,)
            confidence - уверенность на каждом окне [0, 1] (N,)
            confirmations - число подтверждений подряд (1, 3, 5)
            threshold - ниже порога удерживаем последнюю подтверждённую команду
            distractor_id - начальная метка, None задаёт начальный NEUTRAL

        Выход:
            out - последняя подтверждённая команда (N,)
            states - состояния автомата (N,): 'DISTRACTOR', 'NEUTRAL', 'TRANSITION', 'ACTIVE_<name>'
    """
    current = neutral_id if distractor_id is None else distractor_id
    pending, run = None, 0
    out, states = np.empty(len(labels), dtype=np.int64), []

    for observed, conf in zip(labels, confidence):
        # Неуверенное окно не меняет команду и прерывает серию подтверждений.
        proposed = int(observed) if conf >= threshold else current

        if proposed == current: # ничерта не поменялось новых кандидатов нет
            pending, run = None, 0
            state = ('DISTRACTOR' if current == distractor_id else
                     'NEUTRAL' if current == neutral_id else f'ACTIVE_{label_names[current]}')
        else: # появился новый кандидат на смену current
            run = run + 1 if pending == proposed else 1
            pending = proposed

            if run >= confirmations: # опа меняем
                current, pending, run = proposed, None, 0
                state = ('DISTRACTOR' if current == distractor_id else
                         'NEUTRAL' if current == neutral_id else f'ACTIVE_{label_names[current]}')
            else: # начинаем преход
                state = 'TRANSITION'

        out[len(states)] = current
        states.append(state)

    return out, np.asarray(states, dtype=object)

def apply_per_record(labels, slices, fn, confidence=None):
    """
        Применяет каузальный фильтр fn к каждой записи отдельно, чтобы границы
        между записями не пересекались и каузальность не нарушалась.

        Вход:
            labels - предсказания модели (N,)
            slices - список slice(start, stop) по одной на запись
            fn - каузальный фильтр: causal_majority или transition_automaton
            confidence - уверенности (N,) для методов, которым они нужны.
                        None, если фильтр их не использует.
        Выход:
            out - предсказания после фильтрации (N,), склеенные в один массив
            states - состояния (N,). Если fn вернул кортеж (out, states) —
                    берём состояния оттуда. Иначе заполняем 'FILTERED'.
    """
    out, states = np.empty(len(labels), dtype=np.int64), np.empty(len(labels), dtype=object)

    for sl in slices:
        answer = fn(labels[sl]) if confidence is None else fn(labels[sl], confidence[sl])
    
        if isinstance(answer, tuple):
            out[sl], states[sl] = answer
        else:
            out[sl], states[sl] = answer, 'FILTERED'

    return out, states

def events(labels, *, active_command_ids=ACTIVE_COMMAND_IDS):
    """
        Находит непрерывные отрезки активных команд в последовательности меток.

        Пробегает по массиву меток слева направо и группирует подряд идущие
        одинаковые значения в события. Событием считается непрерывный отрезок,
        на котором метка не меняется. В результат попадают только отрезки,
        чья метка принадлежит ACTIVE_COMMAND_IDS (FORWARD, LEFT, RIGHT),
        то есть реальные команды. NEUTRAL и DISTRACTOR игнорируются —
        они не являются активными командами.

        Вход:
            labels - метки классов (N,)
        Выход:
            answer - список кортежей (command, start, end):
                    command - код команды из ACTIVE_COMMAND_IDS
                    start - индекс первого окна отрезка (включительно)
                    end - индекс после последнего окна (эксклюзивно)
    """
    answer, start = [], 0

    for t in range(1, len(labels) + 1):
        if t == len(labels) or labels[t] != labels[start]:
            if labels[start] in active_command_ids:
                answer.append((int(labels[start]), start, t))
            start = t

    return answer

def event_matches(reference, predicted, *, iou_threshold=EVENT_IOU_THRESHOLD):
    """
        Сопоставляет предсказанные события с эталонными по IoU-перекрытию.

        Событие считается совпавшим, если:
        1. метки совпадают (одна и та же команда),
        2. IoU (intersection over union) пересечения отрезков >= EVENT_IOU_THRESHOLD.

        Из всех кандидатов для каждого предсказанного события выбирается тот
        эталонный отрезок, у которого IoU максимальный. Жадное сопоставление:
        каждое эталонное событие может быть сопоставлено не более одного раза.

        Вход:
            reference - эталонные события [(label, start, end), ...]
            predicted - предсказанные события [(label, start, end), ...]
        Выход:
            matches - список пар (ref_index, pred_index) для совпавших событий.
                    Индексы указывают на позиции в исходных списках.
    """
    available, matches = set(range(len(reference))), []

    for pi, (lab, ps, pe) in enumerate(predicted):
        candidates = []
        for ri in available:
            rlab, rs, re = reference[ri]
            inter = max(0, min(pe, re)-max(ps, rs))
            union = max(pe, re)-min(ps, rs)

            if lab == rlab and inter / union >= iou_threshold:
                candidates.append((inter / union, ri))

        if candidates:
            _, ri = max(candidates); available.remove(ri); matches.append((ri, pi))

    return matches

def event_f1_column(iou_threshold):
    """Имя колонки для Event-F1 при заданном IoU-пороге."""
    return f'event_f1@{round(100 * iou_threshold):02d}'


def evaluate(y_true, y_pred, slices, *, active_command_ids=ACTIVE_COMMAND_IDS,
             window_step_seconds=WINDOW_STEP_SECONDS, iou_threshold=EVENT_IOU_THRESHOLD,
             f1_iou_thresholds=EVENT_IOU_THRESHOLDS):
    """
        Считает все метрики задания на наборе записей.
        Оцениваем выходные метки команд, а не внутренние состояния автомата.

        Вход:
            y_true - эталонные метки (N,)
            y_pred - предсказанные метки (N,)
            slices - список slice(start, stop) по одной на запись
        Выход:
            dict с метриками:
                frame_accuracy - покадровая точность (п.10.1)
                event_f1@10, event_f1@25, event_f1@50 - F1 по событиям
                    при соответствующих IoU-порогах (п.10.2)
                event_precision, event_recall - точность/полнота по событиям
                    при основном пороге iou_threshold
                false_activations_per_min - ложные активации в минуту (п.10.3)
                median_delay_s - медианная задержка (п.10.4)
                p95_delay_s - 95-й перцентиль задержки (п.10.5)
                double_commands_per_transition - двойные команды на переход (п.10.6)
                missed_commands - пропущенные команды (п.10.7)
                reference_events, predicted_events - вспомогательные счётчики
    """
    refs, preds, matches, offset = [], [], [], 0
    f1_iou_thresholds = tuple(dict.fromkeys(float(x) for x in f1_iou_thresholds))
    matches_by_iou = {threshold: [] for threshold in f1_iou_thresholds}

    for sl in slices:
        ref = events(y_true[sl], active_command_ids=active_command_ids)
        pred = events(y_pred[sl], active_command_ids=active_command_ids)
        local = event_matches(ref, pred, iou_threshold=iou_threshold)
        local_by_iou = {
            threshold: event_matches(ref, pred, iou_threshold=threshold)
            for threshold in f1_iou_thresholds
        }

        ref_base, pred_base = len(refs), len(preds)

        # надо сделать офсет так как по факту мы взяли 6 записей и склеили в 3
        refs.extend([(a, s + offset, e + offset) for a, s, e in ref])
        preds.extend([(a, s + offset, e + offset) for a, s, e in pred])
        matches.extend([(ri + ref_base, pi + pred_base) for ri, pi in local])
        for threshold, local_matches in local_by_iou.items():
            matches_by_iou[threshold].extend(
                (ri + ref_base, pi + pred_base) for ri, pi in local_matches
            )

        offset += sl.stop - sl.start

    tp, nr, npred = len(matches), len(refs), len(preds)
    precision = tp / npred if npred else 0
    recall = tp / nr if nr else 0


    event_f1s = {}
    for threshold, threshold_matches in matches_by_iou.items():
        threshold_precision = len(threshold_matches) / npred if npred else 0
        threshold_recall = len(threshold_matches) / nr if nr else 0
        event_f1s[event_f1_column(threshold)] = (
            2 * threshold_precision * threshold_recall / (threshold_precision + threshold_recall)
            if threshold_precision + threshold_recall else 0
        )
    delays = [max(0, preds[pi][1] - refs[ri][1]) * window_step_seconds for ri, pi in matches]
    doubles = sum(max(0, sum(a == ra and rs <= ps < re for a, ps, _ in preds) - 1) for ra, rs, re in refs)
    duration_min = len(y_true) * window_step_seconds / 60

    return {'frame_accuracy': np.mean(y_true == y_pred), **event_f1s,
            'event_precision': precision, 'event_recall': recall,
            'false_activations_per_min': (npred - tp) / duration_min, 'median_delay_s': np.median(delays) if delays else np.nan,
            'p95_delay_s': np.percentile(delays, 95) if delays else np.nan, 'double_commands_per_transition': doubles / nr if nr else np.nan,
            'missed_commands': nr - tp, 'reference_events': nr, 'predicted_events': npred}

# постройка графиков

def build_plot_reliability_latency_m4(tuning, *, fig_dir=FIG_DIR):
    tradeoff = tuning.query("method == 'M4_two_level_automaton'").copy()

    fig, ax = plt.subplots(figsize=(14, 7))

    colors = {1: 'tab:blue', 2: 'tab:purple', 3: 'tab:orange', 4: 'tab:red', 5: 'tab:green'}
    markers = {1: 'o', 2: 'D', 3: 's', 4: 'v', 5: '^'}
    EPS = 0.01
    ROUND = 3

    LEVEL_OFFSETS = [14, 34, 54, 74, 94]
    X_SHIFTS      = [0, 0, 0, 0, 0]

    for n, part in tradeoff.groupby('confirmations'):
        n = int(n)
        color = colors[n]

        part = part.copy()
        part['x_r'] = part.median_delay_s.round(ROUND)
        part['y_r'] = part.false_activations_per_min.clip(lower=EPS).round(ROUND)

        grouped = (part.groupby(['x_r', 'y_r'])['threshold']
                    .apply(lambda s: sorted(s.unique()))
                    .reset_index())

        xs = grouped.x_r.to_numpy()
        ys = grouped.y_r.to_numpy()

        ax.scatter(xs, ys, s=110, marker=markers[n], color=color,
                edgecolor='black', linewidth=0.9,
                label=f'подтверждений = {n}', zorder=4)

        for i, (_, row) in enumerate(grouped.iterrows()):
            qs = row.threshold
            if len(qs) == 1:
                label = f'q={qs[0]:.2f}'
            else:
                label = 'q=' + ', '.join(f'{q:.2f}' for q in qs)

            yy = row.y_r
            is_zero = yy == EPS

            if is_zero:
                level = i % len(LEVEL_OFFSETS)
                dy = LEVEL_OFFSETS[level]
                dx = X_SHIFTS[level]
                xytext = (dx, dy)
                ha = 'center'
            else:
                xytext, ha = (0, 12), 'center'

            ax.annotate(label,
                        xy=(row.x_r, yy),
                        xytext=xytext,
                        textcoords='offset points',
                        ha=ha, fontsize=10, color=color, fontweight='bold',
                        arrowprops=dict(
                            arrowstyle='-',
                            color=color,
                            lw=3,
                            alpha=0.5,
                            connectionstyle='angle,angleA=90,angleB=0,rad=0',
                        ))

    ax.axhline(EPS, color='black', linestyle='-', linewidth=1.8,
            alpha=0.85, zorder=2)
    ax.text(0.99, EPS * 1.15, 'FP/min = 0',
            transform=ax.get_yaxis_transform(),
            ha='right', va='bottom',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.35',
                    facecolor='white', edgecolor='black', linewidth=1.2),
            zorder=5)

    ax.set_yscale('log')
    ax.set_ylim(EPS / 5, 30)
    ax.set_xlim(0.0, 1.2)

    ax.set_xlabel('Медианная задержка, с', fontweight='bold', fontsize=24)
    ax.set_ylabel('Ложные активации в минуту',
                fontweight='bold', fontsize=24)
    ax.set_title('Надёжность-задержка: двухуровневый автомат',
                fontweight='bold', fontsize=30)

    ax.grid(True, which='both', alpha=0.3)
    ax.grid(True, which='minor', alpha=0.15)
    ax.legend(title='Число подтверждений', fontsize=12, title_fontsize=12,
            loc='upper right', framealpha=0.95)

    fig.tight_layout()
    if fig_dir is not None:
        Path(fig_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(fig_dir) / 'validation_reliability_delay.png',
                    dpi=160, bbox_inches='tight')
    return fig

def build_plot_prediction_reference(predictions, y_test, *, labels=ALL_LABELS,
                                     label_names=LABEL_NAMES, fig_dir=FIG_DIR):
    fig, axes = plt.subplots(2, 2, figsize=(13, 13), constrained_layout=True)

    for ax, (name, pred) in zip(axes.flat, predictions.items()):
        cm = confusion_matrix(y_test, pred, labels=labels)

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar=False,
            ax=ax,
            xticklabels=[label_names[i] for i in labels],
            yticklabels=[label_names[i] for i in labels],
        )

        ax.set_title(name, fontweight='bold')
        ax.set_xlabel('Предсказание', fontweight='bold')
        ax.set_ylabel('Эталон', fontweight='bold')

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')

    if fig_dir is not None:
        Path(fig_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(fig_dir) / 'test_confusion_matrices.png', dpi=160)
    return fig

def build_plot_reliability_latency_compare(comparison, *, fig_dir=FIG_DIR):
    fig, ax = plt.subplots(figsize=(6, 5))

    for _, row in comparison.iterrows():
        # точка
        ax.scatter(
            row.median_delay_s,
            row.false_activations_per_min,
            s=90,
            zorder=3,
        )
        # подпись рядом с точкой (M1, M2, M3, M4)
        ax.annotate(
            row.method.split(':')[0],
            (row.median_delay_s, row.false_activations_per_min),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=10,
        )

    # жирные подписи осей и заголовок
    ax.set_xlabel('Медианная задержка, с', fontweight='bold')
    ax.set_ylabel('Ложные активации / мин', fontweight='bold')
    ax.set_title('test: надёжность–задержка', fontweight='bold')

    fig.tight_layout()
    if fig_dir is not None:
        Path(fig_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(fig_dir) / 'test_reliability_delay.png', dpi=160)
    return fig


# Offline-демонстратор и автоматический тест каузальности

def assert_causal(labels, confidence, confirmations, threshold,
                  *, neutral_id=NEUTRAL_ID, label_names=LABEL_NAMES,
                  distractor_id=DISTRACTOR_ID):
    full, _ = transition_automaton(labels, confidence, confirmations, threshold,
                                  neutral_id=neutral_id, label_names=label_names,
                                  distractor_id=distractor_id)
    for end in range(1, len(labels) + 1):
        prefix, _ = transition_automaton(labels[:end], confidence[:end],
                                         confirmations, threshold,
                                         neutral_id=neutral_id, label_names=label_names,
                                         distractor_id=distractor_id)
        if not np.array_equal(prefix, full[:end]):
            raise AssertionError(f'Нарушение каузальности в окне {end}')
    return True

def cumulative_switches(values):
    # Только смены выходных команд. Внутренние состояния сюда не передаются.
    return np.cumsum(np.r_[False, np.asarray(values)[1:] != np.asarray(values)[:-1]]) if len(values) else np.array([], dtype=int)


def offline_demonstrator_and_automated_causality_test(
        test_slices, num_slices, test_record_ids, two_test, two_test_confidence,
        best_confirmations, best_threshold, y_test, m4_test, m4_states,
        *, zoom_seconds=40.0, zoom_interval=(30, 80), fig_dir=FIG_DIR):
    """Общая запись слева и увеличенный фрагмент M3/M4 справа."""
    demo_slice = test_slices[num_slices]
    reference = np.asarray(y_test[demo_slice])
    raw = np.asarray(two_test[demo_slice])
    stable = np.asarray(m4_test[demo_slice])
    if not len(reference):
        raise ValueError("Для демонстратора нужна непустая запись.")
    assert_causal(raw, two_test_confidence[demo_slice], best_confirmations, best_threshold)
    dt = WINDOW_STEP_SECONDS
    t = np.arange(len(reference)) * dt
    duration = len(reference) * dt

    if zoom_interval is None:
        if not np.isfinite(zoom_seconds) or zoom_seconds <= 0:
            raise ValueError("Длительность увеличенного фрагмента должна быть положительной.")
        width = min(len(reference), max(1, int(round(zoom_seconds / dt))))
        gain = (stable == reference).astype(int) - (raw == reference).astype(int)
        prefix = np.r_[0, np.cumsum(gain)]
        scores = prefix[width:] - prefix[:-width]
        left = int(np.argmax(scores))
        right = left + width
    else:
        lo, hi = zoom_interval
        if not 0 <= lo < hi <= duration:
            raise ValueError("Границы увеличенного фрагмента должны лежать внутри записи.")
        left = int(np.floor(lo / dt))
        right = min(len(reference), int(np.ceil(hi / dt)))
    lo, hi = left * dt, right * dt

    colors = {0: '#9e9e9e', 1: '#e0e0e0', 2: '#1f77b4', 3: '#2ca02c', 4: '#d62728'}
    titles = ['Эталонная команда', 'Выходные команды M3', 'Выходная команда M4']
    tracks = [reference, raw, stable]

    def draw_track(ax, values, begin, end):
        index = begin
        while index < end:
            first, label = index, int(values[index])
            while index < end and int(values[index]) == label:
                index += 1
            ax.broken_barh([(first * dt, (index - first) * dt)], (0.15, 0.7),
                          facecolors=colors[label], edgecolors='white', linewidth=0.5)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.grid(axis='x', alpha=0.25)
        ax.tick_params(labelsize=15)

    fig = plt.figure(figsize=(30, 13))
    grid = fig.add_gridspec(4, 2, width_ratios=[1.8, 1], height_ratios=[1, 1, 1, 1.5],
                           left=0.055, right=0.98, bottom=0.08, top=0.86,
                           hspace=0.65, wspace=0.22)
    axes = [fig.add_subplot(grid[i, 0]) for i in range(4)]
    zoom_axes = [fig.add_subplot(grid[i, 1]) for i in range(3)]
    for ax, zoom_ax, values, title in zip(axes, zoom_axes, tracks, titles):
        draw_track(ax, values, 0, len(values))
        draw_track(zoom_ax, values, left, right)
        ax.set_xlim(0, duration)
        zoom_ax.set_xlim(lo, hi)
        ax.set_title(title, loc='left', fontsize=23, fontweight='bold')
        zoom_ax.set_title(title, loc='left', fontsize=20, fontweight='bold')
        ax.axvspan(lo, hi, facecolor='none', edgecolor='black', linewidth=2, linestyle='--')
    zoom_axes[-1].set_xlabel('Время от начала записи, с', fontsize=21, fontweight='bold')

    for values, label, color in ((raw, 'M3', '#888888'),
                                  (stable, 'M4', '#d62728')):
        axes[3].step(t, cumulative_switches(values), where='post',
                     linewidth=2, color=color, label=label)
    axes[3].set_xlim(0, duration)
    axes[3].axvspan(lo, hi, color='black', alpha=0.07)
    axes[3].set_title('Накопленное число переключений', loc='left', fontsize=23, fontweight='bold')
    axes[3].set_xlabel('Время, с', fontsize=24, fontweight='bold')
    axes[3].set_ylabel('Переключения', fontsize=24, fontweight='bold')
    axes[3].tick_params(labelsize=16)
    axes[3].legend(prop={'size': 20, 'weight': 'bold'}, loc='upper left')
    for ax in axes + zoom_axes:
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight('bold')
    handles = [Patch(facecolor=colors[k], edgecolor='black', label=LABEL_NAMES[k])
               for k in [1, 2, 3, 4, 0]]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.955),
               ncol=5, prop={'size': 24, 'weight': 'bold'}, handlelength=2, handleheight=1.4, frameon=True)
    fig.suptitle('Offline-демонстратор',
                 fontsize=28, fontweight='bold', y=0.995)
    if fig_dir is not None:
        Path(fig_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(fig_dir) / f'offline_demo_rec_{test_record_ids[num_slices]}.png',
                    dpi=160, bbox_inches='tight')
    return fig

def main():
    from dataset import BvhDataset
    from constants import LABEL_MAP
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-preparation', action='store_true',
                        help='Использовать готовые train/conf/test.csv без повторной обработки BVH')
    parser.add_argument(
        "--records",
        nargs="+",
        default=list(RECORD_IDS),
        help='Полные имена без расширения: первые 3 train, четвёртая conf, остальные test',
    )
    args = parser.parse_args()
    if len(args.records) < 5 or len(set(args.records)) != len(args.records):
        parser.error('--records требует минимум 5 разных записей: 3 train, 1 conf, остальные test')

    records_ids = tuple(args.records)
    train_record_ids = records_ids[:3]
    conf_record_ids = records_ids[3:4]
    test_record_ids = records_ids[4:]
    if not args.skip_preparation:
        BvhDataset(args.records, output_dir=DATA_DIR, label_map=LABEL_MAP)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print(args.records)


    # Загрузка датасетов
    X_train, y_train, train_meta = load_split('train')
    X_conf, y_conf, conf_meta = load_split('conf')
    X_test, y_test, test_meta = load_split('test')

    _, train_sizes = record_slices(train_record_ids, train_meta)
    conf_slices, conf_sizes = record_slices(conf_record_ids, conf_meta)
    test_slices, test_sizes = record_slices(test_record_ids, test_meta)

    print(f'train={X_train.shape}, conf={X_conf.shape}, test={X_test.shape}')
    print('Окон по записям:', dict(zip(records_ids, [*train_sizes, *conf_sizes, *test_sizes])))
    display(pd.DataFrame({'split': ['train', 'conf', 'test'], 'windows': [len(y_train), len(y_conf), len(y_test)]}).assign(duration_min=lambda d: d.windows * WINDOW_STEP_SECONDS / 60))

    # Сопоставляем прототипы со всеми пятью классами по train.
    X_cmd, y_cmd = X_train, y_train

    # одноуровневый паплайн

    one_vq = KMeans(n_clusters=N_ACTION_PROTOTYPES, n_init=20, random_state=RANDOM_STATE).fit(X_cmd)
    one_mapping, one_score = fit_cluster_mapping(one_vq.labels_, y_cmd, N_ACTION_PROTOTYPES)

    # двухуроыенвыый паплайн

    sub_vq = KMeans(n_clusters=N_SUBACTION_PROTOTYPES, n_init=20, random_state=RANDOM_STATE).fit(X_cmd)
    sub_weights = np.bincount(sub_vq.labels_, minlength=N_SUBACTION_PROTOTYPES).astype(float)

    action_vq = KMeans(n_clusters=N_ACTION_PROTOTYPES, n_init=20, random_state=RANDOM_STATE).fit(sub_vq.cluster_centers_, sample_weight=sub_weights)
    z_to_a = action_vq.labels_

    two_mapping, _ = fit_cluster_mapping(z_to_a[sub_vq.labels_], y_cmd, N_ACTION_PROTOTYPES)

    print('Одноуровневое сопоставление:', one_mapping)
    display(pd.DataFrame(one_score, columns=[LABEL_NAMES[c] for c in COMMANDS]))
    print('Двухуровневое сопоставление:', two_mapping, '; z->a:', z_to_a.tolist())
    print('Размеры поддействий:', sub_weights.astype(int).tolist())

    # Настройка только на conf. Для M2 сохраняем максимум Event-F1@50.
    # Режим M4 выбирается через M4_SELECTION_MODE в constants.py.
    # f1 максимизирует F1, reliability использует сравнение с M1–M3.

    one_conf = predict_one_level(X_conf, one_mapping, one_vq)
    two_conf, two_confidence = predict_two_level(X_conf, sub_vq, z_to_a, two_mapping, with_confidence=True)

    rows = []
    for window in MAJORITY_WINDOWS:
        pred, _ = apply_per_record(one_conf, conf_slices, lambda x: causal_majority(x, window))
        rows.append({'method':'M2_one_level_majority', 'window':window, 'confirmations':np.nan, 'threshold':np.nan, **evaluate(y_conf, pred, conf_slices)})

    for confirmations, threshold in product(CONFIRMATION_WINDOWS, CONFIDENCE_THRESHOLDS):
        pred, _ = apply_per_record(two_conf, conf_slices, lambda x, c: transition_automaton(x, c, confirmations, threshold), two_confidence)
        rows.append({'method':'M4_two_level_automaton', 'window':np.nan, 'confirmations':confirmations, 'threshold':threshold, **evaluate(y_conf, pred, conf_slices)})

    tuning = pd.DataFrame(rows)

    def choose(df):
        return df.sort_values(['event_f1@50', 'false_activations_per_min', 'p95_delay_s', 'median_delay_s'], ascending=[False, True, True, True], na_position='last').iloc[0]

    best_m2 = choose(tuning.query("method == 'M2_one_level_majority'"))
    baseline_f1 = max(
        evaluate(y_conf, one_conf, conf_slices)["event_f1@50"],
        best_m2["event_f1@50"],
        evaluate(y_conf, two_conf, conf_slices)["event_f1@50"],
    )
    best_m4 = choose_m4_params(
        tuning.query("method == 'M4_two_level_automaton'"), baseline_f1=baseline_f1,
    )
    print(f'Режим M4: {best_m4["selection_mode"]}. Conf: лучший M1–M3 = {baseline_f1:.6f}, '
          f'лучший M4 = {best_m4["conf_best_m4_f1@50"]:.6f}, '
          f'нижняя граница M4 = {best_m4["conf_min_m4_f1@50"]:.6f}')
    best_window = int(best_m2.window); best_confirmations, best_threshold = int(best_m4.confirmations), float(best_m4.threshold)
    tuning.to_csv(RESULTS_DIR / 'validation_tuning.csv', index=False)
    display(tuning.sort_values(['method','event_f1@50'], ascending=[True,False]))

    print(f'Зафиксированы параметры: M2 window={best_window}; M4 confirmations={best_confirmations}, threshold={best_threshold:.2f}')

    # постройка графика надёжность-задержка
    plt.close(build_plot_reliability_latency_m4(tuning))


    ## Финальный расчёт на test

    one_test = predict_one_level(X_test, one_mapping, one_vq)
    two_test, two_test_confidence = predict_two_level(X_test, sub_vq, z_to_a, two_mapping, with_confidence=True)

    m2_test, _ = apply_per_record(one_test, test_slices, lambda x: causal_majority(x, best_window))
    m4_test, m4_states = apply_per_record(two_test, test_slices, lambda x,c: transition_automaton(x,c,best_confirmations,best_threshold), two_test_confidence)

    predictions = {'M1: 1-level, raw':one_test, f'M2: 1-level, majority (w={best_window})':m2_test, 'M3: 2-level, raw':two_test, f'M4: 2-level, automaton (n={best_confirmations}, q={best_threshold:.2f})':m4_test}
    comparison = pd.DataFrame([{'method':name, **evaluate(y_test,pred,test_slices)} for name,pred in predictions.items()])
    comparison.to_csv(RESULTS_DIR / 'test_method_comparison.csv', index=False)

    display(comparison.round(3))

    # постройка графика предсказание-эталон
    plt.close(build_plot_prediction_reference(predictions, y_test))

    # постройка графика ложные активации мидеанная задержка
    plt.close(build_plot_reliability_latency_compare(comparison))

    print(f'Размер test_slices {len(test_slices)}')

    for i in range(0, len(test_slices)):
        offline_demonstrator_and_automated_causality_test(test_slices, i, test_record_ids, two_test, two_test_confidence, best_confirmations, best_threshold, y_test, m4_test, m4_states)
        plt.close('all')

if __name__ == "__main__":
    main()
