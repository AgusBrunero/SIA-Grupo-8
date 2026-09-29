"""Runner de experimentos para los Ej. 2 y 3.

Cada experimento es la config base de configs/<ej>.json pisada por una variante de un
barrido, repetida con varias semillas. Cada corrida se guarda como una línea de
resultados/digitos/<ej>/corridas.jsonl (config + historial por época + tiempo) y el
análisis (analisis_digitos.py) solo lee esos archivos.

Si se interrumpe, al volver a correr se saltean las corridas ya guardadas.
"""
import os
# Un hilo de BLAS por proceso: se paraleliza entre corridas, no dentro de cada una
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import digitos
from perceptron.mlp import MLP
from perceptron.entrenamiento import TrainConfig, train, predict

ROOT = Path(__file__).parent
CONFIGS = ROOT / 'configs'
RESULTS = ROOT / 'resultados' / 'digitos'
MODELS = ROOT / 'modelos'

MODEL_KEYS = ['layers', 'hidden', 'output', 'beta', 'loss', 'init']
TRAIN_KEYS = list(TrainConfig.__dataclass_fields__)


# ---------- datos de cada ejercicio ----------

def dataset(data_cfg: dict):
    """Devuelve X_train, y_train, X_val, y_val según la sección 'data' de la config."""
    X, y = digitos.load(data_cfg['train'][0])
    for extra in data_cfg['train'][1:]:
        X2, y2 = digitos.load(extra)
        if data_cfg.get('dedupe', True):
            X, y, _ = digitos.combine_without_duplicates(X, y, X2, y2)
        else:
            X, y = np.concatenate([X, X2]), np.concatenate([y, y2])
    tr, va = digitos.stratified_split(y, data_cfg['val_fraction'], data_cfg['split_seed'])
    if data_cfg.get('restrict_train_to'):
        # Ablación del Ej. 3: misma validación, pero se entrena solo con las imágenes que
        # ya estaban en ese dataset (mide cuánto aportan los datos nuevos)
        X_old, _ = digitos.load(data_cfg['restrict_train_to'])
        old = {row.tobytes() for row in X_old}
        tr = np.array([i for i in tr if X[i].tobytes() in old])
    return X[tr], y[tr], X[va], y[va]


def balance(X, y, rng):
    """Oversampling: agrega copias al azar de las clases chicas hasta la mediana de las presentes.
    Se conservan todas las muestras originales; solo se suman repeticiones."""
    counts = digitos.class_counts(y)
    target = int(np.median(counts[counts > 0]))
    extra = [rng.choice(np.flatnonzero(y == c), target - counts[c])
             for c in range(digitos.N_CLASSES) if 0 < counts[c] < target]
    idx = np.concatenate([np.arange(len(y)), *extra])
    return X[idx], y[idx]


# ---------- una corrida ----------

def build_model(cfg: dict) -> MLP:
    return MLP(**{k: cfg[k] for k in MODEL_KEYS}, seed=cfg['seed'], dtype=np.float32)


def train_config(cfg: dict) -> TrainConfig:
    return TrainConfig(**{k: cfg[k] for k in TRAIN_KEYS if k in cfg})


def augment_fn(cfg: dict):
    aug = cfg.get('augment')
    return (lambda X, rng: digitos.augment(X, rng, **aug)) if aug else None


def fit(cfg: dict, data, verbose=False):
    X_tr, y_tr, X_va, y_va = data
    if cfg.get('balance'):
        X_tr, y_tr = balance(X_tr, y_tr, np.random.default_rng(cfg['seed']))
    model = build_model(cfg)
    history = train(model, X_tr, digitos.one_hot(y_tr), train_config(cfg), X_va, digitos.one_hot(y_va),
                    augment_fn=augment_fn(cfg), verbose=verbose)
    return model, history


def per_class_recall(y, pred):
    return [float(np.mean(pred[y == c] == c)) if np.any(y == c) else None for c in range(digitos.N_CLASSES)]


def run_one(args):
    cfg, data_cfg = args
    data = dataset(data_cfg | cfg.get('data', {}))
    model, history = fit(cfg, data)
    pred = predict(model, data[2])
    # El test se registra solo como referencia de "producción": la selección se hace con validación
    X_test, y_test = digitos.load('digits_test')
    pred_test = predict(model, X_test)
    return {'group': cfg['group'], 'variant': cfg['variant'], 'seed': cfg['seed'], 'config': cfg,
            'n_params': model.n_params(), 'history': history,
            'val_recall_por_clase': per_class_recall(data[3], pred),
            'test_accuracy': float(np.mean(pred_test == y_test)),
            'test_recall_por_clase': per_class_recall(y_test, pred_test)}


# ---------- barridos ----------

def expand(spec: dict) -> list[dict]:
    """Config base × (variantes de cada barrido) × semillas."""
    runs = []
    for group, variants in spec['sweeps'].items():
        for variant in variants:
            name = variant.pop('name', None) or ', '.join(f'{k}={v}' for k, v in variant.items())
            variant['name'] = name
            # En las ablaciones cada paso se acumula sobre el anterior
            cfg = spec['base'] | variant
            for seed in spec['seeds']:
                runs.append(cfg | {'group': group, 'variant': name, 'seed': seed})
    return runs


def run_key(cfg: dict) -> str:
    return json.dumps({k: v for k, v in cfg.items() if k != 'name'}, sort_keys=True)


def run_sweeps(name: str, workers: int | None = None) -> Path:
    spec = json.loads((CONFIGS / f'{name}.json').read_text())
    out = RESULTS / name / 'corridas.jsonl'
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {run_key(json.loads(line)['config']) for line in out.read_text().splitlines() if line}
    pending = [cfg for cfg in expand(spec) if run_key(cfg) not in done]
    print(f'[{name}] {len(pending)} corridas pendientes ({len(done)} ya hechas)', flush=True)

    digitos_cache_warmup(spec['data'])
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(workers) as pool, out.open('a') as f:
        futures = [pool.submit(run_one, (cfg, spec['data'])) for cfg in pending]
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            f.write(json.dumps(rec) + '\n')
            f.flush()
            h = rec['history']
            print(f"  [{i}/{len(pending)}] {rec['group']:14s} {rec['variant']:40s} seed={rec['seed']} "
                  f"val_acc={max(h['val_acc']):.4f} ({h['time'][-1]:.0f}s)", flush=True)
    return out


def digitos_cache_warmup(data_cfg):
    # Parsear los CSV una vez antes de lanzar los procesos (si no, todos parsean a la vez)
    for n in data_cfg['train']:
        digitos.load(n)
    digitos.load('digits_test')


def confusion_matrix(y, pred, n=digitos.N_CLASSES) -> np.ndarray:
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (y, pred), 1)
    return cm


def classification_report(y, pred) -> dict:
    cm = confusion_matrix(y, pred)
    tp = np.diag(cm)
    with np.errstate(invalid='ignore', divide='ignore'):
        precision = np.where(cm.sum(0) > 0, tp / cm.sum(0), 0.0)
        recall = np.where(cm.sum(1) > 0, tp / cm.sum(1), 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return {'accuracy': float(np.mean(pred == y)), 'balanced_accuracy': float(recall.mean()),
            'macro_f1': float(f1.mean()), 'precision': precision.tolist(), 'recall': recall.tolist(),
            'f1': f1.tolist(), 'soporte': cm.sum(1).tolist(), 'confusion': cm.tolist()}


def final(name: str, key: str = 'final') -> dict:
    """Entrena la config final (sobre train, con early stopping en validación), la guarda en
    modelos/ y la evalúa UNA vez en digits_test (el 'mundo real')."""
    s = spec(name)
    cfg = s['base'] | s[key] | {'group': key, 'variant': key}
    cfg['seed'] = s[key].get('seed', 0)
    data = dataset(s['data'] | cfg.get('data', {}))
    print(f'[{name}] {key}: {cfg}', flush=True)
    model, history = fit(cfg, data, verbose=True)

    X_test, y_test = digitos.load('digits_test')
    report = classification_report(y_test, predict(model, X_test))
    report['val'] = classification_report(data[3], predict(model, data[2]))
    record = {'config': cfg, 'n_params': model.n_params(), 'history': history, 'test': report}
    out = RESULTS / name / f'{key}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))
    model.save(MODELS / f'{name}_{key}.npz', extra={'config': cfg, 'test_accuracy': report['accuracy']})
    print(f"[{name}] {key}: test accuracy={report['accuracy']:.4f}  balanced={report['balanced_accuracy']:.4f}  "
          f"recall por clase={np.round(report['recall'], 3).tolist()}", flush=True)
    return record


def main(name: str):
    import argparse
    parser = argparse.ArgumentParser(description=f'Experimentos del {name}')
    parser.add_argument('etapa', choices=['barridos', 'final', 'todo'], nargs='?', default='todo')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--extra', nargs='*', default=[], help='otras claves de modelo final en la config')
    args = parser.parse_args()
    if args.etapa in ('barridos', 'todo'):
        run_sweeps(name, args.workers)
    if args.etapa in ('final', 'todo'):
        for key in ['final', *args.extra]:
            final(name, key)


def load_runs(name: str) -> list[dict]:
    path = RESULTS / name / 'corridas.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def spec(name: str) -> dict:
    return json.loads((CONFIGS / f'{name}.json').read_text())
