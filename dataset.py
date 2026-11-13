import bvhio
import pandas as pd
import glm
import numpy as np
import gc
import hashlib
import json
import inspect
from tempfile import NamedTemporaryFile
from zipfile import BadZipFile
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from constants import (FPS, STEP, BVH_DATA_DIR,
                       SPLIT_DATA_DIR, RECORD_IDS, RANDOM_STATE)

class BvhDataset:
    # Загружаем датасет, определяем основные суставы делаем csv
    def __init__(self, main_joints=None, list_names=None, train_val_test_div = [3, 1, 1], *, data_dir=BVH_DATA_DIR,
                 output_dir=SPLIT_DATA_DIR, label_map=None, cut_target_slices=False):
        """
            main_joints - какие суставы используем
            list_names - имя bvh датасетов которые грузим
        """

        self.cut_target_slices = cut_target_slices
        self.train_val_test_div = train_val_test_div
        
        self.main_joints = main_joints
        self.label_map = None if label_map is None else dict(label_map)
        if isinstance(list_names, (str, bytes, set, frozenset, dict)):
            raise TypeError('Передайте список или кортеж имён: порядок задаёт train, conf и test')
        self.list_names = list(RECORD_IDS if list_names is None else list_names)
        assert sum(train_val_test_div) == len(self.list_names)
        if any(not isinstance(name, str) or not name.strip()
               or Path(name).name != name or name in ('.', '..')
               or name.lower().endswith(('.bvh', '.txt'))
               for name in self.list_names):
            raise ValueError('Указывайте полные имена файлов без пути и расширения, например rec_4_operator_2_chr01_MAYA')

        self.data_dir = Path(data_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        missing = [self.data_dir / f'{num}.{ext}'
                   for num in self.list_names for ext in ('bvh', 'txt')
                   if not (self.data_dir / f'{num}.{ext}').is_file()]

        if missing:
            raise FileNotFoundError('Не найдены входные файлы:\n' + '\n'.join(map(str, missing)))

        self.max_target_slices = self.__calc_max_target_slices() if cut_target_slices else None

        annotations = {num: self.__read_annotations(num) for num in self.list_names}
        if self.label_map is None:
            classes = sorted({row['command'] for rows in annotations.values()
                              for row in rows} - {'DISTRACTOR'})
            self.label_map = {name: index for index, name in enumerate(['DISTRACTOR', *classes])}
        if 'DISTRACTOR' not in self.label_map:
            raise ValueError('В label_map нужен класс DISTRACTOR для промежутков между действиями')
        if (any(not isinstance(value, (int, np.integer)) or value < 0
                for value in self.label_map.values())
                or len(set(self.label_map.values())) != len(self.label_map)):
            raise ValueError('Коды классов в label_map должны быть разными неотрицательными целыми числами')
        self.label_names = {int(value): key for key, value in self.label_map.items()}
        print(f'Классы датасета: {self.label_map}', flush=True)
        # Кэшируем окна до PCA, чтобы состав train не влиял на кэш записи.
        self.csv_dataset_joint = []
        cache_dir = self.output_dir / '.bvh_cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        for num in self.list_names:
            fingerprint = self.__record_fingerprint(num)
            cache_name = hashlib.sha256(str(self.data_dir / num).encode()).hexdigest()[:24]
            cache_path = cache_dir / f'{cache_name}.npz'
            cached = None
            if cache_path.is_file():
                try:
                    with np.load(cache_path, allow_pickle=False) as saved:
                        if saved['fingerprint'].item() == fingerprint:
                            cached = pd.DataFrame({
                                'record_id': num,
                                'window_start_frame': saved['starts'],
                                'window_end_frame': saved['ends'],
                                'features': list(saved['features']),
                                'target': saved['targets'],
                            })
                except (OSError, ValueError, KeyError, EOFError, BadZipFile):
                    print(f'Кэш {num} повреждён, пересчитываю запись', flush=True)
            if cached is not None:
                print(f'Запись {num}: загружена из кэша', flush=True)
                self.csv_dataset_joint.append(cached)
                continue

            record = BvhDataset.__new__(BvhDataset)
            record.main_joints = self.main_joints
            record.list_names = [num]
            record.data_dir = self.data_dir
            record.label_map = self.label_map
            record._annotations = {num: annotations[num]}
            record.__build_movement_speed()
            record.__build_command_datasets()
            record.__build_windows()
            record._annotations = {num: annotations[num]}
            record.cut_target_slices = self.cut_target_slices
            record.max_target_slices = self.max_target_slices
            windows = record.csv_dataset_joint[0]
            # Сначала отдельный файл, затем атомарная замена готового кэша.
            with NamedTemporaryFile(dir=cache_dir, suffix='.tmp', delete=False) as temp:
                temp_path = Path(temp.name)
            try:
                with temp_path.open('wb') as stream:
                    np.savez(stream, fingerprint=fingerprint,
                             starts=windows['window_start_frame'].to_numpy(),
                             ends=windows['window_end_frame'].to_numpy(),
                             features=np.vstack(windows['features']),
                             targets=windows['target'].to_numpy(dtype=str))
                temp_path.replace(cache_path)
            finally:
                temp_path.unlink(missing_ok=True)
            self.csv_dataset_joint.append(windows)
            del record

        self.__build_test_train_conf()

    def __record_fingerprint(self, num):
        digest = hashlib.sha256()
        # Кэш окон зависит от их расчёта, но не от PCA и настроек классификатора.
        for method in (self.__read_annotations, self.__build_movement_speed,
                       self.__build_command_datasets, self.__build_windows):
            digest.update(inspect.getsource(method).encode())
        for path in (self.data_dir / f'{num}.bvh',
                     self.data_dir / f'{num}.txt'):
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
        digest.update(json.dumps(
            [FPS, STEP, self.main_joints, self.cut_target_slices, self.max_target_slices],
            sort_keys=True
        ).encode())
        return digest.hexdigest()


    def __build_movement_speed(self):
        self.csv_dataset_joint = []

        for file_idx, num in enumerate(self.list_names):
            print(f'Обработка BVH {file_idx + 1}/{len(self.list_names)}: {num}', flush=True)
            root = bvhio.readAsHierarchy(str(self.data_dir / f'{num}.bvh'))
            joints = {j.Name: j for j, _, _ in root.layout() if j.Name in self.main_joints}
            hips = joints["Hips"]

            prev = {}
            _, max_frame = root.getKeyframeRange()
            frame_count = max_frame + 1
            # Один компактный массив вместо списка словарей для всех кадров.
            values = np.empty((frame_count, 2 + 12 * len(self.main_joints)), dtype=np.float32)
            columns = None

            for frame in range(frame_count):
                root.loadPose(frame, recursive=True)
                inv_hips_rot = glm.inverse(hips.Rotation)

                row = {"file": file_idx, "frame": frame}

                for name in self.main_joints:
                    joint = joints[name]

                    rel_pos = joint.PositionWorld - hips.PositionWorld
                    rel_up  = inv_hips_rot * joint.UpWorld

                    cur = (rel_pos.x, rel_pos.y, rel_pos.z,
                        rel_up.x,  rel_up.y,  rel_up.z)

                    row[f"{name}_px"] = cur[0]
                    row[f"{name}_py"] = cur[1]
                    row[f"{name}_pz"] = cur[2]
                    row[f"{name}_ux"] = cur[3]
                    row[f"{name}_uy"] = cur[4]
                    row[f"{name}_uz"] = cur[5]

                    if frame == 0:
                        row[f"d_{name}_px"] = 0.0
                        row[f"d_{name}_py"] = 0.0
                        row[f"d_{name}_pz"] = 0.0
                        row[f"d_{name}_ux"] = 0.0
                        row[f"d_{name}_uy"] = 0.0
                        row[f"d_{name}_uz"] = 0.0
                    else:
                        p = prev[name]
                        row[f"d_{name}_px"] = cur[0] - p[0]
                        row[f"d_{name}_py"] = cur[1] - p[1]
                        row[f"d_{name}_pz"] = cur[2] - p[2]
                        row[f"d_{name}_ux"] = cur[3] - p[3]
                        row[f"d_{name}_uy"] = cur[4] - p[4]
                        row[f"d_{name}_uz"] = cur[5] - p[5]

                    prev[name] = cur

                if columns is None:
                    columns = list(row)
                values[frame] = list(row.values())

            frame_df = pd.DataFrame(values, columns=columns)
            frame_df[['file', 'frame']] = frame_df[['file', 'frame']].astype(np.int64)
            self.csv_dataset_joint.append(frame_df)
            del root, joints, hips, joint, values, frame_df
            gc.collect()


    def __calc_max_target_slices(self):
        min_len = float('inf') 
        
        for num in self.list_names:
            path = self.data_dir / f'{num}.txt'
            valid_lines = 0
            
            with path.open(encoding='utf-8') as stream:
                for i, line in enumerate(stream):
                    if i < 2:
                        continue
                    if len(line.split()) == 3:
                        valid_lines += 1

            if valid_lines < min_len:
                min_len = valid_lines

        return int(min_len) if min_len != float('inf') else 0

    def __read_annotations(self, num):
        path = self.data_dir / f'{num}.txt'
        rows = []
        with path.open(encoding='utf-8') as stream:
            i = 0
            for line in stream:
                if self.max_target_slices is not None and i >= self.max_target_slices:
                    break
                
                parts = line.split()
                if len(parts) != 3:
                    continue
                start, end = int(parts[0]), int(parts[1])
                command = parts[2]
                
                if self.label_map is not None and command not in self.label_map:
                    raise ValueError(f'{path}: неизвестный класс {command!r}. Передайте нужный label_map.')
                if not 0 <= start <= end:
                    raise ValueError(f'{path}: некорректный интервал {start}–{end}')
                
                rows.append({'command': command, 'frame_start': parts[0], 'frame_end': parts[1]})
                i += 1
                
        if not rows:
            raise ValueError(f'В файле {path} нет интервалов разметки')
        return rows

    # Мы обрезаем кадры когда оператор влючает запись сессии и выключает формируем датасеты команд и их кадры начало и конца
    def __build_command_datasets(self):
        self.csv_dataset_labels = []

        for idx, num in enumerate(self.list_names):
            rows = self._annotations[num] if hasattr(self, '_annotations') else self.__read_annotations(num)
            self.csv_dataset_labels.append(pd.DataFrame(rows))

            frame_start = min(int(row['frame_start']) for row in rows)
            frame_end = max(int(row['frame_end']) for row in rows)

            print(frame_start, rows[0])
            print(frame_end, rows[len(rows) - 1])

            self.csv_dataset_joint[idx] = self.csv_dataset_joint[idx][(self.csv_dataset_joint[idx]['frame'] >= frame_start)]
            self.csv_dataset_joint[idx] = self.csv_dataset_joint[idx][(self.csv_dataset_joint[idx]['frame'] <= frame_end)]


    # Разбиваю на окна по 1 секуде (90 кадров) с шагом в 0.1 секуда. Размечаю окна (пока берём тупо по большенству не известно на сколько это коректно)
    def __build_windows(self):
        for idx, dataset in enumerate(self.csv_dataset_joint):
            for name in self.main_joints:
                cols = [f"d_{name}_px", f"d_{name}_py", f"d_{name}_pz",
                        f"d_{name}_ux", f"d_{name}_uy", f"d_{name}_uz"]
                dataset[f"energy_{name}"] = (dataset[cols] ** 2).sum(axis=1)

            for side in ("Left", "Right"):
                p_arm  = dataset[[f'{side}Arm_px',     f'{side}Arm_py',     f'{side}Arm_pz']].to_numpy()
                p_fore = dataset[[f'{side}ForeArm_px', f'{side}ForeArm_py', f'{side}ForeArm_pz']].to_numpy()
                p_hand = dataset[[f'{side}Hand_px',    f'{side}Hand_py',    f'{side}Hand_pz']].to_numpy()

                upper = p_fore - p_arm
                lower = p_hand - p_fore
                u = upper / (np.linalg.norm(upper, axis=1, keepdims=True) + 1e-8)
                v = lower / (np.linalg.norm(lower, axis=1, keepdims=True) + 1e-8)
                dataset[f'elbow_angle_{side}'] = np.arccos(np.clip((u * v).sum(axis=1), -1.0, 1.0))

            feature_cols = [
                c for c in dataset.columns
                if c != "frame" and c != "file" and not c.startswith("energy_")
            ]
            energy_cols  = [f"energy_{name}" for name in self.main_joints]

            frames = dataset["frame"].to_numpy()
            frame_labels = np.full(len(frames), "DISTRACTOR", dtype=object)
            for _, lr in self.csv_dataset_labels[idx].iterrows():
                mask = (frames >= int(lr["frame_start"])) & (frames <= int(lr["frame_end"]))
                frame_labels[mask] = lr["command"]

            feature_values = dataset[feature_cols].to_numpy()
            energy_values = dataset[energy_cols].to_numpy()
            rows = []
            for start in range(0, len(dataset) - FPS + 1, STEP):
                stop = start + FPS
                flat = feature_values[start:stop].flatten()
                energy = energy_values[start:stop]
                e_joints = energy.mean(axis=0)
                e_total = energy.sum(axis=1).mean()

                wl = frame_labels[start : start + FPS]
                vals, counts = np.unique(wl, return_counts=True)
                top = counts.max()
                winners = vals[counts == top]
                target = winners[0] if len(winners) == 1 else (wl[-1] if wl[-1] in winners else winners[0])

                row = {
                    "record_id": self.list_names[idx],
                    "window_start_frame": int(frames[start]),
                    "window_end_frame": int(frames[stop - 1]),
                    "features": np.concatenate([flat, e_joints, [e_total]]).astype(np.float32),
                    "target": target,
                }
                rows.append(row)

            if not rows:
                raise ValueError(f'Запись {self.list_names[idx]} короче одного окна')
            self.csv_dataset_joint[idx] = pd.DataFrame(rows)

    # Формируем обучение, настройку и финальный тест
    def __build_test_train_conf(self):
        def to_xy(df):
            """DataFrame -> (X, y) с числовыми метками"""
            X = np.vstack(df["features"].to_numpy())
            y = df["target"].map(self.label_map).to_numpy()
            return X, y

        num_train = self.train_val_test_div[0]
        num_val = self.train_val_test_div[1]
        num_test = self.train_val_test_div[2]

        # --- TRAIN ---
        X_train, y_train, train_meta = [], [], []
        for i in range(num_train):
            X, y = to_xy(self.csv_dataset_joint[i])
            X_train.append(X)
            y_train.append(y)
            train_meta.append(self.csv_dataset_joint[i][["record_id", "window_start_frame", "window_end_frame"]])
        X_train = np.vstack(X_train)
        del X
        y_train = np.concatenate(y_train)
        train_meta = pd.concat(train_meta, ignore_index=True)

        print(f'PCA на train: {X_train.shape}, float32', flush=True)
        scaler = StandardScaler(copy=False).fit(X_train)
        scaler.transform(X_train)
        pca = PCA(n_components=16, svd_solver='randomized', random_state=RANDOM_STATE)
        X_train = pca.fit_transform(X_train)
        
        for i in range(num_train):
            self.csv_dataset_joint[i] = self.csv_dataset_joint[i].drop(columns='features')
        gc.collect()

        # --- VALIDATION (CONF) ---
        X_conf, y_conf, conf_meta = [], [], []
        # Проходим ровно num_val файлов, начиная с конца train
        for i in range(num_train, num_train + num_val):
            X, y = to_xy(self.csv_dataset_joint[i])
            X_conf.append(pca.transform(scaler.transform(X)))
            self.csv_dataset_joint[i] = self.csv_dataset_joint[i].drop(columns='features')
            y_conf.append(y)
            conf_meta.append(self.csv_dataset_joint[i][["record_id", "window_start_frame", "window_end_frame"]])
        
        # Склеиваем (на случай, если val файлов больше 1)
        if X_conf:
            X_conf = np.vstack(X_conf)
            y_conf = np.concatenate(y_conf)
            conf_meta = pd.concat(conf_meta, ignore_index=True)

        # --- TEST ---
        X_test, y_test, test_meta = [], [], []
        # Начинаем ровно после train + val
        for i in range(num_train + num_val, len(self.list_names)):
            X, y = to_xy(self.csv_dataset_joint[i])
            X_test.append(pca.transform(scaler.transform(X)))
            del X
            self.csv_dataset_joint[i] = self.csv_dataset_joint[i].drop(columns='features')
            y_test.append(y)
            test_meta.append(self.csv_dataset_joint[i][["record_id", "window_start_frame", "window_end_frame"]])
        
        if X_test:
            X_test = np.vstack(X_test)
            y_test = np.concatenate(y_test)
            test_meta = pd.concat(test_meta, ignore_index=True)

        def to_df(X, y, meta):
            df = meta.reset_index(drop=True).copy()
            df = pd.concat([df, pd.DataFrame(X, columns=[f"pc{i+1}" for i in range(X.shape[1])])], axis=1)
            df["target"] = y
            return df

        self.train_df = to_df(X_train, y_train, train_meta)
        self.conf_df  = to_df(X_conf,  y_conf, conf_meta)
        self.test_df  = to_df(X_test,  y_test, test_meta)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_df.to_csv(self.output_dir / 'train.csv', index=False)
        self.conf_df.to_csv(self.output_dir / 'conf.csv', index=False)
        self.test_df.to_csv(self.output_dir / 'test.csv', index=False)
        with (self.output_dir / 'label_map.json').open('w', encoding='utf-8') as stream:
            json.dump({name: int(value) for name, value in self.label_map.items()},
                      stream, ensure_ascii=False, indent=2)

    def get_datasets(self):
        return self.train_df, self.conf_df, self.test_df
