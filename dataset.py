import bvhio
import pandas as pd
import glm
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from constants import LABEL_MAP, FPS, STEP, MAIN_JOINTS

class BvhDataset:
    # Загружаем датасет, определяем основные суставы делаем csv
    def __init__(self, list_names : list = ['2_9', '2_10', '2_12', '2_14', '2_15', '2_16']):
        #assert len(list_names) == 6, f'Ожидается 6 записей, получено {len(list_names)}'
        self.dataset = []
        self.list_names = list_names
        for num in list_names:
            self.dataset.append(bvhio.readAsHierarchy(f'dataset/rec_{num}_chr01_MAYA.bvh'))

        # начинаем разметку и построения датасетов
        self.__build_movement_speed()
        self.__build_command_datasets()
        self.__build_windows()
        self.__build_test_train_conf()


    def __build_movement_speed(self):
        self.csv_dataset_joint = []

        for file_idx, root in enumerate(self.dataset):
            joints = {j.Name: j for j, _, _ in root.layout() if j.Name in MAIN_JOINTS}
            hips = joints["Hips"]

            prev = {}
            rows = []
            _, max_frame = root.getKeyframeRange()
            frame_count = max_frame + 1

            for frame in range(frame_count):
                root.loadPose(frame, recursive=True)
                inv_hips_rot = glm.inverse(hips.Rotation)

                row = {"file": file_idx, "frame": frame}

                for name in MAIN_JOINTS:
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

                rows.append(row)

            self.csv_dataset_joint.append(pd.DataFrame(rows))


    # Мы обрезаем кадры когда оператор влючает запись сессии и выключает формируем датасеты команд и их кадры начало и конца
    def __build_command_datasets(self):
        self.csv_dataset_labels = []

        for idx, num in enumerate(self.list_names):
            name = f'dataset/rec_{num}_chr01_MAYA.txt'

            with open(name, 'r', encoding='utf-8') as f:
                rows = []

                for line in f.readlines():
                    if len(line.split()) == 3:
                        l = line.split()
                        row = {"command": l[2], "frame_start": l[0], "frame_end": l[1]}
                        rows.append(row)

                self.csv_dataset_labels.append(pd.DataFrame(rows))

                frame_start = int(rows[0]["frame_start"])
                frame_end = int(rows[len(rows) - 1]["frame_end"])

                print(frame_start, rows[0])
                print(frame_end, rows[len(rows) - 1])

                self.csv_dataset_joint[idx] = self.csv_dataset_joint[idx][(self.csv_dataset_joint[idx]['frame'] >= frame_start)]
                self.csv_dataset_joint[idx] = self.csv_dataset_joint[idx][(self.csv_dataset_joint[idx]['frame'] <= frame_end)]


    # Разбиваю на окна по 1 секуде (90 кадров) с шагом в 0.1 секуда. Размечаю окна (пока берём тупо по большенству не известно на сколько это коректно)
    def __build_windows(self):
        for idx, dataset in enumerate(self.csv_dataset_joint):
            for name in MAIN_JOINTS:
                cols = [f"d_{name}_px", f"d_{name}_py", f"d_{name}_pz",
                        f"d_{name}_ux", f"d_{name}_uy", f"d_{name}_uz"]
                dataset[f"energy_{name}"] = (dataset[cols] ** 2).sum(axis=1)

            feature_cols = [
                c for c in dataset.columns
                if c != "frame" and c != "file" and not c.startswith("energy_")
            ]
            energy_cols  = [f"energy_{name}" for name in MAIN_JOINTS]

            frames = dataset["frame"].to_numpy()
            frame_labels = np.full(len(frames), "DISTRACTOR", dtype=object)
            for _, lr in self.csv_dataset_labels[idx].iterrows():
                mask = (frames >= int(lr["frame_start"])) & (frames <= int(lr["frame_end"]))
                frame_labels[mask] = lr["command"]

            rows = []
            for start in range(0, len(dataset) - FPS + 1, STEP):
                win = dataset.iloc[start : start + FPS]

                flat = win[feature_cols].to_numpy().flatten()

                e_joints = win[energy_cols].to_numpy().mean(axis=0)

                e_total = win[energy_cols].to_numpy().sum(axis=1).mean()

                wl = frame_labels[start : start + FPS]
                vals, counts = np.unique(wl, return_counts=True)
                top = counts.max()
                winners = vals[counts == top]
                target = winners[0] if len(winners) == 1 else (wl[-1] if wl[-1] in winners else winners[0])

                row = {
                    "record_id": self.list_names[idx],
                    "window_start_frame": int(win["frame"].iloc[0]),
                    "window_end_frame": int(win["frame"].iloc[-1]),
                    "features": np.concatenate([flat, e_joints, [e_total]]),
                    "target": target,
                }
                rows.append(row)

            self.csv_dataset_joint[idx] = pd.DataFrame(rows)

    # Формируем обучение, настройку и финальный тест
    def __build_test_train_conf(self):
        def to_xy(df):
            """DataFrame -> (X, y) с числовыми метками"""
            X = np.vstack(df["features"].to_numpy())
            y = df["target"].map(LABEL_MAP).to_numpy()
            return X, y


        X_train, y_train, train_meta = [], [], []
        for i in [0, 1, 2]:
            X, y = to_xy(self.csv_dataset_joint[i])
            X_train.append(X)
            y_train.append(y)
            train_meta.append(self.csv_dataset_joint[i][["record_id", "window_start_frame", "window_end_frame"]])
        X_train = np.vstack(X_train)
        y_train = np.concatenate(y_train)
        train_meta = pd.concat(train_meta, ignore_index=True)

        X_conf, y_conf = to_xy(self.csv_dataset_joint[3])
        conf_meta = self.csv_dataset_joint[3][["record_id", "window_start_frame", "window_end_frame"]].reset_index(drop=True)

        X_test, y_test, test_meta = [], [], []
        for i in range(4, len(self.list_names)): # [4, 5]:
            X, y = to_xy(self.csv_dataset_joint[i])
            X_test.append(X)
            y_test.append(y)
            test_meta.append(self.csv_dataset_joint[i][["record_id", "window_start_frame", "window_end_frame"]])
        X_test = np.vstack(X_test)
        y_test = np.concatenate(y_test)
        test_meta = pd.concat(test_meta, ignore_index=True)

        scaler = StandardScaler().fit(X_train)
        pca = PCA(n_components=16).fit(scaler.transform(X_train))

        X_train = pca.transform(scaler.transform(X_train))
        X_conf = pca.transform(scaler.transform(X_conf))
        X_test = pca.transform(scaler.transform(X_test))

        def to_df(X, y, meta):
            df = meta.reset_index(drop=True).copy()
            df = pd.concat([df, pd.DataFrame(X, columns=[f"pc{i+1}" for i in range(X.shape[1])])], axis=1)
            df["target"] = y
            return df

        self.train_df = to_df(X_train, y_train, train_meta)
        self.conf_df  = to_df(X_conf,  y_conf, conf_meta)
        self.test_df  = to_df(X_test,  y_test, test_meta)

        self.train_df.to_csv("dataset/train.csv", index=False)
        self.conf_df.to_csv("dataset/conf.csv",  index=False)
        self.test_df.to_csv("dataset/test.csv",  index=False)

    def get_datasets(self):
        return self.train_df, self.conf_df, self.test_df