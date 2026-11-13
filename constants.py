from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
BVH_DATA_DIR = PROJECT_ROOT / 'dataset' / 'dataset_my'
SPLIT_DATA_DIR = PROJECT_ROOT / 'dataset'
RESULTS_DIR = PROJECT_ROOT / 'results_my'

RECORD_IDS = ('rec_2_9_chr01_MAYA', 'rec_2_10_chr01_MAYA', 'rec_3_10_chr01_MAYA',
              'rec_3_5_chr01_MAYA', 'rec_2_15_chr01_MAYA', 'rec_3_6_chr01_MAYA')
TRAIN_RECORD_IDS = RECORD_IDS[:3]
CONF_RECORD_IDS = RECORD_IDS[3:4]
TEST_RECORD_IDS = RECORD_IDS[4:]

LABEL_MAP = {
    "DISTRACTOR":     0,
    "ACTIVE_NEUTRAL": 1,
    "ACTIVE_FORWARD": 2,
    "ACTIVE_LEFT":    3,
    "ACTIVE_RIGHT":   4,
}

FPS = 90
STEP = int(FPS / 10)

DISTRACTOR_ID = LABEL_MAP["DISTRACTOR"]

COMMANDS = [DISTRACTOR_ID, LABEL_MAP["ACTIVE_NEUTRAL"],
            LABEL_MAP["ACTIVE_FORWARD"],
            LABEL_MAP["ACTIVE_LEFT"],
            LABEL_MAP["ACTIVE_RIGHT"]]

RANDOM_STATE = 42
WINDOW_SECONDS = 1.0
WINDOW_STEP_SECONDS = STEP / FPS

N_ACTION_PROTOTYPES = len(COMMANDS)
N_SUBACTION_PROTOTYPES = 2 * N_ACTION_PROTOTYPES

NEUTRAL_ID = LABEL_MAP["ACTIVE_NEUTRAL"]
ACTIVE_COMMAND_IDS = (
    LABEL_MAP["ACTIVE_FORWARD"],
    LABEL_MAP["ACTIVE_LEFT"],
    LABEL_MAP["ACTIVE_RIGHT"],
)


MAJORITY_WINDOWS = (1, 2, 3, 4, 5)
CONFIRMATION_WINDOWS = (1, 2, 3, 4, 5)
# Относительное снижение F1: 0.05 означает не более 5% от лучшего на conf.
M4_F1_TOLERANCE = 0.05
# reliability: правило сравнения с M1–M3 и относительным допуском.
# f1: стандартный выбор по максимуму F1, затем ложным активациям и задержкам.
M4_SELECTION_MODE = "f1"
CONFIDENCE_THRESHOLDS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


EVENT_IOU_THRESHOLD = 0.50
# Event-F1 показывается при трёх уровнях строгости сопоставления сегментов.
# Основной порог 0.50 остаётся базой для precision/recall, задержек и подбора
# параметров автоматов.
EVENT_IOU_THRESHOLDS = (0.10, 0.25, 0.50)
