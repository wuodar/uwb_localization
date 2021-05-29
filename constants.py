import os
from glob import glob

RAW_TRAIN_DATA_PATHS = glob("../pomiary/**/*stat*.xlsx", recursive=True)
RAW_VAL_DATA_PATHS = glob("../pomiary/**/*random*.xlsx", recursive=True)
RAW_TEST_DATA_PATHS = list(
    set(glob("./pomiary/**/*.xlsx", recursive=True))
    - set(RAW_VAL_DATA_PATHS)
    - set(RAW_TRAIN_DATA_PATHS)
)

COLUMNS = [
    "data__coordinates__x",
    "data__coordinates__y",
    "reference__x",
    "reference__y",
]

OUT_PATH = os.path.join("..","experiments")
