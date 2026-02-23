from __future__ import annotations

import sys
from pathlib import Path

from PySide6 import QtWidgets

from training.app.main_window import TrainerMainWindow
from training.core import TrainingConfig


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    config = TrainingConfig.default(repo_root=Path(__file__).resolve().parents[2])
    window = TrainerMainWindow(config=config)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

