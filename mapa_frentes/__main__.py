"""Entry point: python -m mapa_frentes"""

import sys
import logging

from PyQt5.QtWidgets import QApplication

from mapa_frentes.config import load_config
from mapa_frentes.gui.main_window import MainWindow
from mapa_frentes.gui.front_editor import FrontEditor


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_config()

    app = QApplication(sys.argv)
    app.setApplicationName("MAPA FRENTES")
    app.setOrganizationName("MeteoGalicia")

    window = MainWindow(cfg)

    # Crear editor de frentes y conectarlo
    editor = FrontEditor(window)
    window.set_editor(editor)

    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
