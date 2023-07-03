import json
import os

from shared import ROOT_DIR

DECEPTION_RELEVANT_PERSONAS = ["Machiavellianism", "No Shut Down"]


def open_personas():
    files = os.listdir(os.path.join(os.path.dirname(__file__), "persona_data"))
    all_personas_data = {}
    for file in files:
        with open(ROOT_DIR / "persona_data" / file) as f:
            all_personas_data[file] = [
                json.loads(content) for content in f.read().splitlines()
            ]

    return all_personas_data
