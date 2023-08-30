import configparser
# import json
import os
from types import SimpleNamespace

from ia_ui_items import get_inp_model_ids, get_sam_model_ids


class IAConfig:
    SECTIONS = SimpleNamespace(
        DEFAULT=configparser.DEFAULTSECT,
        USER="USER",
    )

    KEYS = SimpleNamespace(
        SAM_MODEL_ID="sam_model_id",
        INP_MODEL_ID="inp_model_id",
    )

    PATHS = SimpleNamespace(
        INI=os.path.join(os.path.dirname(os.path.realpath(__file__)), "ia_config.ini"),
    )

    global_args = {}

    def __init__(self):
        self.ids_dict = {}
        self.ids_dict[IAConfig.KEYS.SAM_MODEL_ID] = {
            "list": get_sam_model_ids(),
            "index": 1,
        }
        self.ids_dict[IAConfig.KEYS.INP_MODEL_ID] = {
            "list": get_inp_model_ids(),
            "index": 0,
        }


ia_config = IAConfig()


def setup_ia_config_ini():
    ia_config_ini = configparser.ConfigParser(defaults={})
    if os.path.isfile(IAConfig.PATHS.INI):
        ia_config_ini.read(IAConfig.PATHS.INI, encoding="utf-8")

    changed = False
    for key, ids_info in ia_config.ids_dict.items():
        if not ia_config_ini.has_option(IAConfig.SECTIONS.DEFAULT, key):
            if len(ids_info["list"]) > ids_info["index"]:
                ia_config_ini[IAConfig.SECTIONS.DEFAULT][key] = ids_info["list"][ids_info["index"]]
                changed = True
        else:
            if len(ids_info["list"]) > ids_info["index"] and ia_config_ini[IAConfig.SECTIONS.DEFAULT][key] != ids_info["list"][ids_info["index"]]:
                ia_config_ini[IAConfig.SECTIONS.DEFAULT][key] = ids_info["list"][ids_info["index"]]
                changed = True

    if changed:
        with open(IAConfig.PATHS.INI, "w", encoding="utf-8") as f:
            ia_config_ini.write(f)


def get_ia_config(key, section=IAConfig.SECTIONS.DEFAULT):
    setup_ia_config_ini()

    ia_config_ini = configparser.ConfigParser(defaults={})
    ia_config_ini.read(IAConfig.PATHS.INI, encoding="utf-8")

    if ia_config_ini.has_option(section, key):
        return ia_config_ini[section][key]

    section = IAConfig.SECTIONS.DEFAULT
    if ia_config_ini.has_option(section, key):
        return ia_config_ini[section][key]

    return None


def get_ia_config_index(key, section=IAConfig.SECTIONS.DEFAULT):
    value = get_ia_config(key, section)

    ids_dict = ia_config.ids_dict
    if value is None:
        if key in ids_dict.keys():
            ids_info = ids_dict[key]
            return ids_info["index"]
        else:
            return 0
    else:
        if key in ids_dict.keys():
            ids_info = ids_dict[key]
            return ids_info["list"].index(value) if value in ids_info["list"] else ids_info["index"]
        else:
            return 0


def set_ia_config(key, value, section=IAConfig.SECTIONS.DEFAULT):
    setup_ia_config_ini()

    ia_config_ini = configparser.ConfigParser(defaults={})
    ia_config_ini.read(IAConfig.PATHS.INI, encoding="utf-8")

    if ia_config_ini.has_option(section, key) and ia_config_ini[section][key] == value:
        return

    if section != IAConfig.SECTIONS.DEFAULT and not ia_config_ini.has_section(section):
        ia_config_ini[section] = {}

    try:
        ia_config_ini[section][key] = value
    except Exception:
        ia_config_ini[section] = {}
        ia_config_ini[section][key] = value

    with open(IAConfig.PATHS.INI, "w", encoding="utf-8") as f:
        ia_config_ini.write(f)
