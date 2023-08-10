import configparser
# import json
import os
from types import SimpleNamespace

from ia_ui_items import get_inp_model_ids, get_sam_model_ids


class IAConfig:
    SECTIONS = SimpleNamespace(
        DEFAULT="DEFAULT",
        USER="USER",
    )

    KEYS = SimpleNamespace(
        SAM_MODEL_ID="sam_model_id",
        INP_MODEL_ID="inp_model_id",
    )

    WEBUI_KEYS = SimpleNamespace(
        SAM_MODEL_ID="inpaint_anything/Segment Anything Model ID/value",
        INP_MODEL_ID="inpaint_anything/Inpainting Model ID/value",
    )

    PATHS = SimpleNamespace(
        INI=os.path.join(os.path.dirname(os.path.realpath(__file__)), "ia_config.ini"),
    )


def setup_ia_config_ini():
    if not os.path.isfile(IAConfig.PATHS.INI):
        ia_config_ini = configparser.ConfigParser()

        sam_model_ids = get_sam_model_ids()
        sam_model_index = 1
        inp_model_ids = get_inp_model_ids()
        inp_model_index = 0

        ia_config_ini[IAConfig.SECTIONS.DEFAULT] = {
            IAConfig.KEYS.SAM_MODEL_ID: sam_model_ids[sam_model_index],
            IAConfig.KEYS.INP_MODEL_ID: inp_model_ids[inp_model_index],
        }
        with open(IAConfig.PATHS.INI, "w", encoding="utf-8") as f:
            ia_config_ini.write(f)


def get_ia_config(key, section=IAConfig.SECTIONS.DEFAULT):
    setup_ia_config_ini()

    ia_config_ini = configparser.ConfigParser()
    ia_config_ini.read(IAConfig.PATHS.INI, encoding="utf-8")

    if ia_config_ini.has_option(section, key):
        return ia_config_ini[section][key]

    section = IAConfig.SECTIONS.DEFAULT
    if ia_config_ini.has_option(section, key):
        return ia_config_ini[section][key]

    return None


def get_ia_config_index(key, section=IAConfig.SECTIONS.DEFAULT):
    value = get_ia_config(key, section)

    if value is None:
        return None

    if key == IAConfig.KEYS.SAM_MODEL_ID:
        sam_model_ids = get_sam_model_ids()
        idx = sam_model_ids.index(value) if value in sam_model_ids else 1
    elif key == IAConfig.KEYS.INP_MODEL_ID:
        inp_model_ids = get_inp_model_ids()
        idx = inp_model_ids.index(value) if value in inp_model_ids else 0
    else:
        idx = None

    return idx


def set_ia_config(key, value, section=IAConfig.SECTIONS.DEFAULT):
    setup_ia_config_ini()

    ia_config_ini = configparser.ConfigParser()
    ia_config_ini.read(IAConfig.PATHS.INI, encoding="utf-8")

    if section != IAConfig.SECTIONS.DEFAULT and not ia_config_ini.has_section(section):
        ia_config_ini[section] = {}
    else:
        if ia_config_ini.has_option(section, key) and ia_config_ini[section][key] == value:
            return

    try:
        ia_config_ini[section][key] = value
    except Exception:
        ia_config_ini[section] = {}
        ia_config_ini[section][key] = value

    with open(IAConfig.PATHS.INI, "w", encoding="utf-8") as f:
        ia_config_ini.write(f)
