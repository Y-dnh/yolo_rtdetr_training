"""
Обгортка над ruamel.yaml з API, сумісним з PyYAML (safe_load, dump).
Використовується, якщо стандартний 'import yaml' не працює (наприклад, зламаний PyYAML 6 на Windows).
"""
from ruamel.yaml import YAML


def _safe_load(stream):
    """Аналог yaml.safe_load(stream)."""
    y = YAML(typ="safe")
    return y.load(stream)


def _dump(data, stream=None, **kwargs):
    """Аналог yaml.dump(data, stream, ...). stream — файловий об'єкт."""
    y = YAML(typ="safe")
    y.default_flow_style = kwargs.get("default_flow_style", False)
    y.allow_unicode = kwargs.get("allow_unicode", True)
    if stream is not None:
        y.dump(data, stream)
    else:
        from io import StringIO
        buf = StringIO()
        y.dump(data, buf)
        return buf.getvalue()


class _YamlModule:
    safe_load = staticmethod(_safe_load)
    dump = staticmethod(_dump)


yaml = _YamlModule()
