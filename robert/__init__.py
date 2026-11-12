__all__ = ("RobertModel",)


def __getattr__(name: str):
    if name == "RobertModel":
        from robert.api import RobertModel

        return RobertModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
