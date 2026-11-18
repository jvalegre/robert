"""Desktop entry point for the CheckML-only GUI."""

try:
    from robert.gui_easyrob.easyrob_launcher import main as launch_gui
except ImportError:
    from easyrob_launcher import main as launch_gui


def main():
    """Open the CheckML desktop workspace."""
    launch_gui(mode="checkml")


if __name__ == "__main__":
    main()
