"""The GUI exposes ROBERT's all_models option without changing its default."""

def test_all_models_option_is_visible_and_off_by_default(disposed_easyrob_window):
    window = disposed_easyrob_window
    assert not window.all_models_toggle.isChecked()
    assert "full PDF report" in window.all_models_label.text()


def test_all_models_flag_follows_main_toggle_for_every_workflow(disposed_easyrob_window):
    window = disposed_easyrob_window
    for workflow in ("Full Workflow", "GENERATE", "PREDICT", "REPORT"):
        window.workflow_selector.setCurrentText(workflow)
        window.all_models_toggle.setChecked(True)
        window._collect_robert_gui_values()
        command = window.build_robert_command("C:/runs/input.csv")
        assert command.count("--all_models True") == 1
        assert command.count("--all_models") == 1

        window.all_models_toggle.setChecked(False)
        window._collect_robert_gui_values()
        command = window.build_robert_command("C:/runs/input.csv")
        assert command.count("--all_models False") == 1
        assert command.count("--all_models") == 1
