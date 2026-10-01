"""Pure navigation policy for the consolidated results workspace."""


RESULT_VIEW_ORDER = ("Report", "Predictions", "Images", "Interactive plots")


def choose_available_result_view(availability, current_view, has_selection):
    """Keep a selected available view, or choose the first available view."""
    if has_selection and availability.get(current_view, False):
        return current_view
    return next(
        (view for view in RESULT_VIEW_ORDER if availability.get(view, False)),
        None,
    )
