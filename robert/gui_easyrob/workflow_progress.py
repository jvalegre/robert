"""Track AQME preparation and the visible milestones of a ROBERT run."""

import re


STAGES = ("CURATE", "GENERATE", "VERIFY", "PREDICT", "REPORT")
_TIME_MARKER = re.compile(r"\bTime (CURATE|GENERATE|VERIFY|PREDICT):\s*\d", re.IGNORECASE)


class WorkflowProgress:
    """Convert confirmed process output into stage states for the GUI."""

    def __init__(self, workflow, include_aqme=False):
        self.include_aqme = include_aqme or workflow == "AQME"
        self.selected = () if workflow == "AQME" else STAGES if workflow == "Full Workflow" else (workflow,)
        self.completed = set()
        self.failed = False
        self.stopped = False
        self.verify_time_seen = False
        self.predict_time_seen = False

    def observe(self, line):
        """Record a stage only when its completion marker appears in output."""
        if self.include_aqme and (
            re.search(r"\bTime AQME:\s*\d", line, re.IGNORECASE)
            or "Starting data curation with the CURATE module" in line
        ):
            self.complete_aqme()
        match = _TIME_MARKER.search(line)
        if match and match.group(1).upper() in self.selected:
            stage = match.group(1).upper()
            if stage == "VERIFY":
                self.verify_time_seen = True
            elif stage == "PREDICT":
                self.predict_time_seen = True
            else:
                self.completed.add(stage)
        if self.selected == STAGES and "with the PREDICT module" in line:
            self.completed.add("VERIFY")
        if self.selected == STAGES and "Starting REPORT module" in line:
            self.completed.add("PREDICT")

    def complete_aqme(self):
        """Confirm that descriptor generation finished before advancing."""
        if self.include_aqme:
            self.completed.add("AQME")

    def finish(self, exit_code, report_created=False):
        """Confirm report creation and mark an incomplete run as failed."""
        self.stopped = exit_code == -1
        if self.selected == ("VERIFY",) and exit_code == 0 and self.verify_time_seen:
            self.completed.add("VERIFY")
        if self.selected == ("PREDICT",) and exit_code == 0 and self.predict_time_seen:
            self.completed.add("PREDICT")
        if "REPORT" in self.selected and exit_code == 0 and report_created:
            self.completed.add("REPORT")
        self.failed = (
            exit_code != 0
            or (self.include_aqme and "AQME" not in self.completed)
            or any(stage not in self.completed for stage in self.selected)
        )

    def aqme_status(self):
        """Return the optional preparation stage independently of ROBERT."""
        if not self.include_aqme:
            return "inactive"
        if "AQME" in self.completed:
            return "done"
        if self.stopped:
            return "stopped"
        return "failed" if self.failed else "active"

    def statuses(self):
        """Return stage states in display order."""
        current = next((stage for stage in self.selected if stage not in self.completed), None)
        waiting_for_aqme = self.include_aqme and "AQME" not in self.completed
        states = []
        for stage in STAGES:
            if stage not in self.selected:
                states.append("inactive")
            elif stage in self.completed:
                states.append("done")
            elif waiting_for_aqme:
                states.append("pending")
            elif stage == current:
                states.append("stopped" if self.stopped else "failed" if self.failed else "active")
            else:
                states.append("pending")
        return tuple(states)
