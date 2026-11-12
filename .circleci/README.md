# PyPI releases

The `check-pr-version` job runs for pull request events targeting `master`. It compares the final `ROBERT_VERSION` in `robert/_version.py` with the current version on `master` and fails unless the pull request raises it. Intermediate version changes in the pull request do not matter. This job does not use the PyPI token.

The `publish-to-pypi` job runs only for `master`, after the core and GUI test jobs succeed. It checks PyPI and publishes a source distribution and wheel only when the final version is newer than every published `robert` release. An already published or older version fails the job.

Before enabling these jobs:

1. In CircleCI **Project Settings → Project Setup**, use a GitHub App trigger that runs for pull requests opened or pushed to and for default-branch pushes. The PR version job needs a pull request event; a push-only trigger cannot run it.
2. After the first PR build reports `check-pr-version`, make that status check required for merges into `master` in GitHub branch protection. Require PR branches to be up to date before merging so the check runs again if `master` changes. Without the required status check, a failing PR check does not block the merge.
3. In the existing `robert` project on PyPI, create an API token scoped to that project.
4. In CircleCI **Organization Settings → Contexts**, create a context named `pypi-publish`. Add the token, including its `pypi-` prefix, as `TWINE_PASSWORD`.
5. Restrict the context to this CircleCI project. Add the expression restriction `pipeline.git.branch == "master" and not job.ssh.enabled` so other branches and SSH reruns cannot access the token.

To release, change `ROBERT_VERSION` in `robert/_version.py` to a version higher than the one on `master` and merge the PR after its required check passes. The resulting `master` pipeline publishes the final version if it is also newer than the versions on PyPI. If PyPI cannot be queried, the release job fails without uploading. If the token or context is missing, the release job cannot publish.
## Tests and coverage

The active Linux workflow has two test jobs. `build-and-test-core` runs the scientific workflows on three parallel nodes. `build-and-test-gui` runs fast tests, the classic GUI and bot suites, the remaining `tests/test_*.py` files, all `robert/gui_easyrob/test_*.py` files, and a separate Chromium smoke process for the local web dashboard. The GUI job uses Xvfb and checks that Qt and WeasyPrint can render before the tests start.

`coverage-gate` waits for both test jobs, combines their relative-path coverage files, and requires at least 80% line coverage of ROBERT's Python code. It also reports coverage for `robert/gui_easyrob` separately. JavaScript is exercised by Chromium smoke tests, but its lines are not included in Python coverage. The PyPI publish job still runs only on `master`, after both test jobs and the coverage gate pass.
