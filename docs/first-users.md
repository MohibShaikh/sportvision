# First-user trial

Initial target: developers reviewing short, steady basketball practice clips with distinct jersey colors. The guided app does not establish model accuracy; use this trial to discover whether it saves review time.

Ask five volunteers to install from the release or a specific source commit, process their own clip, check team assignments, and download a report. Avoid guiding them through each screen: record where they get stuck. No volunteers have been recruited as part of this implementation.

Record locally, with their consent:

| Participant | Installed unaided? | Minutes to first output | Team grouping useful? | Main failure | Returned with second clip? |
| --- | --- | --- | --- | --- | --- |
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

Proposed decision: improve installation if fewer than four finish unaided. Improve the core review task before adding sports if fewer than three return with another clip. These are experiment goals, not predictions or industry benchmarks.

For accuracy, create a consented fixed-camera clip set and manually label sampled frames: visible balls, player boxes, real team, identity across frames, and possession where unambiguous. Keep these labels independent of detector output. Report ball recall, team accuracy on matched players, identity changes, attributed-frame coverage, and possession agreement. Record sport, camera, resolution, model, and hardware. Compare chosen colors against automatic grouping on identical frames.

The cached basketball smoke clip is only a runtime check; the current tests use synthetic reference colors for team-order correctness. Neither is an accuracy result. Do not label the project validated until the independent annotations and results exist.

Before publication: choose a new package version, build and install its wheel in a fresh environment, check the community activation aliases, publish the package, and replace the README's local-only install notice. The community listing already exists: https://github.com/roboflow/inference/pull/2163. The new browser app is not published yet.
