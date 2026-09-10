# SportVision audit — 7 September 2026

SportVision is a useful starting point for developers experimenting with sports footage. It is not yet a validated match-analysis product, and there is no evidence here of high traction. The most valuable next step is to prove one narrow workflow on representative footage with outside users.

## Findings and fixes

| Finding | Impact | Change |
| --- | --- | --- |
| RF-DETR received BGR images and YOLO's COCO IDs | Incorrect colors and missing/misclassified people and balls | Convert to RGB; map pretrained RF-DETR IDs 1/37 separately from YOLO IDs 0/32 |
| Missing RF-DETR silently selected YOLO; inference errors became empty detections | A failed or different model looked like a working analysis | Explicit model selection and actionable exceptions |
| Tracker exceptions produced fresh sequential IDs | Distances and trajectories could appear valid without tracking | Propagate failure; test actual ID continuity |
| Possession started at 50/50 | Invented statistics without an observation | Empty estimate until valid attribution; explicitly label overlay estimates |
| First classified frame assigned everyone to team 0 | Biased first observation | Predict immediately after fitting; use -1 for unknowns |
| Empty player list after fitting crashed | Normal scene transitions could stop processing | Handle empty predictions |
| Pipeline treated every non-player as a ball | Referee/goalkeeper detections could generate possession | Require ball class 1 |
| Skipped frames updated analytics using stale boxes | Repeated observations presented as new evidence | Reuse annotations only; expose fresh_detections |
| Workflow classifier clustered balls and failed on too few players | Contaminated teams or crashed on sparse scenes | Filter players; defer fitting until enough exist |
| Workflow possessing_team meant historical leader | Current possession remained wrong after turnovers or disappearance | Return current attributed team, or -1; preserve historical shares |
| Distance block returned zero without IDs | Missing tracking looked like stationary players | Require persistent IDs |
| Demo counted balls as players and called last-segment speed an average; manual smoke script invented teams | Convincing but unsupported summaries | Shared pipeline runner with explicit method, ball coverage, and track-ID count |
| README claimed real-time operation, minimal quality loss, and automatic analytics beyond the pipeline | Expectations exceeded implementation | Describe actual scope and calibration/measurement requirements; runnable clip example |
| Notebook duplicated the old analytics and retained previous outputs | Inconsistent behavior and stale evidence | Call the shared runner and clear execution outputs |

## Verification

The original suite passed 69 tests despite the defects. The final suite passes 81 tests; Ruff lint, formatting, and git diff whitespace checks pass. Regression coverage now checks backend conventions/failure behavior, missing teams, player disappearance, stale frames, current possession, and absent tracking. The video integration path now uses ByteTrack instead of assigning detection-array indices as identities.

A local YOLOv8n smoke run processed the first 150 frames of the cached basketball clip in 16.34 seconds (9.18 processing FPS, including initialization), with a tracked ball on 17 frames and 5 player track IDs. It returned no possession estimate. Visual inspection of frame 130 also showed both differently dressed players assigned the same team color, so team classification on this clip remains unreliable. This is evidence that the runner works and reports missing evidence honestly; it is not a model-accuracy benchmark. No ground-truth labels were available. The new RF-DETR adapter was tested with controlled model outputs; RF-DETR itself is not installed in this environment. Colab and the live Roboflow execution engine were not exercised.

## Would people use it?

Developers and students are plausible users: a small Python API and reusable Workflow blocks can reduce setup work. Coaches currently have to manage Python, detector selection, camera assumptions, and ambiguous outputs. The repository has no upload/review interface, team correction, match history, or validated events. The present product is unlikely to retain that audience without significant additional work.

There is interest in the category, but substantial overlap: [Roboflow sports](https://github.com/roboflow/sports) exposes reusable sports-vision tools and publicly shows about 5.3k stars and 657 forks. [SportVision's public repository](https://github.com/MohibShaikh/sportvision) shows 2 stars and 0 forks at this audit. These are awareness signals, not active-user or revenue measurements. [Veo's product documentation](https://support.veo.com/hc/en-us/articles/11873714080657-Veo-Analytics-2-Overview-and-features) describes an established coach-oriented analytics workflow. This comparison suggests competing on a broad feature list will be difficult; it does not prove users will reject this project.

A plausible differentiator is an easily installed, inspectable sports-video pipeline with reusable Workflow blocks and clear data-quality reporting. That remains a hypothesis.

## Remaining product work

1. Pick one sport and camera setup. Evaluate labeled clips for ball recall, tracking identity changes, team classification, and possession error; report runtime with hardware and settings.
2. Support manual team initialization/correction and sport-specific ball/player detection. Pixel proximity between box centers is a weak possession model, especially for airborne balls and basketball.
3. Validate Workflow engine loading, state lifetime, and a complete tracking-equipped workflow. Direct Python block tests do not establish deployed engine compatibility.
4. Treat calibration as a separate validated feature: field-plane anchors, camera motion, and coordinate units need explicit handling. Current utility availability is not reliable physical analytics.
5. Recruit five target developers to process their own clips without assistance. Measure successful first runs, time to output, and repeat use. A proposed gate is at least three returning with a second clip before expanding scope; this is a product decision, not an industry benchmark.

Compatibility changes: callers must handle empty possession dictionaries and backend/tracking exceptions. The demo's output JSON is now a smaller, explicit summary rather than unvalidated speed/distance output. Existing user changes to uv.lock were left intact. These edits are local; hosted README, published packages, and the Colab repository checkout will reflect them only after release.

## Follow-up: guided first run

The local checkout now includes an optional Streamlit app (`pip install -e ".[app]"`, then `sportvision app`). It provides video upload, source preview, chosen jersey colors or automatic grouping, bounded review length, live progress, a final frame preview, and annotated-video/report downloads. Changing inputs clears old results; failures are displayed with a recovery path. No hosted app or new PyPI release was published.

The packaged video runner powers the CLI example, notebook path through the example, and browser app. Manual colors use a continuous Lab color comparison rather than exact histogram-bin matches; this is a heuristic that still needs evaluation on independently labeled clips. Output overlays use blue/red for legibility even when jerseys are white/black.

The community listing was confirmed: Roboflow merged https://github.com/roboflow/inference/pull/2163 on March 26, 2026. Both `WORKFLOWS_PLUGINS=sportvision` and `sportvision.workflows` now expose plugin loaders. The app's dependencies are optional; uv.lock was updated for them while retaining the existing project version correction.

See first-users.md for the proposed five-user trial and labeling protocol. No outside users were contacted and no independent accuracy benchmark was claimed.

## Follow-up: evidence review and missed balls — 11 September 2026

The clip reviewer now supports clickable player boxes, reversible team/ignore assignments for confirmed track IDs, frame seeking, and a colored possession-evidence timeline. Rejected observations explain missing/ambiguous balls, unknown teams, distance limits, and unconfirmed tracks. Unconfirmed ID -1 is never a correction target: multiple detections can share it.

A visible ball can still be missed by the detector. Users can mark its position on the selected frame or explicitly confirm that a team has possession. Both edits remain distinct from measured detections and are labeled manual in the report and video. They affect only the selected frame. Manual confirmation does not inflate the original model's ball-detection coverage. Undo restores the original measurements.

Corrections invalidate the previous video download; rebuilding renders cached observations without another detector run. Reports retain measured team IDs, correction records, source SHA-256, detector settings, and per-frame evidence. The evaluator checks source identity, reports sample sizes, and flags corrected results. Three provisional assistant annotations exercise evaluation; they still require independent human review and do not establish accuracy.

Verification: 105 tests pass, including UI correction/undo flows, coordinate validation, rejected unconfirmed track IDs, preserved frame position, timeline boundaries, source matching, and cached video export. Lint/format/whitespace checks pass. The wheel includes the clickable component assets. A real clip was processed through the browser and the new evidence wording was checked. No new package release was published.
