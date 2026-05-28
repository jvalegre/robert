# json_schema_notes.md

This file records working ideas for ROBERT JSON outputs.

The schema is experimental until confirmed through implementation and testing.

---

## Design Goals

The JSON files should be:

- easy for humans to read,
- easy for a UI to load,
- easy for an LLM to explain,
- traceable back to ROBERT outputs,
- stable enough to use across runs,
- explicit about what is known and unknown.

---

## Basic Module Summary Schema

```json
{
  "schema_version": "0.1",
  "module": "CURATE",
  "artifact_type": "module_summary",
  "status": "completed",
  "inputs": {},
  "outputs": {
    "files_created": []
  },
  "parameters": {},
  "direct_robert_values": {},
  "derived_helper_values": {},
  "warnings": [],
  "notes": []
}
```

---

## Basic Run Context Schema

```json
{
  "schema_version": "0.1",
  "artifact_type": "run_context",
  "robert_version": null,
  "run_id": null,
  "created_at": null,
  "command": null,
  "modules": {
    "curate": {},
    "generate": {},
    "verify": {},
    "predict": {},
    "report": {}
  },
  "score": {},
  "warnings": [],
  "file_manifest": []
}
```

---

## Basic File Manifest Schema

```json
{
  "schema_version": "0.1",
  "artifact_type": "file_manifest",
  "run_id": null,
  "created_at": null,
  "files": [
    {
      "path": "PREDICT/PREDICT_data.dat",
      "name": "PREDICT_data.dat",
      "extension": ".dat",
      "module": "PREDICT",
      "size_bytes": null,
      "modified_at": null,
      "description": "ROBERT prediction summary data"
    }
  ]
}
```

---

## Open Schema Questions

- Should each module write its own JSON file?
- Should REPORT assemble the full `run_context.json`?
- Should JSON export be controlled by a command-line option?
- Should JSON files include full values or only summaries?
- Should images be listed only, or should image metadata be captured?
- Should `.dat` file contents be copied into JSON or only referenced?
- Should file timestamps use local time or UTC?
- Should output folders be timestamped by run?
