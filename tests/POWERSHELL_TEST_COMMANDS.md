# PowerShell Commands for Unit Tests

Use these commands from the repository root in PowerShell.

## 0) One-time setup in your shell session

```powershell
$env:PYTHONPATH = "."
```

> This ensures `app/...` imports resolve during test runs.

## 1) Run all unit tests

```powershell
pytest -q tests/unit
```

## 2) Run tests in a specific path

```powershell
pytest -q tests/unit/test_query_service.py
```

You can also run a whole subfolder, for example:

```powershell
pytest -q tests/unit
```

## 3) Run one specific test

```powershell
pytest -q tests/unit/test_query_service.py::QueryServiceTests::test_query_handles_empty_index
```

## Optional: run all tests in repo (unit + others)

```powershell
pytest -q
```
