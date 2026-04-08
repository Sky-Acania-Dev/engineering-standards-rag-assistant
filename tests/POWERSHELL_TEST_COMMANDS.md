# PowerShell Commands for Unit Tests

Use these commands from the repository root in PowerShell.

## 0) One-time setup in your shell session

```powershell
$env:PYTHONPATH = "."
```

> This ensures `app/...` imports resolve during test runs.

## 1) Run all unit tests

```powershell
py -m unittest discover -s tests/unit -p "test*.py"
```

## 2) Run tests in a specific path (test_query_service here as an example)

```powershell
py -m unittest tests.unit.test_query_service
```

You can also run a whole subfolder, for example:

```powershell
py -m unittest discover -s tests/unit -p "test*.py"
```

## 3) Run one specific test

```powershell
py -m unittest tests.unit.test_query_service.QueryServiceTests.test_query_handles_empty_index
```

## Optional: run all tests in repo (unit + others)

```powershell
py -m unittest discover
```
