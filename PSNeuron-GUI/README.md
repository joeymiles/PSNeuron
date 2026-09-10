# PSNeuron Educational GUI

WinForms front-end for `PSNeuron.psm1` — XOR and sine one-step-ahead forecast demos, with plain-English teaching tips and a slow-walk of inputs through the network.

## Run

PowerShell 7+ required:

```powershell
pwsh -STA -File .\Start-PSNeuronGUI.ps1
```

From repo root:

```powershell
pwsh -STA -File .\PSNeuron-GUI\Start-PSNeuronGUI.ps1
```

Smoke:

```powershell
pwsh -STA -File .\Start-PSNeuronGUI.ps1 -SmokeTest
```

The launcher imports `..\PSNeuron.psm1` by default.
