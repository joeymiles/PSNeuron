#requires -Version 5.1
<#
.SYNOPSIS
  Educational WinForms GUI for PSNeuron.psm1 (XOR + sine one-step forecast).
.NOTES
  Prefer: pwsh -STA -File Start-PSNeuronGUI.ps1
  Imports ../PSNeuron.psm1 relative to this script; Adapter constructs classes in module scope.
  v1.1: ToolTips, tip line, Help, canvas click/hover/dblclick/right-click, Space/Esc/R.
  v1.2: Epoch narrator, loss->0 explainer, weight-impact tips, Walk-inputs slow-mo, richer hover.
#>
[CmdletBinding()]
param(
    [switch]$SmokeTest
)

# PSNeuron.psm1 uses PS7 ternary (? :); re-launch under pwsh if needed.
$pwsh = Join-Path $env:ProgramFiles 'PowerShell\7\pwsh.exe'
if (-not (Test-Path -LiteralPath $pwsh)) {
    $cmd = Get-Command pwsh -ErrorAction SilentlyContinue
    if ($cmd) { $pwsh = $cmd.Source }
}
if ($PSVersionTable.PSVersion.Major -lt 7) {
    if (-not $pwsh -or -not (Test-Path -LiteralPath $pwsh)) {
        Write-Error 'PSNeuron.psm1 requires PowerShell 7+ (pwsh). Install PowerShell 7 and retry.'
        exit 1
    }
    $argsList = @('-NoProfile', '-STA', '-ExecutionPolicy', 'Bypass', '-File', $PSCommandPath)
    if ($SmokeTest) { $argsList += '-SmokeTest' }
    $p = Start-Process -FilePath $pwsh -ArgumentList $argsList -Wait -PassThru -NoNewWindow
    exit $p.ExitCode
}

# Ensure STA for WinForms when launched without -STA
if ([Threading.Thread]::CurrentThread.GetApartmentState() -ne 'STA') {
    $exe = if ($PSVersionTable.PSVersion.Major -ge 7 -and $pwsh -and (Test-Path -LiteralPath $pwsh)) { $pwsh } else { (Get-Process -Id $PID).Path }
    $argsList = @('-NoProfile', '-STA', '-ExecutionPolicy', 'Bypass', '-File', $PSCommandPath)
    if ($SmokeTest) { $argsList += '-SmokeTest' }
    $p = Start-Process -FilePath $exe -ArgumentList $argsList -Wait -PassThru -NoNewWindow
    exit $p.ExitCode
}

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[System.Windows.Forms.Application]::EnableVisualStyles()
[System.Windows.Forms.Application]::SetCompatibleTextRenderingDefault($false)

$script:ModulePath = (Join-Path $PSScriptRoot '..\PSNeuron.psm1')
if (-not (Test-Path -LiteralPath $script:ModulePath)) {
    [System.Windows.Forms.MessageBox]::Show("PSNeuron.psm1 not found:`n$script:ModulePath", 'PSNeuron GUI') | Out-Null
    exit 1
}
try {
    Import-Module -Name $script:ModulePath -Force -ErrorAction Stop
} catch {
    Write-Error "Failed to Import-Module PSNeuron.psm1: $_"
    if (-not $SmokeTest) {
        [System.Windows.Forms.MessageBox]::Show("Failed to import PSNeuron.psm1:`n$_", 'PSNeuron GUI') | Out-Null
    }
    exit 1
}

$script:AdapterPath = Join-Path $PSScriptRoot 'PSNeuron.Adapter.ps1'
if (-not (Test-Path -LiteralPath $script:AdapterPath)) {
    Write-Error "Missing adapter: $script:AdapterPath"
    exit 1
}
. $script:AdapterPath

# --- limits (documented in README / hint) ---
$script:MaxHiddenLayers = 3
$script:MaxNeuronsPerHidden = 12
$script:SineWindow = 4

# --- training state ---
$script:nn = $null
$script:demoName = 'XOR'          # XOR | SineForecast
$script:activation = 'Sigmoid'
$script:layoutHidden = @(2)       # hidden layer sizes only; output always 1 for demos
$script:outputCount = 1
$script:inputFeatures = 2
$script:learningRate = 0.5
$script:epoch = 0
$script:loss = 0.0
$script:running = $false
$script:selectedLayer = -1
$script:selectedIndex = -1
$script:lastActivations = @{}
$script:trainData = @()
$script:vizSampleIndex = 0
$script:suppressEvents = $false
$script:weightSyncTick = 0

# --- UX / hit-test state (v1.1) ---
$script:nodeRadius = 18
$script:nodePositions = @{}
$script:edgeHits = @()
$script:lossSlice = @()
$script:lossSliceStartEpoch = 0
$script:flashKey = $null
$script:flashUntil = [datetime]::MinValue
$script:lastHoverKey = $null
$script:ctxMenuLayer = -1
$script:ctxMenuIndex = -1

# --- Teaching / walk state (v1.2) ---
$script:prevLoss = $null
$script:walkArmed = $false
$script:walkSteps = @()
$script:walkIndex = -1
$script:walkHighlightKey = $null
$script:walkHighlightEdge = $null   # @{ Layer; Neuron; WIdx }
$script:ghostActivations = @{}
$script:lastHoverEdgeKey = $null
$script:narratorEpochSkip = 0

function Get-FullLayout {
    return [int[]](@($script:layoutHidden) + @($script:outputCount))
}

function Get-LayoutLabel {
    $parts = @("In=$($script:inputFeatures)")
    for ($i = 0; $i -lt $script:layoutHidden.Count; $i++) {
        $parts += "H$i=$($script:layoutHidden[$i])"
    }
    $parts += "Out=$($script:outputCount)"
    return ($parts -join '  ')
}

function Convert-ToFloatRow {
    param($Row)
    # Normalize a demo row to float[] (guards against PS nesting quirks)
    $vals = @($Row)
    $out = [float[]]::new($vals.Count)
    for ($i = 0; $i -lt $vals.Count; $i++) {
        $out[$i] = [float]$vals[$i]
    }
    return $out
}

function Load-DemoData {
    if ($script:demoName -eq 'SineForecast') {
        $script:inputFeatures = $script:SineWindow
        $raw = @(Get-PSNeuronDemoRows -Demo SineForecast -SineWindow $script:SineWindow -SineCount 48)
    }
    else {
        $script:demoName = 'XOR'
        $script:inputFeatures = 2
        $raw = @(Get-PSNeuronDemoRows -Demo XOR)
    }
    $script:trainData = @()
    foreach ($r in $raw) {
        $script:trainData += , (Convert-ToFloatRow $r)
    }
    if ($script:trainData.Count -lt 1) {
        throw "Demo data empty for $($script:demoName)"
    }
    # Sanity: each row must be inputFeatures + 1 (target)
    $need = $script:inputFeatures + 1
    if (@($script:trainData[0]).Count -ne $need) {
        throw "Demo row width $(@($script:trainData[0]).Count) != expected $need"
    }
    if ($script:vizSampleIndex -ge $script:trainData.Count) {
        $script:vizSampleIndex = 0
    }
}

function New-DemoNetwork {
    param(
        [int[]]$HiddenLayout = $script:layoutHidden,
        [switch]$PreserveWeights
    )
    $old = $script:nn
    $script:layoutHidden = @([int[]]$HiddenLayout | ForEach-Object { [Math]::Max(1, [int]$_) })
    if ($script:layoutHidden.Count -lt 1) { $script:layoutHidden = @(2) }
    if ($script:layoutHidden.Count -gt $script:MaxHiddenLayers) {
        $script:layoutHidden = $script:layoutHidden[0..($script:MaxHiddenLayers - 1)]
    }
    for ($i = 0; $i -lt $script:layoutHidden.Count; $i++) {
        if ($script:layoutHidden[$i] -gt $script:MaxNeuronsPerHidden) {
            $script:layoutHidden[$i] = $script:MaxNeuronsPerHidden
        }
    }
    $layout = Get-FullLayout
    $script:nn = New-PSNeuronNetwork -Layout $layout `
        -LearningRate ([float]$script:learningRate) `
        -InputFeatureCount $script:inputFeatures `
        -ActivationFunction $script:activation
    if ($PreserveWeights -and $null -ne $old) {
        Copy-PSNeuronOverlappingWeights -OldNet $old -NewNet $script:nn
    }
    $script:epoch = 0
    $script:loss = 0.0
    $script:selectedLayer = -1
    $script:selectedIndex = -1
    $script:lastActivations = @{}
    if ($script:nn.LossHistory) { $script:nn.ClearLossHistory() }
}

function Get-NetworkMse {
    if ($null -eq $script:nn) { return 0.0 }
    $data = New-PSNeuronFloatJagged -Rows $script:trainData
    return [float]$script:nn.CalculateLoss($data)
}

function Get-VizInput {
    $row = $script:trainData[$script:vizSampleIndex]
    $xin = [float[]]::new($script:inputFeatures)
    for ($i = 0; $i -lt $script:inputFeatures; $i++) { $xin[$i] = [float]$row[$i] }
    return $xin
}

function Format-SampleLabel {
    param([int]$Index)
    $row = $script:trainData[$Index]
    $ins = for ($i = 0; $i -lt $script:inputFeatures; $i++) { '{0:F2}' -f [float]$row[$i] }
    $tgt = [float]$row[$script:inputFeatures]
    return ("#{0} [{1}] -> {2:F2}" -f $Index, ($ins -join ','), $tgt)
}

function Capture-Activations {
    param([float[]]$InputValues)
    $script:lastActivations = @{}
    for ($i = 0; $i -lt $InputValues.Count; $i++) {
        $script:lastActivations["I,$i"] = [float]$InputValues[$i]
    }
    for ($L = 0; $L -lt $script:nn.Network.Count; $L++) {
        for ($n = 0; $n -lt $script:nn.Network[$L].Count; $n++) {
            $script:lastActivations["$L,$n"] = [float]$script:nn.Network[$L][$n].Value
        }
    }
}

function Invoke-OneEpoch {
    param([switch]$QuietNarrator)
    if ($null -eq $script:nn) { return }
    $lossBefore = if ($null -ne $script:prevLoss) { [float]$script:prevLoss } else { [float]$script:loss }
    if ($script:epoch -eq 0 -and $null -eq $script:prevLoss) {
        $lossBefore = Get-NetworkMse
    }
    $order = 0..($script:trainData.Count - 1) | Sort-Object { Get-Random }
    foreach ($idx in $order) {
        $row = $script:trainData[$idx]
        $xin = [float[]]::new($script:inputFeatures)
        for ($i = 0; $i -lt $script:inputFeatures; $i++) { $xin[$i] = [float]$row[$i] }
        $yout = [float[]]@([float]$row[$script:inputFeatures])
        $script:nn.TrainSample($xin, $yout)
    }
    $script:epoch++
    $script:loss = Get-NetworkMse
    if ($script:nn.PSObject.Methods['AppendLoss']) {
        $script:nn.AppendLoss([float]$script:loss)
    }
    $vizIn = Get-VizInput
    $null = $script:nn.GetOutput($vizIn)
    Capture-Activations -InputValues $vizIn
    $script:prevLoss = [float]$script:loss
    if (-not $QuietNarrator) {
        Write-EpochNarrator -LossBefore $lossBefore -LossAfter ([float]$script:loss) -EpochNum $script:epoch
    }
}

function Update-StatusLabels {
    $lblEpoch.Text = "Epoch: $($script:epoch)"
    $lblLoss.Text = ("Loss (MSE): {0:F6}" -f $script:loss)
    $lblLayout.Text = "Layout: $(Get-LayoutLabel)  LR=$($script:learningRate)  Act=$($script:activation)"
    Update-LossExplainLabel
}

function Refresh-NeuronCombo {
    $prevL = $script:selectedLayer
    $prevN = $script:selectedIndex
    $cmbNeuron.Items.Clear()
    if ($null -eq $script:nn) { return }
    $sel = 0
    for ($L = 0; $L -lt $script:nn.Network.Count; $L++) {
        $layerName = if ($L -eq $script:nn.Network.Count - 1) { 'Out' } else { "H$L" }
        for ($n = 0; $n -lt $script:nn.Network[$L].Count; $n++) {
            $item = "$layerName L$L N$n|$L|$n"
            [void]$cmbNeuron.Items.Add($item)
            if ($prevL -eq $L -and $prevN -eq $n) { $sel = $cmbNeuron.Items.Count - 1 }
        }
    }
    if ($cmbNeuron.Items.Count -gt 0) {
        $script:suppressEvents = $true
        try { $cmbNeuron.SelectedIndex = $sel } finally { $script:suppressEvents = $false }
    }
}

function Refresh-LayerCombo {
    $cmbAddLayer.Items.Clear()
    for ($i = 0; $i -lt $script:layoutHidden.Count; $i++) {
        [void]$cmbAddLayer.Items.Add("H$i (n=$($script:layoutHidden[$i]))|$i")
    }
    if ($cmbAddLayer.Items.Count -gt 0 -and $cmbAddLayer.SelectedIndex -lt 0) {
        $cmbAddLayer.SelectedIndex = 0
    }
}

function Refresh-SampleCombo {
    $script:suppressEvents = $true
    try {
        $cmbSample.Items.Clear()
        $maxShow = [Math]::Min(24, $script:trainData.Count)
        for ($i = 0; $i -lt $maxShow; $i++) {
            [void]$cmbSample.Items.Add((Format-SampleLabel -Index $i))
        }
        if ($script:trainData.Count -gt $maxShow) {
            [void]$cmbSample.Items.Add(("... +{0} more (pick 0..{1} via index)" -f ($script:trainData.Count - $maxShow), ($script:trainData.Count - 1)))
        }
        if ($cmbSample.Items.Count -gt 0) {
            $idx = [Math]::Min($script:vizSampleIndex, $maxShow - 1)
            $cmbSample.SelectedIndex = [Math]::Max(0, $idx)
        }
    }
    finally { $script:suppressEvents = $false }
}

function Load-WeightEditors {
    param([int]$FocusWeightIndex = -1)
    $pnlWeights.Controls.Clear()
    if ($cmbNeuron.SelectedItem -eq $null) { return }
    $parts = [string]$cmbNeuron.SelectedItem -split '\|'
    if ($parts.Count -lt 3) { return }
    $L = [int]$parts[1]
    $n = [int]$parts[2]
    $script:selectedLayer = $L
    $script:selectedIndex = $n
    $neuron = $script:nn.Network[$L][$n]
    $w = $neuron.Weights
    $y = 4
    $focusTrack = $null
    for ($i = 0; $i -lt $w.Count; $i++) {
        $isBias = ($i -eq $w.Count - 1)
        $labelText = if ($isBias) { 'bias' } else { "w$i" }
        $lbl = New-Object System.Windows.Forms.Label
        $lbl.Text = $labelText
        $lbl.Location = New-Object System.Drawing.Point(4, ($y + 4))
        $lbl.Size = New-Object System.Drawing.Size(36, 20)
        $nud = New-Object System.Windows.Forms.NumericUpDown
        $nud.DecimalPlaces = 4
        $nud.Minimum = -10
        $nud.Maximum = 10
        $nud.Increment = 0.05
        $nud.Location = New-Object System.Drawing.Point(44, $y)
        $nud.Size = New-Object System.Drawing.Size(90, 22)
        $val = [decimal][Math]::Max(-10, [Math]::Min(10, [double]$w[$i]))
        $nud.Value = $val
        $track = New-Object System.Windows.Forms.TrackBar
        $track.Minimum = -1000
        $track.Maximum = 1000
        $track.TickFrequency = 200
        $track.Location = New-Object System.Drawing.Point(140, $y)
        $track.Size = New-Object System.Drawing.Size(160, 30)
        $track.Value = [int][Math]::Max(-1000, [Math]::Min(1000, [Math]::Round([double]$w[$i] * 100)))
        $wi = $i
        $nud.Tag = @{ Layer = $L; Index = $n; W = $wi; Sync = $true }
        $track.Tag = @{ Layer = $L; Index = $n; W = $wi; Sync = $true }
        $tipW = Get-WeightImpactTip -Layer $L -Neuron $n -WIdx $wi -IsBias:$isBias
        if ($toolTips) {
            $toolTips.SetToolTip($nud, $tipW)
            $toolTips.SetToolTip($track, $tipW)
            $toolTips.SetToolTip($lbl, $tipW)
        }
        $nud.add_ValueChanged({
            param($sender, $e)
            $t = $sender.Tag
            if (-not $t.Sync) { return }
            $script:nn.Network[$t.Layer][$t.Index].Weights[$t.W] = [float]$sender.Value
            foreach ($c in $pnlWeights.Controls) {
                if ($c -is [System.Windows.Forms.TrackBar] -and $c.Tag.Layer -eq $t.Layer -and $c.Tag.Index -eq $t.Index -and $c.Tag.W -eq $t.W) {
                    $c.Tag.Sync = $false
                    $c.Value = [int][Math]::Max(-1000, [Math]::Min(1000, [Math]::Round([double]$sender.Value * 100)))
                    $c.Tag.Sync = $true
                }
            }
            $wCount = $script:nn.Network[$t.Layer][$t.Index].Weights.Count
            $impact = Get-WeightImpactTip -Layer $t.Layer -Neuron $t.Index -WIdx $t.W -IsBias:($t.W -eq ($wCount - 1))
            Set-TipLine -Text $impact
            Set-TeachPanel -Text $impact
            $canvas.Invalidate()
        }.GetNewClosure())
        $track.add_Scroll({
            param($sender, $e)
            $t = $sender.Tag
            if (-not $t.Sync) { return }
            $fv = [float]($sender.Value / 100.0)
            $script:nn.Network[$t.Layer][$t.Index].Weights[$t.W] = $fv
            foreach ($c in $pnlWeights.Controls) {
                if ($c -is [System.Windows.Forms.NumericUpDown] -and $c.Tag.Layer -eq $t.Layer -and $c.Tag.Index -eq $t.Index -and $c.Tag.W -eq $t.W) {
                    $c.Tag.Sync = $false
                    $dv = [decimal][Math]::Max(-10, [Math]::Min(10, [double]$fv))
                    $c.Value = $dv
                    $c.Tag.Sync = $true
                }
            }
            $wCount = $script:nn.Network[$t.Layer][$t.Index].Weights.Count
            $impact = Get-WeightImpactTip -Layer $t.Layer -Neuron $t.Index -WIdx $t.W -IsBias:($t.W -eq ($wCount - 1))
            Set-TipLine -Text $impact
            Set-TeachPanel -Text $impact
            $canvas.Invalidate()
        }.GetNewClosure())
        $pnlWeights.Controls.Add($lbl)
        $pnlWeights.Controls.Add($nud)
        $pnlWeights.Controls.Add($track)
        if ($FocusWeightIndex -eq $i) { $focusTrack = $track }
        $y += 36
    }
    $pnlWeights.AutoScrollMinSize = New-Object System.Drawing.Size(0, ($y + 8))
    if ($null -ne $focusTrack) {
        try { $null = $focusTrack.Focus() } catch {}
        $pnlWeights.ScrollControlIntoView($focusTrack)
    }
}

function Sync-WeightEditorsLive {
    # Update existing editors from network without rebuild (keeps selection / focus friendly)
    if ($script:selectedLayer -lt 0 -or $null -eq $script:nn) { return }
    $L = $script:selectedLayer
    $n = $script:selectedIndex
    if ($L -ge $script:nn.Network.Count) { return }
    if ($n -ge $script:nn.Network[$L].Count) { return }
    # Skip if user is editing a NumericUpDown
    $active = [System.Windows.Forms.Form]::ActiveForm
    if ($active) {
        $fc = $active.ActiveControl
        if ($fc -is [System.Windows.Forms.NumericUpDown] -and $fc.Parent -eq $pnlWeights) { return }
        if ($fc -is [System.Windows.Forms.TrackBar] -and $fc.Parent -eq $pnlWeights) { return }
    }
    $w = $script:nn.Network[$L][$n].Weights
    foreach ($c in @($pnlWeights.Controls)) {
        if ($c -is [System.Windows.Forms.NumericUpDown]) {
            $t = $c.Tag
            if ($null -eq $t) { continue }
            if ($t.Layer -ne $L -or $t.Index -ne $n) { continue }
            if ($t.W -ge $w.Count) { continue }
            $dv = [decimal][Math]::Max(-10, [Math]::Min(10, [double]$w[$t.W]))
            if ($c.Value -ne $dv) {
                $t.Sync = $false
                $c.Value = $dv
                $t.Sync = $true
            }
        }
        elseif ($c -is [System.Windows.Forms.TrackBar]) {
            $t = $c.Tag
            if ($null -eq $t) { continue }
            if ($t.Layer -ne $L -or $t.Index -ne $n) { continue }
            if ($t.W -ge $w.Count) { continue }
            $tv = [int][Math]::Max(-1000, [Math]::Min(1000, [Math]::Round([double]$w[$t.W] * 100)))
            if ($c.Value -ne $tv) {
                $t.Sync = $false
                $c.Value = $tv
                $t.Sync = $true
            }
        }
    }
}


function Set-TipLine {
    param(
        [string]$Text,
        [ValidateSet('info','warn','ok')]
        [string]$Level = 'info'
    )
    if (-not $lblTip) { return }
    $lblTip.Text = $Text
    switch ($Level) {
        'warn' { $lblTip.ForeColor = [System.Drawing.Color]::Orange }
        'ok'   { $lblTip.ForeColor = [System.Drawing.Color]::LightGreen }
        default { $lblTip.ForeColor = [System.Drawing.Color]::Gainsboro }
    }
}

function Update-ContextWarnings {
    if ($script:learningRate -ge 1.5) {
        Set-TipLine -Level warn -Text ("High LR ({0:F3}): updates may overshoot - try 0.1-0.5 for teaching." -f $script:learningRate)
        return $true
    }
    if ($script:activation -eq 'ReLU' -and $script:demoName -eq 'XOR') {
        Set-TipLine -Level warn -Text 'ReLU + XOR can stall (dead units). Prefer Sigmoid for this demo.'
        return $true
    }
    return $false
}

function Set-TeachPanel {
    param(
        [string]$Text,
        [ValidateSet('info','warn','ok')]
        [string]$Level = 'info'
    )
    if (-not $lblTeach) { return }
    $lblTeach.Text = $Text
    switch ($Level) {
        'warn' { $lblTeach.ForeColor = [System.Drawing.Color]::Orange }
        'ok'   { $lblTeach.ForeColor = [System.Drawing.Color]::LightGreen }
        default { $lblTeach.ForeColor = [System.Drawing.Color]::Gainsboro }
    }
}

function Update-LossExplainLabel {
    if (-not $lblLossExplain) { return }
    $near = ([float]$script:loss -le 0.00005)
    if ($near) {
        $lblLossExplain.Text = 'Loss ~0.0000 = predictions match the quiz answers on this demo (not a money number). Nice!'
        $lblLossExplain.ForeColor = [System.Drawing.Color]::LightGreen
    }
    else {
        $lblLossExplain.Text = 'Loss = average "how wrong" the guesses are. Training nudges weights to drive it toward 0.0000.'
        $lblLossExplain.ForeColor = [System.Drawing.Color]::Silver
    }
}

function Get-SourceActForWeight {
    param([int]$Layer, [int]$WIdx)
    if ($null -eq $script:nn) { return 0.0 }
    if ($Layer -eq 0) {
        $key = "I,$WIdx"
        if ($script:lastActivations.ContainsKey($key)) { return [float]$script:lastActivations[$key] }
        $vin = Get-VizInput
        if ($WIdx -ge 0 -and $WIdx -lt $vin.Count) { return [float]$vin[$WIdx] }
        return 0.0
    }
    $prevKey = "$($Layer - 1),$WIdx"
    if ($script:lastActivations.ContainsKey($prevKey)) { return [float]$script:lastActivations[$prevKey] }
    if ($Layer - 1 -lt $script:nn.Network.Count -and $WIdx -lt $script:nn.Network[$Layer - 1].Count) {
        return [float]$script:nn.Network[$Layer - 1][$WIdx].Value
    }
    return 0.0
}

function Get-SourceLabelForWeight {
    param([int]$Layer, [int]$WIdx)
    if ($Layer -eq 0) { return ("I{0}" -f $WIdx) }
    $prevL = $Layer - 1
    # Source is the previous network layer (hidden or, if Layer is Out, last hidden).
    return ("H{0}:{1}" -f $prevL, $WIdx)
}

function Get-NeuronPlainName {
    param([int]$Layer, [int]$Index)
    if ($null -eq $script:nn) { return ("L{0}:{1}" -f $Layer, $Index) }
    if ($Layer -eq $script:nn.Network.Count - 1) { return ("Out:{0}" -f $Index) }
    return ("H{0}:{1}" -f $Layer, $Index)
}

function Get-PushPhrase {
    param([double]$Contribution)
    if ($Contribution -gt 0.05) { return 'pushing the next neuron UP' }
    if ($Contribution -lt -0.05) { return 'pushing the next neuron DOWN' }
    return 'barely pushing either way'
}

function Get-WeightImpactTip {
    param(
        [int]$Layer,
        [int]$Neuron,
        [int]$WIdx,
        [switch]$IsBias
    )
    if ($null -eq $script:nn) { return 'Pick a neuron to see what a weight does.' }
    $dest = Get-NeuronPlainName -Layer $Layer -Index $Neuron
    $wArr = $script:nn.Network[$Layer][$Neuron].Weights
    if ($IsBias -or $WIdx -eq ($wArr.Count - 1)) {
        $b = [float]$wArr[$wArr.Count - 1]
        $dir = if ($b -ge 0) { 'easier to fire (higher activation)' } else { 'harder to fire (lower activation)' }
        return ("Bias on {0} (value {1:F3}): a built-in nudge with no input. Raising it makes {0} {2}." -f $dest, $b, $dir)
    }
    $src = Get-SourceLabelForWeight -Layer $Layer -WIdx $WIdx
    $w = [float]$wArr[$WIdx]
    $srcAct = Get-SourceActForWeight -Layer $Layer -WIdx $WIdx
    $contrib = $w * $srcAct
    $listen = if ($w -ge 0) {
        "Raising this weight makes $dest listen MORE to $src on this sample."
    } else {
        "This weight is negative - raising it (toward 0 or +) reduces how much $src suppresses $dest."
    }
    $push = Get-PushPhrase -Contribution $contrib
    return ("{0} Weight {1}->{2} = {3:F3}. On this viz sample, {1} is {4:F3}, so contribution~{5:F3} ({6}). Watch {2}'s activation after Pulse." -f $listen, $src, $dest, $w, $srcAct, $contrib, $push)
}

function Get-NeuronPreActivation {
    param([int]$Layer, [int]$Index)
    if ($null -eq $script:nn) { return 0.0 }
    $neuron = $script:nn.Network[$Layer][$Index]
    $z = 0.0
    if ($Layer -eq 0) {
        $vin = Get-VizInput
        for ($i = 0; $i -lt $script:inputFeatures; $i++) {
            $z += [double]$neuron.Weights[$i] * [double]$vin[$i]
        }
    }
    else {
        $prev = $script:nn.Network[$Layer - 1]
        for ($i = 0; $i -lt $prev.Count; $i++) {
            $prevAct = [float]$prev[$i].Value
            if ($script:lastActivations.ContainsKey("$($Layer-1),$i")) {
                $prevAct = [float]$script:lastActivations["$($Layer-1),$i"]
            }
            $z += [double]$neuron.Weights[$i] * [double]$prevAct
        }
    }
    $z += [double]$neuron.GetBias()
    return [float]$z
}

function Write-EpochNarrator {
    param(
        [float]$LossBefore,
        [float]$LossAfter,
        [int]$EpochNum
    )
    $delta = $LossAfter - $LossBefore
    $why = if ([Math]::Abs($delta) -lt 1e-9) {
        'guesses barely changed'
    } elseif ($delta -lt 0) {
        'guesses got closer to the quiz answers'
    } else {
        'guesses got farther (common early on or with a high learning rate)'
    }
    $nRows = $script:trainData.Count
    $line = ("Epoch {0}: tried all {1} quiz rows, nudged weights, loss {2:F4}->{3:F4} because {4}." -f `
        $EpochNum, $nRows, $LossBefore, $LossAfter, $why)
    if ($LossAfter -le 0.00005) {
        $line += ' Loss near 0.0000 means the net''s answers match the targets on this demo.'
    }
    # Continuous Start: tip every epoch; teach panel every epoch (cheap).
    Set-TipLine -Level ok -Text $line
    Set-TeachPanel -Level ok -Text $line
    Update-LossExplainLabel
}

function Stop-WalkUi {
    if ($walkTimer) { [void]$walkTimer.Stop() }
    $script:walkArmed = $false
    $script:walkSteps = @()
    $script:walkIndex = -1
    $script:walkHighlightKey = $null
    $script:walkHighlightEdge = $null
    if ($chkWalk -and $chkWalk.Checked) {
        $script:suppressEvents = $true
        try { $chkWalk.Checked = $false } finally { $script:suppressEvents = $false }
    }
    return
}

function Build-WalkSteps {
    $steps = New-Object System.Collections.Generic.List[object]
    if ($null -eq $script:nn) { return @() }
    Apply-ForwardViz
    $sample = Format-SampleLabel -Index $script:vizSampleIndex
    $steps.Add(@{
        Kind = 'intro'
        Key = $null
        Edge = $null
        Tip = ("Walk: following viz sample {0} left->right. Each pause shows one input, edge, or neuron." -f $sample)
    })
    for ($i = 0; $i -lt $script:inputFeatures; $i++) {
        $act = 0.0
        if ($script:lastActivations.ContainsKey("I,$i")) { $act = [float]$script:lastActivations["I,$i"] }
        $steps.Add(@{
            Kind = 'input'
            Key = "I,$i"
            Edge = $null
            Tip = ("Input I{0} = {1:F3}. This number is the feature leaving the left side; next it rides edges into the first hidden layer." -f $i, $act)
        })
    }
    for ($L = 0; $L -lt $script:nn.Network.Count; $L++) {
        for ($n = 0; $n -lt $script:nn.Network[$L].Count; $n++) {
            $dest = Get-NeuronPlainName -Layer $L -Index $n
            $prevCount = if ($L -eq 0) { $script:inputFeatures } else { $script:nn.Network[$L - 1].Count }
            for ($p = 0; $p -lt $prevCount; $p++) {
                $src = Get-SourceLabelForWeight -Layer $L -WIdx $p
                $w = [float]$script:nn.Network[$L][$n].Weights[$p]
                $srcAct = Get-SourceActForWeight -Layer $L -WIdx $p
                $contrib = $w * $srcAct
                $push = Get-PushPhrase -Contribution $contrib
                $fromKey = if ($L -eq 0) { "I,$p" } else { "$($L-1),$p" }
                $steps.Add(@{
                    Kind = 'edge'
                    Key = "$L,$n"
                    Edge = @{ Layer = $L; Neuron = $n; WIdx = $p; FromKey = $fromKey }
                    Tip = ("Edge {0}->{1}: weight={2:F3}, source={3:F3}, contribution~{4:F3} ({5})." -f $src, $dest, $w, $srcAct, $contrib, $push)
                })
            }
            $z = Get-NeuronPreActivation -Layer $L -Index $n
            $act = [float]$script:nn.Network[$L][$n].Value
            if ($script:lastActivations.ContainsKey("$L,$n")) { $act = [float]$script:lastActivations["$L,$n"] }
            $feel = if ($act -ge 0.7) { 'firing strongly' } elseif ($act -ge 0.35) { 'medium' } elseif ($act -gt 0.05) { 'weak' } else { 'mostly quiet' }
            $steps.Add(@{
                Kind = 'node'
                Key = "$L,$n"
                Edge = $null
                Tip = ("{0}: raw mix z~{1:F3} -> activation={2:F3} ({3}). Activation is what the next layer sees." -f $dest, $z, $act, $feel)
            })
        }
    }
    $outL = $script:nn.Network.Count - 1
    $outAct = [float]$script:nn.Network[$outL][0].Value
    if ($script:lastActivations.ContainsKey("$outL,0")) { $outAct = [float]$script:lastActivations["$outL,0"] }
    $row = $script:trainData[$script:vizSampleIndex]
    $tgt = [float]$row[$script:inputFeatures]
    $steps.Add(@{
        Kind = 'done'
        Key = "$outL,0"
        Edge = $null
        Tip = ("Done. Output guess={0:F3}, quiz target={1:F3}. Loss averages how far guesses miss targets - we train toward 0.0000." -f $outAct, $tgt)
    })
    return @($steps.ToArray())
}

function Show-WalkStep {
    if ($script:walkIndex -lt 0 -or $script:walkIndex -ge $script:walkSteps.Count) {
        Stop-WalkUi
        Set-TipLine -Text 'Walk finished. Toggle Walk inputs again, or Pulse / Step to keep exploring.'
        Set-TeachPanel -Text 'Walk finished. Loss is average how-wrong; weights are the dials that change each neuron''s guess.'
        if ($canvas) { $canvas.Invalidate() }
        return
    }
    $step = $script:walkSteps[$script:walkIndex]
    $script:walkHighlightKey = $step.Key
    $script:walkHighlightEdge = $step.Edge
    if ($step.Kind -eq 'node' -or $step.Kind -eq 'done') {
        $parts = [string]$step.Key -split ','
        if ($parts.Count -eq 2 -and $parts[0] -ne 'I') {
            # soft-select without clobbering teach text too hard
            $script:selectedLayer = [int]$parts[0]
            $script:selectedIndex = [int]$parts[1]
        }
    }
    Set-TipLine -Level ok -Text ("Walk {0}/{1}: {2}" -f ($script:walkIndex + 1), $script:walkSteps.Count, $step.Tip)
    Set-TeachPanel -Level ok -Text $step.Tip
    if ($flashTimer) {
        $script:flashKey = $step.Key
        $script:flashUntil = [datetime]::UtcNow.AddMilliseconds(650)
        $flashTimer.Start()
    }
    if ($canvas) { $canvas.Invalidate() }
}

function Start-WalkUi {
    if ($null -eq $script:nn) { return }
    Stop-TrainingUi
    $script:walkSteps = @(Build-WalkSteps)
    if ($script:walkSteps.Count -lt 1) { return }
    $script:walkArmed = $true
    $script:walkIndex = 0
    Show-WalkStep
    if ($walkTimer) {
        $walkTimer.Interval = 850
        $walkTimer.Start()
    }
}

function Advance-WalkStep {
    if (-not $script:walkArmed) { return }
    $script:walkIndex++
    Show-WalkStep
}

function Build-NetworkGeometry {
    param([int]$Width, [int]$Height)
    $positions = @{}
    $edges = @()
    if ($null -eq $script:nn) {
        $script:nodePositions = $positions
        $script:edgeHits = $edges
        return
    }
    $padX = 50
    $padY = 40
    $layerCount = 1 + $script:nn.Network.Count
    for ($i = 0; $i -lt $script:inputFeatures; $i++) {
        $x = $padX
        $y = $padY + (($Height - 2 * $padY) * ($i + 1) / ($script:inputFeatures + 1))
        $positions["I,$i"] = New-Object System.Drawing.PointF($x, $y)
    }
    for ($L = 0; $L -lt $script:nn.Network.Count; $L++) {
        $count = $script:nn.Network[$L].Count
        $x = $padX + (($Width - 2 * $padX) * ($L + 1) / ($layerCount - 1))
        for ($n = 0; $n -lt $count; $n++) {
            $y = $padY + (($Height - 2 * $padY) * ($n + 1) / ($count + 1))
            $positions["$L,$n"] = New-Object System.Drawing.PointF($x, $y)
        }
    }
    for ($n = 0; $n -lt $script:nn.Network[0].Count; $n++) {
        $neuron = $script:nn.Network[0][$n]
        for ($i = 0; $i -lt $script:inputFeatures; $i++) {
            $p1 = $positions["I,$i"]
            $p2 = $positions["0,$n"]
            $edges += , @{
                X1 = [double]$p1.X; Y1 = [double]$p1.Y
                X2 = [double]$p2.X; Y2 = [double]$p2.Y
                Layer = 0; Neuron = $n; WIdx = $i; W = [double]$neuron.Weights[$i]
            }
        }
    }
    for ($L = 1; $L -lt $script:nn.Network.Count; $L++) {
        for ($n = 0; $n -lt $script:nn.Network[$L].Count; $n++) {
            $neuron = $script:nn.Network[$L][$n]
            $prevCount = $script:nn.Network[$L - 1].Count
            for ($p = 0; $p -lt $prevCount; $p++) {
                $p1 = $positions["$($L-1),$p"]
                $p2 = $positions["$L,$n"]
                $edges += , @{
                    X1 = [double]$p1.X; Y1 = [double]$p1.Y
                    X2 = [double]$p2.X; Y2 = [double]$p2.Y
                    Layer = $L; Neuron = $n; WIdx = $p; W = [double]$neuron.Weights[$p]
                }
            }
        }
    }
    $script:nodePositions = $positions
    $script:edgeHits = $edges
}

function Find-NodeAtPoint {
    param([int]$X, [int]$Y)
    $best = $null
    $bestD = [double]::MaxValue
    $r = [double]$script:nodeRadius + 4
    foreach ($key in @($script:nodePositions.Keys)) {
        $pt = $script:nodePositions[$key]
        $dx = [double]$pt.X - $X
        $dy = [double]$pt.Y - $Y
        $d = [Math]::Sqrt($dx * $dx + $dy * $dy)
        if ($d -le $r -and $d -lt $bestD) {
            $bestD = $d
            $best = $key
        }
    }
    return $best
}

function Get-DistanceToSegment {
    param([double]$Px, [double]$Py, [double]$X1, [double]$Y1, [double]$X2, [double]$Y2)
    $vx = $X2 - $X1
    $vy = $Y2 - $Y1
    $wx = $Px - $X1
    $wy = $Py - $Y1
    $c1 = $vx * $wx + $vy * $wy
    if ($c1 -le 0) { return [Math]::Sqrt($wx * $wx + $wy * $wy) }
    $c2 = $vx * $vx + $vy * $vy
    if ($c2 -le $c1) {
        $dx = $Px - $X2; $dy = $Py - $Y2
        return [Math]::Sqrt($dx * $dx + $dy * $dy)
    }
    $b = $c1 / $c2
    $bx = $X1 + $b * $vx
    $by = $Y1 + $b * $vy
    $dx = $Px - $bx; $dy = $Py - $by
    return [Math]::Sqrt($dx * $dx + $dy * $dy)
}

function Find-EdgeAtPoint {
    param([int]$X, [int]$Y, [double]$MaxDist = 8.0)
    $best = $null
    $bestD = [double]::MaxValue
    foreach ($e in $script:edgeHits) {
        $d = Get-DistanceToSegment -Px $X -Py $Y -X1 $e.X1 -Y1 $e.Y1 -X2 $e.X2 -Y2 $e.Y2
        if ($d -le $MaxDist -and $d -lt $bestD) {
            $bestD = $d
            $best = $e
        }
    }
    return $best
}

function Select-NetworkNeuron {
    param(
        [int]$Layer,
        [int]$Index,
        [string]$Reason = ''
    )
    if ($null -eq $script:nn) { return }
    if ($Layer -lt 0 -or $Layer -ge $script:nn.Network.Count) { return }
    if ($Index -lt 0 -or $Index -ge $script:nn.Network[$Layer].Count) { return }
    $script:selectedLayer = $Layer
    $script:selectedIndex = $Index
    $want = "|$Layer|$Index"
    $found = -1
    for ($i = 0; $i -lt $cmbNeuron.Items.Count; $i++) {
        if ([string]$cmbNeuron.Items[$i] -like "*$want") { $found = $i; break }
    }
    if ($found -ge 0) {
        $script:suppressEvents = $true
        try { $cmbNeuron.SelectedIndex = $found } finally { $script:suppressEvents = $false }
    }
    Load-WeightEditors
    $canvas.Invalidate()
    $layerName = if ($Layer -eq $script:nn.Network.Count - 1) { 'Out' } else { "H$Layer" }
    $act = [float]$script:nn.Network[$Layer][$Index].Value
    if ($script:lastActivations.ContainsKey("$Layer,$Index")) { $act = $script:lastActivations["$Layer,$Index"] }
    $bias = [float]$script:nn.Network[$Layer][$Index].GetBias()
    $msg = if ($Reason) { $Reason } else {
        ("Selected {0}:{1} - activation={2:F3}, bias={3:F3}. Drag a slider: raising a weight makes this neuron listen more to that input." -f $layerName, $Index, $act, $bias)
    }
    Set-TipLine -Text $msg
    Set-TeachPanel -Text $msg
}

function Focus-WeightSlider {
    param([int]$WeightIndex)
    Load-WeightEditors -FocusWeightIndex $WeightIndex
    if ($script:selectedLayer -ge 0 -and $null -ne $script:nn) {
        $wCount = $script:nn.Network[$script:selectedLayer][$script:selectedIndex].Weights.Count
        $isBias = ($WeightIndex -eq ($wCount - 1))
        $tip = Get-WeightImpactTip -Layer $script:selectedLayer -Neuron $script:selectedIndex -WIdx $WeightIndex -IsBias:$isBias
        Set-TipLine -Text $tip
        Set-TeachPanel -Text $tip
    }
}

function Invoke-PulseForward {
    if ($null -eq $script:nn) { return }
    # Ghost = activations before this pulse (before/after teaching)
    $script:ghostActivations = @{}
    foreach ($k in @($script:lastActivations.Keys)) {
        $script:ghostActivations[$k] = [float]$script:lastActivations[$k]
    }
    Apply-ForwardViz
    if ($script:selectedLayer -ge 0) {
        $script:flashKey = "$($script:selectedLayer),$($script:selectedIndex)"
    }
    else {
        $L = $script:nn.Network.Count - 1
        $script:flashKey = "$L,0"
    }
    $script:flashUntil = [datetime]::UtcNow.AddMilliseconds(400)
    if ($flashTimer) { $flashTimer.Start() }
    $canvas.Invalidate()
    $sample = Format-SampleLabel -Index $script:vizSampleIndex
    $outL = $script:nn.Network.Count - 1
    $outNow = [float]$script:lastActivations["$outL,0"]
    $ghostNote = ''
    if ($script:ghostActivations.ContainsKey("$outL,0")) {
        $outWas = [float]$script:ghostActivations["$outL,0"]
        if ([Math]::Abs($outNow - $outWas) -gt 1e-4) {
            $ghostNote = (" Dashed ring = before Pulse (out {0:F3}->{1:F3})." -f $outWas, $outNow)
        }
        else {
            $ghostNote = ' Dashed ring = before Pulse (little change).'
        }
    }
    $msg = ("Forward pulse on viz sample {0} - node colors = activations now.{1}" -f $sample, $ghostNote)
    Set-TipLine -Level ok -Text $msg
    Set-TeachPanel -Level ok -Text $msg
}

function Show-PSNeuronHelp {
    $text = @"
PSNeuron graph (scannable)
- Left -> right: Inputs -> Hidden layer(s) -> Output
- Node fill = activation on the current viz sample
- Yellow ring = selected neuron (edit weights on the right)
- Cyan ring = Walk-inputs focus; dashed gray = Pulse before/after ghost
- Edges: blue = positive weight, red = negative; thicker = larger |w|
- Loss strip (bottom) = MSE per epoch (lower is better)

Why drive loss toward 0.0000?
- Loss is the average "how wrong" the net's guesses are on the quiz rows
- 0.0000 means predictions match the targets on this demo
- It is NOT a money / dollar number - just a teaching score for wrongness
- Each epoch tries every quiz row and nudges weights so guesses get closer

Weights (the dials)
- A weight says how strongly one neuron listens to another
- Raising a positive weight makes the destination listen more to that source
- Bias is a built-in nudge with no input
- Click an edge or drag a slider - the teach panel explains the impact

Walk inputs
- Toggle "Walk inputs" to slow-walk one viz sample left->right
- Pauses on each input, edge, and neuron with plain-English z / activation / contribution

Teaching tips
- XOR learns well with Sigmoid + small net (e.g. 2 hidden)
- ReLU + XOR can stall (dead units) - watch the tip line
- Add neuron/layer transplants overlapping weights; new links stay random
- High learning rate (>~1.5) can overshoot

Keyboard (when not typing in a number box)
- Space = Step one epoch
- Esc = Stop training (also stops Walk)
- R = Reset net

Canvas
- Click node = select - Double-click / Pulse = forward + flash
- Click edge ~ focus that weight - Right-click neuron = menu
- Hover node/edge = activation, weight, pushing up/down
- Click loss strip = tip for that epoch's loss
"@
    [System.Windows.Forms.MessageBox]::Show($text, 'PSNeuron GUI - Help (v1.2)', 'OK', 'Information') | Out-Null
}

function Test-KeyboardBlocked {
    if (-not $form) { return $true }
    $fc = $form.ActiveControl
    if ($null -eq $fc) { return $false }
    if ($fc -is [System.Windows.Forms.NumericUpDown]) { return $true }
    if ($fc -is [System.Windows.Forms.TextBox]) { return $true }
    if ($fc -is [System.Windows.Forms.ComboBox] -and $fc.DroppedDown) { return $true }
    if ($fc.Parent -is [System.Windows.Forms.NumericUpDown]) { return $true }
    return $false
}

function Randomize-NeuronWeights {
    param([int]$Layer, [int]$Index)
    if ($null -eq $script:nn) { return }
    $neuron = $script:nn.Network[$Layer][$Index]
    $w = $neuron.GetWeights()
    for ($i = 0; $i -lt $w.Count; $i++) {
        $w[$i] = [float](Get-Random -Minimum -1.0 -Maximum 1.0)
    }
    $neuron.SetWeights($w)
}

function Zero-NeuronBias {
    param([int]$Layer, [int]$Index)
    if ($null -eq $script:nn) { return }
    $script:nn.Network[$Layer][$Index].SetBias([float]0)
}

function Get-LossTipAtX {
    param([int]$X, [int]$Width)
    $hist = $script:lossSlice
    if ($null -eq $hist -or $hist.Count -lt 1) { return $null }
    $padL = 8; $padR = 8
    $plotW = [Math]::Max(10, $Width - $padL - $padR)
    $t = ([double]($X - $padL) / $plotW)
    $t = [Math]::Max(0.0, [Math]::Min(1.0, $t))
    $idx = [int][Math]::Round($t * ($hist.Count - 1))
    $epochNum = $script:lossSliceStartEpoch + $idx
    $val = [float]$hist[$idx]
    return @{ Epoch = $epochNum; Loss = $val }
}

function Get-ActivationColor([float]$v) {
    # Map roughly [0,1]; tanh may be negative - fold via (v+1)/2 when needed
    $t = [double]$v
    if ($t -lt 0 -or $t -gt 1) { $t = ($t + 1.0) / 2.0 }
    $t = [Math]::Max(0.0, [Math]::Min(1.0, $t))
    $r = [int](255 * $t)
    $g = [int](80 + 120 * (1.0 - [Math]::Abs($t - 0.5) * 2))
    $b = [int](255 * (1.0 - $t))
    return [System.Drawing.Color]::FromArgb(255, $r, $g, $b)
}

function Draw-Network {
    param($g, $width, $height)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $g.Clear([System.Drawing.Color]::FromArgb(255, 30, 30, 36))
    if ($null -eq $script:nn) { return }

    Build-NetworkGeometry -Width $width -Height $height
    $positions = $script:nodePositions
    $radius = $script:nodeRadius

    $pen = New-Object System.Drawing.Pen([System.Drawing.Color]::Gray, 1)
    $walkEdge = $script:walkHighlightEdge
    foreach ($e in $script:edgeHits) {
        $w = [double]$e.W
        $mag = [Math]::Min(1.0, [Math]::Abs($w))
        $alpha = [int](40 + 200 * $mag)
        $thick = [single](0.5 + 3.5 * $mag)
        $col = if ($w -ge 0) {
            [System.Drawing.Color]::FromArgb($alpha, 100, 200, 255)
        } else {
            [System.Drawing.Color]::FromArgb($alpha, 255, 120, 120)
        }
        $isWalkEdge = ($null -ne $walkEdge -and [int]$e.Layer -eq [int]$walkEdge.Layer -and [int]$e.Neuron -eq [int]$walkEdge.Neuron -and [int]$e.WIdx -eq [int]$walkEdge.WIdx)
        if ($isWalkEdge) {
            $col = [System.Drawing.Color]::FromArgb(255, 255, 220, 80)
            $thick = [single]([Math]::Max($thick, 4.0))
        }
        $pen.Color = $col
        $pen.Width = $thick
        $g.DrawLine($pen, [single]$e.X1, [single]$e.Y1, [single]$e.X2, [single]$e.Y2)
    }
    $pen.Dispose()

    $font = New-Object System.Drawing.Font('Segoe UI', 8)
    $brushText = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::White)
    $sf = New-Object System.Drawing.StringFormat
    $sf.Alignment = [System.Drawing.StringAlignment]::Center
    $sf.LineAlignment = [System.Drawing.StringAlignment]::Center

    $flashing = ($null -ne $script:flashKey -and [datetime]::UtcNow -lt $script:flashUntil)

    for ($i = 0; $i -lt $script:inputFeatures; $i++) {
        $pt = $positions["I,$i"]
        $ikey = "I,$i"
        $act = 0.0
        if ($script:lastActivations.ContainsKey($ikey)) { $act = $script:lastActivations[$ikey] }
        $fill = Get-ActivationColor $act
        $brush = New-Object System.Drawing.SolidBrush($fill)
        $g.FillEllipse($brush, ($pt.X - $radius), ($pt.Y - $radius), (2 * $radius), (2 * $radius))
        $brush.Dispose()
        if ($script:ghostActivations.ContainsKey($ikey) -and [Math]::Abs([float]$script:ghostActivations[$ikey] - [float]$act) -gt 1e-4) {
            $ghostPen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(160, 180, 180, 180), 2)
            $ghostPen.DashStyle = [System.Drawing.Drawing2D.DashStyle]::Dash
            $g.DrawEllipse($ghostPen, ($pt.X - $radius - 6), ($pt.Y - $radius - 6), (2 * $radius + 12), (2 * $radius + 12))
            $ghostPen.Dispose()
        }
        $isWalk = ($script:walkHighlightKey -eq $ikey)
        if ($isWalk) {
            $walkPen = New-Object System.Drawing.Pen([System.Drawing.Color]::Cyan, 3)
            $g.DrawEllipse($walkPen, ($pt.X - $radius - 4), ($pt.Y - $radius - 4), (2 * $radius + 8), (2 * $radius + 8))
            $walkPen.Dispose()
        }
        $g.DrawEllipse([System.Drawing.Pens]::WhiteSmoke, ($pt.X - $radius), ($pt.Y - $radius), (2 * $radius), (2 * $radius))
        $g.DrawString(("I{0}`n{1:F2}" -f $i, $act), $font, $brushText, (New-Object System.Drawing.RectangleF(($pt.X - 28), ($pt.Y - 28), 56, 56)), $sf)
    }
    for ($L = 0; $L -lt $script:nn.Network.Count; $L++) {
        for ($n = 0; $n -lt $script:nn.Network[$L].Count; $n++) {
            $pt = $positions["$L,$n"]
            $act = [float]$script:nn.Network[$L][$n].Value
            if ($script:lastActivations.ContainsKey("$L,$n")) { $act = $script:lastActivations["$L,$n"] }
            $fill = Get-ActivationColor $act
            $brush = New-Object System.Drawing.SolidBrush($fill)
            $g.FillEllipse($brush, ($pt.X - $radius), ($pt.Y - $radius), (2 * $radius), (2 * $radius))
            $brush.Dispose()
            $nkey = "$L,$n"
            $selected = ($L -eq $script:selectedLayer -and $n -eq $script:selectedIndex)
            $isFlash = ($flashing -and $script:flashKey -eq $nkey)
            $isWalk = ($script:walkHighlightKey -eq $nkey)
            if ($script:ghostActivations.ContainsKey($nkey) -and [Math]::Abs([float]$script:ghostActivations[$nkey] - [float]$act) -gt 1e-4) {
                $ghostPen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(160, 180, 180, 180), 2)
                $ghostPen.DashStyle = [System.Drawing.Drawing2D.DashStyle]::Dash
                $g.DrawEllipse($ghostPen, ($pt.X - $radius - 6), ($pt.Y - $radius - 6), (2 * $radius + 12), (2 * $radius + 12))
                $ghostPen.Dispose()
            }
            if ($isWalk) {
                $walkPen = New-Object System.Drawing.Pen([System.Drawing.Color]::Cyan, 3)
                $g.DrawEllipse($walkPen, ($pt.X - $radius - 4), ($pt.Y - $radius - 4), (2 * $radius + 8), (2 * $radius + 8))
                $walkPen.Dispose()
            }
            if ($isFlash) {
                $flashPen = New-Object System.Drawing.Pen([System.Drawing.Color]::DeepSkyBlue, 4)
                $g.DrawEllipse($flashPen, ($pt.X - $radius - 3), ($pt.Y - $radius - 3), (2 * $radius + 6), (2 * $radius + 6))
                $flashPen.Dispose()
            }
            $edgePen = if ($selected) { New-Object System.Drawing.Pen([System.Drawing.Color]::Yellow, 3) } else { [System.Drawing.Pens]::WhiteSmoke }
            $g.DrawEllipse($edgePen, ($pt.X - $radius), ($pt.Y - $radius), (2 * $radius), (2 * $radius))
            if ($selected) { $edgePen.Dispose() }
            $tag = if ($L -eq $script:nn.Network.Count - 1) { 'O' } else { "H$L" }
            $g.DrawString(("{0}:{1}`n{2:F2}" -f $tag, $n, $act), $font, $brushText, (New-Object System.Drawing.RectangleF(($pt.X - 30), ($pt.Y - 28), 60, 56)), $sf)
        }
    }
    $font.Dispose()
    $brushText.Dispose()
    $sf.Dispose()

    $leg = New-Object System.Drawing.Font('Segoe UI', 8)
    $sampleTxt = Format-SampleLabel -Index $script:vizSampleIndex
    $walkTag = if ($script:walkArmed) { ' | WALK' } else { '' }
    $g.DrawString(("$($script:demoName) | blue=w>0 red=w<0 | node=activation | viz $sampleTxt | click node/edge$walkTag"), $leg,
        [System.Drawing.Brushes]::Gray, 8, ($height - 22))
    $leg.Dispose()
}

function Draw-LossChart {
    param($g, $width, $height)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $g.Clear([System.Drawing.Color]::FromArgb(255, 24, 24, 28))
    $hist = @()
    if ($null -ne $script:nn -and $null -ne $script:nn.LossHistory) {
        $hist = @($script:nn.LossHistory)
    }
    $font = New-Object System.Drawing.Font('Segoe UI', 8)
    $g.DrawString('Loss / epoch  (click for tip)', $font, [System.Drawing.Brushes]::LightGray, 4, 2)
    if ($hist.Count -lt 1) {
        $script:lossSlice = @()
        $g.DrawString('(train to populate)', $font, [System.Drawing.Brushes]::DimGray, 4, 22)
        $font.Dispose()
        return
    }
    $maxShow = [Math]::Min(200, $hist.Count)
    $slice = $hist[($hist.Count - $maxShow)..($hist.Count - 1)]
    $script:lossSlice = @($slice)
    $script:lossSliceStartEpoch = $script:epoch - $slice.Count + 1
    if ($script:lossSliceStartEpoch -lt 1) { $script:lossSliceStartEpoch = 1 }
    $minV = ($slice | Measure-Object -Minimum).Minimum
    $maxV = ($slice | Measure-Object -Maximum).Maximum
    if ($maxV -le $minV) { $maxV = $minV + 1e-6 }
    $padL = 8; $padR = 8; $padT = 20; $padB = 8
    $plotW = [Math]::Max(10, $width - $padL - $padR)
    $plotH = [Math]::Max(10, $height - $padT - $padB)
    $pen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(255, 80, 200, 120), 1.5)
    $pts = New-Object 'System.Collections.Generic.List[System.Drawing.PointF]'
    for ($i = 0; $i -lt $slice.Count; $i++) {
        $x = $padL + ($plotW * $i / [Math]::Max(1, $slice.Count - 1))
        $norm = ([double]$slice[$i] - $minV) / ($maxV - $minV)
        $y = $padT + $plotH * (1.0 - $norm)
        [void]$pts.Add((New-Object System.Drawing.PointF([single]$x, [single]$y)))
    }
    if ($pts.Count -ge 2) {
        $g.DrawLines($pen, $pts.ToArray())
    }
    elseif ($pts.Count -eq 1) {
        $g.FillEllipse([System.Drawing.Brushes]::LimeGreen, ($pts[0].X - 2), ($pts[0].Y - 2), 4, 4)
    }
    $last = [float]$slice[-1]
    $g.DrawString(("{0:F4}  (n={1})" -f $last, $hist.Count), $font, [System.Drawing.Brushes]::WhiteSmoke, ($width - 120), 2)
    $pen.Dispose()
    $font.Dispose()
}

function Refresh-AllUi {
    param([switch]$RebuildWeights)
    Update-StatusLabels
    Refresh-LayerCombo
    Refresh-NeuronCombo
    if ($RebuildWeights) { Load-WeightEditors } else { Sync-WeightEditorsLive }
    $canvas.Invalidate()
    $lossChart.Invalidate()
}

function Apply-ForwardViz {
    if ($null -eq $script:nn) { return }
    $vizIn = Get-VizInput
    $null = $script:nn.GetOutput($vizIn)
    Capture-Activations -InputValues $vizIn
}

# --- UI ---
$form = New-Object System.Windows.Forms.Form
$form.Text = 'PSNeuron Educational GUI v1.2'
$form.Size = New-Object System.Drawing.Size(1100, 760)
$form.StartPosition = 'CenterScreen'
$form.MinimumSize = New-Object System.Drawing.Size(900, 640)
$form.KeyPreview = $true

$toolTips = New-Object System.Windows.Forms.ToolTip
$toolTips.AutoPopDelay = 10000
$toolTips.InitialDelay = 350
$toolTips.ReshowDelay = 150
$toolTips.ShowAlways = $true

$status = New-Object System.Windows.Forms.StatusStrip
$status.SizingGrip = $false
$lblTip = New-Object System.Windows.Forms.ToolStripStatusLabel
$lblTip.Spring = $true
$lblTip.TextAlign = [System.Drawing.ContentAlignment]::MiddleLeft
$lblTip.ForeColor = [System.Drawing.Color]::Gainsboro
$lblTip.Text = 'Ready - click Help for graph legend & shortcuts.'
[void]$status.Items.Add($lblTip)

$splitMain = New-Object System.Windows.Forms.SplitContainer
$splitMain.Dock = 'Fill'
$splitMain.Orientation = 'Horizontal'
$splitMain.SplitterDistance = 520
$splitMain.Panel2MinSize = 90

$canvas = New-Object System.Windows.Forms.Panel
$canvas.Dock = 'Fill'
$canvas.BackColor = [System.Drawing.Color]::FromArgb(255, 30, 30, 36)
$canvas.TabStop = $true
$canvas.Add_Paint({
    param($sender, $e)
    Draw-Network -g $e.Graphics -width $sender.ClientSize.Width -height $sender.ClientSize.Height
})

$lossChart = New-Object System.Windows.Forms.Panel
$lossChart.Dock = 'Fill'
$lossChart.BackColor = [System.Drawing.Color]::FromArgb(255, 24, 24, 28)
$lossChart.Add_Paint({
    param($sender, $e)
    Draw-LossChart -g $e.Graphics -width $sender.ClientSize.Width -height $sender.ClientSize.Height
})

$splitMain.Panel1.Controls.Add($canvas)
$splitMain.Panel2.Controls.Add($lossChart)

$right = New-Object System.Windows.Forms.Panel
$right.Dock = 'Right'
$right.Width = 360
$right.AutoScroll = $true
$right.Padding = New-Object System.Windows.Forms.Padding(8)

$y = 8
function Add-Right([System.Windows.Forms.Control]$c) {
    $right.Controls.Add($c)
}

$lblTitle = New-Object System.Windows.Forms.Label
$lblTitle.Text = 'Controls'
$lblTitle.Font = New-Object System.Drawing.Font('Segoe UI', 11, [System.Drawing.FontStyle]::Bold)
$lblTitle.Location = New-Object System.Drawing.Point(8, $y)
$lblTitle.AutoSize = $true
Add-Right $lblTitle
$btnHelp = New-Object System.Windows.Forms.Button
$btnHelp.Text = 'Help'
$btnHelp.Location = New-Object System.Drawing.Point(260, ($y - 2))
$btnHelp.Size = New-Object System.Drawing.Size(60, 26)
Add-Right $btnHelp
$y += 28

# Demo
$lblDemo = New-Object System.Windows.Forms.Label
$lblDemo.Text = 'Demo:'
$lblDemo.Location = New-Object System.Drawing.Point(8, ($y + 3))
$lblDemo.AutoSize = $true
$cmbDemo = New-Object System.Windows.Forms.ComboBox
$cmbDemo.DropDownStyle = 'DropDownList'
$cmbDemo.Location = New-Object System.Drawing.Point(70, $y)
$cmbDemo.Size = New-Object System.Drawing.Size(250, 24)
[void]$cmbDemo.Items.Add('XOR')
[void]$cmbDemo.Items.Add('SineForecast')
$cmbDemo.SelectedIndex = 0
Add-Right $lblDemo; Add-Right $cmbDemo
$y += 30

# Activation
$lblAct = New-Object System.Windows.Forms.Label
$lblAct.Text = 'Activation:'
$lblAct.Location = New-Object System.Drawing.Point(8, ($y + 3))
$lblAct.AutoSize = $true
$cmbAct = New-Object System.Windows.Forms.ComboBox
$cmbAct.DropDownStyle = 'DropDownList'
$cmbAct.Location = New-Object System.Drawing.Point(90, $y)
$cmbAct.Size = New-Object System.Drawing.Size(120, 24)
[void]$cmbAct.Items.Add('Sigmoid')
[void]$cmbAct.Items.Add('ReLU')
[void]$cmbAct.Items.Add('tanh')
$cmbAct.SelectedIndex = 0
Add-Right $lblAct; Add-Right $cmbAct
$y += 30

# Learning rate
$lblLr = New-Object System.Windows.Forms.Label
$lblLr.Text = 'Learning rate:'
$lblLr.Location = New-Object System.Drawing.Point(8, ($y + 3))
$lblLr.AutoSize = $true
$nudLr = New-Object System.Windows.Forms.NumericUpDown
$nudLr.DecimalPlaces = 3
$nudLr.Minimum = 0.001
$nudLr.Maximum = 5
$nudLr.Increment = 0.05
$nudLr.Value = [decimal]0.5
$nudLr.Location = New-Object System.Drawing.Point(110, $y)
$nudLr.Size = New-Object System.Drawing.Size(80, 22)
Add-Right $lblLr; Add-Right $nudLr
$y += 32

$btnStart = New-Object System.Windows.Forms.Button
$btnStart.Text = 'Start'
$btnStart.Location = New-Object System.Drawing.Point(8, $y)
$btnStart.Size = New-Object System.Drawing.Size(70, 28)
$btnStop = New-Object System.Windows.Forms.Button
$btnStop.Text = 'Stop'
$btnStop.Location = New-Object System.Drawing.Point(84, $y)
$btnStop.Size = New-Object System.Drawing.Size(70, 28)
$btnStop.Enabled = $false
$btnStep = New-Object System.Windows.Forms.Button
$btnStep.Text = 'Step'
$btnStep.Location = New-Object System.Drawing.Point(160, $y)
$btnStep.Size = New-Object System.Drawing.Size(70, 28)
$btnPulse = New-Object System.Windows.Forms.Button
$btnPulse.Text = 'Pulse'
$btnPulse.Location = New-Object System.Drawing.Point(236, $y)
$btnPulse.Size = New-Object System.Drawing.Size(80, 28)
Add-Right $btnStart; Add-Right $btnStop; Add-Right $btnStep; Add-Right $btnPulse
$y += 32

$chkWalk = New-Object System.Windows.Forms.CheckBox
$chkWalk.Text = 'Walk inputs (slow-mo left->right)'
$chkWalk.Location = New-Object System.Drawing.Point(8, $y)
$chkWalk.Size = New-Object System.Drawing.Size(310, 22)
$chkWalk.ForeColor = [System.Drawing.Color]::Gainsboro
Add-Right $chkWalk
$y += 26

$btnReset = New-Object System.Windows.Forms.Button
$btnReset.Text = 'Reset net'
$btnReset.Location = New-Object System.Drawing.Point(8, $y)
$btnReset.Size = New-Object System.Drawing.Size(148, 28)
$btnAddLayer = New-Object System.Windows.Forms.Button
$btnAddLayer.Text = 'Add hidden layer'
$btnAddLayer.Location = New-Object System.Drawing.Point(166, $y)
$btnAddLayer.Size = New-Object System.Drawing.Size(148, 28)
Add-Right $btnReset; Add-Right $btnAddLayer
$y += 36

$lblAdd = New-Object System.Windows.Forms.Label
$lblAdd.Text = 'Add neuron to:'
$lblAdd.Location = New-Object System.Drawing.Point(8, ($y + 3))
$lblAdd.AutoSize = $true
$cmbAddLayer = New-Object System.Windows.Forms.ComboBox
$cmbAddLayer.DropDownStyle = 'DropDownList'
$cmbAddLayer.FormattingEnabled = $true
$cmbAddLayer.Location = New-Object System.Drawing.Point(110, $y)
$cmbAddLayer.Size = New-Object System.Drawing.Size(120, 24)
$cmbAddLayer.Add_Format({
    param($s, $e)
    if ($e.ListItem) { $e.Value = ([string]$e.ListItem).Split('|')[0] }
})
$btnAdd = New-Object System.Windows.Forms.Button
$btnAdd.Text = 'Add neuron'
$btnAdd.Location = New-Object System.Drawing.Point(236, $y)
$btnAdd.Size = New-Object System.Drawing.Size(90, 26)
Add-Right $lblAdd; Add-Right $cmbAddLayer; Add-Right $btnAdd
$y += 34

$lblEpoch = New-Object System.Windows.Forms.Label
$lblEpoch.Location = New-Object System.Drawing.Point(8, $y)
$lblEpoch.Size = New-Object System.Drawing.Size(320, 18)
$lblEpoch.Text = 'Epoch: 0'
Add-Right $lblEpoch
$y += 20

$lblLoss = New-Object System.Windows.Forms.Label
$lblLoss.Location = New-Object System.Drawing.Point(8, $y)
$lblLoss.Size = New-Object System.Drawing.Size(320, 18)
$lblLoss.Text = 'Loss (MSE): 0'
Add-Right $lblLoss
$y += 18

$lblLossExplain = New-Object System.Windows.Forms.Label
$lblLossExplain.Location = New-Object System.Drawing.Point(8, $y)
$lblLossExplain.Size = New-Object System.Drawing.Size(330, 32)
$lblLossExplain.ForeColor = [System.Drawing.Color]::Silver
$lblLossExplain.Text = 'Loss = average "how wrong" the guesses are. Training nudges weights toward 0.0000.'
Add-Right $lblLossExplain
$y += 34

$lblTeach = New-Object System.Windows.Forms.Label
$lblTeach.Location = New-Object System.Drawing.Point(8, $y)
$lblTeach.Size = New-Object System.Drawing.Size(330, 72)
$lblTeach.BorderStyle = 'FixedSingle'
$lblTeach.BackColor = [System.Drawing.Color]::FromArgb(255, 28, 28, 34)
$lblTeach.ForeColor = [System.Drawing.Color]::Gainsboro
$lblTeach.Text = "Teach panel: Step/Start narrates each epoch. Walk inputs slow-walks one sample. Click a weight/edge to see what it does."
Add-Right $lblTeach
$y += 78

$lblLayout = New-Object System.Windows.Forms.Label
$lblLayout.Location = New-Object System.Drawing.Point(8, $y)
$lblLayout.Size = New-Object System.Drawing.Size(330, 36)
$lblLayout.Text = 'Layout:'
Add-Right $lblLayout
$y += 40

$lblSample = New-Object System.Windows.Forms.Label
$lblSample.Text = 'Viz sample (forward heat):'
$lblSample.Location = New-Object System.Drawing.Point(8, $y)
$lblSample.AutoSize = $true
Add-Right $lblSample
$y += 20

$cmbSample = New-Object System.Windows.Forms.ComboBox
$cmbSample.DropDownStyle = 'DropDownList'
$cmbSample.Location = New-Object System.Drawing.Point(8, $y)
$cmbSample.Size = New-Object System.Drawing.Size(320, 24)
Add-Right $cmbSample
$y += 30

$lblPick = New-Object System.Windows.Forms.Label
$lblPick.Text = 'Select neuron (edit weights live):'
$lblPick.Location = New-Object System.Drawing.Point(8, $y)
$lblPick.AutoSize = $true
Add-Right $lblPick
$y += 20

$cmbNeuron = New-Object System.Windows.Forms.ComboBox
$cmbNeuron.DropDownStyle = 'DropDownList'
$cmbNeuron.Location = New-Object System.Drawing.Point(8, $y)
$cmbNeuron.Size = New-Object System.Drawing.Size(320, 24)
$cmbNeuron.FormattingEnabled = $true
$cmbNeuron.Add_Format({
    param($s, $e)
    if ($e.ListItem) { $e.Value = ([string]$e.ListItem).Split('|')[0] }
})
Add-Right $cmbNeuron
$y += 28

$pnlWeights = New-Object System.Windows.Forms.Panel
$pnlWeights.Location = New-Object System.Drawing.Point(8, $y)
$pnlWeights.Size = New-Object System.Drawing.Size(320, 160)
$pnlWeights.Anchor = 'Top,Left,Right'
$pnlWeights.AutoScroll = $true
$pnlWeights.BorderStyle = 'FixedSingle'
Add-Right $pnlWeights
$y += 168

$lblHint = New-Object System.Windows.Forms.Label
$lblHint.Text = "Limits: <=$($script:MaxHiddenLayers) hidden, <=$($script:MaxNeuronsPerHidden)/layer; Out=1. Space/Esc/R shortcuts. Click graph nodes & edges. See Help."
$lblHint.Location = New-Object System.Drawing.Point(8, $y)
$lblHint.Size = New-Object System.Drawing.Size(330, 56)
$lblHint.ForeColor = [System.Drawing.Color]::DimGray
Add-Right $lblHint

# ToolTips on major controls
$toolTips.SetToolTip($cmbDemo, 'Demo task: XOR (2 inputs) or SineForecast (window->next value). Changing demo rebuilds the net.')
$toolTips.SetToolTip($lblDemo, 'Demo task: XOR or SineForecast.')
$toolTips.SetToolTip($cmbAct, 'Activation applied per neuron. Sigmoid is safest for XOR; ReLU can stall on XOR.')
$toolTips.SetToolTip($lblAct, 'Neuron activation function.')
$toolTips.SetToolTip($nudLr, 'Learning rate (step size). Live via setter. High values (>~1.5) may overshoot.')
$toolTips.SetToolTip($lblLr, 'How large each weight update is.')
$toolTips.SetToolTip($btnStart, 'Start continuous training (one epoch per timer tick). Esc or Stop to halt.')
$toolTips.SetToolTip($btnStop, 'Stop continuous training. Esc also stops.')
$toolTips.SetToolTip($btnStep, 'Run exactly one training epoch. Shortcut: Space.')
$toolTips.SetToolTip($btnPulse, 'Forward-pass the current viz sample and flash the selected (or output) node.')
$toolTips.SetToolTip($btnReset, 'Re-initialize weights (same layout). Shortcut: R.')
$toolTips.SetToolTip($btnAddLayer, 'Append a hidden layer (max 3). Overlapping weights are transplanted.')
$toolTips.SetToolTip($btnAdd, 'Add a neuron to the chosen hidden layer. Overlapping weights transplanted; new links random.')
$toolTips.SetToolTip($cmbAddLayer, 'Which hidden layer receives the new neuron.')
$toolTips.SetToolTip($cmbSample, 'Training row used for forward-pass heat on the graph. Changing it updates node colors.')
$toolTips.SetToolTip($lblSample, 'Viz sample drives node activation colors.')
$toolTips.SetToolTip($cmbNeuron, 'Pick a neuron to edit its incoming weights/bias. Yellow ring marks selection on the graph.')
$toolTips.SetToolTip($lblPick, 'Neuron picker for live weight editing.')
$toolTips.SetToolTip($pnlWeights, 'Incoming weights + bias for the selected neuron. Synced live while training.')
$toolTips.SetToolTip($canvas, 'Graph: inputs->hidden->out. Click node=select, edge~weight, double-click/Pulse=forward, right-click=menu.')
$toolTips.SetToolTip($lossChart, "MSE per epoch. Click/scrub for that epoch's loss tip.")
$toolTips.SetToolTip($btnHelp, 'Short help: graph meaning, tips, keyboard shortcuts.')
$toolTips.SetToolTip($lblHint, 'Quick limits reminder - open Help for the full legend.')
$toolTips.SetToolTip($lblEpoch, 'Completed training epochs.')
$toolTips.SetToolTip($lblLoss, 'Mean squared error on the full demo set.')
$toolTips.SetToolTip($lblLayout, 'Current topology, learning rate, and activation.')
$toolTips.SetToolTip($lblLossExplain, 'Plain English: loss is average how-wrong; 0.0000 means predictions match targets on this demo - not money.')
$toolTips.SetToolTip($lblTeach, 'Living teach panel: epoch narrator, weight impact, and Walk-inputs narration.')
$toolTips.SetToolTip($chkWalk, 'Slow-walk the current viz sample through the net (input -> edges -> neurons -> output) with pauses and tip text.')

$form.Controls.Add($splitMain)
$form.Controls.Add($right)
$form.Controls.Add($status)

# Context menu for neurons
$ctxNeuron = New-Object System.Windows.Forms.ContextMenuStrip
$miSelect = New-Object System.Windows.Forms.ToolStripMenuItem('Select for weight editing')
$miRand = New-Object System.Windows.Forms.ToolStripMenuItem("Randomize this neuron's weights")
$miZero = New-Object System.Windows.Forms.ToolStripMenuItem('Zero bias')
[void]$ctxNeuron.Items.Add($miSelect)
[void]$ctxNeuron.Items.Add($miRand)
[void]$ctxNeuron.Items.Add($miZero)

$timer = New-Object System.Windows.Forms.Timer
$timer.Interval = 40

$flashTimer = New-Object System.Windows.Forms.Timer
$flashTimer.Interval = 50
$flashTimer.Add_Tick({
    if ([datetime]::UtcNow -ge $script:flashUntil) {
        $script:flashKey = $null
        $flashTimer.Stop()
    }
    $canvas.Invalidate()
})

$walkTimer = New-Object System.Windows.Forms.Timer
$walkTimer.Interval = 850
$walkTimer.Add_Tick({
    if (-not $script:walkArmed) { $walkTimer.Stop(); return }
    Advance-WalkStep
})

$timer.Add_Tick({
    if (-not $script:running) { return }
    Invoke-OneEpoch
    Update-StatusLabels
    $canvas.Invalidate()
    $lossChart.Invalidate()
    $script:weightSyncTick++
    # Live weight panel sync every tick (skips when user is dragging/editing)
    Sync-WeightEditorsLive
})

function Stop-TrainingUi {
    $script:running = $false
    [void]$timer.Stop()
    $btnStart.Enabled = $true
    $btnStop.Enabled = $false
    return
}

$btnStart.Add_Click({
    if ($script:walkArmed) { Stop-WalkUi }
    $script:running = $true
    $btnStart.Enabled = $false
    $btnStop.Enabled = $true
    $timer.Start()
    Set-TipLine -Level ok -Text 'Training... watch the teach panel / loss strip. Each epoch tries all quiz rows and nudges weights. Stop or Esc anytime.'
    Set-TeachPanel -Level ok -Text 'Training started. Loss is average how-wrong. We nudge weights each epoch to push loss toward 0.0000 (matching the quiz answers).'
})

$btnStop.Add_Click({
    Stop-TrainingUi
    Load-WeightEditors
    $canvas.Invalidate()
    $lossChart.Invalidate()
    Update-LossExplainLabel
    Set-TipLine -Text 'Stopped. Step for one epoch, Walk inputs for a slow tour, or Pulse to refresh activations.'
    Set-TeachPanel -Text ("Stopped at epoch {0}, loss {1:F4}. Loss->0 means guesses match targets on this demo (not money)." -f $script:epoch, $script:loss)
})

$btnStep.Add_Click({
    if ($script:running) { return }
    if ($script:walkArmed) { Stop-WalkUi }
    Invoke-OneEpoch
    Refresh-AllUi -RebuildWeights
    # Narrator already set tip/teach inside Invoke-OneEpoch
})

$btnPulse.Add_Click({ Invoke-PulseForward })
$btnHelp.Add_Click({ Show-PSNeuronHelp })

$chkWalk.Add_CheckedChanged({
    if ($script:suppressEvents) { return }
    if ($chkWalk.Checked) {
        Start-WalkUi
    }
    else {
        if ($script:walkArmed) {
            if ($walkTimer) { $walkTimer.Stop() }
            $script:walkArmed = $false
            $script:walkSteps = @()
            $script:walkIndex = -1
            $script:walkHighlightKey = $null
            $script:walkHighlightEdge = $null
            Set-TipLine -Text 'Walk cleared. Toggle again to restart, or Step/Start to train.'
            Set-TeachPanel -Text 'Walk cleared. Loss = average how-wrong; weights are dials that change how strongly neurons listen.'
            $canvas.Invalidate()
        }
    }
})

$btnAdd.Add_Click({
    Stop-TrainingUi
    $layerIdx = 0
    if ($cmbAddLayer.SelectedItem) {
        $parts = [string]$cmbAddLayer.SelectedItem -split '\|'
        if ($parts.Count -ge 2) { $layerIdx = [int]$parts[1] }
    }
    if ($layerIdx -lt 0 -or $layerIdx -ge $script:layoutHidden.Count) { $layerIdx = 0 }
    if ($script:layoutHidden[$layerIdx] -ge $script:MaxNeuronsPerHidden) {
        [System.Windows.Forms.MessageBox]::Show(
            "Hidden layer H$layerIdx already at max $($script:MaxNeuronsPerHidden) neurons.",
            'PSNeuron GUI') | Out-Null
        return
    }
    $newHidden = [int[]]@($script:layoutHidden)
    $newHidden[$layerIdx] = $newHidden[$layerIdx] + 1
    New-DemoNetwork -HiddenLayout $newHidden -PreserveWeights
    Apply-ForwardViz
    $script:loss = Get-NetworkMse
    Refresh-SampleCombo
    Refresh-AllUi -RebuildWeights
    Set-TipLine -Level ok -Text 'Added neuron - overlapping weights transplanted; new connections stay random-init.'
})

$btnAddLayer.Add_Click({
    Stop-TrainingUi
    if ($script:layoutHidden.Count -ge $script:MaxHiddenLayers) {
        [System.Windows.Forms.MessageBox]::Show(
            "Max hidden layers is $($script:MaxHiddenLayers) (layout like @(H0,H1,H2,Out)).",
            'PSNeuron GUI') | Out-Null
        return
    }
    $newHidden = [int[]](@($script:layoutHidden) + @(2))
    New-DemoNetwork -HiddenLayout $newHidden -PreserveWeights
    Apply-ForwardViz
    $script:loss = Get-NetworkMse
    Refresh-SampleCombo
    Refresh-AllUi -RebuildWeights
    Set-TipLine -Level ok -Text 'Added hidden layer - prior layers transplanted where indices overlap; new layer random.'
})

$btnReset.Add_Click({
    Stop-TrainingUi
    if ($script:walkArmed) { Stop-WalkUi }
    New-DemoNetwork -HiddenLayout $script:layoutHidden
    $script:prevLoss = $null
    $script:ghostActivations = @{}
    Apply-ForwardViz
    $script:loss = Get-NetworkMse
    Refresh-SampleCombo
    Refresh-AllUi -RebuildWeights
    Set-TipLine -Text 'Network reset - fresh random weights, epoch/loss cleared.'
    Set-TeachPanel -Text 'Reset: new random dials (weights). Train again and watch loss fall toward 0.0000 as guesses improve.'
})

$cmbNeuron.Add_SelectedIndexChanged({
    if ($script:suppressEvents) { return }
    Load-WeightEditors
    $canvas.Invalidate()
    if ($script:selectedLayer -ge 0 -and $null -ne $script:nn) {
        $L = $script:selectedLayer; $n = $script:selectedIndex
        $layerName = if ($L -eq $script:nn.Network.Count - 1) { 'Out' } else { "H$L" }
        $act = [float]$script:nn.Network[$L][$n].Value
        if ($script:lastActivations.ContainsKey("$L,$n")) { $act = $script:lastActivations["$L,$n"] }
        $bias = [float]$script:nn.Network[$L][$n].GetBias()
        Set-TipLine -Text ("Selected {0}:{1} - act={2:F3}, bias={3:F3}. Sliders edit incoming weights." -f $layerName, $n, $act, $bias)
    }
})

$cmbSample.Add_SelectedIndexChanged({
    if ($script:suppressEvents) { return }
    if ($cmbSample.SelectedIndex -lt 0) { return }
    $maxShow = [Math]::Min(24, $script:trainData.Count)
    if ($cmbSample.SelectedIndex -ge $maxShow) { return }
    $script:vizSampleIndex = $cmbSample.SelectedIndex
    Apply-ForwardViz
    $canvas.Invalidate()
    Set-TipLine -Text ("Viz sample {0} - node colors show this forward pass. Pulse to flash." -f (Format-SampleLabel -Index $script:vizSampleIndex))
})

$cmbDemo.Add_SelectedIndexChanged({
    if ($script:suppressEvents) { return }
    Stop-TrainingUi
    $script:demoName = [string]$cmbDemo.SelectedItem
    Load-DemoData
    # Sensible default layouts per demo
    if ($script:demoName -eq 'SineForecast') {
        $script:layoutHidden = @(6)
    }
    else {
        $script:layoutHidden = @(2)
    }
    New-DemoNetwork -HiddenLayout $script:layoutHidden
    Apply-ForwardViz
    $script:loss = Get-NetworkMse
    Refresh-SampleCombo
    Refresh-AllUi -RebuildWeights
    Set-TipLine -Text ("Demo $($script:demoName) loaded. Start or Step to train; Pulse to visualize one sample.")
    [void](Update-ContextWarnings)
})

$cmbAct.Add_SelectedIndexChanged({
    if ($script:suppressEvents) { return }
    $script:activation = [string]$cmbAct.SelectedItem
    if ($null -eq $script:nn) { return }
    # Live activation change without full rebuild (keeps weights)
    $script:nn.SetActivationFunction($script:activation)
    Apply-ForwardViz
    Update-StatusLabels
    $canvas.Invalidate()
    if (-not (Update-ContextWarnings)) {
        Set-TipLine -Text ("Activation set to $($script:activation) (weights kept).")
    }
})

$nudLr.Add_ValueChanged({
    if ($script:suppressEvents) { return }
    $script:learningRate = [float]$nudLr.Value
    if ($null -ne $script:nn) {
        $script:nn.SetLearningRate([float]$script:learningRate)
    }
    Update-StatusLabels
    if (-not (Update-ContextWarnings)) {
        Set-TipLine -Text ("Learning rate = {0:F3} (live)." -f $script:learningRate)
    }
})


# --- Canvas interactions ---
$canvas.Add_MouseMove({
    param($sender, $e)
    if ($null -eq $script:nn) { return }
    if ($script:nodePositions.Count -lt 1) {
        Build-NetworkGeometry -Width $sender.ClientSize.Width -Height $sender.ClientSize.Height
    }
    $key = Find-NodeAtPoint -X $e.X -Y $e.Y
    if ($null -ne $key) {
        $edgeHoverKey = "N:$key"
        if ($edgeHoverKey -eq $script:lastHoverKey) { return }
        $script:lastHoverKey = $edgeHoverKey
        $script:lastHoverEdgeKey = $null
        $parts = $key -split ','
        if ($parts[0] -eq 'I') {
            $i = [int]$parts[1]
            $act = 0.0
            if ($script:lastActivations.ContainsKey($key)) { $act = $script:lastActivations[$key] }
            $feel = if ($act -ge 0.7) { 'strong signal' } elseif ($act -ge 0.35) { 'medium signal' } elseif ($act -gt 0.05) { 'weak signal' } else { 'near zero' }
            $tip = ("Input I{0}: value={1:F3} ({2}). Flows into the first hidden layer along the edges." -f $i, $act, $feel)
        }
        else {
            $L = [int]$parts[0]; $n = [int]$parts[1]
            $name = Get-NeuronPlainName -Layer $L -Index $n
            $act = [float]$script:nn.Network[$L][$n].Value
            if ($script:lastActivations.ContainsKey($key)) { $act = $script:lastActivations[$key] }
            $bias = [float]$script:nn.Network[$L][$n].GetBias()
            $z = Get-NeuronPreActivation -Layer $L -Index $n
            $feel = if ($act -ge 0.7) { 'firing strongly' } elseif ($act -ge 0.35) { 'medium' } elseif ($act -gt 0.05) { 'weak' } else { 'mostly quiet' }
            $tip = ("{0}: activation={1:F3} ({2}), z~{3:F3}, bias={4:F3}. Activation is what the next layer hears." -f $name, $act, $feel, $z, $bias)
        }
        $toolTips.Show($tip, $canvas, ($e.X + 16), ($e.Y + 16), 3500)
        return
    }
    $edge = Find-EdgeAtPoint -X $e.X -Y $e.Y
    if ($null -ne $edge) {
        $ek = "E:$($edge.Layer),$($edge.Neuron),$($edge.WIdx)"
        if ($ek -eq $script:lastHoverKey) { return }
        $script:lastHoverKey = $ek
        $script:lastHoverEdgeKey = $ek
        $src = Get-SourceLabelForWeight -Layer ([int]$edge.Layer) -WIdx ([int]$edge.WIdx)
        $dest = Get-NeuronPlainName -Layer ([int]$edge.Layer) -Index ([int]$edge.Neuron)
        $w = [double]$edge.W
        # refresh live weight
        try { $w = [double]$script:nn.Network[[int]$edge.Layer][[int]$edge.Neuron].Weights[[int]$edge.WIdx] } catch {}
        $srcAct = Get-SourceActForWeight -Layer ([int]$edge.Layer) -WIdx ([int]$edge.WIdx)
        $contrib = $w * $srcAct
        $push = Get-PushPhrase -Contribution $contrib
        $tip = ("Edge {0}->{1}: weight={2:F3}, source={3:F3}, contribution~{4:F3} ({5})." -f $src, $dest, $w, $srcAct, $contrib, $push)
        $toolTips.Show($tip, $canvas, ($e.X + 16), ($e.Y + 16), 3500)
        return
    }
    if ($null -ne $script:lastHoverKey) {
        $script:lastHoverKey = $null
        $script:lastHoverEdgeKey = $null
        $toolTips.Hide($canvas)
    }
})

$canvas.Add_MouseLeave({
    $script:lastHoverKey = $null
    $toolTips.Hide($canvas)
})

$canvas.Add_MouseClick({
    param($sender, $e)
    if ($null -eq $script:nn) { return }
    Build-NetworkGeometry -Width $sender.ClientSize.Width -Height $sender.ClientSize.Height
    if ($e.Button -eq [System.Windows.Forms.MouseButtons]::Right) {
        $key = Find-NodeAtPoint -X $e.X -Y $e.Y
        if ($null -eq $key) { return }
        $parts = $key -split ','
        if ($parts[0] -eq 'I') {
            Set-TipLine -Text 'Input nodes have no trainable weights - pick a hidden/output neuron.'
            return
        }
        $script:ctxMenuLayer = [int]$parts[0]
        $script:ctxMenuIndex = [int]$parts[1]
        $ctxNeuron.Show($canvas, $e.Location)
        return
    }
    if ($e.Button -ne [System.Windows.Forms.MouseButtons]::Left) { return }
    $key = Find-NodeAtPoint -X $e.X -Y $e.Y
    if ($null -ne $key) {
        $parts = $key -split ','
        if ($parts[0] -eq 'I') {
            Set-TipLine -Text ("Input I{0} - feeds the first hidden layer; select a neuron to edit weights." -f [int]$parts[1])
            return
        }
        Select-NetworkNeuron -Layer ([int]$parts[0]) -Index ([int]$parts[1])
        return
    }
    $edge = Find-EdgeAtPoint -X $e.X -Y $e.Y
    if ($null -ne $edge) {
        $impact = Get-WeightImpactTip -Layer ([int]$edge.Layer) -Neuron ([int]$edge.Neuron) -WIdx ([int]$edge.WIdx)
        Select-NetworkNeuron -Layer ([int]$edge.Layer) -Index ([int]$edge.Neuron) -Reason $impact
        Focus-WeightSlider -WeightIndex ([int]$edge.WIdx)
    }
})

$canvas.Add_MouseDoubleClick({
    param($sender, $e)
    if ($e.Button -ne [System.Windows.Forms.MouseButtons]::Left) { return }
    if ($null -eq $script:nn) { return }
    Build-NetworkGeometry -Width $sender.ClientSize.Width -Height $sender.ClientSize.Height
    $key = Find-NodeAtPoint -X $e.X -Y $e.Y
    if ($null -ne $key) {
        $parts = $key -split ','
        if ($parts[0] -ne 'I') {
            Select-NetworkNeuron -Layer ([int]$parts[0]) -Index ([int]$parts[1])
        }
        $script:flashKey = $key
        $script:flashUntil = [datetime]::UtcNow.AddMilliseconds(400)
        $flashTimer.Start()
    }
    Invoke-PulseForward
})

$miSelect.Add_Click({
    if ($script:ctxMenuLayer -ge 0) {
        Select-NetworkNeuron -Layer $script:ctxMenuLayer -Index $script:ctxMenuIndex
    }
})
$miRand.Add_Click({
    if ($script:ctxMenuLayer -lt 0) { return }
    Stop-TrainingUi
    Randomize-NeuronWeights -Layer $script:ctxMenuLayer -Index $script:ctxMenuIndex
    Select-NetworkNeuron -Layer $script:ctxMenuLayer -Index $script:ctxMenuIndex `
        -Reason 'Randomized this neuron''s weights (incl. bias) in [-1,1).'
    Apply-ForwardViz
    $canvas.Invalidate()
})
$miZero.Add_Click({
    if ($script:ctxMenuLayer -lt 0) { return }
    Zero-NeuronBias -Layer $script:ctxMenuLayer -Index $script:ctxMenuIndex
    Select-NetworkNeuron -Layer $script:ctxMenuLayer -Index $script:ctxMenuIndex `
        -Reason 'Bias set to 0 for this neuron.'
    Apply-ForwardViz
    $canvas.Invalidate()
})

$lossChart.Add_MouseClick({
    param($sender, $e)
    $info = Get-LossTipAtX -X $e.X -Width $sender.ClientSize.Width
    if ($null -eq $info) {
        Set-TipLine -Text 'Loss strip empty - train a few epochs first.'
        return
    }
    Set-TipLine -Text ("Epoch ~{0}: loss (average how-wrong) = {1:F6}. Lower is closer to matching the quiz answers." -f $info.Epoch, $info.Loss)
})
$lossChart.Add_MouseMove({
    param($sender, $e)
    if ($e.Button -ne [System.Windows.Forms.MouseButtons]::Left) { return }
    $info = Get-LossTipAtX -X $e.X -Width $sender.ClientSize.Width
    if ($null -ne $info) {
        Set-TipLine -Text ("Epoch ~{0}: loss (average how-wrong) = {1:F6}. Lower is closer to matching the quiz answers." -f $info.Epoch, $info.Loss)
    }
})

$form.Add_KeyDown({
    param($sender, $e)
    if (Test-KeyboardBlocked) { return }
    switch ($e.KeyCode) {
        'Space' {
            if (-not $script:running) {
                if ($script:walkArmed) { Stop-WalkUi }
                Invoke-OneEpoch
                Refresh-AllUi -RebuildWeights
            }
            $e.Handled = $true
            $e.SuppressKeyPress = $true
        }
        'Escape' {
            if ($script:walkArmed) {
                Stop-WalkUi
                Set-TipLine -Text 'Walk stopped (Esc).'
                Set-TeachPanel -Text 'Walk stopped.'
                $canvas.Invalidate()
            }
            elseif ($script:running) {
                Stop-TrainingUi
                Load-WeightEditors
                $canvas.Invalidate()
                $lossChart.Invalidate()
                Set-TipLine -Text 'Stopped (Esc).'
            }
            $e.Handled = $true
        }
        'R' {
            if (-not $script:running) {
                New-DemoNetwork -HiddenLayout $script:layoutHidden
                Apply-ForwardViz
                $script:loss = Get-NetworkMse
                Refresh-SampleCombo
                Refresh-AllUi -RebuildWeights
                Set-TipLine -Text 'Network reset (R).'
            }
            $e.Handled = $true
        }
    }
})

$form.Add_FormClosing({
    $script:running = $false
    $script:walkArmed = $false
    $timer.Stop()
    $flashTimer.Stop()
    if ($walkTimer) { $walkTimer.Stop(); $walkTimer.Dispose() }
    $timer.Dispose()
    $flashTimer.Dispose()
    $toolTips.Dispose()
})

$form.Add_Shown({
    try { $splitMain.SplitterDistance = [Math]::Max(200, $form.ClientSize.Height - 160) } catch {}
})

# Init
Load-DemoData
New-DemoNetwork -HiddenLayout @(2)
Apply-ForwardViz
$script:loss = Get-NetworkMse
Refresh-SampleCombo
Refresh-LayerCombo
Refresh-NeuronCombo
Update-StatusLabels
Load-WeightEditors
Update-LossExplainLabel
Set-TipLine -Text 'Ready - Start/Step train with an epoch narrator; Walk inputs slow-walks one sample; Help explains loss->0.'
Set-TeachPanel -Text 'Welcome: each epoch tries all quiz rows and nudges weights. Loss is average how-wrong (not money). Walk inputs shows numbers traveling left->right.'

if ($SmokeTest) {
    if ($null -eq $script:nn) { Write-Error 'Smoke failed: NeuralNetwork not constructed'; exit 2 }
    $bmp = New-Object System.Drawing.Bitmap(640, 400)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    Draw-Network -g $g -width 640 -height 400
    $g.Dispose()
    $bmp.Dispose()
    if ($script:nodePositions.Count -lt 1) { Write-Error 'Smoke failed: node hit-map empty'; exit 13 }
    if ($script:edgeHits.Count -lt 1) { Write-Error 'Smoke failed: edge hit-map empty'; exit 14 }
    $bmp2 = New-Object System.Drawing.Bitmap(400, 80)
    $g2 = [System.Drawing.Graphics]::FromImage($bmp2)
    Draw-LossChart -g $g2 -width 400 -height 80
    $g2.Dispose()
    $bmp2.Dispose()

    Invoke-OneEpoch
    if ($script:epoch -lt 1) { Write-Error 'Smoke failed: epoch did not advance'; exit 3 }
    if ($null -eq $script:nn.LossHistory -or $script:nn.LossHistory.Count -lt 1) {
        Write-Error 'Smoke failed: LossHistory empty after epoch'; exit 5
    }
    Update-StatusLabels
    $epochAfterStep = $script:epoch
    $lossAfterStep = $script:loss

    # Hit-test / UX helpers
    Build-NetworkGeometry -Width 640 -Height 400
    $pt0 = $script:nodePositions['0,0']
    $found = Find-NodeAtPoint -X ([int]$pt0.X) -Y ([int]$pt0.Y)
    if ($found -ne '0,0') { Write-Error "Smoke failed: Find-NodeAtPoint got $found"; exit 15 }
    $edge0 = $script:edgeHits[0]
    $mx = [int](($edge0.X1 + $edge0.X2) / 2)
    $my = [int](($edge0.Y1 + $edge0.Y2) / 2)
    $ef = Find-EdgeAtPoint -X $mx -Y $my -MaxDist 12
    if ($null -eq $ef) { Write-Error 'Smoke failed: Find-EdgeAtPoint'; exit 16 }

    Select-NetworkNeuron -Layer 0 -Index 0
    if ($script:selectedLayer -ne 0 -or $script:selectedIndex -ne 0) {
        Write-Error 'Smoke failed: Select-NetworkNeuron'; exit 17
    }
    Randomize-NeuronWeights -Layer 0 -Index 0
    Zero-NeuronBias -Layer 0 -Index 0
    if ([Math]::Abs([float]$script:nn.Network[0][0].GetBias()) -gt 1e-6) {
        Write-Error 'Smoke failed: Zero-NeuronBias'; exit 18
    }
    Set-TipLine -Text 'smoke tip'
    if ($lblTip.Text -ne 'smoke tip') { Write-Error 'Smoke failed: Set-TipLine'; exit 19 }

    $bmp3 = New-Object System.Drawing.Bitmap(400, 80)
    $g3 = [System.Drawing.Graphics]::FromImage($bmp3)
    Draw-LossChart -g $g3 -width 400 -height 80
    $g3.Dispose(); $bmp3.Dispose()
    $lt = Get-LossTipAtX -X 50 -Width 400
    if ($null -eq $lt) { Write-Error 'Smoke failed: Get-LossTipAtX'; exit 20 }

    # Add-neuron with weight transplant
    $h0 = $script:layoutHidden[0]
    $wBefore = [float[]]@($script:nn.Network[0][0].Weights)
    New-DemoNetwork -HiddenLayout @(($h0 + 1)) -PreserveWeights
    $wAfter = [float[]]@($script:nn.Network[0][0].Weights)
    $copied = $true
    $lim = [Math]::Min($wBefore.Count, $wAfter.Count)
    for ($i = 0; $i -lt $lim; $i++) {
        if ([Math]::Abs($wBefore[$i] - $wAfter[$i]) -gt 1e-6) { $copied = $false; break }
    }
    if (-not $copied) { Write-Error 'Smoke failed: weight transplant mismatch'; exit 6 }
    if ($script:layoutHidden[0] -ne ($h0 + 1)) { Write-Error 'Smoke failed: add-neuron rebuild'; exit 4 }

    # Multi-hidden
    New-DemoNetwork -HiddenLayout @(3, 2) -PreserveWeights
    $full = Get-FullLayout
    if ($full.Count -ne 3 -or $full[0] -ne 3 -or $full[1] -ne 2 -or $full[2] -ne 1) {
        Write-Error "Smoke failed: multi-hidden layout got [$full]"; exit 7
    }

    # Activation + LR setters
    $script:nn.SetActivationFunction('tanh')
    $script:nn.SetLearningRate([float]0.25)
    if ($script:nn.GetActivationFunction() -ne 'tanh') { Write-Error 'Smoke failed: activation setter'; exit 8 }
    if ([Math]::Abs($script:nn.GetLearningRate() - 0.25) -gt 1e-6) { Write-Error 'Smoke failed: LR setter'; exit 9 }

    # Sine demo path
    $script:demoName = 'SineForecast'
    Load-DemoData
    New-DemoNetwork -HiddenLayout @(4)
    if ($script:inputFeatures -ne $script:SineWindow) { Write-Error 'Smoke failed: sine input features'; exit 10 }
    if ($script:trainData.Count -lt 4) { Write-Error 'Smoke failed: sine data'; exit 11 }
    Invoke-OneEpoch
    Apply-ForwardViz
    $script:loss = Get-NetworkMse
    Invoke-PulseForward
    Build-NetworkGeometry -Width 640 -Height 400

    # TrainSample alias callable
    $row0 = $script:trainData[0]
    $xin0 = [float[]]::new($script:inputFeatures)
    for ($i = 0; $i -lt $script:inputFeatures; $i++) { $xin0[$i] = [float]$row0[$i] }
    $yout0 = [float[]]@([float]$row0[$script:inputFeatures])
    try { $script:nn.TrainSample($xin0, $yout0) }
    catch { Write-Error "Smoke failed: TrainSample alias missing/broken: $_"; exit 12 }

    # v1.2 teaching surface
    if (-not $lblTeach) { Write-Error 'Smoke failed: lblTeach missing'; exit 21 }
    if (-not $lblLossExplain) { Write-Error 'Smoke failed: lblLossExplain missing'; exit 22 }
    if (-not $chkWalk) { Write-Error 'Smoke failed: chkWalk missing'; exit 23 }
    Set-TeachPanel -Text 'smoke teach'
    if ($lblTeach.Text -ne 'smoke teach') { Write-Error 'Smoke failed: Set-TeachPanel'; exit 24 }
    Update-LossExplainLabel
    if ([string]::IsNullOrWhiteSpace($lblLossExplain.Text)) { Write-Error 'Smoke failed: loss explain empty'; exit 25 }
    if ($lblTip.Text -notmatch 'Epoch|nudge|loss' -and $epochAfterStep -ge 1) {
        # Narrator ran on first Invoke-OneEpoch; tip may have been overwritten by later Set-TipLine('smoke tip')
    }
    # Re-run one epoch and require narrator language
    Invoke-OneEpoch
    if ($lblTip.Text -notmatch 'Epoch\s+\d+' -or $lblTip.Text -notmatch 'quiz rows') {
        Write-Error ("Smoke failed: epoch narrator tip got: {0}" -f $lblTip.Text); exit 26
    }
    if ($lblTeach.Text -notmatch 'Epoch\s+\d+') {
        Write-Error ("Smoke failed: epoch narrator teach panel got: {0}" -f $lblTeach.Text); exit 27
    }
    $impact = Get-WeightImpactTip -Layer 0 -Neuron 0 -WIdx 0
    if ($impact -notmatch 'listen|weight|Bias') {
        Write-Error ("Smoke failed: weight impact tip weak: {0}" -f $impact); exit 28
    }
    Focus-WeightSlider -WeightIndex 0
    $sineEpochSaved = $script:epoch
    $sineLossSaved = $script:loss
    # Walk inputs (advance a few steps without waiting on timer)
    $script:demoName = 'XOR'
    Load-DemoData
    New-DemoNetwork -HiddenLayout @(2)
    Apply-ForwardViz
    $built = @(Build-WalkSteps)
    if ($built.Count -lt 5) { Write-Error ("Smoke failed: walk steps too few ({0})" -f $built.Count); exit 29 }
    $script:walkSteps = $built
    $script:walkArmed = $true
    $script:walkIndex = 0
    Show-WalkStep
    if (-not $script:walkHighlightKey -and $script:walkSteps[0].Kind -ne 'intro') {
        Write-Error 'Smoke failed: walk highlight missing on first content step'; exit 30
    }
    # Advance past intro to a real highlight
    $script:walkIndex = 1
    Show-WalkStep
    if ([string]::IsNullOrWhiteSpace([string]$script:walkHighlightKey)) {
        Write-Error 'Smoke failed: walk highlight key empty after advance'; exit 31
    }
    $bmpW = New-Object System.Drawing.Bitmap(640, 400)
    $gW = [System.Drawing.Graphics]::FromImage($bmpW)
    Draw-Network -g $gW -width 640 -height 400
    $gW.Dispose(); $bmpW.Dispose()
    Stop-WalkUi
    if ($script:walkArmed) { Write-Error 'Smoke failed: Stop-WalkUi left armed'; exit 32 }
    Invoke-PulseForward
    if ($script:ghostActivations.Count -lt 1) { Write-Error 'Smoke failed: Pulse ghost activations empty'; exit 33 }

    Write-Host ("SMOKE OK v1.2 stepEpoch={0} stepLoss={1} layout=[{2}] sineEpoch={3} sineLoss={4} act={5} lr={6} nodes={7} edges={8} walkSteps={9} ps={10} module={11}" -f `
        $epochAfterStep, $lossAfterStep, ((Get-FullLayout) -join ','), $sineEpochSaved, $sineLossSaved, `
        $script:nn.GetActivationFunction(), $script:nn.GetLearningRate(), $script:nodePositions.Count, $script:edgeHits.Count, `
        $built.Count, $PSVersionTable.PSVersion, $script:ModulePath)
    $form.Dispose()
    exit 0
}

[System.Windows.Forms.Application]::Run($form)
