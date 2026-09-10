# This small neural network is a plain feed-forward MLP.
# It can make one-step-ahead forecasts from a fixed window of recent values if you
# feed the last N samples as inputs and train the single output to predict the next value.
# Educational surface: public getters/setters for weights, bias, activations, layout, LR;
# activation is chosen at construction (not hardcoded); loss history for charting.

class Neuron {
    [int]     hidden $NumberOfInputs
    [float[]] hidden $Weights
    [Neuron[]] hidden $InputNeurons
    [bool]    hidden $IsInput
    [float[]] hidden $NetworkErrors
    [float]   hidden $LearningRate
    [string]  hidden $ActivationFunction
    [float]   hidden $LastZ

    [float]   $Value

    Neuron(
        [Neuron[]]$InputNeurons,
        [int]$NumberOfInputs,
        [float]$LearningRate,
        [string]$ActivationFunction = "Sigmoid"
    ) {
        $this.InputNeurons = $InputNeurons
        $this.NumberOfInputs = $NumberOfInputs
        $this.Value = 0.0
        $this.NetworkErrors = @()
        $this.LearningRate = $LearningRate
        $this.ActivationFunction = $ActivationFunction
        $this.IsInput = (-not $InputNeurons) -or ($InputNeurons.Count -eq 0)

        # Initialize weights (including bias)
        $this.Weights = [float[]]::new($NumberOfInputs + 1)
        for ($i = 0; $i -lt $this.Weights.Count; $i++) {
            # Random uniform in [-1, 1)
            $this.Weights[$i] = Get-Random -Minimum -1.0 -Maximum 1.0
        }
    }

    # --- teaching / GUI surface ---
    [float[]] GetWeights() { return [float[]]@($this.Weights) }

    SetWeights([float[]]$NewWeights) {
        if ($null -eq $NewWeights -or $NewWeights.Count -ne $this.Weights.Count) {
            throw "Weight count must be $($this.Weights.Count) (inputs + bias)."
        }
        for ($i = 0; $i -lt $NewWeights.Count; $i++) {
            $this.Weights[$i] = [float]$NewWeights[$i]
        }
    }

    [float] GetBias() { return $this.Weights[$this.NumberOfInputs] }

    SetBias([float]$Bias) { $this.Weights[$this.NumberOfInputs] = $Bias }

    [float] GetWeight([int]$Index) {
        if ($Index -lt 0 -or $Index -ge $this.Weights.Count) {
            throw "Weight index out of range: $Index"
        }
        return $this.Weights[$Index]
    }

    SetWeight([int]$Index, [float]$Value) {
        if ($Index -lt 0 -or $Index -ge $this.Weights.Count) {
            throw "Weight index out of range: $Index"
        }
        $this.Weights[$Index] = $Value
    }

    [float] GetActivation() { return $this.Value }

    [string] GetActivationFunction() { return $this.ActivationFunction }

    SetActivationFunction([string]$Name) {
        $n = "Sigmoid"
        if (-not [string]::IsNullOrWhiteSpace($Name)) {
            $t = $Name.Trim().ToLower()
            if ($t -eq "sigmoid") { $n = "Sigmoid" }
            elseif ($t -eq "relu") { $n = "ReLU" }
            elseif ($t -eq "tanh") { $n = "tanh" }
            else { throw "Unsupported activation function: $Name (use Sigmoid, ReLU, tanh)" }
        }
        $this.ActivationFunction = $n
    }

    [float] GetLearningRate() { return $this.LearningRate }

    SetLearningRate([float]$Rate) { $this.LearningRate = $Rate }

    [int] GetInputCount() { return $this.NumberOfInputs }

    hidden [float] ApplyActivation([float]$z) {
        switch ($this.ActivationFunction) {
            "Sigmoid" { return [float](1.0 / (1.0 + [math]::Exp(-$z))) }
            "ReLU" { return [float]([math]::Max(0.0, $z)) }
            "tanh" { return [float]([math]::Tanh($z)) }
            default {
                throw "Unsupported activation function: $($this.ActivationFunction)"
            }
        }
        return 0.0 # Fallback return to satisfy all code paths
    }

    [float] GetOutput([float[]]$InputValues) {
        if ($InputValues.Count -ne $this.NumberOfInputs) {
            throw "Input feature count does not match the expected number of inputs ($($this.NumberOfInputs))."
        }
        $z = 0.0
        for ($i = 0; $i -lt $this.NumberOfInputs; $i++) { $z += $this.Weights[$i] * $InputValues[$i] }
        $z += $this.Weights[$this.NumberOfInputs]  # bias
        $this.LastZ = $z
        $this.Value = $this.ApplyActivation($z)
        return $this.Value
    }

    [float] GetOutput() {
        $z = 0.0
        for ($i = 0; $i -lt $this.NumberOfInputs; $i++) { $z += $this.Weights[$i] * $this.InputNeurons[$i].Value }
        $z += $this.Weights[$this.NumberOfInputs]  # bias
        $this.LastZ = $z
        $this.Value = $this.ApplyActivation($z)
        return $this.Value
    }

    AddError([float]$Err) { $this.NetworkErrors += $Err }

    [float] GetDerivative() {
        switch ($this.ActivationFunction) {
            "Sigmoid" { return $this.Value * (1.0 - $this.Value) }
            "ReLU" { return ($this.LastZ -gt 0.0) ? 1.0 : 0.0 }
            "tanh" { return 1.0 - ($this.Value * $this.Value) }
            default {
                throw "Unsupported activation function: $($this.ActivationFunction)"
            }
        }
        return 0.0 # Fallback return to satisfy all code paths
    }

    [float] GetError([float]$target) {
        return ($this.Value - $target) * $this.GetDerivative()
    }

    backPropagate([float]$target) {
        $err = $this.GetError($target)
        for ($i = 0; $i -lt $this.NumberOfInputs; $i++) {
            $this.InputNeurons[$i].AddError($err * $this.Weights[$i])
            $this.Weights[$i] -= $this.InputNeurons[$i].Value * $err * $this.LearningRate
        }
        # bias update
        $this.Weights[$this.NumberOfInputs] -= $err * $this.LearningRate
    }

    [float] GetError() {
        $sum = 0.0
        foreach ($ne in $this.NetworkErrors) { $sum += $ne }
        $this.NetworkErrors = @()
        return $sum * $this.GetDerivative()
    }

    backPropagate() {
        $err = $this.GetError()
        for ($i = 0; $i -lt $this.NumberOfInputs; $i++) {
            $this.InputNeurons[$i].AddError($err * $this.Weights[$i])
            $this.Weights[$i] -= $this.InputNeurons[$i].Value * $err * $this.LearningRate
        }
        $this.Weights[$this.NumberOfInputs] -= $err * $this.LearningRate
    }

    backPropagate([float[]]$NetworkInputs) {
        $err = $this.GetError()
        for ($i = 0; $i -lt $this.NumberOfInputs; $i++) {
            $this.Weights[$i] -= $NetworkInputs[$i] * $err * $this.LearningRate
        }
        $this.Weights[$this.NumberOfInputs] -= $err * $this.LearningRate
    }
}

function Get-PSNeuronNormalizedActivation {
    param([string]$Name)
    if ([string]::IsNullOrWhiteSpace($Name)) { return "Sigmoid" }
    switch -Regex ($Name.Trim()) {
        '^(?i)sigmoid$' { return "Sigmoid" }
        '^(?i)relu$'    { return "ReLU" }
        '^(?i)tanh$'    { return "tanh" }
        default { throw "Unsupported activation function: $Name (use Sigmoid, ReLU, tanh)" }
    }
}

class NeuralNetwork {
    [Neuron[][]] hidden $Network
    [int]        hidden $InputFeatureCount
    [float]      hidden $LearningRate
    [string]     hidden $ActivationFunction
    [int[]]      hidden $Layout
    # Loss per completed epoch (MSE on last validation/train set passed to step helpers)
    [System.Collections.Generic.List[float]] $LossHistory

    NeuralNetwork (
        [int[]]$NetworkLayout,
        [float]$LearningRate,
        [int] $InputFeatureCount
    ) {
        $this.InitNetwork($NetworkLayout, $LearningRate, $InputFeatureCount, "Sigmoid")
    }

    NeuralNetwork (
        [int[]]$NetworkLayout,
        [float]$LearningRate,
        [int] $InputFeatureCount,
        [string]$ActivationFunction
    ) {
        $this.InitNetwork($NetworkLayout, $LearningRate, $InputFeatureCount, $ActivationFunction)
    }

    hidden InitNetwork(
        [int[]]$NetworkLayout,
        [float]$LearningRate,
        [int]$InputFeatureCount,
        [string]$ActivationFunction
    ) {
        $act = "Sigmoid"
        if (-not [string]::IsNullOrWhiteSpace($ActivationFunction)) {
            $t = $ActivationFunction.Trim().ToLower()
            if ($t -eq "sigmoid") { $act = "Sigmoid" }
            elseif ($t -eq "relu") { $act = "ReLU" }
            elseif ($t -eq "tanh") { $act = "tanh" }
            else { throw "Unsupported activation function: $ActivationFunction (use Sigmoid, ReLU, tanh)" }
        }
        $this.Network = @()
        $this.InputFeatureCount = $InputFeatureCount
        $this.LearningRate = [float]$LearningRate
        $this.ActivationFunction = $act
        $this.Layout = [int[]]@($NetworkLayout)
        $this.LossHistory = [System.Collections.Generic.List[float]]::new()

        for ($i = 0; $i -lt $NetworkLayout.Count; $i++) {
            $layer = @()
            for ($n = 0; $n -lt $NetworkLayout[$i]; $n++) {
                if ($i -eq 0) {
                    $layer += [Neuron]::new([Neuron[]]@(), $this.InputFeatureCount, [float]$LearningRate, $act)
                }
                else {
                    $layer += [Neuron]::new([Neuron[]]$this.Network[$i - 1], $NetworkLayout[$i - 1], [float]$LearningRate, $act)
                }
            }
            $this.Network += , $layer
        }
    }

    # --- teaching / GUI surface ---
    [Neuron[][]] GetNetwork() { return $this.Network }

    [int[]] GetLayout() { return [int[]]@($this.Layout) }

    [int] GetInputFeatureCount() { return $this.InputFeatureCount }

    [float] GetLearningRate() { return $this.LearningRate }

    SetLearningRate([float]$Rate) {
        $this.LearningRate = [float]$Rate
        foreach ($layer in $this.Network) {
            foreach ($neuron in $layer) {
                $neuron.SetLearningRate([float]$Rate)
            }
        }
    }

    [string] GetActivationFunction() { return $this.ActivationFunction }

    SetActivationFunction([string]$Name) {
        $act = "Sigmoid"
        if (-not [string]::IsNullOrWhiteSpace($Name)) {
            $t = $Name.Trim().ToLower()
            if ($t -eq "sigmoid") { $act = "Sigmoid" }
            elseif ($t -eq "relu") { $act = "ReLU" }
            elseif ($t -eq "tanh") { $act = "tanh" }
            else { throw "Unsupported activation function: $Name (use Sigmoid, ReLU, tanh)" }
        }
        $this.ActivationFunction = $act
        foreach ($layer in $this.Network) {
            foreach ($neuron in $layer) {
                $neuron.SetActivationFunction($act)
            }
        }
    }

    [float[]] GetActivations([int]$LayerIndex) {
        $layer = $this.Network[$LayerIndex]
        $vals = [float[]]::new($layer.Count)
        for ($i = 0; $i -lt $layer.Count; $i++) { $vals[$i] = $layer[$i].Value }
        return $vals
    }

    [float[]] GetAllActivations() {
        $list = [System.Collections.Generic.List[float]]::new()
        foreach ($layer in $this.Network) {
            foreach ($neuron in $layer) { [void]$list.Add($neuron.Value) }
        }
        return $list.ToArray()
    }

    [float[]] GetLossHistoryArray() {
        if ($null -eq $this.LossHistory) { return [float[]]@() }
        return $this.LossHistory.ToArray()
    }

    ClearLossHistory() {
        if ($null -eq $this.LossHistory) {
            $this.LossHistory = [System.Collections.Generic.List[float]]::new()
        }
        else {
            $this.LossHistory.Clear()
        }
    }

    AppendLoss([float]$Loss) {
        if ($null -eq $this.LossHistory) {
            $this.LossHistory = [System.Collections.Generic.List[float]]::new()
        }
        [void]$this.LossHistory.Add([float]$Loss)
    }

    [Neuron] GetNeuron([int]$LayerIndex, [int]$NeuronIndex) {
        return $this.Network[$LayerIndex][$NeuronIndex]
    }

    [float[]] GetNeuronWeights([int]$LayerIndex, [int]$NeuronIndex) {
        return $this.Network[$LayerIndex][$NeuronIndex].GetWeights()
    }

    SetNeuronWeights([int]$LayerIndex, [int]$NeuronIndex, [float[]]$Weights) {
        $this.Network[$LayerIndex][$NeuronIndex].SetWeights($Weights)
    }

    [float[]] GetOutput([float[]]$InputValues) {
        $output = @()
        for ($i = 0; $i -lt $this.Network.Count; $i++) {
            $layer = $this.Network[$i]
            if ($i -eq 0) {
                foreach ($neuron in $layer) { $null = $neuron.GetOutput($InputValues) }
            }
            else {
                foreach ($neuron in $layer) { $null = $neuron.GetOutput() }
            }
            if ($i -eq $this.Network.Count - 1) {
                foreach ($neuron in $layer) { $output += $neuron.Value }
            }
        }
        return $output
    }

    [float] CalculateLoss ([float[][]]$ValidationData) {
        $totalLoss = 0.0
        foreach ($row in $ValidationData) {
            $inputData = $row | Select-Object -First $this.InputFeatureCount
            $outputData = $row | Select-Object -Skip  $this.InputFeatureCount
            $null = $this.GetOutput($inputData)
            $totalLoss += $this.CalculateSampleLoss($outputData)
        }
        return $totalLoss / [math]::Max(1, $ValidationData.Count)
    }

    [float] CalculateSampleLoss ([float[]]$targetValues) {
        $outputLayer = $this.Network[-1]
        $loss = 0.0
        for ($j = 0; $j -lt $outputLayer.Count; $j++) {
            $loss += 0.5 * [math]::Pow($outputLayer[$j].Value - $targetValues[$j], 2)
        }
        return $loss
    }

    [array[]] CloneWeights() {
        $clonedWeights = @()
        foreach ($layer in $this.Network) {
            $clonedLayer = @()
            foreach ($neuron in $layer) { $clonedLayer += , $neuron.Weights.Clone() }
            $clonedWeights += , $clonedLayer
        }
        return $clonedWeights
    }

    RestoreWeights([array[]]$clonedWeights) {
        for ($i = 0; $i -lt $this.Network.Count; $i++) {
            for ($j = 0; $j -lt $this.Network[$i].Count; $j++) {
                $this.Network[$i][$j].Weights = $clonedWeights[$i][$j].Clone()
            }
        }
    }

    TrainNeuron ([float[]]$InputValues, [float[]]$TargetValues) {
        $null = $this.GetOutput($InputValues)
        for ($i = $this.Network.Count - 1; $i -ge 0; $i--) {
            $layer = $this.Network[$i]
            for ($n = 0; $n -lt $layer.Count; $n++) {
                if ($i -eq $this.Network.Count - 1) {
                    $layer[$n].backPropagate($TargetValues[$n])
                }
                elseif ($i -eq 0) {
                    $layer[$n].backPropagate($InputValues)
                }
                else {
                    $layer[$n].backPropagate()
                }
            }
        }
    }

    # Clearer alias — same as TrainNeuron (keeps Adapter / existing callers working)
    TrainSample ([float[]]$InputValues, [float[]]$TargetValues) {
        $this.TrainNeuron($InputValues, $TargetValues)
    }

    # One epoch of SGD over TrainingData; appends mean loss on LossData (or TrainingData) to LossHistory.
    [float] TrainEpoch ([float[][]]$TrainingData, [float[][]]$LossData) {
        $TrainingSet = $TrainingData | Sort-Object { Get-Random }
        foreach ($row in $TrainingSet) {
            $InputData = $row | Select-Object -First $this.InputFeatureCount
            $OutputData = $row | Select-Object -Skip  $this.InputFeatureCount
            $this.TrainNeuron($InputData, $OutputData)
        }
        $eval = if ($null -ne $LossData -and $LossData.Count -gt 0) { $LossData } else { $TrainingData }
        $loss = $this.CalculateLoss($eval)
        $this.AppendLoss($loss)
        return $loss
    }

    [float] TrainEpoch ([float[][]]$TrainingData) {
        return $this.TrainEpoch($TrainingData, $TrainingData)
    }

    TrainNetwork ([float[][]]$TrainingData, [int]$Epochs, [float[][]]$ValidationData, [int]$EarlyStoppingThreshold) {
        $bestWeights = $this.CloneWeights()
        $bestValidationLoss = [double]::MaxValue
        $noImprovement = 0

        for ($i = 1; $i -le $Epochs; $i++) {
            # Shuffle training data
            $TrainingSet = $TrainingData | Sort-Object { Get-Random }

            foreach ($row in $TrainingSet) {
                $InputData = $row | Select-Object -First $this.InputFeatureCount
                $OutputData = $row | Select-Object -Skip  $this.InputFeatureCount
                $this.TrainNeuron($InputData, $OutputData)
            }

            # Validation / early stopping
            $validationLoss = $this.CalculateLoss($ValidationData)
            $this.AppendLoss([float]$validationLoss)
            if ($validationLoss -lt $bestValidationLoss) {
                $bestValidationLoss = $validationLoss
                $bestWeights = $this.CloneWeights()
                $noImprovement = 0
            }
            else {
                $noImprovement++
            }

            if ($noImprovement -ge $EarlyStoppingThreshold) {
                Write-Host "Early stopping: No improvement for $EarlyStoppingThreshold epochs (epoch $i)."
                $this.RestoreWeights($bestWeights)
                break
            }
        }
    }
}

# --- Demo data helpers (XOR + sine one-step forecast) ---

function Get-PSNeuronXorData {
    <#
    .SYNOPSIS
      Classic 2-input XOR rows: [x0, x1, target]
    #>
    , @([float]0, [float]0, [float]0)
    , @([float]0, [float]1, [float]1)
    , @([float]1, [float]0, [float]1)
    , @([float]1, [float]1, [float]0)
}

function Get-PSNeuronSineForecastData {
    <#
    .SYNOPSIS
      One-step-ahead sine forecast rows from a fixed window.
    .DESCRIPTION
      Builds sliding windows over sin(t). Each row is [x0..x{Window-1}, next].
      Values are mapped to ~[0,1] so Sigmoid outputs are a natural fit; with tanh
      you may prefer centering, but the GUI demos stay in [0,1] for simplicity.
    .PARAMETER Window
      Number of past samples used as inputs (default 4).
    .PARAMETER Count
      Number of training rows to emit (default 48).
    .PARAMETER Step
      Angle step in radians between samples (default 0.25).
    #>
    param(
        [int]$Window = 4,
        [int]$Count = 48,
        [double]$Step = 0.25
    )
    if ($Window -lt 1) { throw "Window must be >= 1" }
    if ($Count -lt 1) { throw "Count must be >= 1" }
    $need = $Window + $Count
    $series = [float[]]::new($need)
    for ($i = 0; $i -lt $need; $i++) {
        # Map sin from [-1,1] -> [0,1]
        $series[$i] = [float](0.5 + 0.5 * [math]::Sin($i * $Step))
    }
    for ($r = 0; $r -lt $Count; $r++) {
        $row = [float[]]::new($Window + 1)
        for ($w = 0; $w -lt $Window; $w++) {
            $row[$w] = $series[$r + $w]
        }
        $row[$Window] = $series[$r + $Window]
        Write-Output -NoEnumerate $row
    }
}

Export-ModuleMember -Function @(
    'Get-PSNeuronXorData',
    'Get-PSNeuronSineForecastData'
)
