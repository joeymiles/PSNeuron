# Thin helpers around PSNeuron.psm1 (classes are not exported by Import-Module).
# Dot-source only after Import-Module of PSNeuron.psm1 under PowerShell 7+.
# Does not invent a second network — constructs [NeuralNetwork] inside module scope.

function Get-PSNeuronModule {
    $m = Get-Module -Name 'PSNeuron' -ErrorAction SilentlyContinue |
        Where-Object { $_.Path -and ($_.Path -like '*PSNeuron.psm1') } |
        Select-Object -First 1
    if (-not $m) {
        throw 'PSNeuron module is not loaded. Import PSNeuron.psm1 first.'
    }
    return $m
}

function New-PSNeuronNetwork {
    param(
        [Parameter(Mandatory)] [int[]] $Layout,
        [Parameter(Mandatory)] [float] $LearningRate,
        [Parameter(Mandatory)] [int] $InputFeatureCount,
        [string] $ActivationFunction = 'Sigmoid'
    )
    $m = Get-PSNeuronModule
    # Classes live in module scope; construct there, return instance to caller.
    return & $m {
        param($Layout, $LearningRate, $InputFeatureCount, $ActivationFunction)
        [NeuralNetwork]::new(
            [int[]]$Layout,
            [float]$LearningRate,
            [int]$InputFeatureCount,
            [string]$ActivationFunction
        )
    } $Layout $LearningRate $InputFeatureCount $ActivationFunction
}

function New-PSNeuronFloatJagged {
    param([Parameter(Mandatory)] [object[]] $Rows)
    $arr = [float[][]]::new($Rows.Count)
    for ($i = 0; $i -lt $Rows.Count; $i++) {
        $arr[$i] = [float[]]@($Rows[$i])
    }
    return $arr
}

function Copy-PSNeuronOverlappingWeights {
    <#
    .SYNOPSIS
      Transplant overlapping weights from OldNet into NewNet.
      Same layer index / neuron index / weight index kept; new connections stay random-init.
    #>
    param(
        [Parameter(Mandatory)] $OldNet,
        [Parameter(Mandatory)] $NewNet
    )
    $oldLayers = $OldNet.Network.Count
    $newLayers = $NewNet.Network.Count
    $layerLimit = [Math]::Min($oldLayers, $newLayers)
    for ($L = 0; $L -lt $layerLimit; $L++) {
        $oldCount = $OldNet.Network[$L].Count
        $newCount = $NewNet.Network[$L].Count
        $nLimit = [Math]::Min($oldCount, $newCount)
        for ($n = 0; $n -lt $nLimit; $n++) {
            $oldW = $OldNet.Network[$L][$n].Weights
            $newW = $NewNet.Network[$L][$n].Weights
            $wLimit = [Math]::Min($oldW.Count, $newW.Count)
            # Copy overlapping input weights; if bias indices match (same NumberOfInputs),
            # the last slot is bias and is included in Min count when input counts match.
            for ($w = 0; $w -lt $wLimit; $w++) {
                $NewNet.Network[$L][$n].Weights[$w] = $oldW[$w]
            }
            # If new neuron has more inputs, trailing weights (beyond old) stay random;
            # bias of new neuron: if input count grew, bias is at a new index — leave random.
            # If input count identical, bias was already copied as last overlapping slot.
        }
    }
}

function Get-PSNeuronDemoRows {
    param(
        [ValidateSet('XOR', 'SineForecast')]
        [string] $Demo = 'XOR',
        [int] $SineWindow = 4,
        [int] $SineCount = 48
    )
    # Emit rows as separate pipeline objects so @() rebuilds a proper jagged list.
    switch ($Demo) {
        'XOR' {
            if (Get-Command Get-PSNeuronXorData -ErrorAction SilentlyContinue) {
                Get-PSNeuronXorData
            }
            else {
                , @([float]0, [float]0, [float]0)
                , @([float]0, [float]1, [float]1)
                , @([float]1, [float]0, [float]1)
                , @([float]1, [float]1, [float]0)
            }
        }
        'SineForecast' {
            if (-not (Get-Command Get-PSNeuronSineForecastData -ErrorAction SilentlyContinue)) {
                throw 'Get-PSNeuronSineForecastData not available; update PSNeuron.psm1.'
            }
            Get-PSNeuronSineForecastData -Window $SineWindow -Count $SineCount
        }
    }
}
