# Sujano Static Issue - Root Cause Analysis

## Symptoms
- Python sujano: **PERFECT** audio
- Swift sujano: "voice for first sound (prompt) then static (generated region)"

## Key Findings

### Audio Amplitudes
- Python: [-0.795, 1.033] - loud but clean
- Swift: [-0.496, 0.708] - loud, distorted

### Mel Values (Swift)
- ODE output: [-12.76, 4.37] - reasonable range
- Generated mel: [-5.27, **1.78**] - **POSITIVE VALUES!**

### Problem
Swift generated mel has **positive values (max=1.78)** which is invalid for log-scale mels.
Normal mel values should be negative (log of magnitude < 1).

## Hypothesis

The encoder identity init fix (gamma=1) works for samantha but **overcorrects for sujano**.

Possible causes:
1. **Encoder too strong**: Identity norm (gamma=1) amplifies sujano's naturally bright characteristics
2. **Voice conditioning**: Sujano's speaker_emb or speech_emb_matrix causes decoder to produce extreme values  
3. **ODE solver**: The brighter encoder output causes ODE to diverge for sujano

## Next Steps

1. Compare Python vs Swift encoder outputs for sujano
2. Check if Python uses a different norm strategy
3. Try scaling down the encoder output specifically for bright voices
4. Investigate if finalProj needs clamping/clipping

