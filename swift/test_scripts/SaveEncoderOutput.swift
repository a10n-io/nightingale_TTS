import Foundation
import MLX
import Nightingale

print("Loading model...")
let modelsPath = "../models/chatterbox"
let flowPath = "\(modelsPath)/s3gen.safetensors"
let flowWeights = try! MLX.loadArrays(url: URL(fileURLWithPath: flowPath))
let s3gen = S3Gen(flowWeights: flowWeights, vocoderWeights: nil)

// Load voice
let voiceDir = "../baked_voices/samantha"
let voiceArrays = try! MLX.loadArrays(url: URL(fileURLWithPath: voiceDir + "/baked_voice.safetensors"))
let speakerEmb = voiceArrays["t3.speaker_emb"]!
let speechEmbMatrix = voiceArrays["gen.embedding"]!
let promptToken = voiceArrays["gen.prompt_token"]!
let promptFeat = voiceArrays["gen.prompt_feat"]!

// Load Python tokens
let tokensPath = "../test_audio/cross_validate/python_speech_tokens.safetensors"
let tokensArrays = try! MLX.loadArrays(url: URL(fileURLWithPath: tokensPath))
let speechTokens = tokensArrays["speech_tokens"]!

print("\nGenerating mel with debug=true to save encoder output...")

let mel = s3gen.generateMelFromTokens(
    speechTokens: speechTokens,
    speechEmbMatrix: speechEmbMatrix,
    speakerEmb: speakerEmb,
    promptToken: promptToken,
    promptFeat: promptFeat,
    temperature: 1.0,
    cfgStrength: 1.0,
    debug: true  // This will trigger forensic output saving
)

print("\n✅ Complete - encoder output should be saved to test_audio/forensic/")
