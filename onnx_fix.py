"""
ONNX Runtime Fix for SimSwap
This module patches the ONNX Runtime to automatically set providers
"""
import onnxruntime as ort
import os

# Store original InferenceSession
_original_InferenceSession = ort.InferenceSession

def patched_InferenceSession(model_path_or_bytes, sess_options=None, providers=None, provider_options=None, **kwargs):
    """
    Patched InferenceSession that automatically sets providers if not specified
    """
    if providers is None:
        # Get available providers
        available_providers = ort.get_available_providers()
        
        # Prefer CUDA if available, otherwise use CPU
        if 'CUDAExecutionProvider' in available_providers:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
    
    # Call original InferenceSession with providers
    return _original_InferenceSession(
        model_path_or_bytes, 
        sess_options=sess_options, 
        providers=providers, 
        provider_options=provider_options, 
        **kwargs
    )

# Apply the patch
ort.InferenceSession = patched_InferenceSession

print("ONNX Runtime patch applied successfully")
