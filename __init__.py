import torch.nn.functional as F
import torchvision.transforms.functional as TF

import comfy.utils
import comfy.sd
import comfy.diffusers_load
import comfy.samplers
import comfy.controlnet


class SVD_txt2vid_ConditioningwithLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip": ("CLIP", ),
                              "samples_to": ("LATENT",),
                              "motion_bucket_id": ("INT", {"default": 127, "min": 1, "max": 1023}),
                              "fps": ("INT", {"default": 6, "min": 1, "max": 1024}),
                              "augmentation_level": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "text1": ("STRING", {"multiline": True}),
                              "text2": ("STRING", {"multiline": True}),
                             }}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")

    FUNCTION = "latentencode"

    CATEGORY = "conditioning/video_models"

    def latentencode(self, clip, motion_bucket_id, fps, augmentation_level, text1, text2, samples_to):
        # Prepare latents for conditioning
        samples_out = samples_to.copy()
        samples = samples_to["samples"]
        t = samples

        # Prepare positive & negative conditioning/Copied directly from CLIPTextEncode
        tokens1 = clip.tokenize(text1)
        cond, pooled_cond = clip.encode_from_tokens(tokens1, return_pooled=True)
        tokens2 = clip.tokenize(text2)
        uncond, pooled_uncond = clip.encode_from_tokens(tokens2, return_pooled=True)

        # Conditioning with added latent as conditioning frame (in the end the frame is the same as a standard CHW tensor)
        positive = [[cond, {"pooled_output": pooled_cond, "motion_bucket_id": motion_bucket_id, "fps": fps, "augmentation_level": augmentation_level, "concat_latent_image": t}]]
        negative = [[uncond, {"pooled_output": pooled_uncond, "motion_bucket_id": motion_bucket_id, "fps": fps, "augmentation_level": augmentation_level, "concat_latent_image": t}]]
        samples_out[samples] = t

        return (positive, negative, samples_out)
    
NODE_CLASS_MAPPINGS = {
    "SVD_txt2vid_ConditioningwithLatent": SVD_txt2vid_ConditioningwithLatent,
}