def load_stable_diffusion_pretrained(state_dict, temporal_attention=True):
    import collections
    sd_new = collections.OrderedDict()
    keys = list(state_dict.keys())
    for k in keys:
        if k.startswith("diffusion_model."):
            k_new = k.replace("diffusion_model.", "")
            if "input_blocks.3.0.op." in k_new:
                k_new = k_new.replace("0.op", "op")
            if temporal_attention:
                k_new = k_new.replace("middle_block.2", "middle_block.3")
                k_new = k_new.replace("output_blocks.5.2", "output_blocks.5.3")
                k_new = k_new.replace("output_blocks.8.2", "output_blocks.8.3")
            sd_new[k_new] = state_dict[k]
    return sd_new
