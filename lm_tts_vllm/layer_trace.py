"""Temporary hooks for a single prefill; never installed during normal inference."""
import torch


def install(backbone, engine):
    values, handles = {}, []
    def save(name, value, layer):
        if isinstance(value, tuple):
            value = value[0]
        # Keep all tokens in layer zero to diagnose attention inputs; subsequent
        # layers keep the last token to limit diagnostic CPU transfers.
        tokens = value.shape[0] if engine else value.shape[1]
        value = value.reshape(tokens, -1)
        values[name] = value.detach().cpu().clone() if layer == 0 else value[-1:].detach().cpu().clone()

    def output(module, name, layer):
        handles.append(module.register_forward_hook(lambda m, a, v: save(name, v, layer)))

    for index, layer in enumerate(backbone.layers):
        prefix = f"layers.{index}."
        attn = layer.self_attn
        for name in ["input_layernorm", "post_attention_layernorm"]:
            output(getattr(layer, name), prefix + name, index)
        if engine:
            def qkv(m, args, result, p=prefix, i=index, a=attn):
                for name, value in zip(["q_proj", "k_proj", "v_proj"],
                                       result[0].split([a.q_size, a.kv_size, a.kv_size], -1)):
                    save(p + "self_attn." + name, value, i)
            handles.append(attn.qkv_proj.register_forward_hook(qkv))
            def attention_inputs(m, args, p=prefix, i=index):
                for name, value in zip(["query", "key", "value"], args):
                    save(p + "attention_" + name, value, i)
            handles.append(attn.attn.register_forward_pre_hook(attention_inputs))
        else:
            for name in ["q_proj", "k_proj", "v_proj"]:
                output(getattr(attn, name), prefix + "self_attn." + name, index)
        for name in ["q_norm", "k_norm", "o_proj"]:
            output(getattr(attn, name), prefix + "self_attn." + name, index)
        handles.append(attn.o_proj.register_forward_pre_hook(
            lambda m, args, p=prefix, i=index: save(p + "attention_output", args[0], i)))
        output(layer.mlp, prefix + "mlp", index)
        def layer_output(m, args, result, p=prefix, i=index):
            value = result[0] + result[1] if engine else result[0]
            save(p + "output", value, i)
        handles.append(layer.register_forward_hook(layer_output))
    output(backbone.norm, "norm", len(backbone.layers))
    return values, handles


def begin_worker(model):
    model._layer_trace = install(model.model, engine=True)


def end_worker(model):
    values, handles = model._layer_trace
    for handle in handles:
        handle.remove()
    del model._layer_trace
    # NumPy transport avoids cross-runtime CUDA/tensor multiprocessing reducers.
    return {name: value.float().numpy() for name, value in values.items()}
