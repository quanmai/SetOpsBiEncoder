import importlib

def init_hf_bert_biencoder(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError("Please install transformers")
    from .hf_models import get_bert_biencoder_components

    return get_bert_biencoder_components(args, **kwargs)


BIENCODER_INITIALIZERS = {
    "hf_bert": init_hf_bert_biencoder
}

def init_comp(initializers_dict, type, args, **kwargs):
    if type in initializers_dict:
        return initializers_dict[type](args, **kwargs)
    raise RuntimeError("unsupported model type: {}".format(type))



def init_biencoder_components(enconder_type: str, args, **kwargs):
    return init_comp(BIENCODER_INITIALIZERS, enconder_type, args, **kwargs)