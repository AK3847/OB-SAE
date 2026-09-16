"""Work around a peft bug that makes OFT unusable on bitsandbytes-quantized layers.

peft 0.20.0: `OFTModel._create_new_module` calls every dispatcher as
`dispatcher(target, adapter_name, oft_config=oft_config, **kwargs)`. The default dispatcher
renames it correctly (`Linear(target, adapter_name, config=oft_config, ...)`), but the bnb
dispatchers forward their kwargs verbatim:

    new_module = Linear4bit(target, adapter_name, **fourbit_kwargs)

`Linear4bit.__init__` takes `config` as a required positional argument, so `oft_config` falls
through into `**kwargs` and the call raises:

    TypeError: Linear4bit.__init__() missing 1 required positional argument: 'config'

This aliases `oft_config` to `config` before the dispatcher runs. The layer classes accept
`**kwargs`, so the leftover `oft_config` is ignored. `r` is already passed correctly by
`_create_and_replace`, and block size, coft, eps and block_share are read from `config` inside
`update_layer`, so nothing else has to change.

`OFTModel._create_new_module` imports the dispatchers lazily inside the function, so patching
the module attribute takes effect.
"""
import functools


def apply() -> bool:
    """Patch the bnb OFT dispatchers. Returns True if anything was patched."""
    try:
        from peft.tuners.oft import bnb as oft_bnb
    except ImportError:
        return False

    patched_any = False
    for name in ("dispatch_bnb_4bit", "dispatch_bnb_8bit"):
        original = getattr(oft_bnb, name, None)
        if original is None or getattr(original, "_oft_config_alias_patch", False):
            continue

        def wrap(original):
            @functools.wraps(original)
            def dispatcher(target, adapter_name, **kwargs):
                if "config" not in kwargs and "oft_config" in kwargs:
                    kwargs["config"] = kwargs["oft_config"]
                return original(target, adapter_name, **kwargs)

            dispatcher._oft_config_alias_patch = True
            return dispatcher

        setattr(oft_bnb, name, wrap(original))
        patched_any = True

    return patched_any
