"""LoRA weight matching -- maps LoRA state dict keys to model keys.

Supports LoRA, LoHa, LoKr, GLoRA, diff, and direct-weight formats.
"""

from __future__ import annotations

from typing import Any


def match_lora(
    lora: dict[str, Any],
    to_load: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Match LoRA weights to model keys.

    Args:
        lora: Raw LoRA state dict loaded from a .safetensors file.
        to_load: Mapping from LoRA key prefixes to model weight keys.

    Returns:
        A tuple of (patch_dict, remaining_dict) where patch_dict maps
        model keys to (format, weights) tuples for model_patcher, and
        remaining_dict contains unmatched LoRA keys.
    """
    patch_dict: dict[str, Any] = {}
    loaded_keys: set[str] = set()

    for x in to_load:
        real_load_key = to_load[x]
        if real_load_key in lora:
            patch_dict[real_load_key] = ("fooocus", lora[real_load_key])
            loaded_keys.add(real_load_key)
            continue

        alpha = _extract_alpha(lora, x, loaded_keys)
        _try_lora_format(lora, x, real_load_key, alpha, patch_dict, loaded_keys)
        _try_loha_format(lora, x, real_load_key, alpha, patch_dict, loaded_keys)
        _try_lokr_format(lora, x, real_load_key, patch_dict, loaded_keys)
        _try_glora_format(lora, x, real_load_key, alpha, patch_dict, loaded_keys)
        _try_diff_format(lora, x, real_load_key, patch_dict, loaded_keys)

    remaining_dict = {k: v for k, v in lora.items() if k not in loaded_keys}
    return patch_dict, remaining_dict


def _extract_alpha(lora: dict[str, Any], prefix: str, loaded_keys: set[str]) -> Any | None:
    alpha_name = f"{prefix}.alpha"
    if alpha_name in lora:
        loaded_keys.add(alpha_name)
        return lora[alpha_name].item()
    return None


def _try_lora_format(
    lora: dict[str, Any],
    prefix: str,
    load_key: str,
    alpha: Any | None,
    patch_dict: dict[str, Any],
    loaded_keys: set[str],
) -> None:
    regular = f"{prefix}.lora_up.weight"
    diffusers = f"{prefix}_lora.up.weight"
    transformers = f"{prefix}.lora_linear_layer.up.weight"

    a_name: str | None = None
    b_name: str = ""
    mid_name: str | None = None

    if regular in lora:
        a_name = regular
        b_name = f"{prefix}.lora_down.weight"
        mid_name = f"{prefix}.lora_mid.weight"
    elif diffusers in lora:
        a_name = diffusers
        b_name = f"{prefix}_lora.down.weight"
    elif transformers in lora:
        a_name = transformers
        b_name = f"{prefix}.lora_linear_layer.down.weight"

    if a_name is None:
        return

    if b_name not in lora:
        return

    mid = None
    if mid_name is not None and mid_name in lora:
        mid = lora[mid_name]
        loaded_keys.add(mid_name)

    patch_dict[load_key] = ("lora", (lora[a_name], lora[b_name], alpha, mid))
    loaded_keys.add(a_name)
    loaded_keys.add(b_name)


def _try_loha_format(
    lora: dict[str, Any],
    prefix: str,
    load_key: str,
    alpha: Any | None,
    patch_dict: dict[str, Any],
    loaded_keys: set[str],
) -> None:
    w1a = f"{prefix}.hada_w1_a"
    if w1a not in lora:
        return

    w1b = f"{prefix}.hada_w1_b"
    w2a = f"{prefix}.hada_w2_a"
    w2b = f"{prefix}.hada_w2_b"
    t1_name = f"{prefix}.hada_t1"
    t2_name = f"{prefix}.hada_t2"

    required = {w1b, w2a, w2b}
    if not required.issubset(lora):
        return

    t1 = None
    t2 = None
    if t1_name in lora:
        if t2_name not in lora:
            return
        t1 = lora[t1_name]
        t2 = lora[t2_name]
        loaded_keys.add(t1_name)
        loaded_keys.add(t2_name)

    patch_dict[load_key] = (
        "loha",
        (lora[w1a], lora[w1b], alpha, lora[w2a], lora[w2b], t1, t2),
    )
    loaded_keys.update({w1a, w1b, w2a, w2b})


def _try_lokr_format(
    lora: dict[str, Any],
    prefix: str,
    load_key: str,
    patch_dict: dict[str, Any],
    loaded_keys: set[str],
) -> None:
    names = {
        "w1": f"{prefix}.lokr_w1",
        "w2": f"{prefix}.lokr_w2",
        "w1a": f"{prefix}.lokr_w1_a",
        "w1b": f"{prefix}.lokr_w1_b",
        "w2a": f"{prefix}.lokr_w2_a",
        "w2b": f"{prefix}.lokr_w2_b",
        "t2": f"{prefix}.lokr_t2",
    }

    values: dict[str, Any] = {}
    for key, name in names.items():
        if name in lora:
            values[key] = lora[name]
            loaded_keys.add(name)

    has_any = any(k in values for k in ("w1", "w2", "w1a", "w2a"))
    if not has_any:
        return

    patch_dict[load_key] = (
        "lokr",
        (
            values.get("w1"),
            values.get("w2"),
            None,  # alpha not used in lokr
            values.get("w1a"),
            values.get("w1b"),
            values.get("w2a"),
            values.get("w2b"),
            values.get("t2"),
        ),
    )


def _try_glora_format(
    lora: dict[str, Any],
    prefix: str,
    load_key: str,
    alpha: Any | None,
    patch_dict: dict[str, Any],
    loaded_keys: set[str],
) -> None:
    a1 = f"{prefix}.a1.weight"
    if a1 not in lora:
        return

    a2 = f"{prefix}.a2.weight"
    b1 = f"{prefix}.b1.weight"
    b2 = f"{prefix}.b2.weight"

    required = {a2, b1, b2}
    if not required.issubset(lora):
        return

    patch_dict[load_key] = ("glora", (lora[a1], lora[a2], lora[b1], lora[b2], alpha))
    loaded_keys.update({a1, a2, b1, b2})


def _try_diff_format(
    lora: dict[str, Any],
    prefix: str,
    load_key: str,
    patch_dict: dict[str, Any],
    loaded_keys: set[str],
) -> None:
    w_norm_name = f"{prefix}.w_norm"
    w_norm = lora.get(w_norm_name)

    if w_norm is not None:
        loaded_keys.add(w_norm_name)
        patch_dict[load_key] = ("diff", (w_norm,))

        b_norm_name = f"{prefix}.b_norm"
        b_norm = lora.get(b_norm_name)
        if b_norm is not None:
            loaded_keys.add(b_norm_name)
            bias_key = f"{load_key[: -len('.weight')]}.bias"
            patch_dict[bias_key] = ("diff", (b_norm,))

    diff_name = f"{prefix}.diff"
    diff_weight = lora.get(diff_name)
    if diff_weight is not None:
        patch_dict[load_key] = ("diff", (diff_weight,))
        loaded_keys.add(diff_name)

    diff_bias_name = f"{prefix}.diff_b"
    diff_bias = lora.get(diff_bias_name)
    if diff_bias is not None:
        bias_key = f"{load_key[: -len('.weight')]}.bias"
        patch_dict[bias_key] = ("diff", (diff_bias,))
        loaded_keys.add(diff_bias_name)
