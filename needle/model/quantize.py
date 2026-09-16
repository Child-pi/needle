import functools
import math
import re

import numpy as np
import jax
import jax.numpy as jnp


def _leaf_key(path):
    """Return the owning parameter key, skipping Flax metadata wrappers."""
    for entry in reversed(path):
        if hasattr(entry, "key"):
            return str(entry.key)
    return ""


def fake_quant(w, group_size=128, bits=4):
    qmax = 2 ** (bits - 1) - 1    
    D = w.shape[-1]
    pad = (-D) % group_size
    wp = jnp.pad(w, [(0, 0)] * (w.ndim - 1) + [(0, pad)]) if pad else w
    g = wp.reshape(*wp.shape[:-1], -1, group_size).astype(jnp.float32)
    absmax = jnp.max(jnp.abs(g), axis=-1, keepdims=True)
    scale = jnp.where(absmax > 0, absmax / qmax, 1.0)
    q = jnp.clip(jnp.round(g / scale), -qmax - 1, qmax) * scale
    q = q.reshape(wp.shape).astype(w.dtype)
    if pad:
        q = q[..., :D]
    return w + jax.lax.stop_gradient(q - w)  


def fake_quant_act(x):
    return fake_quant(x, x.shape[-1], ACT_BITS)


def cq_fake_quant_kv(x, bits, group=64):
    return x + jax.lax.stop_gradient(cq_quantize(x, bits, group) - x)


def a8_fake_quant_kv(x):
    """Per-head symmetric A8 KV quantization matching the native cache."""
    return fake_quant(x, x.shape[-1], 8)


def maybe_quant_query(x, quant):
    """Per-head A8 query quantization matching the native attention kernel."""
    if not ACT_BITS:
        return x
    return jax.lax.cond(quant, a8_fake_quant_kv, lambda t: t, x)


def maybe_quant_kv(x, quant):
    if not KV_BITS:
        return x
    quantize = (a8_fake_quant_kv if KV_BITS >= 8 else
                lambda t: cq_fake_quant_kv(t, KV_BITS, _KV_GROUP))
    return jax.lax.cond(
        quant, quantize, lambda t: t, x)


QAT_EVERY = 0
_WEIGHT_GROUP = 128
_WEIGHT_BITS = 4
ACT_BITS = 8
KV_BITS = 0
_KV_GROUP = 64


def configure_qat(every, weight_group=128, weight_bits=4):
    global QAT_EVERY, _WEIGHT_GROUP, _WEIGHT_BITS
    QAT_EVERY, _WEIGHT_GROUP, _WEIGHT_BITS = int(every), int(weight_group), int(weight_bits)


def configure_deploy(act_bits=8, kv_bits=8, kv_group=64):
    global ACT_BITS, KV_BITS, _KV_GROUP
    kv_bits = int(kv_bits)
    changed = (ACT_BITS, KV_BITS, _KV_GROUP) != (int(act_bits), int(kv_bits), int(kv_group))
    ACT_BITS, KV_BITS, _KV_GROUP = int(act_bits), int(kv_bits), int(kv_group)
    if changed:
        jax.clear_caches()


def quantize_params_configured(params):
    return cq_ste_params(params, _WEIGHT_BITS, _WEIGHT_GROUP)


def deploy_quantize(params, config):
    spec = getattr(config, "weight_bits", "") or ""
    if spec:
        bits_map, default_bits = parse_bits_map(spec)
        return cq_mixed_params(params, bits_map, default_bits), spec
    return (cq_quantize_params(params, _WEIGHT_BITS, _WEIGHT_GROUP),
            f"CQ W{_WEIGHT_BITS}")


def weight_bits():
    return _WEIGHT_BITS


def maybe_quant_weights(params, do_quantize):
    if not QAT_EVERY:
        return params
    return jax.lax.cond(
        do_quantize, quantize_params_configured, lambda p: p, params)


@functools.lru_cache(maxsize=None)
def _lloyd_max_gaussian(bits, iters=200, samples=400000, seed=0):
    if bits == 1:
        c = math.sqrt(2.0 / math.pi)
        return np.array([-c, c])
    levels = 1 << bits
    x = np.sort(np.random.RandomState(seed).randn(samples))
    c = x[((np.arange(levels) + 0.5) / levels * samples).astype(int)].astype(np.float64)
    for _ in range(iters):
        bnd = (c[:-1] + c[1:]) / 2.0
        idx = np.searchsorted(bnd, x)
        for k in range(levels):
            m = idx == k
            if m.any():
                c[k] = x[m].mean()
    return np.sort(c)

TERNARY_BITS = 1.58
_TERNARY_CB = np.array([-1.2240064, 0.0, 1.2240064])


@functools.lru_cache(maxsize=None)
def _cq_codebook_np(bits, group_size):
    cb = _TERNARY_CB if bits == TERNARY_BITS else _lloyd_max_gaussian(bits)
    return (cb / np.sqrt(group_size)).astype(np.float32)


@functools.lru_cache(maxsize=None)
def _cq_hadamard_np(group_size):
    H = np.array([[1.0]], dtype=np.float32)
    while H.shape[0] < group_size:
        H = np.block([[H, H], [H, -H]])
    return (H / np.sqrt(group_size)).astype(np.float32)


def _cq_nearest(x, cb):
    flat = x.reshape(-1)
    pos = jnp.clip(jnp.searchsorted(cb, flat), 1, cb.shape[0] - 1)
    left, right = cb[pos - 1], cb[pos]
    idx = jnp.where(jnp.abs(flat - left) <= jnp.abs(flat - right), pos - 1, pos)
    return cb[idx].reshape(x.shape)


def cq_quantize(w, bits, group_size=128, codebook=None):
    cb = codebook if codebook is not None else jnp.asarray(_cq_codebook_np(bits, group_size))
    D, g = w.shape[-1], group_size
    pad = (-D) % g
    wp = jnp.pad(w, [(0, 0)] * (w.ndim - 1) + [(0, pad)]) if pad else w
    groups = wp.reshape(*wp.shape[:-1], -1, g).astype(jnp.float32)
    H = jnp.asarray(_cq_hadamard_np(g))
    rot = groups @ H
    norm = jnp.sqrt(jnp.sum(rot ** 2, axis=-1, keepdims=True))
    unit = rot / jnp.maximum(norm, 1e-12)
    norm = norm.astype(jnp.float16).astype(jnp.float32)
    deq = (_cq_nearest(unit, cb) * norm) @ H
    deq = deq.reshape(wp.shape).astype(w.dtype)
    return deq[..., :D] if pad else deq


def _is_quant_leaf(path, leaf):
    key = _leaf_key(path)
    return ((key in ("kernel", "embedding") or key.startswith("mhc_phi"))
            and getattr(leaf, "ndim", 0) >= 2)


def _reduces_second_last(path):
    key = _leaf_key(path)
    return key == "kernel" or key.startswith("mhc_phi")


def cq_quantize_params(params, bits, group_size=128):
    return _map_quant_leaves(
        params, lambda w, i: cq_quantize(w, bits, group_size))


AB_KEY = "ab_scales"


def _ab_of(params):
    return params.get(AB_KEY) if hasattr(params, "get") else None


def ab_init_params(params, seed=0, signs=False):
    """a = b = 1 (exact no-op at step 0; the PTQ default) or, with signs=True,
    a = b = random +-1 (randomized-Hadamard init for from-scratch/SFT runs).
    a*b = 1 either way, so effective FP weights are unchanged. A tree that
    already carries AB (a resumed QAPT checkpoint) is returned unchanged."""
    if _ab_of(params):
        return dict(params)
    rng = np.random.RandomState(seed)
    ab = {}

    def visit(path, leaf):
        if _is_quant_leaf(path, leaf):
            r = leaf.shape[-2] if _reduces_second_last(path) else leaf.shape[-1]
            v = ((rng.randint(0, 2, r) * 2 - 1) if signs else np.ones(r))
            ab[leaf_name(path)] = {"a": jnp.asarray(v, jnp.float32),
                                   "b": jnp.asarray(v, jnp.float32)}
        return leaf

    jax.tree_util.tree_map_with_path(visit, params)
    out = dict(params)
    out[AB_KEY] = ab
    return out


def freeze_quant_grads(grads):
    """Zero the gradients of every CQ-quantized leaf (signs stay frozen);
    norms, gates, AB scales and other smooth params keep theirs."""
    def z(path, leaf):
        return jnp.zeros_like(leaf) if _is_quant_leaf(path, leaf) else leaf
    return jax.tree_util.tree_map_with_path(z, grads)


def ab_fold_params(params):
    """FP-effective tree of an AB-carrying model (w <- a*b*w, AB_KEY dropped):
    the clean-forward reference of the function the quant transforms deploy."""
    if not _ab_of(params):
        return params
    folded = dict(_map_quant_leaves(params, lambda w, i: w))
    folded.pop(AB_KEY)
    return folded


def cq_model_bytes(params, bits, group_size=128):
    acc = {"q": 0, "groups": 0, "other": 0}

    def visit(path, leaf):
        n = int(np.prod(leaf.shape))
        if _is_quant_leaf(path, leaf):
            acc["q"] += n
            per = leaf.shape[-2] if _reduces_second_last(path) else leaf.shape[-1]
            acc["groups"] += (n // per) * (-(-per // group_size))
        else:
            acc["other"] += n
        return leaf

    jax.tree_util.tree_map_with_path(visit, params)
    return acc["q"] * bits / 8 + acc["groups"] * 2 + acc["other"] * 2


def model_bytes_fp16(params):
    return 2 * sum(int(np.prod(l.shape)) for l in jax.tree_util.tree_leaves(params))




CQ_GROUP_SIZE = 128
CQ_BITS = (1, 2, 3, 4)
MIN_BITS, MAX_BITS = 1, 8


def canonical_tensor_name(name):
    n = re.sub(r"^layer\d+\.", "attn.", name)
    n = re.sub(r"^engrams?_?(\d+)[./]?", r"engram\1.", n)
    if n.startswith("stack/"):
        n = n[len("stack/"):]
    n = n.replace("layers/block/self_attn/", "attn.")
    n = n.replace("mtp_block/self_attn/", "mtp.")
    if n.endswith("/kernel"):
        n = n[:-len("/kernel")]
    if n == "embedding/embedding":
        n = "embedding"
    n = re.sub(r"^(engram\d+\.)embedding$", r"\1tables", n)
    return n.replace("/", ".")


def _bits_for(name, bits_map, default_bits):
    canon = canonical_tensor_name(name)
    hits = [k for k in bits_map if canon.startswith(k)]
    return bits_map[max(hits, key=len)] if hits else default_bits


@functools.lru_cache(maxsize=None)
def cq_distortion(bits, group_size=CQ_GROUP_SIZE, rows=2048, seed=0):
    w = np.random.RandomState(seed).randn(rows, 4 * group_size).astype(np.float32)
    deq = np.asarray(cq_quantize(jnp.asarray(w), bits, group_size))
    return float(np.mean((deq - w) ** 2) / np.mean(w ** 2))


def noise_scale(bits, group_size=CQ_GROUP_SIZE):
    b = min(max(float(bits), MIN_BITS), MAX_BITS)
    lo, hi = int(np.floor(b)), int(np.ceil(b))
    d_lo = cq_distortion(lo, group_size)
    if hi == lo:
        return float(np.sqrt(d_lo))
    d_hi = cq_distortion(hi, group_size)
    t = b - lo
    return float(np.sqrt(np.exp((1.0 - t) * np.log(d_lo) + t * np.log(d_hi))))


def add_cq_noise(w, key, scale, group_size=CQ_GROUP_SIZE):
    D, g = w.shape[-1], group_size
    pad = (-D) % g
    wp = jnp.pad(w, [(0, 0)] * (w.ndim - 1) + [(0, pad)]) if pad else w
    groups = wp.reshape(*wp.shape[:-1], -1, g).astype(jnp.float32)
    rms = jnp.sqrt(jnp.mean(jnp.square(groups), axis=-1, keepdims=True))
    sigma = jax.lax.stop_gradient(rms) * scale
    eps = jax.random.normal(key, groups.shape, dtype=jnp.float32)
    noisy = (groups + sigma * eps).reshape(wp.shape).astype(w.dtype)
    return noisy[..., :D] if pad else noisy


def _map_quant_leaves(params, fn):
    """fn(leaf_with_reduction_axis_last, index) over the CQ-quantized leaves;
    AB scales, when present, sandwich fn: leaf -> a * fn(b * leaf)."""
    ab = _ab_of(params)
    counter = [0]

    def q(path, leaf):
        if not _is_quant_leaf(path, leaf):
            return leaf
        i = counter[0]
        counter[0] += 1
        s = ab.get(leaf_name(path)) if ab else None

        def apply(w):
            out = fn(w * s["b"] if s else w, i)
            return out * s["a"] if s else out

        if _reduces_second_last(path):
            return jnp.swapaxes(apply(jnp.swapaxes(leaf, -1, -2)), -1, -2)
        return apply(leaf)

    return jax.tree_util.tree_map_with_path(q, params)


def noise_params(params, key, scale, group_size=CQ_GROUP_SIZE):
    return _map_quant_leaves(
        params,
        lambda w, i: add_cq_noise(w, jax.random.fold_in(key, i), scale, group_size),
    )


def noise_params_only(params, key, scale, name, group_size=CQ_GROUP_SIZE):
    names = [n for n, _ in quant_leaf_names(params)]
    return _map_quant_leaves(
        params,
        lambda w, i: (add_cq_noise(w, key, scale, group_size)
                      if names[i].startswith(name) else w),
    )


def leaf_name(path):
    # Partitioned/AxisMetadata leaves add a trailing GetAttrKey("value").
    # That is container metadata, not part of the checkpoint parameter name.
    return "/".join(
        str(p.key) if hasattr(p, "key") else str(p.idx)
        for p in path
        if hasattr(p, "key") or hasattr(p, "idx")
    )


def quant_leaf_names(params):
    out = []

    def visit(path, leaf):
        if _is_quant_leaf(path, leaf):
            out.append((leaf_name(path), int(np.prod(leaf.shape))))
        return leaf

    jax.tree_util.tree_map_with_path(visit, params)
    return out


def parse_bits_map(spec):
    m, default = {}, None
    for part in str(spec or "").split(","):
        if not part.strip():
            continue
        k, v = part.split("=", 1)
        b = TERNARY_BITS if float(v) == TERNARY_BITS else int(v)
        if k.strip() == "default":
            default = b
        else:
            m[canonical_tensor_name(k.strip())] = b
    if default is None:
        raise ValueError(f"bits map {spec!r} needs a default=<b> entry")
    bad = {b for b in list(m.values()) + [default]
           if b not in CQ_BITS and b != TERNARY_BITS}
    if bad:
        raise ValueError(f"bits map {spec!r} has unsupported widths {sorted(bad)}")
    return m, default


def cq_mixed_params(params, bits_map, default_bits, group_size=CQ_GROUP_SIZE):
    names = [n for n, _ in quant_leaf_names(params)]

    def fn(w, i):
        b = _bits_for(names[i], bits_map, default_bits)
        return cq_quantize(w, b, group_size)

    return _map_quant_leaves(params, fn)


def cq_mixed_stats(params, bits_map, default_bits, group_size=CQ_GROUP_SIZE):
    qbits = qn = 0
    other = 0
    groups = 0

    def visit(path, leaf):
        nonlocal qbits, qn, other, groups
        n = int(np.prod(leaf.shape))
        if _is_quant_leaf(path, leaf):
            b = _bits_for(leaf_name(path), bits_map, default_bits)
            qbits += b * n
            qn += n
            per = leaf.shape[-2] if _reduces_second_last(path) else leaf.shape[-1]
            groups += (n // per) * (-(-per // group_size))
        else:
            other += n
        return leaf

    jax.tree_util.tree_map_with_path(visit, params)
    mb = (qbits / 8 + groups * 2 + other * 2) / 1e6
    return qbits / max(qn, 1), mb


def cq_ste(w, bits, group_size=CQ_GROUP_SIZE):
    return w + jax.lax.stop_gradient(cq_quantize(w, bits, group_size) - w)


HEAD_BITS = 4


def head_weight(w, quant, reduce_second_last=False, group_size=CQ_GROUP_SIZE):
    """A probe-head matrix as the engine sees it: CQ at HEAD_BITS under QAT,
    straight-through for the gradient; kernels quantize along their input axis."""
    def fake(x):
        if reduce_second_last:
            return jnp.swapaxes(
                cq_ste(jnp.swapaxes(x, -1, -2), HEAD_BITS, group_size), -1, -2)
        return cq_ste(x, HEAD_BITS, group_size)
    return jax.lax.cond(quant, fake, lambda x: x, w)


def cq_ste_params(params, bits, group_size=CQ_GROUP_SIZE):
    return _map_quant_leaves(params, lambda w, i: cq_ste(w, bits, group_size))


def cq_ste_mixed_params(params, bits_map, default_bits, group_size=CQ_GROUP_SIZE):
    names = [n for n, _ in quant_leaf_names(params)]
    return _map_quant_leaves(
        params,
        lambda w, i: cq_ste(w, _bits_for(names[i], bits_map, default_bits), group_size))


def kurtosis_penalty(params, group_size=None):
    g = _WEIGHT_GROUP if group_size is None else group_size
    ab = _ab_of(params)
    H = jnp.asarray(_cq_hadamard_np(g))
    terms = []

    def visit(path, leaf):
        if not _is_quant_leaf(path, leaf):
            return leaf
        s = ab.get(leaf_name(path)) if ab else None
        w = jnp.swapaxes(leaf, -1, -2) if _reduces_second_last(path) else leaf
        if s is not None:
            w = w * s["b"]
        pad = (-w.shape[-1]) % g
        wp = jnp.pad(w, [(0, 0)] * (w.ndim - 1) + [(0, pad)]) if pad else w
        rot = wp.reshape(*wp.shape[:-1], -1, g).astype(jnp.float32) @ H
        m2 = jnp.mean(rot ** 2, axis=-1)
        m4 = jnp.mean(rot ** 4, axis=-1)
        kurt = m4 / jnp.maximum(m2 ** 2, 1e-12)
        dev = (kurt - 3.0) ** 2
        terms.append((jnp.sum(dev), dev.size))
        return leaf

    jax.tree_util.tree_map_with_path(visit, params)
    return sum(t for t, _ in terms) / max(sum(n for _, n in terms), 1)


def attractor_penalty(params, bits=None, group_size=None):
    b = _WEIGHT_BITS if bits is None else bits
    g = _WEIGHT_GROUP if group_size is None else group_size
    terms = []

    def fn(w, i):
        q = jax.lax.stop_gradient(cq_quantize(w, b, g))
        d = (w - q).astype(jnp.float32)
        ref = jax.lax.stop_gradient(w.astype(jnp.float32))
        terms.append((jnp.sum(d * d), jnp.sum(ref * ref)))
        return w

    _map_quant_leaves(params, fn)
    num = sum(t for t, _ in terms)
    den = sum(r for _, r in terms)
    return num / jnp.maximum(den, 1e-12)
