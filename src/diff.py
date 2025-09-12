
import difflib

def mark_diff(orig: str, corr: str):
    sm = difflib.SequenceMatcher(a=orig.split(), b=corr.split())
    out_a, out_b = [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            out_a += orig.split()[i1:i2]
            out_b += corr.split()[j1:j2]
        elif op in ("replace", "delete"):
            out_a += [f"[{w}]" for w in orig.split()[i1:i2]]
        if op in ("replace", "insert"):
            out_b += [f"**{w}**" for w in corr.split()[j1:j2]]
    return " ".join(out_a), " ".join(out_b)
