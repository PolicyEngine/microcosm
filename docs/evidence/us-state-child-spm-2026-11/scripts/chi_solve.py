"""Build P support: the published solver plus microcosm's chi-square design-weight penalty. Usage: chi_solve.py <lambda>"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.expanduser("~/reviews/pe30-population-2026-09-27/scripts"))
import calib_resolve as cr

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run", "chi")
os.makedirs(OUT, exist_ok=True)
lam = float(sys.argv[1])


def solve(
    A,  # noqa: N803
    b,
    lw,
    w0,
    lam,
    epochs=6000,
    lr=0.02,
    ratio=5.0,
    cap=1.0,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
):
    keep = lw > 0
    A = A[keep].astype(np.float32).tocsr()
    AT = A.T.tocsr()
    b = b[keep].astype(np.float32)
    s = np.maximum(np.abs(b), 1.0).astype(np.float32)
    lwn = (lw[keep] / lw[keep].sum()).astype(np.float32)
    log_w = np.log(w0).astype(np.float32)
    upper = np.log(ratio * w0).astype(np.float32)
    total0 = float(w0.sum())
    dsum = float(w0.sum())
    m = np.zeros_like(log_w)
    v = np.zeros_like(log_w)
    d = w0.astype(np.float32)
    for ep in range(epochs):
        w = np.exp(log_w)
        est = A @ w
        z = (est - b) / s
        az = np.abs(z)
        ge = (lwn * np.sign(z) * (az < cap) / s).astype(np.float32)
        g = (AT @ ge) * w
        # d/dlog w of lambda*sum(d*(w/d-1)^2)/sum(d) = lambda*2*(w/d-1)*w/sum(d)
        g = g + lam * 2 * (w / d - 1) * w / dsum
        t = ep + 1
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        bc1 = 1 - beta1**t
        bc2 = 1 - beta2**t
        log_w = (log_w - (lr / bc1) * m / (np.sqrt(v) / np.sqrt(bc2) + eps)).astype(
            np.float32
        )
        log_w = np.minimum(log_w, upper)
        cur = float(np.exp(log_w).sum(dtype=np.float64))
        if cur > 0:
            log_w = (log_w + np.float32(np.log(total0 / cur))).astype(np.float32)
            log_w = np.minimum(log_w, upper)
    final = np.exp(log_w.astype(np.float64))
    final = np.minimum(final, ratio * w0)
    return cr.project_to_total(final, total0, ratio, w0)


diag, T, A, z, lw, fam, b = cr.load()
lw = lw.copy()
lw[fam == "jct"] = 0.0
w0 = z["initial_household_weight"].astype(np.float64)
t = time.time()
w = solve(A, b, lw, w0, lam)
keep = lw > 0
est = A @ w
e = np.abs(est - b) / np.maximum(np.abs(b), 1.0)
print(
    lam,
    "seconds",
    round(time.time() - t),
    "loss",
    round(float((np.minimum(e[keep], 1) * lw[keep]).sum() / lw[keep].sum()), 4),
    "within10",
    round(float((e[keep] < 0.1).mean()), 3),
    "kish",
    round(w.sum() ** 2 / (w**2).sum()),
    flush=True,
)
np.savez_compressed(f"{OUT}/l{lam}.npz", weights=w)
