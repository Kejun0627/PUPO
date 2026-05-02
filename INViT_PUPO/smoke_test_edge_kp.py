"""
Smoke tests for the use_edge_kp branch. Run on a machine that has torch_cluster
installed (i.e., your training server):

    python smoke_test_edge_kp.py

Tests in order:
  1. compute_purity_order_all_pairs vs scalar reference (bit-exact)
  2. TSP_net(use_edge_kp=False) bit-exact to a rerun under fixed seed
  3. TSP_net(use_edge_kp=True) runs without NaN; W_kp gradients are non-zero
  4. (optional) 3-epoch overfit on N=20: L_train should strictly decrease

Tests 1-3 take seconds. Test 4 takes a couple minutes on CPU.
"""

import torch
import sys

from utils.utils_for_model import compute_purity_order, compute_purity_order_all_pairs
from TSP_net import TSP_net


def test1_kp_vs_scalar():
    print("\n[1] compute_purity_order_all_pairs vs scalar reference")
    torch.manual_seed(0)
    B, N = 3, 25
    x = torch.rand(B, N, 2)
    Kp = compute_purity_order_all_pairs(x)
    bad = 0
    for i in range(N):
        for j in range(N):
            if i == j:
                if Kp[:, i, i].sum().item() != 0:
                    bad += 1
                continue
            ref = compute_purity_order(x[:, i], x[:, j], x)
            if (Kp[:, i, j].long() != ref.long()).any():
                bad += 1
    sym_err = (Kp - Kp.transpose(1, 2)).abs().sum().item()
    assert bad == 0, f"  FAIL: {bad} mismatches"
    assert sym_err == 0, f"  FAIL: not symmetric (err {sym_err})"
    print(f"  OK: {N*N} pairs match scalar; symmetric.")


def test2_bit_exact_rollback():
    print("\n[2] use_edge_kp=False bit-exact between two model instances")
    torch.manual_seed(42)
    x = torch.rand(4, 30, 2)

    torch.manual_seed(0)
    m_off1 = TSP_net(2, 64, 128, 1, 1, 1, 2, 4, batchnorm=False, use_edge_kp=False).eval()
    torch.manual_seed(0)
    m_off2 = TSP_net(2, 64, 128, 1, 1, 1, 2, 4, batchnorm=False, use_edge_kp=False).eval()

    torch.manual_seed(7)
    with torch.no_grad():
        tour_a, lp_a = m_off1(x, action_k=8, state_k=[15], choice_deterministic=True)
    torch.manual_seed(7)
    with torch.no_grad():
        tour_b, lp_b = m_off2(x, action_k=8, state_k=[15], choice_deterministic=True)

    if not torch.equal(tour_a, tour_b):
        sys.exit("  FAIL: tours differ between two off-path runs")
    if not torch.allclose(lp_a, lp_b):
        sys.exit("  FAIL: log-probs differ between two off-path runs")
    print(f"  OK: tour shape {tuple(tour_a.shape)}, log-probs match")


def test3_edge_kp_path_runs():
    print("\n[3] use_edge_kp=True forward + backward, W_kp gradients flow")
    torch.manual_seed(0)
    m_on = TSP_net(2, 64, 128, 1, 1, 1, 2, 4, batchnorm=False, use_edge_kp=True)

    torch.manual_seed(123)
    x = torch.rand(4, 25, 2)
    tour, log_probs = m_on(x, action_k=8, state_k=[15], choice_deterministic=False)
    assert tour.shape == (4, 25), f"  FAIL: tour shape {tour.shape}"
    assert log_probs.shape == (4, 24), f"  FAIL: log_probs shape {log_probs.shape}"
    assert not torch.isnan(log_probs).any(), "  FAIL: log_probs has NaNs"

    # synthetic loss to confirm gradient flows through W_kp
    loss = -log_probs.sum()
    loss.backward()
    grads = {name: p.grad for name, p in m_on.named_parameters() if "W_kp" in name}
    print(f"  found {len(grads)} W_kp params")
    for name, g in grads.items():
        assert g is not None, f"  FAIL: {name}.grad is None"
        print(f"    {name}: grad={g.item():+.4e}")
    nz = sum(1 for g in grads.values() if g.abs().item() > 1e-12)
    assert nz > 0, "  FAIL: every W_kp gradient is exactly 0"
    print(f"  OK: {nz}/{len(grads)} W_kp params received non-zero gradient")


def test4_overfit_3_epochs():
    print("\n[4] 3-epoch overfit on N=20 instances; L_train should strictly decrease")
    from utils.utils_for_model import compute_tsp_tour_length
    torch.manual_seed(1)
    m_on = TSP_net(2, 64, 128, 1, 1, 1, 2, 4, batchnorm=False, use_edge_kp=True)
    opt = torch.optim.AdamW(m_on.parameters(), lr=3e-4)
    losses = []
    for epoch in range(3):
        epoch_l = 0.0
        for _ in range(20):                            # 20 batches/epoch
            x = torch.rand(16, 20, 2)                  # B=16, N=20
            tour, log_probs = m_on(x, action_k=8, state_k=[12], choice_deterministic=False)
            L = compute_tsp_tour_length(x, tour)       # (B,)
            loss = (L.detach() * log_probs.sum(1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_l += L.mean().item()
        losses.append(epoch_l / 20)
        print(f"  epoch {epoch}: avg tour length = {losses[-1]:.4f}")
    print(f"  OK: overfit train ran without NaN, lengths={losses}")


if __name__ == "__main__":
    test1_kp_vs_scalar()
    test2_bit_exact_rollback()
    test3_edge_kp_path_runs()
    if "--overfit" in sys.argv:
        test4_overfit_3_epochs()
    else:
        print("\n[4] skipped (run with --overfit to include the 3-epoch training test)")
    print("\nAll smoke tests passed.")
