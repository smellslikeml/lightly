from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch import distributed as torch_dist
from torch.distributed import nn as torch_dist_nn

from lightly.loss import MCRLoss
from lightly.models.modules.heads import ProjectionHead


def l2_normalize(features: Tensor) -> Tensor:
    """L2-normalizes features along the last dimension, as the SimDINO head does."""
    return F.normalize(features, dim=-1, p=2)


def random_views(
    num_views: int, batch_size: int = 8, feature_dim: int = 16
) -> list[Tensor]:
    """Creates L2-normalized features for the given number of views."""
    return [
        l2_normalize(torch.randn(batch_size, feature_dim)) for _ in range(num_views)
    ]


class ReferenceMCRLoss(nn.Module):
    """MCRLoss from the SimDINO reference implementation.

    Port (modulo type annotations and formatting) of the MCRLoss class in [0]
    (MIT license, Copyright (c) 2025 Ziyang Wu), used to verify the numerical
    parity of the lightly implementation. Unlike the lightly version, it takes
    the concatenated multi-crop features and assumes two teacher views.

    - [0]: https://github.com/RobinWu218/SimDINO/blob/main/simdino/main_dino.py
    """

    def __init__(
        self,
        ncrops: int,
        reduce_cov: int = 0,
        expa_type: int = 0,
        eps: float = 0.5,
        coeff: float = 1.0,
    ) -> None:
        super().__init__()
        self.ncrops = ncrops
        self.eps = eps
        self.coeff = coeff
        self.reduce_cov = reduce_cov
        self.expa_type = expa_type

    def forward(
        self, student_feat: Tensor, teacher_feat: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        student_feat = student_feat.view(self.ncrops, -1, student_feat.shape[-1])
        teacher_feat = teacher_feat.view(2, -1, teacher_feat.shape[-1])

        comp_loss = self.calc_compression(student_feat, teacher_feat)
        if self.expa_type == 0:  # only compute expansion on global views
            expa_loss = self.calc_expansion(student_feat[: len(teacher_feat)])
        elif self.expa_type == 1:
            expa_loss = self.calc_expansion(
                (student_feat[: len(teacher_feat)] + teacher_feat) / 2
            )
        loss = -self.coeff * comp_loss - expa_loss
        return loss, comp_loss.detach(), expa_loss.detach()

    def calc_compression(
        self, student_feat_list: Tensor, teacher_feat_list: Tensor
    ) -> Tensor:
        sim = F.cosine_similarity(
            teacher_feat_list.unsqueeze(1), student_feat_list.unsqueeze(0), dim=-1
        )
        # The reference zeroes the same-view (diagonal) pairs through a strided
        # view; zeroing them by index computes the same entries.
        diag = min(len(teacher_feat_list), len(student_feat_list))
        sim[torch.arange(diag), torch.arange(diag)] = 0

        n_loss_terms = len(teacher_feat_list) * len(student_feat_list) - min(
            len(teacher_feat_list), len(student_feat_list)
        )
        comp_loss = sim.mean(2).sum() / n_loss_terms
        return comp_loss

    def calc_expansion(self, feat_list: Tensor) -> Tensor:
        num_views = len(feat_list)
        m, p = feat_list[0].shape

        cov_list = torch.stack([W.T.matmul(W) for W in feat_list])
        N = 1
        if torch_dist.is_initialized():
            N = torch_dist.get_world_size()
            if self.reduce_cov == 1:
                cov_list = torch_dist_nn.all_reduce(cov_list)
        scalar = p / (m * N * self.eps)
        I = torch.eye(p, device=cov_list[0].device)
        loss = cov_list.new_zeros(())
        for i in range(num_views):
            chol = torch.linalg.cholesky_ex(I + scalar * cov_list[i])[0]
            loss += chol.diagonal().log().sum()
        loss /= num_views
        loss *= (p + N * m) / (p * N * m)
        return loss


class TestMCRLoss:
    def test_forward__returns_scalar_loss(self) -> None:
        criterion = MCRLoss()
        teacher_out = random_views(2)
        student_out = random_views(6)
        loss = criterion(teacher_out=teacher_out, student_out=student_out)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_forward__expa_type_1_returns_scalar_loss(self) -> None:
        criterion = MCRLoss(expa_type=1)
        teacher_out = random_views(2)
        student_out = random_views(6)
        loss = criterion(teacher_out=teacher_out, student_out=student_out)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda")
    def test_forward_cuda(self) -> None:
        criterion = MCRLoss()
        teacher_out = [view.cuda() for view in random_views(2)]
        student_out = [view.cuda() for view in random_views(6)]
        loss = criterion(teacher_out=teacher_out, student_out=student_out)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_forward__gradient_flows_to_student_and_teacher(self) -> None:
        # SimDINO does not detach the teacher features, unlike DINOLoss.
        criterion = MCRLoss()
        teacher_features = [torch.randn(8, 16, requires_grad=True) for _ in range(2)]
        student_features = [torch.randn(8, 16, requires_grad=True) for _ in range(6)]
        teacher_out = [l2_normalize(features) for features in teacher_features]
        student_out = [l2_normalize(features) for features in student_features]
        loss = criterion(teacher_out=teacher_out, student_out=student_out)
        loss.backward()
        for features in teacher_features + student_features:
            assert features.grad is not None
            assert torch.isfinite(features.grad).all()

    def test_calc_compression__is_maximal_for_aligned_features(self) -> None:
        # If all views hold the same features, every cross-view pair has
        # cosine similarity one.
        criterion = MCRLoss()
        view = [l2_normalize(torch.randn(8, 16))] * 3
        comp = criterion.calc_compression(
            teacher_feat_list=view[:2], student_feat_list=view
        )
        assert torch.allclose(comp, torch.ones_like(comp))

    def test_calc_expansion__is_positive_for_normalized_features(self) -> None:
        # The covariance of normalized features is positive semi-definite, so
        # the log-determinant of I + scalar * Z^T Z is positive.
        criterion = MCRLoss()
        feat_list = random_views(2)
        expansion = criterion.calc_expansion(feat_list=feat_list)
        assert expansion.item() > 0.0

    def test_forward__works_when_distributed_is_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The distributed guard must short-circuit so that the loss also works
        # without an initialized process group and on torch builds that ship
        # without distributed support.
        teacher_out = random_views(2)
        student_out = random_views(6)
        criterion = MCRLoss(reduce_cov=True)
        expected = criterion(teacher_out=teacher_out, student_out=student_out)

        monkeypatch.setattr(torch_dist, "is_available", lambda: False)
        loss = criterion(teacher_out=teacher_out, student_out=student_out)
        assert torch.allclose(loss, expected)

    def test_init__raises_for_invalid_arguments(self) -> None:
        with pytest.raises(ValueError, match="eps must be greater than 0"):
            MCRLoss(eps=0.0)
        with pytest.raises(ValueError, match="expa_type must be 0 or 1"):
            MCRLoss(expa_type=2)

    def test_forward__raises_for_single_view(self) -> None:
        criterion = MCRLoss()
        view = random_views(1)
        with pytest.raises(ValueError, match="no cross-view terms"):
            criterion(teacher_out=view, student_out=view)

    def test_forward__raises_when_student_sees_fewer_views_than_teacher(self) -> None:
        criterion = MCRLoss()
        teacher_out = random_views(2)
        student_out = random_views(1)
        with pytest.raises(ValueError, match="at least as many views"):
            criterion(teacher_out=teacher_out, student_out=student_out)


class TestMCRLossParity:
    """Numerical parity with the reference SimDINO implementation on a fixed seed."""

    @staticmethod
    def _seeded_inputs() -> tuple[list[Tensor], list[Tensor]]:
        torch.manual_seed(42)
        teacher_out = random_views(2)
        student_out = random_views(6)
        return teacher_out, student_out

    def test_forward__matches_reference_expa_type_0(self) -> None:
        teacher_out, student_out = self._seeded_inputs()
        loss_fn = MCRLoss(expa_type=0)
        reference = ReferenceMCRLoss(ncrops=len(student_out), expa_type=0)

        loss = loss_fn(teacher_out=teacher_out, student_out=student_out)
        ref_loss, ref_comp, ref_exp = reference(
            torch.cat(student_out), torch.cat(teacher_out)
        )

        assert torch.allclose(loss, ref_loss, atol=1e-6)
        comp = loss_fn.calc_compression(
            teacher_feat_list=teacher_out, student_feat_list=student_out
        )
        assert torch.allclose(comp, ref_comp, atol=1e-6)
        exp = loss_fn.calc_expansion(feat_list=student_out[: len(teacher_out)])
        assert torch.allclose(exp, ref_exp, atol=1e-6)

    def test_forward__matches_reference_expa_type_1(self) -> None:
        teacher_out, student_out = self._seeded_inputs()
        loss_fn = MCRLoss(expa_type=1)
        reference = ReferenceMCRLoss(ncrops=len(student_out), expa_type=1)

        loss = loss_fn(teacher_out=teacher_out, student_out=student_out)
        ref_loss, _, ref_exp = reference(
            torch.cat(student_out), torch.cat(teacher_out)
        )

        assert torch.allclose(loss, ref_loss, atol=1e-6)
        smoothed = [
            (student + teacher) / 2
            for student, teacher in zip(student_out[: len(teacher_out)], teacher_out)
        ]
        exp = loss_fn.calc_expansion(feat_list=smoothed)
        assert torch.allclose(exp, ref_exp, atol=1e-6)

    def test_forward__matches_reference_custom_hyperparameters(self) -> None:
        teacher_out, student_out = self._seeded_inputs()
        loss_fn = MCRLoss(eps=0.2, coeff=2.0, expa_type=1)
        reference = ReferenceMCRLoss(
            ncrops=len(student_out), expa_type=1, eps=0.2, coeff=2.0
        )

        loss = loss_fn(teacher_out=teacher_out, student_out=student_out)
        ref_loss, ref_comp, ref_exp = reference(
            torch.cat(student_out), torch.cat(teacher_out)
        )

        assert torch.allclose(loss, ref_loss, atol=1e-6)
        comp = loss_fn.calc_compression(
            teacher_feat_list=teacher_out, student_feat_list=student_out
        )
        assert torch.allclose(comp, ref_comp, atol=1e-6)
        smoothed = [
            (student + teacher) / 2
            for student, teacher in zip(student_out[: len(teacher_out)], teacher_out)
        ]
        exp = loss_fn.calc_expansion(feat_list=smoothed)
        assert torch.allclose(exp, ref_exp, atol=1e-6)


class TestMCRLossWithProjectionHead:
    def test_forward__with_simplified_dino_head(self) -> None:
        # SimDINO's head is a projection head bottleneck without DINO's
        # prototype layer, followed by L2 normalization.
        head = ProjectionHead(
            [
                (16, 32, None, nn.GELU()),
                (32, 16, None, None),
            ]
        )
        loss_fn = MCRLoss()

        torch.manual_seed(0)
        teacher_out = [l2_normalize(head(torch.randn(8, 16))) for _ in range(2)]
        student_out = [l2_normalize(head(torch.randn(8, 16))) for _ in range(6)]

        loss = loss_fn(teacher_out=teacher_out, student_out=student_out)
        loss.backward()

        assert loss.ndim == 0
        assert torch.isfinite(loss)
        for param in head.parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()
