"""MCR loss from SimDINO.

- [0]: SimDINO: Simplifying DINO via Coding Rate Regularization, 2025, https://arxiv.org/abs/2502.10385
- [1]: https://github.com/RobinWu218/SimDINO
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import distributed as torch_dist
from torch.distributed import nn as torch_dist_nn
from torch.nn import Module


class MCRLoss(Module):
    """Implementation of the SimDINO loss [0] using coding rate regularization.

    SimDINO (v1) replaces DINO's prototype layer, centering, and softmax
    cross-entropy with a coding rate loss [2] on the projection head features.
    It combines two terms: a compression term that pulls student and teacher
    features of the same image together across views, and an expansion term
    that maximizes the coding rate of the features to prevent collapse.

    This implementation is ported from the reference implementation [1] (MIT
    license, Copyright (c) 2025 Ziyang Wu). It expects L2-normalized features
    as produced by the simplified DINO head of SimDINO: a projection head
    bottleneck without the prototype layer, followed by L2 normalization.
    DINO's multi-crop transform is reused as is; the teacher only sees the
    global views while the student sees all of them.

    - [0]: SimDINO: Simplifying DINO via Coding Rate Regularization, 2025, https://arxiv.org/abs/2502.10385
    - [1]: https://github.com/RobinWu218/SimDINO
    - [2]: Learning Diverse and Discriminative Representations via the Principle of Maximal Coding Rate Reduction, 2020, https://arxiv.org/abs/2006.08558

    Attributes:
        eps:
            Epsilon of the coding rate expansion term. The expansion term uses
            the scalar ``feature_dim / (batch_size * world_size * eps)``.
        coeff:
            Coefficient of the compression term.
        expa_type:
            Features the expansion term is computed on: ``0`` uses the
            student's global views only, ``1`` uses the average of the student
            and teacher global views (smoothing). The reference class defaults
            to ``0`` while its training script uses ``1``.
        reduce_cov:
            If True, the feature covariance matrices are summed across all
            distributed processes before the expansion term is computed, so
            that the coding rate is estimated on the global batch. Has no
            effect if no default process group is initialized.

    Examples:
        >>> # initialize loss function
        >>> loss_fn = MCRLoss()
        >>>
        >>> # generate two global and several local views with a DINO transform
        >>> views = transform(images)
        >>>
        >>> # embed the global views with the teacher and all views with the
        >>> # student; both heads end with L2 normalization
        >>> teacher_out = [teacher(view) for view in views[:2]]
        >>> student_out = [student(view) for view in views]
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(teacher_out, student_out)
    """

    def __init__(
        self,
        eps: float = 0.5,
        coeff: float = 1.0,
        expa_type: int = 0,
        reduce_cov: bool = False,
    ) -> None:
        """Initializes the MCRLoss module.

        Args:
            eps:
                Epsilon of the coding rate expansion term.
            coeff:
                Coefficient of the compression term.
            expa_type:
                Whether to smooth the expansion term: ``0`` computes it on the
                student's global views only, ``1`` on the average of the
                student and teacher global views.
            reduce_cov:
                If True, sum the feature covariance matrices across all
                distributed processes.

        Raises:
            ValueError: If eps is not greater than 0 or expa_type is not 0
                or 1.
        """
        super().__init__()
        if eps <= 0:
            raise ValueError(f"eps must be greater than 0, got {eps}")
        if expa_type not in (0, 1):
            raise ValueError(f"expa_type must be 0 or 1, got {expa_type}")
        self.eps = eps
        self.coeff = coeff
        self.expa_type = expa_type
        self.reduce_cov = reduce_cov

    def forward(self, teacher_out: list[Tensor], student_out: list[Tensor]) -> Tensor:
        """Computes the SimDINO loss as negative compression minus expansion.

        Args:
            teacher_out:
                List of tensors with shape (batch_size, feature_dim)
                containing the L2-normalized teacher features. Each tensor
                must represent one global view of the batch.
            student_out:
                List of tensors with shape (batch_size, feature_dim)
                containing the L2-normalized student features. Each tensor
                must represent one view of the batch and the first views must
                correspond to the teacher's global views.

        Returns:
            The combined compression and expansion loss.

        Raises:
            ValueError: If the student sees fewer views than the teacher, or
                if there are no cross-view pairs to compute the compression
                term from.
        """
        if len(student_out) < len(teacher_out):
            raise ValueError(
                "MCRLoss requires the student to see at least as many views "
                f"as the teacher, got {len(student_out)} student view(s) and "
                f"{len(teacher_out)} teacher view(s)."
            )

        comp_loss = self.calc_compression(
            teacher_feat_list=teacher_out, student_feat_list=student_out
        )
        if self.expa_type == 0:
            # Expansion on the student's global views only.
            expa_features = student_out[: len(teacher_out)]
        else:
            # Smoothed expansion on the mean of student and teacher views.
            expa_features = [
                (student + teacher) / 2
                for student, teacher in zip(
                    student_out[: len(teacher_out)], teacher_out
                )
            ]
        expa_loss = self.calc_expansion(feat_list=expa_features)
        return -self.coeff * comp_loss - expa_loss

    def calc_compression(
        self, teacher_feat_list: list[Tensor], student_feat_list: list[Tensor]
    ) -> Tensor:
        """Computes the compression term between teacher and student features.

        The term is the mean cosine similarity between teacher and student
        features over all cross-view pairs, excluding the pairs where both
        networks see the same view.

        Args:
            teacher_feat_list:
                List of tensors with shape (batch_size, feature_dim)
                containing the teacher features of the global views.
            student_feat_list:
                List of tensors with shape (batch_size, feature_dim)
                containing the student features of all views.

        Returns:
            The compression term.

        Raises:
            ValueError: If there are no cross-view pairs to compute the
                compression term from.
        """
        teacher = torch.stack(teacher_feat_list)
        student = torch.stack(student_feat_list)
        # t = n_views_teacher, s = n_views_student, b = batch_size
        sim = F.cosine_similarity(teacher.unsqueeze(1), student.unsqueeze(0), dim=-1)
        # Zero the entries where student and teacher see the same view. Viewed
        # row-major over (t, s), the same-view pairs form the diagonal.
        diag = min(teacher.shape[0], student.shape[0])
        sim[torch.arange(diag), torch.arange(diag)] = 0

        n_loss_terms = teacher.shape[0] * student.shape[0] - min(
            teacher.shape[0], student.shape[0]
        )
        if n_loss_terms == 0:
            raise ValueError(
                "MCRLoss requires at least two views in total (the diagonal "
                "matching the same view index is excluded), but got "
                f"{len(teacher_feat_list)} teacher view(s) and "
                f"{len(student_feat_list)} student view(s), which leaves no "
                "cross-view terms to compute the compression term from."
            )
        return sim.mean(dim=2).sum() / n_loss_terms

    def calc_expansion(self, feat_list: list[Tensor]) -> Tensor:
        """Computes the expansion (coding rate) term of the loss.

        For each view, the term is the sum of the logarithms of the diagonal
        of the Cholesky factor of ``I + scalar * Z^T Z``, i.e. half its
        log-determinant, computed stably without forming eigenvalues. The
        term is averaged over the views and scaled by the balancing factor
        ``(p + N * m) / (p * N * m)`` from the reference implementation. Here
        ``p`` is the feature dimension, ``m`` the local batch size, and ``N``
        the world size.

        Args:
            feat_list:
                List of tensors with shape (batch_size, feature_dim)
                containing the features of one or more views.

        Returns:
            The expansion term.
        """
        features = torch.stack(feat_list)
        num_views, batch_size, feature_dim = features.shape

        # Per-view feature covariance, shape (num_views, feature_dim, feature_dim).
        cov = torch.stack([W.T.matmul(W) for W in feat_list])

        world_size = 1
        if torch_dist.is_available() and torch_dist.is_initialized():
            world_size = torch_dist.get_world_size()
            if self.reduce_cov:
                # Autograd-aware all_reduce so that gradients flow across
                # ranks, ref #1920.
                cov = torch_dist_nn.all_reduce(cov)

        scalar = feature_dim / (batch_size * world_size * self.eps)
        identity = torch.eye(feature_dim, device=cov.device)
        # Sum over views of log(det(L)) where L is the Cholesky factor of
        # I + scalar * cov, equal to half the log-determinant per view.
        chol = torch.linalg.cholesky_ex(identity + scalar * cov)[0]
        loss = chol.diagonal(dim1=-2, dim2=-1).log().sum() / num_views
        return loss * (feature_dim + world_size * batch_size) / (
            feature_dim * world_size * batch_size
        )
