# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for CLIP preservation-loss regularization."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from nvidia_tao_pytorch.multimodal.clip.model.pl_clip_model import CLIPPlModel
from nvidia_tao_pytorch.multimodal.clip.model.preservation_loss import (
    build_preservation_loss,
)


class _TinyCLIP(nn.Module):
    """Small CLIP-shaped model used to test teacher preservation behavior."""

    def __init__(self):
        super().__init__()
        self.vision = nn.Linear(4, 4, bias=False)
        self.text = nn.Linear(4, 4, bias=False)

    def forward(self, image=None, text=None):
        return F.normalize(self.vision(image), dim=-1), F.normalize(self.text(text), dim=-1)


def _regularization_config():
    return SimpleNamespace(
        enabled=True,
        embedding_mse_weight=0.05,
        cosine_weight=0.05,
        similarity_weight=0.1,
    )


def test_preservation_teacher_is_frozen_and_reports_zero_for_matching_model():
    """The copied pre-LoRA teacher is immutable and yields zero drift initially."""
    model = _TinyCLIP()
    preservation = build_preservation_loss(model, _regularization_config())
    preservation.train()
    image, text = torch.randn(3, 4), torch.randn(3, 4)
    student_image, student_text = model(image=image, text=text)
    losses = preservation(student_image, student_text, image, text)

    assert not preservation.teacher.training
    assert all(not parameter.requires_grad for parameter in preservation.teacher.parameters())
    assert losses['embedding_mse'].item() == pytest.approx(0.0)
    assert losses['cosine'].item() == pytest.approx(0.0)
    assert losses['similarity'].item() == pytest.approx(0.0)
    assert losses['preservation_total'].item() == pytest.approx(0.0)


def test_training_step_adds_preservation_and_keeps_teacher_gradient_free():
    """The real training-step composition adds preservation only to the student."""
    model = _TinyCLIP()
    preservation = build_preservation_loss(model, _regularization_config())
    image, text = torch.randn(3, 4), torch.randn(3, 4)
    student_image = torch.randn(3, 4, requires_grad=True)
    student_text = torch.randn(3, 4, requires_grad=True)
    contrastive_loss = torch.tensor(2.0, requires_grad=True)
    logged = {}
    schedules = {
        tower: {'lr': 1e-4, 'warmup': 0, 'scheduler': 'constant'}
        for tower in ('vision', 'text')
    }
    fake_module = SimpleNamespace(
        _forward_pass=lambda batch: None,
        _backward=lambda outputs, batch: (
            contrastive_loss, torch.tensor(1.0), student_image, student_text,
        ),
        optimizer=SimpleNamespace(param_groups=[{'_tower': 'vision'}, {'_tower': 'text'}]),
        _tower_schedules=schedules,
        global_step=0,
        max_steps=1,
        preservation_loss=preservation,
        triplet_loss_weight=0.0,
        log=lambda name, value, **kwargs: logged.setdefault(name, value),
    )

    total_loss = CLIPPlModel.training_step(fake_module, (image, text))
    expected = contrastive_loss + preservation(
        student_image, student_text, image, text,
    )['preservation_total']
    torch.testing.assert_close(total_loss, expected)
    assert total_loss.item() > contrastive_loss.item()
    total_loss.backward()
    assert student_image.grad is not None
    assert student_text.grad is not None
    assert all(parameter.grad is None for parameter in preservation.teacher.parameters())
    assert 'train/emb_mse_loss' in logged
    assert 'train/cosine_loss' in logged
    assert 'train/sim_loss' in logged
