# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen-teacher geometry preservation losses for CLIP fine-tuning."""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from nvidia_tao_pytorch.core.tlt_logging import logging


class PreservationLoss(nn.Module):
    """Compare student embeddings to a frozen pretrained CLIP teacher."""

    def __init__(self, teacher_model, regularization_config):
        super().__init__()
        self.teacher = teacher_model
        self.embedding_mse_weight = regularization_config.embedding_mse_weight
        self.cosine_weight = regularization_config.cosine_weight
        self.similarity_weight = regularization_config.similarity_weight

    def train(self, mode=True):
        """Keep the frozen teacher deterministic when Lightning enters train mode."""
        super().train(mode)
        self.teacher.eval()
        return self

    @torch.no_grad()
    def _teacher_forward(self, image, text):
        outputs = self.teacher(image=image, text=text)
        return outputs[0], outputs[1]

    def forward(self, student_image_features, student_text_features, image, text):
        """Return unweighted components and their weighted preservation sum."""
        teacher_image_features, teacher_text_features = self._teacher_forward(image, text)
        image_embedding_mse = F.mse_loss(
            student_image_features, teacher_image_features
        )
        text_embedding_mse = F.mse_loss(
            student_text_features, teacher_text_features
        )
        embedding_mse = (image_embedding_mse + text_embedding_mse) / 2.0
        image_cosine = 1.0 - F.cosine_similarity(
            student_image_features, teacher_image_features, dim=-1
        ).mean()
        text_cosine = 1.0 - F.cosine_similarity(
            student_text_features, teacher_text_features, dim=-1
        ).mean()
        cosine = (image_cosine + text_cosine).clamp_min(0.0) / 2.0
        similarity = F.mse_loss(
            student_image_features @ student_text_features.T,
            teacher_image_features @ teacher_text_features.T,
        )
        total = sum((
            self.embedding_mse_weight * embedding_mse,
            self.cosine_weight * cosine,
            self.similarity_weight * similarity,
        ))
        return {
            'preservation_total': total,
            'embedding_mse': embedding_mse,
            'cosine': cosine,
            'similarity': similarity,
        }


def build_preservation_loss(model, regularization_config):
    """Copy and freeze ``model`` before LoRA is injected, when enabled."""
    if not regularization_config.enabled:
        return None
    teacher = copy.deepcopy(model)
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    teacher.eval()
    logging.info(
        'Created frozen CLIP preservation teacher: emb_mse=%s, cosine=%s, sim=%s.',
        regularization_config.embedding_mse_weight,
        regularization_config.cosine_weight,
        regularization_config.similarity_weight,
    )
    return PreservationLoss(teacher, regularization_config)
