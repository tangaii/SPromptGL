"""Deterministic CPU-float64 classifier-sensitive subspaces and Jacobian geometry."""
from __future__ import annotations

import math

import numpy as np
import torch


def _canonical_sign(vector):
    index = int(vector.abs().argmax())
    return -vector if vector[index] < 0 else vector


def _ordered_basis(projector, dimension, tolerance):
    columns = []
    for index in range(projector.shape[0]):
        vector = projector[:, index].clone()
        for _ in range(2):
            for previous in columns:
                vector -= previous.dot(vector) * previous
        norm = vector.norm()
        if float(norm) <= tolerance:
            continue
        columns.append(_canonical_sign(vector / norm))
        if len(columns) == dimension:
            break
    if len(columns) != dimension:
        raise RuntimeError(f"Could not canonicalize eigenspace of dimension {dimension}")
    return torch.stack(columns, dim=1)


def classifier_sensitive_basis(cam, gradient, labels_unused, basis_u, rank=8, cluster_tolerance=1e-8):
    """Build V from legal rows only. Labels are accepted for audit shape checks and never used."""
    if cam.ndim != 2 or cam.shape[1] != 144 or gradient.shape != cam.shape:
        raise ValueError("Expected matching legal CAM144 and classifier-Jacobian rows")
    if len(labels_unused) != len(cam):
        raise ValueError("Legal-row provenance length differs from the CAM batch")
    u = basis_u.detach().cpu().double()
    rows = gradient.detach().cpu().double()
    rows_perp = rows - (rows @ u) @ u.T
    covariance = rows_perp.T @ rows_perp
    covariance = (covariance + covariance.T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True, stable=True)
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

    selected, selected_values = [], []
    start = 0
    while start < len(eigenvalues) and len(selected) < rank:
        reference = max(abs(float(eigenvalues[start])), 1e-30)
        stop = start + 1
        while stop < len(eigenvalues):
            scale = max(reference, abs(float(eigenvalues[stop])), 1e-30)
            if abs(float(eigenvalues[stop] - eigenvalues[start])) > cluster_tolerance * scale:
                break
            stop += 1
        projector = eigenvectors[:, start:stop] @ eigenvectors[:, start:stop].T
        canonical = _ordered_basis(projector, stop - start, tolerance=1e-12)
        take = min(rank - len(selected), canonical.shape[1])
        selected.extend(canonical[:, :take].unbind(dim=1))
        selected_values.extend([eigenvalues[start:stop].mean()] * take)
        start = stop

    if len(selected) != rank:
        raise RuntimeError("Classifier-sensitive covariance has insufficient rank")
    candidate = torch.stack(selected, dim=1)
    candidate -= u @ (u.T @ candidate)
    candidate = reorthogonalize(candidate, u)
    return candidate, torch.stack(selected_values), {
        "covariance_trace": float(torch.trace(covariance)),
        "covariance_eigenvalues_top8": [float(value) for value in selected_values],
        "perpendicular_gradient_energy": float(rows_perp.square().sum()),
    }


def reorthogonalize(candidate, basis_u):
    candidate = candidate.detach().cpu().double().clone()
    u = basis_u.detach().cpu().double()
    candidate -= u @ (u.T @ candidate)
    columns = []
    for raw in candidate.unbind(dim=1):
        value = raw.clone()
        value -= u @ (u.T @ value)
        for _ in range(2):
            for prior in columns:
                value -= prior.dot(value) * prior
        norm = value.norm()
        if float(norm) <= 1e-12:
            raise RuntimeError("Supplementary classifier subspace became numerically rank deficient")
        columns.append(_canonical_sign(value / norm))
    return torch.stack(columns, dim=1)


def update_subspace(current, candidate, basis_u, rate):
    current = current.detach().cpu().double()
    candidate = reorthogonalize(candidate, basis_u)
    left, _, right = torch.linalg.svd(candidate.T @ current, full_matrices=False)
    aligned = candidate @ (left @ right)
    blended = (1 - rate) * current + rate * aligned
    return reorthogonalize(blended, basis_u)


def decompose_jacobian(gradient, basis_u):
    u = basis_u.to(device=gradient.device, dtype=gradient.dtype)
    parallel = (gradient @ u) @ u.T
    perpendicular = gradient - parallel
    reconstruction = (parallel + perpendicular - gradient).norm(dim=-1)
    parallel_norm = parallel.norm(dim=-1)
    perpendicular_norm = perpendicular.norm(dim=-1)
    orthogonality = (parallel * perpendicular).sum(dim=-1).abs()
    return {"parallel": parallel, "perpendicular": perpendicular,
            "parallel_norm": parallel_norm, "perpendicular_norm": perpendicular_norm,
            "reconstruction_error": reconstruction, "orthogonality_error": orthogonality}


def normalize_component(component, floor):
    norm = component.norm(dim=-1, keepdim=True)
    return component / norm.clamp_min(floor), norm


def coverage_ratio(gradient, basis):
    basis = basis.to(device=gradient.device, dtype=gradient.dtype)
    projected = (gradient @ basis) @ basis.T
    total = gradient.square().sum(-1).clamp_min(torch.finfo(gradient.dtype).tiny)
    return projected.square().sum(-1) / total
