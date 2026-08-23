#!/usr/bin/env python3
"""Stream a numerical parameter-delta audit between two HF safetensor actors."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from safetensors import safe_open


def _files(root):
    root = Path(root).expanduser().resolve()
    files = sorted(path for path in root.rglob('*.safetensors') if path.is_file())
    if not files:
        raise FileNotFoundError(f'no .safetensors files found under {root}')
    return root, files


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _relative_map(root, paths):
    return {str(path.relative_to(root)): path for path in paths}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    reference_root, reference_files = _files(args.reference)
    candidate_root, candidate_files = _files(args.candidate)
    reference_map = _relative_map(reference_root, reference_files)
    candidate_map = _relative_map(candidate_root, candidate_files)
    if set(reference_map) != set(candidate_map):
        raise ValueError('checkpoint safetensor shard names differ')

    total_elements = 0
    changed_elements = 0
    delta_sq_sum = 0.0
    reference_sq_sum = 0.0
    max_abs_delta = 0.0
    tensor_count = 0
    file_audits = []
    for relative_path in sorted(reference_map):
        reference_path = reference_map[relative_path]
        candidate_path = candidate_map[relative_path]
        with safe_open(str(reference_path), framework='pt', device='cpu') as reference_handle, \
                safe_open(str(candidate_path), framework='pt', device='cpu') as candidate_handle:
            reference_keys = set(reference_handle.keys())
            candidate_keys = set(candidate_handle.keys())
            if reference_keys != candidate_keys:
                raise ValueError(f'parameter keys differ in shard {relative_path}')
            for key in sorted(reference_keys):
                reference = reference_handle.get_tensor(key)
                candidate = candidate_handle.get_tensor(key)
                if reference.shape != candidate.shape:
                    raise ValueError(f'parameter shape differs for {key}')
                delta = candidate.float() - reference.float()
                total_elements += delta.numel()
                changed_elements += int(torch.count_nonzero(delta).item())
                delta_sq_sum += float(torch.sum(delta.double() ** 2).item())
                reference_sq_sum += float(torch.sum(reference.double() ** 2).item())
                max_abs_delta = max(max_abs_delta, float(torch.max(torch.abs(delta)).item()))
                tensor_count += 1
        file_audits.append({
            'relative_path': relative_path,
            'reference_sha256': _sha256(reference_path),
            'candidate_sha256': _sha256(candidate_path),
        })

    payload = {
        'reference': str(reference_root),
        'candidate': str(candidate_root),
        'tensor_count': tensor_count,
        'parameter_element_count': total_elements,
        'changed_element_count': changed_elements,
        'changed_element_fraction': changed_elements / total_elements if total_elements else 0.0,
        'delta_l2_norm': math.sqrt(delta_sq_sum),
        'reference_l2_norm': math.sqrt(reference_sq_sum),
        'relative_delta_l2_norm': math.sqrt(delta_sq_sum / reference_sq_sum) if reference_sq_sum else 0.0,
        'max_abs_delta': max_abs_delta,
        'shards': file_audits,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
