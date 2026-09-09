from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path
from typing import Iterable

RNA_BASES = frozenset({"A", "C", "G", "U"})
BRACKET_PAIRS = {"(": ")", "[": "]", "{": "}", "<": ">"}
CLOSE_TO_OPEN = {close: open_ for open_, close in BRACKET_PAIRS.items()}
BRACKET_FAMILIES = tuple(BRACKET_PAIRS.items())


def normalize_sequence(sequence: str) -> str:
    return sequence.strip().upper().replace("T", "U")


def ambiguous_base_count(sequence: str) -> int:
    seq = normalize_sequence(sequence)
    return sum(1 for ch in seq if ch not in RNA_BASES)


def sequence_is_unambiguous(sequence: str) -> bool:
    return ambiguous_base_count(sequence) == 0


def sequence_length(sequence: str) -> int:
    return len(normalize_sequence(sequence))


def sequence_hamming_distance(
    left: str,
    right: str,
    *,
    reject_ambiguous: bool = True,
) -> int | None:
    left = normalize_sequence(left)
    right = normalize_sequence(right)
    if len(left) != len(right):
        raise ValueError(f"sequence lengths differ: {len(left)} != {len(right)}")
    if reject_ambiguous and (not sequence_is_unambiguous(left) or not sequence_is_unambiguous(right)):
        return None
    return sum(a != b for a, b in zip(left, right))


def normalized_distance(distance: object, length: object) -> float | None:
    try:
        distance_value = float(distance)
        length_value = float(length)
    except (TypeError, ValueError):
        return None
    if length_value <= 0 or math.isnan(distance_value) or math.isnan(length_value):
        return None
    return distance_value / length_value


def dotbracket_to_pairs(dotbracket: str) -> frozenset[tuple[int, int]]:
    stacks: dict[str, list[int]] = {open_char: [] for open_char in BRACKET_PAIRS}
    pairs: set[tuple[int, int]] = set()
    for idx, ch in enumerate(dotbracket.strip()):
        if ch in BRACKET_PAIRS:
            stacks[ch].append(idx)
        elif ch in CLOSE_TO_OPEN:
            open_char = CLOSE_TO_OPEN[ch]
            if not stacks[open_char]:
                raise ValueError(f"unmatched closing bracket {ch!r} at position {idx}")
            left = stacks[open_char].pop()
            pairs.add((left, idx))
        elif ch in {".", "&"}:
            continue
        else:
            raise ValueError(f"unsupported dot-bracket character {ch!r} at position {idx}")
    for open_char, stack in stacks.items():
        if stack:
            raise ValueError(f"unmatched opening bracket {open_char!r} at positions {stack}")
    return frozenset(pairs)


def bpseq_lines_to_pairs(lines: Iterable[str]) -> frozenset[tuple[int, int]]:
    mapping: dict[int, int] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith(("#", ";", ">")):
            continue
        fields = line.split()
        if len(fields) < 3:
            raise ValueError(f"expected at least 3 BPSEQ columns, got: {raw_line.rstrip()!r}")
        idx = int(fields[0])
        partner = int(fields[2])
        if idx in mapping:
            raise ValueError(f"duplicate BPSEQ index {idx}")
        mapping[idx] = partner

    pairs: set[tuple[int, int]] = set()
    for idx, partner in mapping.items():
        if partner <= 0:
            continue
        if partner not in mapping:
            raise ValueError(f"BPSEQ partner index {partner} missing for index {idx}")
        if mapping[partner] != idx:
            raise ValueError(f"BPSEQ asymmetry: {idx} -> {partner} but {partner} -> {mapping[partner]}")
        if idx < partner:
            pairs.add((idx - 1, partner - 1))
    return frozenset(pairs)


def bpseq_file_to_pairs(path: Path) -> frozenset[tuple[int, int]]:
    with path.open("r", encoding="utf-8") as handle:
        return bpseq_lines_to_pairs(handle)


def pair_crosses(left: tuple[int, int], right: tuple[int, int]) -> bool:
    i, j = left
    k, l = right
    return (i < k < j < l) or (k < i < l < j)


def pair_nested(left: tuple[int, int], right: tuple[int, int]) -> bool:
    i, j = left
    k, l = right
    return (i < k < l < j) or (k < i < j < l)


def normalize_base_pairs(pairs: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    normalized: set[tuple[int, int]] = set()
    for left, right in pairs:
        if left == right:
            raise ValueError(f"invalid base pair with identical endpoints: {(left, right)}")
        i, j = (left, right) if left < right else (right, left)
        normalized.add((i, j))
    return sorted(normalized)


def validate_base_pair_matching(pairs: Iterable[tuple[int, int]]) -> None:
    used: dict[int, tuple[int, int]] = {}
    for left, right in pairs:
        if left == right:
            raise ValueError(f"invalid base pair with identical endpoints: {(left, right)}")
        if left in used:
            raise ValueError(f"nucleotide {left} appears in multiple base pairs: {used[left]} and {(left, right)}")
        if right in used:
            raise ValueError(f"nucleotide {right} appears in multiple base pairs: {used[right]} and {(left, right)}")
        used[left] = (left, right)
        used[right] = (left, right)


def density2_decomposition(
    pairs: Iterable[tuple[int, int]],
    *,
    validate_matching: bool = True,
) -> tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]] | None:
    normalized = normalize_base_pairs(pairs)
    if validate_matching:
        validate_base_pair_matching(normalized)

    if not normalized:
        return frozenset(), frozenset()

    adjacency: list[list[int]] = [[] for _ in range(len(normalized))]
    for i in range(len(normalized)):
        for j in range(i + 1, len(normalized)):
            if pair_crosses(normalized[i], normalized[j]):
                adjacency[i].append(j)
                adjacency[j].append(i)

    colors = [-1] * len(normalized)
    for start in range(len(normalized)):
        if colors[start] != -1:
            continue
        colors[start] = 0
        queue: deque[int] = deque([start])
        while queue:
            vertex = queue.popleft()
            for neighbor in adjacency[vertex]:
                if colors[neighbor] == -1:
                    colors[neighbor] = 1 - colors[vertex]
                    queue.append(neighbor)
                elif colors[neighbor] == colors[vertex]:
                    return None

    signatures = [frozenset(neighbors) for neighbors in adjacency]
    signature_groups: dict[frozenset[int], list[int]] = {}
    for idx, signature in enumerate(signatures):
        if signature:
            signature_groups.setdefault(signature, []).append(idx)

    band_members: list[list[int]] = []
    for indices in signature_groups.values():
        nested_adjacency: dict[int, list[int]] = {idx: [] for idx in indices}
        for pos, left_idx in enumerate(indices):
            for right_idx in indices[pos + 1 :]:
                if pair_nested(normalized[left_idx], normalized[right_idx]):
                    nested_adjacency[left_idx].append(right_idx)
                    nested_adjacency[right_idx].append(left_idx)

        visited_band: set[int] = set()
        for start_idx in indices:
            if start_idx in visited_band:
                continue
            stack = [start_idx]
            visited_band.add(start_idx)
            component: list[int] = []
            while stack:
                current_idx = stack.pop()
                component.append(current_idx)
                for neighbor_idx in nested_adjacency[current_idx]:
                    if neighbor_idx not in visited_band:
                        visited_band.add(neighbor_idx)
                        stack.append(neighbor_idx)
            band_members.append(component)

    if band_members:
        structure_length = max(right for _, right in normalized) + 1
        band_graph: list[set[int]] = [set() for _ in band_members]
        for left_band in range(len(band_members)):
            for right_band in range(left_band + 1, len(band_members)):
                if any(
                    pair_crosses(normalized[left_idx], normalized[right_idx])
                    for left_idx in band_members[left_band]
                    for right_idx in band_members[right_band]
                ):
                    band_graph[left_band].add(right_band)
                    band_graph[right_band].add(left_band)

        visited_component: set[int] = set()
        for start_band in range(len(band_graph)):
            if start_band in visited_component:
                continue
            queue: deque[int] = deque([start_band])
            visited_component.add(start_band)
            component_bands: list[int] = []
            while queue:
                current_band = queue.popleft()
                component_bands.append(current_band)
                for neighbor_band in band_graph[current_band]:
                    if neighbor_band not in visited_component:
                        visited_component.add(neighbor_band)
                        queue.append(neighbor_band)

            for position in range(structure_length):
                covering_bands = 0
                for band_idx in component_bands:
                    if any(
                        normalized[pair_idx][0] <= position <= normalized[pair_idx][1]
                        for pair_idx in band_members[band_idx]
                    ):
                        covering_bands += 1
                        if covering_bands > 2:
                            return None

    component_0 = frozenset(normalized[i] for i, color in enumerate(colors) if color == 0)
    component_1 = frozenset(normalized[i] for i, color in enumerate(colors) if color == 1)
    return canonical_density2_components(component_0, component_1)


def maximal_strict_density2_decomposition(
    pairs: Iterable[tuple[int, int]],
    *,
    validate_matching: bool = True,
) -> tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]] | None:
    """Return strict density-2 lanes oriented to maximize the scaffold lane.

    Each crossing-graph component has two interchangeable colours.  Selecting
    its larger colour for the first result, and assigning isolated pairs there,
    maximizes the total number of scaffold pairs while retaining both lanes as
    pseudoknot-free.  Ties use lexicographic pair order for reproducibility.
    """
    normalized = normalize_base_pairs(pairs)
    if validate_matching:
        validate_base_pair_matching(normalized)
    if density2_decomposition(normalized, validate_matching=False) is None:
        return None

    adjacency: list[list[int]] = [[] for _ in normalized]
    for left in range(len(normalized)):
        for right in range(left + 1, len(normalized)):
            if pair_crosses(normalized[left], normalized[right]):
                adjacency[left].append(right)
                adjacency[right].append(left)

    colours = [-1] * len(normalized)
    g_big: set[tuple[int, int]] = set()
    g_small: set[tuple[int, int]] = set()
    for start in range(len(normalized)):
        if colours[start] != -1:
            continue
        colours[start] = 0
        queue: deque[int] = deque([start])
        component: list[int] = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in adjacency[current]:
                if colours[neighbor] == -1:
                    colours[neighbor] = 1 - colours[current]
                    queue.append(neighbor)

        colour_0 = tuple(normalized[index] for index in component if colours[index] == 0)
        colour_1 = tuple(normalized[index] for index in component if colours[index] == 1)
        choose_colour_0 = len(colour_0) > len(colour_1) or (
            len(colour_0) == len(colour_1) and colour_0 <= colour_1
        )
        big, small = (colour_0, colour_1) if choose_colour_0 else (colour_1, colour_0)
        g_big.update(big)
        g_small.update(small)
    return frozenset(g_big), frozenset(g_small)


def canonical_density2_components(
    component_0: Iterable[tuple[int, int]],
    component_1: Iterable[tuple[int, int]],
) -> tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]]:
    left = frozenset(normalize_base_pairs(component_0))
    right = frozenset(normalize_base_pairs(component_1))
    left_key = (len(left), tuple(sorted(left)))
    right_key = (len(right), tuple(sorted(right)))
    if left_key <= right_key:
        return left, right
    return right, left


def is_density2(pairs: Iterable[tuple[int, int]], *, validate_matching: bool = True) -> bool:
    return density2_decomposition(pairs, validate_matching=validate_matching) is not None


def pairs_to_dotbracket(
    pairs: Iterable[tuple[int, int]],
    length: int,
    open_char: str = "(",
    close_char: str = ")",
) -> str:
    if length < 0:
        raise ValueError(f"length must be non-negative, got {length}")
    if len(open_char) != 1 or len(close_char) != 1:
        raise ValueError("bracket characters must be single characters")
    chars = ["."] * length
    for left, right in normalize_base_pairs(pairs):
        if left < 0 or right < 0 or right >= length:
            raise ValueError(f"pair {(left, right)} does not fit in length {length}")
        if chars[left] != "." or chars[right] != ".":
            raise ValueError(f"pair {(left, right)} conflicts with an existing bracket assignment")
        chars[left] = open_char
        chars[right] = close_char
    return "".join(chars)


def split_target_dotbracket(dotbracket: str) -> tuple[str, str, str | None]:
    target = dotbracket.strip()
    pair_counts: dict[tuple[str, str], int] = {}
    for open_char, close_char in BRACKET_FAMILIES:
        if open_char == "(":
            continue
        pair_counts[(open_char, close_char)] = min(target.count(open_char), target.count(close_char))
    selected_family: tuple[str, str] | None = None
    selected_pairs = 0
    for family, pair_count in pair_counts.items():
        if pair_count > selected_pairs:
            selected_family = family
            selected_pairs = pair_count
    g_chars = {"(", ")", ".", "&"}
    gprime_chars = set(selected_family or ()) | {".", "&"}
    g = "".join(ch if ch in g_chars else "." for ch in target)
    gprime = "".join(ch if ch in gprime_chars else "." for ch in target)
    selected_label = None if selected_family is None or selected_pairs == 0 else f"{selected_family[0]}{selected_family[1]}"
    return g, gprime, selected_label


def base_pair_distance(left: str, right: str) -> int:
    left_pairs = dotbracket_to_pairs(left)
    right_pairs = dotbracket_to_pairs(right)
    return len(left_pairs.symmetric_difference(right_pairs))


def combined_score_from_logs(logp_g: float, logp_branch: float) -> float:
    if math.isnan(logp_g) or math.isnan(logp_branch):
        return math.nan
    return logp_g + logp_branch


__all__ = [
    "RNA_BASES",
    "BRACKET_FAMILIES",
    "BRACKET_PAIRS",
    "ambiguous_base_count",
    "base_pair_distance",
    "bpseq_file_to_pairs",
    "bpseq_lines_to_pairs",
    "canonical_density2_components",
    "combined_score_from_logs",
    "density2_decomposition",
    "dotbracket_to_pairs",
    "is_density2",
    "maximal_strict_density2_decomposition",
    "normalize_base_pairs",
    "normalize_sequence",
    "pair_nested",
    "pair_crosses",
    "pairs_to_dotbracket",
    "sequence_hamming_distance",
    "sequence_length",
    "sequence_is_unambiguous",
    "normalized_distance",
    "split_target_dotbracket",
    "validate_base_pair_matching",
]
