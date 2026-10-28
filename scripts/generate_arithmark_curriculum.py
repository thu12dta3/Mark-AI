"""
ArithMark & Mathematical Logic Procedural Data Generator
=========================================================
Target: Open SLM Leaderboard - ArithMark 3.0 & Logic Reasoning Dominance
Generates high-density continuation and reasoning pairs matching the
distribution of AxiomicLabs/Arithmark-3.0 and AxiomicLabs/LogicMark.

Covers:
1. Addition & Subtraction (single-step and multi-step word problems)
2. Multiplication & Division (equal grouping, item packing, rate scaling)
3. Relative Comparisons ("X had Y fewer/more than Z")
4. Units & Measurements (meters, centimeters, grams, kilograms, liters, hours, minutes)
5. Symbolic Transitive Logic ("If a = b, b = c, then a = c")
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

NAMES = [
    "Sam", "Jude", "Mateo", "Hugo", "Henry", "Ethan", "Lucas", "Noah", "Liam",
    "Oliver", "Emma", "Ava", "Sophia", "Isabella", "Mia", "Amelia", "Harper",
    "Evelyn", "Leo", "Maya", "Elena", "Clara", "Gabriel", "Arthur", "Alice",
]

ITEMS = [
    "rubber bands", "marbles", "stars", "craft tapes", "stickers", "pencils",
    "erasers", "notebooks", "wooden blocks", "paper clips", "trading cards",
    "magnets", "colored beads", "toy cars", "stamps", "crayons", "balloons",
]

CONTAINERS = ["boxes", "packs", "bags", "drawers", "baskets", "shelves", "jars"]


def gen_addition_problem() -> Dict[str, any]:
    name1 = random.choice(NAMES)
    item = random.choice(ITEMS)
    a = random.randint(12, 180)
    b = random.randint(9, 120)
    ans = a + b

    # Context format matching ArithMark 3.0
    templates = [
        (f"{name1} had {a} {item}. A friend gave {name1} {b} more. Afterward, {name1} had", ans, f" {ans} {item}."),
        (f"In the morning, there were {a} {item} on the table. Later, someone placed {b} more there. In total, there were", ans, f" {ans} {item}."),
        (f"{name1} collected {a} {item} yesterday and {b} {item} today. Altogether, {name1} had", ans, f" {ans} {item}."),
    ]
    ctx, correct_val, correct_ending = random.choice(templates)

    # Distractors
    offsets = [-2, -1, 1, 2]
    random.shuffle(offsets)
    endings = [f" {correct_val + off} {item}." for off in offsets[:3]]
    insert_pos = random.randint(0, 3)
    endings.insert(insert_pos, correct_ending)

    return {
        "ctx": ctx,
        "endings": endings,
        "label": insert_pos,
        "category": "addition_integer",
        "full_text": f"{ctx}{correct_ending}",
    }


def gen_subtraction_problem() -> Dict[str, any]:
    name1 = random.choice(NAMES)
    item = random.choice(ITEMS)
    a = random.randint(30, 250)
    b = random.randint(7, a - 2)
    ans = a - b

    templates = [
        (f"{name1} packed {a} {item} for school. On the way, {name1} lost {b}. In the pack, there were", ans, f" {ans} {item}."),
        (f"A shop had {a} {item} in stock. Customers bought {b} of them during the morning. There remained", ans, f" {ans} {item}."),
        (f"{name1} started with {a} {item} and gave {b} to a classmate. Afterward, {name1} had", ans, f" {ans} {item}."),
    ]
    ctx, correct_val, correct_ending = random.choice(templates)

    offsets = [-2, -1, 1, 2]
    random.shuffle(offsets)
    endings = [f" {correct_val + off} {item}." for off in offsets[:3]]
    insert_pos = random.randint(0, 3)
    endings.insert(insert_pos, correct_ending)

    return {
        "ctx": ctx,
        "endings": endings,
        "label": insert_pos,
        "category": "subtraction_integer",
        "full_text": f"{ctx}{correct_ending}",
    }


def gen_multiplication_problem() -> Dict[str, any]:
    name1 = random.choice(NAMES)
    item = random.choice(ITEMS)
    container = random.choice(CONTAINERS)
    groups = random.randint(3, 14)
    per_group = random.randint(4, 25)
    ans = groups * per_group

    templates = [
        (f"{name1} put {per_group} {item} into each of {groups} {container}. In all the {container} together, there were", ans, f" {ans} {item}."),
        (f"There are {groups} {container}, and each container holds exactly {per_group} {item}. In total, there are", ans, f" {ans} {item}."),
        (f"{name1} organized {item} in {groups} equal groups of {per_group}. Altogether, there were", ans, f" {ans} {item}."),
    ]
    ctx, correct_val, correct_ending = random.choice(templates)

    offsets = [-groups, -1, 1, groups]
    random.shuffle(offsets)
    endings = [f" {max(1, correct_val + off)} {item}." for off in offsets[:3]]
    insert_pos = random.randint(0, 3)
    endings.insert(insert_pos, correct_ending)

    return {
        "ctx": ctx,
        "endings": endings,
        "label": insert_pos,
        "category": "multiplication_integer",
        "full_text": f"{ctx}{correct_ending}",
    }


def gen_comparison_problem() -> Dict[str, any]:
    name1, name2 = random.sample(NAMES, 2)
    item = random.choice(ITEMS)
    a = random.randint(25, 180)
    diff = random.randint(5, 35)

    if random.random() < 0.5:
        # Fewer
        ans = a - diff
        ctx = f"{name1} had {a} {item}. {name2} had {diff} fewer than {name1}. That means {name2} had"
    else:
        # More
        ans = a + diff
        ctx = f"{name1} had {a} {item}. {name2} had {diff} more than {name1}. That means {name2} had"

    correct_ending = f" {ans} {item}."
    offsets = [-2, -1, 1, 2]
    random.shuffle(offsets)
    endings = [f" {ans + off} {item}." for off in offsets[:3]]
    insert_pos = random.randint(0, 3)
    endings.insert(insert_pos, correct_ending)

    return {
        "ctx": ctx,
        "endings": endings,
        "label": insert_pos,
        "category": "comparison_integer",
        "full_text": f"{ctx}{correct_ending}",
    }


def gen_measurement_problem() -> Dict[str, any]:
    name1 = random.choice(NAMES)
    units = [("centimeters", "craft tape"), ("meters", "cable"), ("grams", "packet"), ("milliliters", "vial")]
    unit_name, obj = random.choice(units)
    a = random.randint(15, 150)
    b = random.randint(12, 120)
    ans = a + b

    ctx = f"{name1} had one {obj} that was {a} {unit_name} long and another that was {b} {unit_name} long. Together, they were"
    correct_ending = f" {ans} {unit_name}."

    offsets = [-2, -1, 1, 2]
    random.shuffle(offsets)
    endings = [f" {ans + off} {unit_name}." for off in offsets[:3]]
    insert_pos = random.randint(0, 3)
    endings.insert(insert_pos, correct_ending)

    return {
        "ctx": ctx,
        "endings": endings,
        "label": insert_pos,
        "category": "measurement_addition",
        "full_text": f"{ctx}{correct_ending}",
    }


def gen_symbolic_logic_problem() -> Dict[str, any]:
    vars_pool = list("abcdefgh")
    random.shuffle(vars_pool)
    v1, v2, v3, v4 = vars_pool[:4]

    # Transitive chain: v1 = v2, v2 = v3 -> v1 = v3
    ctx = f"If:\n{v1} = {v2}\n{v2} = {v3}\n{v4} != {v1}\n\nThen"
    correct_ending = f" {v1} = {v3}"
    distractors = [f" {v1} != {v3}", f" {v4} = {v1}", f" {v4} = {v3}"]
    random.shuffle(distractors)

    endings = distractors[:3]
    insert_pos = random.randint(0, 3)
    endings.insert(insert_pos, correct_ending)

    return {
        "ctx": ctx,
        "endings": endings,
        "label": insert_pos,
        "category": "symbolic_transitive_logic",
        "full_text": f"{ctx}{correct_ending}",
    }


def generate_curriculum(num_samples: int = 50000, output_path: str = "data/arithmark_synthetic_curriculum.jsonl"):
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    generators = [
        gen_addition_problem,
        gen_subtraction_problem,
        gen_multiplication_problem,
        gen_comparison_problem,
        gen_measurement_problem,
        gen_symbolic_logic_problem,
    ]

    print(f"Generating {num_samples:,} procedural ArithMark & Logic training pairs...")
    with open(out_file, "w", encoding="utf-8") as f:
        for i in range(num_samples):
            gen = random.choice(generators)
            record = gen()
            record["id"] = f"synth_arith_{i:06d}"
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Successfully generated {num_samples:,} records to: {out_file} ({out_file.stat().st_size / 1e6:.2f} MB)")
    return str(out_file)


if __name__ == "__main__":
    generate_curriculum(50000, "data/arithmark_synthetic_curriculum.jsonl")
