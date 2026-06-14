"""Apply proposed gold to both factual + conceptual candidate pools -> qa_set.json.

Gold ids are matched by ::NNNN:: ordinal within each question's own pool, so
prefixes on your machine don't matter. Factual gold is pre-filled (verified
from the first review). Conceptual gold is filled in GOLD_CONCEPTUAL after you
review the conceptual candidates.

Questions absent from a mapping are dropped.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --- Factual set (qa_candidates.json), verified from first review ---
GOLD_FACTUAL = {
    1:  ["0363", "0343"], 2: ["0136", "0137"], 3: ["0273", "0214"],
    4:  ["0738", "0737"], 5: ["0276"],         6: ["0195", "0179"],
    7:  ["0295", "0294"], 8: ["0118", "0273"], 9: ["0312", "0227"],
    10: ["0137"],         11: ["0263"],        12: ["0123"],
    13: ["0791", "0338"], 15: ["0140", "0141"],
    # 14 dropped (no clean no-surcharge answer in pool)
}

# --- Conceptual set (qa_candidates_conceptual.json) ---
# Fill after reviewing: question_index (1-based) -> [ordinals]. Empty = drop.
GOLD_CONCEPTUAL = {
    1: ["0072", "0071"],
    2: ["0184", "0182"],
    3: ["0199", "0294"],
    4: ["0462", "0461"],
    5: ["0187", "0297"],
    6: ["0208", "0207"],
    7: ["0286", "0163"],
    8: ["0214"],
}


def ordinal(cid: str) -> str:
    return cid.split("::")[1]


def build(cand_path, gold_map):
    cand = json.loads(Path(cand_path).read_text())
    out, missing = [], []
    for i, item in enumerate(cand, 1):
        if i not in gold_map or not gold_map[i]:
            continue
        pool = {ordinal(c["chunk_id"]): c["chunk_id"] for c in item["candidates"]}
        gold = []
        for w in gold_map[i]:
            if w in pool:
                gold.append(pool[w])
            else:
                missing.append((cand_path.name, i, w))
        item["gold_chunk_ids"] = gold
        item["_verified"] = True
        item.pop("candidates", None)
        out.append(item)
    return out, missing


def main():
    all_out, all_missing = [], []
    fac, m1 = build(HERE / "qa_candidates.json", GOLD_FACTUAL)
    all_out += fac; all_missing += m1
    concept_path = HERE / "qa_candidates_conceptual.json"
    if concept_path.exists() and GOLD_CONCEPTUAL:
        con, m2 = build(concept_path, GOLD_CONCEPTUAL)
        all_out += con; all_missing += m2
    else:
        print("(conceptual gold not yet filled -- writing factual only)")

    (HERE / "qa_set.json").write_text(json.dumps(all_out, indent=2))
    print(f"Wrote qa_set.json with {len(all_out)} verified questions "
          f"({sum(1 for q in all_out if q['type']=='conceptual')} conceptual)")
    if all_missing:
        print("WARNING missing ordinals:", all_missing)


if __name__ == "__main__":
    main()