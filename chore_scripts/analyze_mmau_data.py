import json
from collections import Counter

# =================== Load id2task mapping ===================

mini_mapping_file = "../../MMAU/mmau-mini-id2task.json"
mapping_file = "../../MMAU/mmau-id2task.json"
with open(mini_mapping_file, "r") as f:
    id2task_mini = json.load(f)
with open(mapping_file, "r") as f:
    id2task_full = json.load(f)

id2task = {**id2task_mini, **id2task_full}  # Merge two mappings

# =================== Analyze MMAU data in Speech-IFEval ===================

filename = "../data/eval_data/closed_ended_questions.jsonl"  # "chain-of-thought.jsonl"
data = []
with open(filename, "r") as f:
    for line in f:
        data.append(json.loads(line))


data = [item for item in data if item["audio_filepath"].lower().startswith("mmau")]

category_set = Counter()
invalid_count = 0
for item in data:
    id = item["audio_filepath"].split("/")[-1].split(".")[0]
    task_info = id2task.get(id, None)
    if task_info is None:
        # raise ValueError(f"ID {id} not found in id2task mapping.")
        print(f"Warning: ID {id} not found in id2task mapping.")
        invalid_count += 1
        continue

    task = task_info["task"]
    category = task_info["category"]
    sub_category = task_info["sub-category"]
    category_set[(task, category, sub_category)] += 1

    if sub_category == "Phonemic Stress Pattern Analysis":
        print(json.dumps(item, indent=2), end="\n\n")

print("Unique (task, category, sub-category) combinations in Speech-IFEval MMAU data:")

# Calculate column widths
tasks = [k[0] for k in category_set]
categories = [k[1] for k in category_set]
sub_categories = [k[2] for k in category_set]

max_task_len = max((len(t) for t in tasks + ["Task"]), default=4)
max_cat_len = max((len(c) for c in categories + ["Category"]), default=8)
max_sub_len = max((len(s) for s in sub_categories + ["Sub-category"]), default=12)

# Print header
header = f"{'Task':<{max_task_len}}  {'Category':<{max_cat_len}}  {'Sub-category':<{max_sub_len}}  {'Count'}"
print(header)
print("-" * len(header))

for task, category, sub_category in sorted(category_set):
    count = category_set[(task, category, sub_category)]
    print(
        f"{task:<{max_task_len}}  {category:<{max_cat_len}}  {sub_category:<{max_sub_len}}  {count}"
    )

print(f"\nTotal invalid entries (IDs not found in mapping): {invalid_count}")
