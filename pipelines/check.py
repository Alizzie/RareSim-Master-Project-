import json
from collections import Counter
from pathlib import Path

# path to your shared artifacts
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHARED_PATH = PROJECT_ROOT / "outputs" / "shared" / "disease_profiles.json"

with SHARED_PATH.open("r", encoding="utf-8") as f:
    disease_profiles = json.load(f)

total = len(disease_profiles)

with_hpo = 0
with_propagated = 0
with_description = 0
no_hpo = 0

namespace_counter = Counter()

for disease_id, profile in disease_profiles.items():
    hpo_terms = profile.get("hpo_terms", [])
    propagated = profile.get("propagated_hpo_terms", [])
    description = profile.get("merged_description")

    # HPO checks
    if len(hpo_terms) > 0:
        with_hpo += 1
    else:
        no_hpo += 1

    if len(propagated) > 0:
        with_propagated += 1

    # description check
    if description and description.strip():
        with_description += 1

    # namespace (OMIM / ORPHA / MONDO etc.)
    if ":" in disease_id:
        namespace = disease_id.split(":")[0]
        namespace_counter[namespace] += 1
    else:
        namespace_counter["UNKNOWN"] += 1

print("===== DISEASE PROFILE STATS =====")
print(f"Total profiles: {total}")
print(f"With HPO terms: {with_hpo}")
print(f"Without HPO terms: {no_hpo}")
print(f"With propagated terms: {with_propagated}")
print(f"With description: {with_description}")

print("\n===== NAMESPACE DISTRIBUTION =====")
for k, v in namespace_counter.items():
    print(f"{k}: {v}")