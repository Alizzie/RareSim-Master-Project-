# Disease ID Normalization and Mapping

## Purpose

Raw sources use different ID formats for the same disease or phenotype.

RareSim normalizes identifiers and maps equivalent disease IDs to canonical IDs where possible.

This is necessary because the same disease may appear as:

```text
ORPHA:123
Orphanet_123
OMIM:123456
MIM:123456
MONDO:0000437
MONDO_0000437
DECIPHER:123
```

Without normalization and mapping, these could be treated as unrelated diseases.

## HPO ID Normalization

HPO normalization is implemented in:

```text
raresim/utils/normalizers.py
```

The function is:

```python
normalize_hpo_id()
```

It converts HPO IDs into the standard format:

```text
HP:0000000
```

Examples:

```text
HP_0001250              -> HP:0001250
http://.../HP_0001250   -> HP:0001250
hp:0001250              -> HP:0001250
```

If an HPO ID cannot be normalized into valid HPO format, it returns `None`.

## Disease ID Normalization

Disease ID normalization is handled by:

```python
normalize_disease_id()
```

Examples:

```text
Orphanet_123    -> ORPHA:123
ORPHA_123       -> ORPHA:123
ORPHANET:123    -> ORPHA:123
MIM:123456      -> OMIM:123456
MONDO_0000437   -> MONDO:0000437
123             -> ORPHA:123
```

Supported prefixes include:

```text
ORPHA
OMIM
MONDO
DECIPHER
DOID
```

The goal is to convert source-specific variants into stable internal display IDs.

## Mapping to ORPHA

Cross-ontology disease mapping is implemented in:

```text
raresim/utils/mapping_utils.py
```

The main generated artifact is:

```text
orpha_mapping_index.json
```

This maps non-ORPHA IDs to ORPHA IDs when a reliable mapping exists.

Example:

```json
{
  "OMIM:301310": "ORPHA:123",
  "MONDO:0000437": "ORPHA:102002"
}
```

## Mapping Sources

The mapping utility supports ORDO, MONDO, and HOOM metadata.

In the current build script, ORDO and MONDO metadata are loaded. HOOM metadata support exists, but hoom_metadata is currently set to an empty dictionary. HOOM is currently used mainly for disease-HPO annotation extraction, not for disease description metadata in the actual build.

The code uses:

```text
xrefs
exact_matches
normalized IDs
```

to detect equivalent disease identifiers.

## Mapping Rules

The mapping logic is conservative.

Rules:

1. ORPHA IDs are treated as canonical anchors.
2. OMIM IDs can be mapped to ORPHA using xrefs.
3. MONDO IDs are mapped to ORPHA only when a reliable exactMatch exists for the MONDO entry itself.
4. ORDO metadata may use exactMatch first, then xref fallback. The mapping utility also supports HOOM metadata, but the current build does not load HOOM metadata.
5. If no reliable ORPHA mapping exists, the normalized original ID is kept.

Important: RareSim does not force every disease ID to ORPHA.

This matters for identifiers such as:

```text
DECIPHER
DOID
MONDO
OMIM
```

which may remain non-ORPHA depending on available mappings.

## Resolving a Disease ID

During profile construction, IDs are resolved with:

```python
resolve_to_orpha()
```

Resolution priority:

1. If the ID is already ORPHA, keep it.
2. If the ID appears in the mapping index, map it to ORPHA.
3. If source metadata contains ORPHA xrefs, use them.
4. Otherwise, keep the original ID.

## Alias-to-Canonical Mapping

During profile construction, the builder also creates:

```text
alias_to_canonical.json
```

This maps source-specific disease IDs to the canonical disease ID used internally.

Example:

```json
{
  "MONDO:0000437": "ORPHA:102002",
  "OMIM:123456": "ORPHA:102002"
}
```

This file is important because evaluation datasets or external tools may use aliases instead of canonical ORPHA IDs.
