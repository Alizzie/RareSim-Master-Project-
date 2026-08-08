## Screenshots

### Input Panel

<img alt="input-hpo-terms" src="https://github.com/user-attachments/assets/18829607-85a5-4a54-895c-8fe00d974331" width="914" style="display:block; margin-top: 24px; margin-bottom: 24px;">

*HPO Terms mode with a populated term set. Included terms appear as green tags; excluded terms (added via the phenotype search dropdown's **− Exclude** button) are shown separately and are filtered out server-side before any similarity method runs.*

<img alt="input-phenotype-search" src="https://github.com/user-attachments/assets/d0897e70-bdeb-4b94-b1fb-0e46a60ac01e" width="914" style="display:block; margin-top: 24px; margin-bottom: 24px;">

*The phenotype search dropdown, open mid-search. Each matching term has **+ Include** and **− Exclude** buttons, letting you build up the patient's HPO set and explicitly rule terms out without leaving the search flow.*

<img alt="input-raw-text-extraction" src="https://github.com/user-attachments/assets/d04d74e3-aa9c-4f81-8469-f0caf9142d0c" width="914" style="display:block; margin-top: 24px; margin-bottom: 24px;">

*Raw Text mode: clinical text entered, an extraction method selected, and the resulting extracted HPO terms shown as tags. The callout highlights that the similarity-method list changes based on input mode — HPO-only methods (Resnik/Lin/JC BMA, set-based methods, and the HPO-based TF-IDF variants) grey out here, while the raw-text TF-IDF variant is only available in this mode. The hybrid TF-IDF variant remains selectable in both.*

### Results Panel

<img alt="results-method-filter" src="https://github.com/user-attachments/assets/a0acdd48-eed9-4c44-a119-e118533b4a9f" width="914" style="display:block; margin-top: 24px; margin-bottom: 24px;">

*Ranked results for 2 similarity methods. The filter tags above the list (one per selected method, plus "All") let you narrow the view — the callout marks the active filter's visual state so it's clear which one is currently selected.*

<img alt="results-expanded-detail" src="https://github.com/user-attachments/assets/de44126b-d90a-4b8d-905f-b9559000554f" width="914" style="display:block; margin-top: 24px; margin-bottom: 24px;">

*A single result card expanded via its detail dropdown, showing shared phenotypes and top term matches between the patient and the candidate disease.*

### Method Comparison

*The method comparison view with all four selected methods expanded individually, showing ranking agreement and divergence across methods for the same patient.*

<img width="914" alt="results-method-comparison-1" src="https://github.com/user-attachments/assets/ca0e2dd6-1d36-4667-8dde-f38620ec1d35" style="display:block; margin-top: 24px; margin-bottom: 24px;">

<img width="914" alt="results-method-comparison-2" src="https://github.com/user-attachments/assets/afcc3c97-3201-465c-bfa9-1282b75bbea9" style="display:block; margin-top: 24px; margin-bottom: 24px;">

<img width="914" alt="results-method-comparison-3" src="https://github.com/user-attachments/assets/16284e95-b1cf-49db-8066-ade513bbe9c7" style="display:block; margin-top: 24px; margin-bottom: 24px;">

<img width="914" alt="results-method-comparison-4" src="https://github.com/user-attachments/assets/9b697bdc-36f4-4937-b26a-1cfa1b286797" style="display:block; margin-top: 24px;">