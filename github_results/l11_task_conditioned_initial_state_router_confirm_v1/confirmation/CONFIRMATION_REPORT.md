# Frozen Initial-State Router Confirmation

Models, preprocessing, feature set, and threshold were frozen on seeds 0-99 before held-out feature collection.

| Task | Discordant | AUC | Accuracy | Balanced accuracy | Router | L11 | SHR |
|---|---:|---:|---:|---:|---:|---:|---:|
| open_drawer | 36 | 0.675 | 0.611 | 0.613 | 46/100 | 44/100 | 40/100 |
| close_drawer | 30 | 0.784 | 0.767 | 0.761 | 77/100 | 76/100 | 62/100 |
| pick_coke_can | 23 | 0.554 | 0.478 | 0.504 | 28/100 | 33/100 | 24/100 |
| move_near | 23 | 0.873 | 0.783 | 0.742 | 71/100 | 62/100 | 67/100 |

- Macro balanced accuracy: **0.655**
- Router total: **222/400**; L11: **215/400**; SHR: **193/400**
- Decision: **GO**
