# L11 Initial-State Representation Probe

Episode outcomes are used only as episode-level routing labels. No timestep receives a copied Harm/Rescue label.

## L11 vs SHR

- Discordant episodes: **114**
- Task-only grouped OOF AUC: **0.473**

| Feature set | Grouped OOF AUC | Balanced acc. | LOTO median AUC | Replay success | GO |
|---|---:|---:|---:|---:|:---:|
| proprio | 0.370 | 0.432 | 0.500 | 229/400 | STOP |
| visual_global | 0.680 | 0.612 | 0.495 | 248/400 | STOP |
| visual_l11 | 0.675 | 0.651 | 0.451 | 252/400 | STOP |
| prompt_action | 0.685 | 0.658 | 0.438 | 253/400 | STOP |
| all_state | 0.643 | 0.646 | 0.503 | 252/400 | STOP |

## L11 vs VANILLA

- Discordant episodes: **118**
- Task-only grouped OOF AUC: **0.608**

| Feature set | Grouped OOF AUC | Balanced acc. | LOTO median AUC | Replay success | GO |
|---|---:|---:|---:|---:|:---:|
| proprio | 0.393 | 0.500 | 0.500 | 238/400 | STOP |
| visual_global | 0.697 | 0.691 | 0.406 | 236/400 | STOP |
| visual_l11 | 0.692 | 0.706 | 0.466 | 233/400 | STOP |
| prompt_action | 0.783 | 0.718 | 0.353 | 244/400 | STOP |
| all_state | 0.745 | 0.718 | 0.331 | 241/400 | STOP |

