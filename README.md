# BrushCraft

A digital sculpting application in Rust, built to hold and sculpt billions of polygons.

Design docs and research live in a separate repo, `~/Code/brushcraft-docs`.

## Layout

| Crate | Responsibility |
|---|---|
| `geom` | Base mesh, subdivision, correction pyramid, tile math, quantization |
| `store` | Project file, out-of-core paging, tile pool, cache budget, spatial index |
| `brushcraft-bench` | Headless M0 scale-spike benchmark CLI |

`sculpt`, `render`, `input` and `app` arrive with their milestones.

## Checks

```sh
scripts/check.sh   # fmt, clippy, nextest, doctests, cargo deny, cargo machete
```
