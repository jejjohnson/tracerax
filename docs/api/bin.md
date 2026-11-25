# Cells and deposits

What particles leave in cells. `Cells` are the boxes between edges, and `deposit` adds one weight per particle to them.

```python
import jax.numpy as jnp
from tracerax.bin import Cells, deposit

# cells that grow with height
cells = Cells(
    (
        jnp.linspace(-100.0, 900.0, 21),
        jnp.linspace(-200.0, 200.0, 9),
        jnp.array([0.0, 10.0, 25.0, 50.0, 100.0, 200.0]),
    )
)
position = jnp.array([[5.0, 0.0, 3.0], [420.0, -60.0, 180.0]])
mass = jnp.array([0.5, 0.5])
deposited = deposit(cells, position, mass, kind="linear")
concentration = deposited / cells.volumes()
```

- **Kinds.** `"nearest"` is a histogram, whose gradient in the positions is zero. `"linear"` is cloud-in-cell, which splits a particle between the centres around it and is differentiable in the positions.
- **Mass.** Both keep the whole weight of every particle inside the edges, on stretched cells and at the walls: a particle within half a cell of the ground counts in full in the lowest cell. What is outside the edges is dropped.
- **Edges.** A cell holds its lower edge and not its upper one, so a particle on an inner edge belongs to the upper cell and one on the last edge is outside. This differs from `numpy.histogram` at the last edge.
- **Axes.** The cells have one axis per position column, in their order. For a map of a plume, pass two axes and `position[:, :2]`.
- **Precision.** Edges are compared in the positions' dtype, and a cell too narrow for it raises, so count both from a local origin in float32. Cells built from concrete edges cost no check at run time.

::: tracerax.bin
