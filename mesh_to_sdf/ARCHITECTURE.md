## Mesh to SDF

`generate_sdf` bruteforces the closest point on the mesh for each query point by checking the distance to each triangle, using `rayon` to parallelize the computation. Further work would be to use a spatial data structure (e.g. bvh) to speed up the computation, and letting the user choose the algorithm (since a bvh requires memory that grows linearly with the number of triangles).

`generate_grid_sdf` uses a binary heap to keep track of the closest triangle for each cell. At initialization, the heap is filled with the closest triangle for each cell. Then, the heap is iteratively updated by checking the distance to the closest triangle for each cell, until the heap is empty.

*Determining sign*: currently the only method is to check the normals of the triangles. A robust method is needed, via raycasting for example. Raycasting can be optimized for the grid by aligning the rays with the grid axes.

Point - triangles methods can be optimized further.