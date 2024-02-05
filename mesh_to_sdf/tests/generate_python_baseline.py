"""This file is used to generate the "ground truth" SDFs for the crate tests."""


# Load some mesh (don't necessarily need trimesh)
import trimesh
mesh = trimesh.load("assets/suzanne.glb").geometry['Suzanne']

query_points = [[0, 0, 0], [1 ,1 ,1 ], [0.1, 0.2, 0.2]]

# pysdf
from pysdf import SDF
sdf = SDF(mesh.vertices, mesh.faces);
sdf_multi_point = sdf(query_points)
print("pysdf", sdf_multi_point)
#pysdf [0.45216727 -0.6997909   0.45411023] # negative is outside in pysdf


import mesh_to_sdf
import numpy as np
query_points = np.array(query_points)
sdf = mesh_to_sdf.mesh_to_sdf(mesh, query_points, surface_point_method='scan', sign_method='normal', bounding_radius=None, scan_count=100, scan_resolution=400, sample_point_count=10000000, normal_sample_count=11)
print("mesh_to_sdf", sdf)
# mesh_to_sdf [-0.40961263  0.6929414  -0.46345082] # negative is inside in mesh_to_sdf

