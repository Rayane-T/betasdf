from mesh_to_sdf import get_surface_point_cloud
import pyrender
import numpy as np
import trimesh


def scale_to_unit_sphere(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    translation= mesh.bounding_box.centroid
    distances = np.linalg.norm(vertices, axis=1)
    if(np.max(distances)>1.0): 
        vertices /= np.max(distances)
        print("body is normalized by: ",np.max(distances))
    else:
        print("no need for normalization")
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces),translation

# Sample some uniform points and some normally distributed around the surface as proposed in the DeepSDF paper
def sample_sdf_near_surface(mesh, number_of_points = 500000, surface_point_method='scan', sign_method='normal', scan_count=100, scan_resolution=400, sample_point_count=10000000, normal_sample_count=11, min_size=0, return_gradients=False):
    mesh, _ = scale_to_unit_sphere(mesh)
    
    if surface_point_method == 'sample' and sign_method == 'depth':
        print("Incompatible methods for sampling points and determining sign, using sign_method='normal' instead.")
        sign_method = 'normal'

    surface_point_cloud = get_surface_point_cloud(mesh, surface_point_method, 1, scan_count, scan_resolution, sample_point_count, calculate_normals=sign_method=='normal' or return_gradients)

    return surface_point_cloud.sample_sdf_near_surface(number_of_points, surface_point_method=='scan', sign_method, normal_sample_count, min_size, return_gradients)

mesh = trimesh.load('./dataset/t_posemesh/t_posemesh_8.obj',process=False)

nb_samples=5000000

points, sdf, gradient = sample_sdf_near_surface(mesh, number_of_points=nb_samples,return_gradients=True)

colors = np.zeros(points.shape)
colors[sdf < 0, 2] = 1
colors[sdf > 0, 0] = 1

# View the sampled mesh 
#cloud = pyrender.Mesh.from_points(points, colors=colors)
#scene = pyrender.Scene()
#scene.add(cloud)
#viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)

surface=np.zeros_like(sdf)

on_surface=int(nb_samples* 47 / 50)

surface=np.zeros_like(sdf)
surface[:on_surface]=1


with open("t_posesampled_8.npy", 'wb') as f:
    np.save(f, points)
    np.save(f, sdf)
    np.save(f,gradient)
    np.save(f,surface)