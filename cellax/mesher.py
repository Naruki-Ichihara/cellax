from pucgen.pucgen import PUC, BaseCell, SphericalInclusion, CylindricalChannel
import meshio
import numpy as np
import os

def delete_mesh(filepath, cell_type="HEX8"):
    mesh = meshio.read(filepath)
    matids = mesh.cell_data_dict["mat_id"][cell_type]
    cells = mesh.cells_dict[cell_type]
    target_matid = 1
    selected = matids == target_matid
    selected_cells = cells[selected]
    unique_points, inverse_indices = np.unique(selected_cells, return_inverse=True)
    new_points = mesh.points[unique_points]
    new_cells = [(cell_type, inverse_indices.reshape(selected_cells.shape))]
    new_cell_data = {"mat_id": [matids[selected]]}
    return new_points, new_cells, new_cell_data


def graph():
    puc = PUC(cell_mat_id=None)
    puc.add(BaseCell(dimension=(1, 1, 1), el_size=0.1, mat_id=2))
    puc.add(CylindricalChannel(dimension=0.15, central_point=(0, 0, 0),
                           direction='x', el_size=0.05, mat_id=1))
    puc.add(CylindricalChannel(dimension=0.15, central_point=(0, 0, 0),
                           direction='y', el_size=0.05, mat_id=1))
    puc.add(CylindricalChannel(dimension=0.15, central_point=(0, 0, 0),
                           direction='z', el_size=0.05, mat_id=1))
    puc('temp.vtk')
    points, cells, cell_data = delete_base('temp.vtk', "tetra")
    os.remove('temp.vtk')
    meshio.write_points_cells(
    "output.vtk",
    points,
    cells,
    cell_data=cell_data
    )
    return points, cells, cell_data

graph()
