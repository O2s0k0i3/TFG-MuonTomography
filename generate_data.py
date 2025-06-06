import numpy as np
import pandas as pd
import optparse
import matplotlib.pyplot as plt
import json

def get_density(material):
    # density of the different materials in g/cm3
    if material == "lead":
        return 11.34
    if material == "copper":
        return 8.96
    if material == "iron":
        return 7.874
    if material == "aluminium":
        return 2.7
    if material == "steel":
        return 7.85

def poca(r1, r2, v1, v2):
    v3 = np.cross(v1, v2)
    v2_cross_v3 = np.cross(v2, v3)
    det = np.dot(v1, -v2_cross_v3)

    if np.abs(det) < 1.0e-6 or np.isnan(det):
        return False, [0, 0, 0]
    inv_det = 1/det
    delta_r = r1 - r2
    v1_cross_v3 = np.cross(v1, v3)
    xpoca1 = r1 + v1 * np.dot(delta_r, v2_cross_v3) * inv_det
    xpoca2 = r2 + v2 * np.dot(delta_r, v1_cross_v3) * inv_det
    v = 0.5 * (xpoca1 + xpoca2)
    return True, v

def read_json(path):
    pipe_number = path.split('.')[0].split('_')[1]
    with open(path, "r") as file:
        content = json.load(file)
    world = content['TheWorld']
    pipe = content['ComplexPipe']

    return {'width': 51.2,
              'height': 51.2,
              'dpi': 10,
              'xcenter': float(pipe['x']),
              'ycenter': float(pipe['y']),
              'inner_radius': float(pipe['innerRadius_1']),
              'outer_radius': float(pipe['outerRadius']),
              'material': pipe['outerMaterial_1'],
              'pipe_number': int(pipe_number)}

def real_pipe(dataset, params, output_path):
    width = params["width"]
    height = params["height"]
    dpi = params["dpi"]
    x = []
    y = []
    z = []
    for _, row in dataset.iterrows():
        r1 = np.asarray([row['x1'], row['y1'], row['z1']])
        r2 = np.asarray([row['x2'], row['y2'], row['z2']])
        v1 = np.asarray([row['vx1'], row['vy1'], row['vz1']])
        v2 = np.asarray([row['vx2'], row['vy2'], row['vz2']])

        if True in np.isnan(v1) or True in np.isnan(v2):
            continue
        v1_normalized = v1/np.linalg.norm(v1)
        v2_normalized = v2/np.linalg.norm(v2)
        val = np.dot(v1_normalized, v2_normalized)
        if val > 1.0:
            val = 1.0
        theta = np.arccos(val)
        # Find the optimal value here to clean some noise
        if theta < 0.06:
            continue

        valid, point = poca(r1, r2, v1, v2)

        if not valid or np.abs(point[0]) > width or np.abs(point[1]) > height:
            continue

        x.append(point[0])
        y.append(point[1])
        z.append(point[2])

    plt.figure(figsize=(width, height), dpi=dpi)
    plt.hist2d(y, z, bins=(100, 100), cmap="binary", range=[[-width/2, width/2], [-height/2, height/2]])
    plt.axis("off")
    plt.margins(0, 0)
    plt.savefig(output_path)

def simulated_pipe(params, output_path):
    dpi = params['dpi']
    # All parameters have to be scaled by the dpi to keep the same image size as the real one.
    width = int(params["width"] * dpi)
    height = int(params["height"] * dpi)
    x_center = params["xcenter"] * dpi
    y_center = params["ycenter"] * dpi
    inner_radius = params["inner_radius"] * dpi
    outer_radius = params["outer_radius"] * dpi
    density = get_density(params["material"])

    x = np.linspace(0, width-1, width)
    y = np.linspace(0, height-1, height)
    x, y = np.meshgrid(x, y)

    # center_x and center_y has the origin in the center of the image
    # the origin of the coordinates is top left, so we need to
    # subtract half the width and half the height here
    dist_from_center = np.sqrt((x + x_center - width/2)**2 + (y + y_center - height/2)**2)

    image = np.zeros((width, height))

    pipe_mask = (dist_from_center >= inner_radius) & (dist_from_center <= outer_radius)
    image[pipe_mask] = density

    plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
    # vmin and vmax are chosen based on the min and max density of the materials used
    plt.imshow(image, vmin=2, vmax=12, cmap="binary")
    plt.axis("off")
    plt.margins(0, 0)
    plt.savefig(output_path)

def main(opts):
    params = read_json(opts.json)
    dataset = pd.read_hdf(opts.hd5)

    real_pipe(dataset, params, opts.output_real)
    simulated_pipe(params, opts.output_sim)

if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option('--hd5')
    parser.add_option('--json')
    parser.add_option('--output-real')
    parser.add_option('--output-sim')
    opts, args = parser.parse_args()
    main(opts)
