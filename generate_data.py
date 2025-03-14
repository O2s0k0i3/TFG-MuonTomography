import numpy as np
import pandas as pd
import optparse
import matplotlib.pyplot as plt
import json

def get_density(material):
    # density of the different materials in g/cm3
    match material:
        case "lead": return 11.35
        case "copper": return 8.96
        case "iron": return 7.874
        case "aluminium": return 2.7
        case "steel": return 7.85

def poca(r1, r2, v1, v2):
    v3 = np.cross(v1, v2)
    v2_cross_v3 = np.cross(v2, v3)
    det = np.dot(v1, -v2_cross_v3)
    if np.abs(det) < 1.0e-6:
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

    return {'width': int(world['xSizeWorld']), 
              'height': int(world['ySizeWorld']), 
              'xcenter': float(pipe['x']), 
              'ycenter': float(pipe['y']), 
              'inner_radius': float(pipe['innerRadius_1']), 
              'outer_radius': float(pipe['outerRadius']), 
              'material': pipe['outerMaterial_1'],
              'pipe_number': int(pipe_number)}

def real_pipe(dataset, params):
    width = params["width"]
    height = params["height"]
    x = []
    y = []
    z = []
    for _, row in dataset.iterrows():
        r1 = np.asarray([row['x1'], row['y1'], row['z1']])
        r2 = np.asarray([row['x2'], row['y2'], row['z2']])
        v1 = np.asarray([row['vx1'], row['vy1'], row['vz1']])
        v2 = np.asarray([row['vx2'], row['vy2'], row['vz2']])
        valid, point = poca(r1, r2, v1, v2)

        if not valid or np.abs(point[0]) > width or np.abs(point[1]) > height:
            continue

        x.append(point[0])
        y.append(point[1])
        z.append(point[2])

    plt.figure(figsize=(width/100, height/100))
    plt.hist2d(x, y, bins=(width, height), cmap="binary")
    plt.axis("off")
    plt.margins(0, 0)
    plt.savefig(f"images/real/real_{params["pipe_number"]}.png")

def simulated_pipe(params):
    width = params["width"]
    height = params["height"]
    density = get_density(params["material"])

    x = np.linspace(0, width-1, width)
    y = np.linspace(0, height-1, height)
    X, Y = np.meshgrid(x, y)

    # center_x and center_y has the origin in the center of the image
    # the origin of the coordinates is top left, so we need to
    # subtract half the width and half the height here
    dist_from_center = np.sqrt((X - params["xcenter"] - width/2)**2 + (Y - params["ycenter"] - height/2)**2)

    image = np.zeros((height, width))

    pipe_mask = (dist_from_center >= params["inner_radius"]) & (dist_from_center <= params["outer_radius"])
    image[pipe_mask] = density

    # codificar el formato de la imagen en el nombre de la misma forma que esté en los datos
    # name = f"sim_{center_x}_{center_y}_{inner_radius}_{outer_radius}_{img_width}_{img_height}_{density}.png"
    name = f"images/simulated/sim_{params["pipe_number"]}.png"
    # vmin y vmax se han escogido en base a las densidades mínima y máxima que se están considerando
    plt.imsave(name, image, vmin=2, vmax=12, cmap="binary")

def main(opts):
    params = read_json(opts.json)
    dataset = pd.read_hdf(opts.hd5)

    real_pipe(dataset, params)
    simulated_pipe(params)

if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option('--hd5')
    parser.add_option('--json')
    opts, args = parser.parse_args()
    main(opts)
