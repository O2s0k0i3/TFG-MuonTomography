import numpy as np
import matplotlib.pyplot as plt

"""

Subirlo a github
mirar como hacer para poner densidades realistas, no entre 0 y 1
una vez tenga las trayectorias generadas, correr este programa para sacar las imagenes y correr POCA.
ambas imagenes tienen que tener las mismas dimensiones (poca y esta)
arreglar el ejemplo para que plotee las graficas finales

"""

def draw_pipe(center_x, center_y, inner_radius, outer_radius, img_width, img_height, density):
    x = np.linspace(0, img_width-1, img_width)
    y = np.linspace(0, img_height-1, img_height)
    X, Y = np.meshgrid(x, y)

    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    image = np.zeros((img_height, img_width))

    pipe_mask = (dist_from_center >= inner_radius) & (dist_from_center <= outer_radius)
    image[pipe_mask] = np.clip(density, 0, 1)

    return image

img_width = 300
img_height = 200
center_x = img_width // 2
center_y = img_height // 3
inner_radius = 40
outer_radius = 60
density = 0.7

pipe_image = draw_pipe(center_x, center_y, inner_radius, outer_radius, img_width, img_height, density)

# Para guardar la imagen:
# plt.imsave("imagen.png", pipe_image, vmin=0, vmax=1, cmap="binary")

# Para mostrar la imagen:
plt.imshow(pipe_image, vmin=0, vmax=1, cmap="binary")
plt.axis('equal')
plt.show()
