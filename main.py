from vedo import *

mesh = Mesh("map_data/Manhattan.obj",)
mesh.texture("map_data/Manhattan.png", scale = 1)
mesh.show()