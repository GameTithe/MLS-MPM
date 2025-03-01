import taichi as ti
import math
ti.init(arch=ti.gpu)

grid_res = 64
grid_size = 1.0 / grid_res
dt = 1e-3


n = 7 
particle_count = n * n 

particle_x = ti.Vector.field(3, dtype= float, shape=(particle_count, ))
particle_v = ti.Vector.field(3, dtype= float, shape=(particle_count, ))
particle_mass = ti.field(dtype=ti.f32, shape=(particle_count, ))

@ti.kernel
def initialise():
    for i in range(particle_count):
        x = i % n  # 2D 변환: x 좌표
        y = i // n  # 2D 변환: y 좌표
        particle_x[i] = [(x - n / 2) * grid_size, (y - n / 2) * grid_size, 0.0]
        particle_v[i] = [0, 0, 0]

      
window = ti.ui.Window("MLS-MPM Simulation", (1024, 1024), vsync = True)
canvas = window.get_canvas() 
scene = ti.ui.Scene()
camera = ti.ui.Camera()
     
          
current_t = 0.0
end_t = 10.0

initialise()

while window.running:
    if(current_t > end_t):
        initialise() 
        current_t = 0.0
    
    # routine 
    
    camera.position(0.0, 0.0, 3)
    camera.lookat(0.0, 0.0, 0.0)
    
    scene.set_camera(camera) 
    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    
    scene.particles(particle_x, radius= 0.01, color=(0.5, 0.42, 0.8))

    canvas.scene(scene)
    window.show()