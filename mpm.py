import taichi as ti
import math
import random

ti.init(arch=ti.gpu)

grid_res = 64
grid_size = 1.0 / grid_res

num_cells = grid_res * grid_res

dt = 1e-3


n = 15 
particle_count = n * n 

Particle = ti.types.struct(
    x = ti.types.vector(3, ti.f32),
    v = ti.types.vector(3, ti.f32),
    mass = ti.f32
) 

particles = Particle.field(shape = particle_count)

Cell = ti.types.struct(
    v = ti.types.vector(3, ti.f32),
    mass = ti.f32
)

grid = Cell.field(shape = num_cells)

 
@ti.kernel
def initialise():  
    for i in range(particle_count):
        x = i % n
        y = i // n 
        
        particles[i].x = [ (x - n / 2) * grid_size, (y - n / 2) * grid_size, 0]
        particles[i].v = [ ti.random() - 0.5, ti.random() - 0.5 + 2.75, 0]
        particles[i].mass = 1.0
          
      
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
    
    scene.particles(particles.x, radius= 0.01, color=(0.5, 0.42, 0.8))

    canvas.scene(scene)
    window.show()