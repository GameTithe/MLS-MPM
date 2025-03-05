import taichi as ti
import math
import random
import numpy as np

ti.init(arch=ti.gpu)

vec3 = ti.math.vec3
ivec3 = ti.math.ivec3
vec2 = ti.math.vec2
mat2 = ti.math.mat2

grid_res = 64
grid_size = 1.0
num_cells = grid_res * grid_res #4096
 
n = 5
particle_count = n * n  
box_x = 30
box_y = 30
#TODO: exchage to 3D
box_center = ti.Vector([grid_res//2, grid_res//2, 0.0])    

 
left = box_center.x - box_x + 2
right = box_center.x + box_x - 2

bottom = box_center.y - box_y + 3
top = box_center.y + box_y - 3

dt = 1e-3

drag_damping = 1
gravity = ti.Vector([0.0, -9.8, 0.0])

Particle = ti.types.struct(
    x = ti.types.vector(3, ti.f32),
    v = ti.types.vector(3, ti.f32),
    mass = ti.f32,
    C = ti.types.matrix(2, 2, ti.f32)
) 
particles = Particle.field(shape = particle_count)
particles_radius = ti.field(dtype=ti.f32, shape=particle_count)


Cell = ti.types.struct(
    v = ti.types.vector(3, ti.f32),
    mass = ti.f32
)

grid = Cell.field(shape = num_cells)

weights = ti.Vector.field(3, dtype=ti.f32, shape =(2,))

@ti.func
def Reset_Grid(): 
    for i in grid:
        grid[i].mass = 0.0
        grid[i].v = ti.Vector([0.0, 0.0, 0.0])
        
        
@ti.kernel
def initialise():  
     
    Reset_Grid() 
    for i in range(particle_count):
        x = i % n
        y = i // n 
        
        particles[i].x = [ (x - n / 2.0) * grid_size, (y - n / 2.0) * grid_size, 0.0]  + box_center 
        particles[i].v =[ ( ti.random() -0.5)  * 10, ( 1 + ti.random()) + 2, 0.0]
        particles[i].mass = ti.random() #ti.random()
        particles_radius[i] = 0.2 # ti.random() * 0.1
        particles[i].C = 0.0
   
 
@ti.func
def P2G():
    
    for i in particles:  
        p = particles[i]  
        
        cell_idx_x = ti.cast(p.x.x, ti.i32)   
        cell_idx_y = ti.cast(p.x.y, ti.i32)  
        
        cell_idx = ti.Vector([cell_idx_x, cell_idx_y, 0.0]) 
        cell_diff = (p.x - cell_idx) - ti.Vector([0.5, 0.5, 0.0])  
        
        weights = [0.5 * pow(0.5 - cell_diff, 2), 0.75 - pow(cell_diff, 2), 0.5 * pow(0.5 + cell_diff, 2)]

        total = 0.0
        for gy in ti.static(range(-1, 2)):
            for gx in ti.static(range(-1, 2)):
                weight = weights[gx + 1][0] * weights[gy + 1][1]
                cell_x = ivec3([cell_idx.x + gx, cell_idx.y + gy, 0])

                grid_idx = cell_x.y * grid_res + cell_x.x
                
                
                if 0 <= grid_idx < num_cells: 
                    cell_dist = (cell_x - p.x) + 0.5
                    Q = p.C @ cell_dist.xy
                    
                    weighted_mass = weight * p.mass
                    
                    grid[grid_idx].mass += weighted_mass
                    
                    grid[grid_idx].v += weighted_mass * (p.v + ivec3([Q.x, Q.y, 0.0]))
                    #print(grid[grid_idx].v)
                     
                    # cell = grid[grid_idx]
                    
                    # cell.mass += weighted_mass  
                    # cell.v += weighted_mass * (p.v + ivec3([Q.x, Q.y, 0.0]))   
                    
                    # grid[grid_idx] = cell       
    
    
    
@ti.func
def GridUpdate(): 
    for i in grid:
        x = i % grid_res
        y = i // grid_res
        
        if(grid[i].mass > 0.0):
            grid[i].v /= grid[i].mass
            grid[i].v += gravity.xyz * dt
            
            if x < 2 or x >= grid_res - 2:
                grid[i].v.x = 0.0
            if y < 2 or y >= grid_res - 2:
                grid[i].v.y = 0.0 
         

@ti.func
def G2P():# G2P  
    for i in range( particle_count ):
        
        p = particles[i]
        p.v = 0.0
        
        cell_idx_y = ti.cast(p.x.y, ti.i32)
        cell_idx_x = ti.cast(p.x.x, ti.i32)   
        cell_idx = ti.Vector([cell_idx_x, cell_idx_y, 0.0])
        
        cell_diff = (p.x - cell_idx) - 0.5  
        weights = [0.5 * pow(0.5 - cell_diff, 2), 0.75 - pow(cell_diff, 2), 0.5 * pow(0.5 + cell_diff, 2)]
       
        B = ti.Matrix.zero(ti.f32, 2, 2)
         
        for gy in ti.static(range(-1, 2)):
            for gx in ti.static(range(-1, 2)):
                weight = weights[gx + 1][0] * weights[gy + 1][1]
                cell_x = ivec3([cell_idx.x + gx, cell_idx.y + gy, cell_idx.z])
            
                grid_idx = ti.cast(cell_x.y, ti.i32) * grid_res + ti.cast(cell_x.x, ti.i32)
                
                if 0 <= grid_idx < num_cells:

                    cell_dist = (cell_x - p.x) + 0.5
                    
                    cell = grid[grid_idx]
                    
                    weighted_velocity = cell.v * weight 
                    term = weighted_velocity.xy.outer_product(cell_dist.xy) 
                    B += term 
                    p.v += vec3(weighted_velocity.x, weighted_velocity.y, 0.0)
         
        p.C = B * 4.0    
        
        cell_index = ti.cast(p.x.y, ti.i32) * grid_res + ti.cast(p.x.x, ti.i32)
        if cell_index >= 0 and cell_index < num_cells :
            cell = grid[cell_index] 
            
            p.x += p.v * dt 
            p.x.x = ti.max(ti.min(p.x.x, right), left)
            p.x.y = ti.max(ti.min(p.x.y, top), bottom) 
            p.x.z = ti.max(ti.min(p.x.z, 0.0), 0.0)
            
            particles[i] = p
@ti.kernel
def Simulate():
     
    Reset_Grid()     
    P2G()   
    GridUpdate()
    G2P()    
    
    for i in particles:
        print(particles[i].x)
     
             
   
window = ti.ui.Window("MLS-MPM Simulation", (1024, 1024), vsync = True)
canvas = window.get_canvas() 
scene = ti.ui.Scene()
camera = ti.ui.Camera()
              

 
pos = ti.Vector.field(3, dtype=ti.f32, shape =(1,))
times = 0.0


pos = np.array([31.5, 31.5, 0])  # 초기 위치 (x, y, z)
vel = np.array([0.0, 0.0, 0.0])  # 초기 속도 (0, 0, 0)

current_t = 0.0
while window.running: 
    
    if current_t < dt:
        initialise()
            
    Simulate()
    
    current_t += dt
    
        
    # routine  
    camera.position(box_center.x, box_center.y, 100) 
    camera.lookat(box_center.x, box_center.y, 0)
    
    scene.set_camera(camera) 
    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    
    scene.particles(particles.x, radius=0,per_vertex_radius= particles_radius, color=(0.5, 0.42, 0.8))
    
    canvas.scene(scene)
    window.show()