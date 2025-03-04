import taichi as ti
import math
import random

ti.init(arch=ti.gpu)

grid_res = 64
grid_size = 1.0
num_cells = grid_res * grid_res #4096
 
n = 10
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

#prevVel = ti.Vector([0.0, 0.0, 0.0])
prevVel = ti.Vector.field(3, dtype=ti.f32, shape=(1, ))  

@ti.kernel
def initialise():  
     
    for i in range(particle_count):
        x = i % n
        y = i // n 
        
        #particles[i].x = [ (x - n / 2) * grid_size * 2, (y - n / 2) * grid_size * 2, 0]* (ti.random() -0.5) + box_center
        particles[i].x = [ (x - n / 2) * grid_size, (y - n / 2) * grid_size, 0.0]  + box_center 
        particles[i].v = [0.0, 0.0, 0.0]
        particles[i].mass = 3.0 #ti.random()
        particles_radius[i] = 0.2 # ti.random() * 0.1
        particles[i].C = 0.0
         
        
@ti.kernel
def Simulate():

    for i in range(num_cells):
        grid[i].mass = 0.0
        grid[i].v = ti.Vector([0.0, 0.0, 0.0])
            
    # P2G (Particle to Grid)
    for i in range(particle_count): 
        
        p = particles[i]  
        
        cell_idx_x = ti.cast(p.x.x, ti.i32)   
        cell_idx_y = ti.cast(p.x.y, ti.i32)  
        
        cell_idx = ti.Vector([cell_idx_x, cell_idx_y, 0.0]) 
        cell_diff = (p.x - cell_idx) - ti.Vector([0.5, 0.5, 0.0])  
        
        # cell_index = cell_idx_y * grid_res + cell_idx_x
        # if cell_index >= 0 and cell_index < num_cells:
        #     cell = grid[cell_index]
        #     cell.v = p.v
        #     cell.mass = p.mass
        #     grid[cell_index] = cell  
        
        
        # Quadratic B-Spline 
        weights[0] = 0.5 * pow(0.5 - cell_diff, 2)
        weights[1] = 0.75 - pow(cell_diff, 2)
        weights[2] = 0.5 * pow(0.5 + cell_diff, 2)
    
        # 주변 9개 셀에 대해 질량과 운동량 업데이트  
        total = 0.0
        for gx in (range(3)):
            for gy in (range(3)):
                 
                weight = weights[gx].x * weights[gy].y 
                cell_x = cell_idx + ti.Vector([gx - 1, gy - 1, 0.0]) 
                
                if(cell_x.x < 0 or cell_x.x >= grid_res): 
                    continue
                
                if(cell_x.y < 0 or cell_x.y >= grid_res): 
                    continue
                
                cell_dist =  ti.Vector([cell_x.x - p.x.x, cell_x.y - p.x.y]) + ti.Vector([0.5, 0.5]) 
                #cell_dist = ti.Vector([p.x.x - cell_x.x, p.x.y - cell_x.y])
                
                # m*B(xi - xp)
                Q = p.C @ cell_dist  
                # Weight  * mp
                mass_contrib = weight * p.mass
                total += mass_contrib
                
                # converting 2D index to 1D 
                cell_index = ti.cast(cell_x.y, ti.i32) * grid_res + ti.cast(cell_x.x, ti.i32)
                cell = grid[cell_index]
                
                cell.mass += mass_contrib 
                cell.v += mass_contrib * ( p.v + ti.Vector([Q.x, Q.y, 0.0])) 
                #cell.v += mass_contrib * p.v 
                grid[cell_index] = cell
                
        print(total) 
         
    #gird velocity update                
    for i in range(num_cells):
        cell = grid[i]  
        
        if  cell.mass > 0.0:  
            # 여러 입자로부터 누적된 velocity이기 때문에
            # 질량으로 나눠준다. 
            
            cell.v /= cell.mass
            cell.v += dt * (gravity)
            
            #boundary condition 
            y = i // grid_res
            x = i % grid_res 
             
            if x < left or x > right:
                cell.v.x = 0.0
                
            if y < bottom or y > top:
                cell.v.y = 0.0
               
            grid[i] = cell  
            
        cell.v.z = 0.0
        grid[i] = cell 
        
         
    # G2P  
    for i in range( particle_count ):
        
        p = particles[i]
        p.v = 0.0
        
        cell_idx_y = ti.cast(p.x.y, ti.i32)
        cell_idx_x = ti.cast(p.x.x, ti.i32)   
        cell_idx = ti.Vector([cell_idx_x, cell_idx_y, 0.0])
        
        cell_diff = (p.x - cell_idx) - 0.5 

        weights[0] = 0.5 * pow(0.5 - cell_diff, 2.0)
        weights[1] = 0.75 - pow(cell_diff, 2.0)
        weights[2] = 0.5 * pow(0.5 + cell_diff, 2.0)
        
        B = ti.Matrix.zero(ti.f32, 2, 2)
        
        for gx in range(3) :
            for gy in range(3) :
                weight = weights[gx].x * weights[gy].y
                 
                cell_x = cell_idx + ti.Vector([gx - 1, gy - 1, 0.0]) 
                cell_index = ti.cast(cell_x.y, ti.i32) * grid_res + ti.cast(cell_x.x, ti.i32)
                
                if cell_index < 0 or cell_index >= num_cells :
                    continue 
                
                dist = (cell_x - p.x) + ti.Vector([0.5, 0.5, 0.0])
                weighted_velocity = grid[cell_index].v * weight
                 
                # term = ti.Matrix([
                #     [weighted_velocity.x * dist.x, weighted_velocity.x * dist.y], 
                #     [weighted_velocity.y * dist.x, weighted_velocity.y * dist.y],  
                # ]) 
                
                term = weighted_velocity.xy.outer_product(dist.xy)
                
                B += term
                p.v += ti.Vector([weighted_velocity.x, weighted_velocity.y, 0.0])
        
        p.C = B * 4.0   
        
        #print("1:", p.v,",", prevVel[0], ",", p.v - prevVel[0])   
         
        cell_index = ti.cast(p.x.y, ti.i32) * grid_res + ti.cast(p.x.x, ti.i32)
        
        if cell_index >= 0 and cell_index < num_cells :
            cell = grid[cell_index]
            #print(cell.v)
            p.x += p.v * dt 
            p.x.x = ti.max(ti.min(p.x.x, right), left)
            p.x.y = ti.max(ti.min(p.x.y, top), bottom) 
            p.x.z = ti.max(ti.min(p.x.z, 0.0), 0.0)
            
            particles[i] = p  
        

  
   
window = ti.ui.Window("MLS-MPM Simulation", (1024, 1024), vsync = True)
canvas = window.get_canvas() 
scene = ti.ui.Scene()
camera = ti.ui.Camera()
              

initialise()
 
while window.running: 
        
    Simulate()
 
    # routine  
    camera.position(box_center.x, box_center.y, 100) 
    camera.lookat(box_center.x, box_center.y, 0)
    
    scene.set_camera(camera) 
    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    
    scene.particles(particles.x, radius=0,per_vertex_radius= particles_radius, color=(0.5, 0.42, 0.8))
    
    canvas.scene(scene)
    window.show()