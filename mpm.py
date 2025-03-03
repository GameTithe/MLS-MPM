import taichi as ti
import math
import random

ti.init(arch=ti.gpu)

grid_res = 64
grid_size = 1.0
num_cells = grid_res * grid_res #4096
 
n = 16
particle_count = n * n  
box_x = 30
box_y = 40
#TODO: exchage to 3D
box_center = ti.Vector([grid_res//2, grid_res//2, 0.0])    

 
left = box_center.x - box_x + 2
right = box_center.x + box_x - 2

bottom = box_center.y - box_y + 3
top = box_center.y + box_y - 3

dt = 3e-3

drag_damping = 1
gravity = ti.Vector([0.0, -9.8, 0.0])

Particle = ti.types.struct(
    x = ti.types.vector(3, ti.f32),
    v = ti.types.vector(3, ti.f32),
    mass = ti.f32,
    C = ti.types.matrix(3, 3, ti.f32)
) 
particles = Particle.field(shape = particle_count)
particles_radius = ti.field(dtype=ti.f32, shape=particle_count)


Cell = ti.types.struct(
    v = ti.types.vector(3, ti.f32),
    mass = ti.f32
)

grid = Cell.field(shape = num_cells)

weights = ti.Vector.field(3, dtype=ti.f32, shape =(3,))
@ti.kernel
def initialise():  
     
    for i in range(particle_count):
        x = i % n
        y = i // n 
        
        #particles[i].x = [ (x - n / 2) * grid_size * 2, (y - n / 2) * grid_size * 2, 0]* (ti.random() -0.5) + box_center
        particles[i].x = [ (x - n / 2) * grid_size, (y - n / 2) * grid_size, 0]  + box_center 
        #particles[i].v = [ 0, 0, 0]
        particles[i].v = [ti.random() * 5, (1 + ti.random()) * 10, 0]
        particles[i].mass = 1 #ti.random()
        particles_radius[i] = 0.5 # ti.random() * 0.1
        particles[i].C = 0
        
         
    
@ti.kernel
def Simulate():
    for i in range(num_cells):
        grid[i].mass = 0
        grid[i].v = ti.Vector([0.0, 0.0, 0.0])

    # P2G (Particle to Grid)
    for i in range(particle_count): 
        
        p = particles[i]
        #cell_idx = ti.cast(p.x.xyz, ti.i32)  
            
        cell_idx_y = ti.cast(p.x.y, ti.i32)  
        cell_idx_x = ti.cast(p.x.x, ti.i32)   
        cell_idx = ti.Vector([cell_idx_x, cell_idx_y, 0])
        
        cell_diff = (p.x - cell_idx) - 0.5 
        
        # Quadratic B-Spline
        weights[0] = 0.5 * ti.pow(0.5 - cell_diff, 2)
        weights[1] = 0.75 - ti.pow(cell_diff, 2)
        weights[2] = 0.5 * ti.pow(0.5 + cell_diff, 2)

        # 주변 9개 셀에 대해 질량과 운동량 업데이트
        for gx in (range(3)):
            for gy in (range(3)):
                #TODO: z값 고려
                weight = weights[gx].x * weights[gy].y 
                cell_x = cell_idx + ti.Vector([gx - 1, gy - 1, 0])
                
                # cell_dist = ti.Matrix([
                #     [cell_x.x - p.x.x + 0.5],
                #     [cell_x.y - p.x.y + 0.5],
                #     [cell_x.z - p.x.z + 0.5]
                # ]) 
                cell_dist = cell_x - p.x + ti.Vector([0.5, 0.5, 0.5]) 
                
                #cell_dist_mat = ti.Matrix([[cell_dist.x], [cell_dist.y], [cell_dist.z]])  # 3x1 행렬 변환
                #Q = p.C @ cell_dist_mat
                Q = p.C @ cell_dist 
                
                # Weight  * mp
                # mi
                mass_contrib = weight * p.mass
                
                # converting 2D index to 1D 
                cell_index = ti.cast(cell_x.y, ti.i32) * grid_res + ti.cast(cell_x.x, ti.i32)

                if cell_index < 0 or cell_index >= num_cells :
                    continue
                
                cell = grid[cell_index]
                
                cell.mass += mass_contrib

                cell.v += mass_contrib * (p.v + Q)
                 
                grid[cell_index] = cell
                
    
    #gird velocity update                
    for i in range(num_cells):
        cell = grid[i]
        if  cell.mass > 0: 
            cell.v /= cell.mass
            cell.v += dt * gravity
            
            #boundary condition 
            x = i // grid_res
            y = i % grid_res
            
             
            if x < left or x > right:
                cell.v.x = 0
                
            if y < bottom or y > top:
                cell.v.y = 0
                #cell.v *= ti.exp(-drag_damping * dt)  
                
            print(cell.v)
            grid[i] = cell
            
        cell.v.z = 0 
        
        grid[i] = cell
         
    # G2P  
    for i in range( particle_count ):
       
        p = particles[i]
        p.v = 0  
        
        cell_idx_y = ti.cast(p.x.y, ti.i32)
        cell_idx_x = ti.cast(p.x.x, ti.i32)   
        cell_idx = ti.Vector([cell_idx_x, cell_idx_y, 0])
        
        cell_diff = (p.x - cell_idx) - 0.5 #ti.Vector([0.5, 0.5, 0.5]) 

        weights[0] = 0.5 * ti.pow(0.5 - cell_diff, 2)
        weights[1] = 0.75 - ti.pow(cell_diff, 2)
        weights[2] = 0.5 * ti.pow(0.5 + cell_diff, 2)
        
        B = ti.Matrix.zero(ti.f32, 3, 3)
        
        for gx in range(3):
            for gy in range(3) :
                weight = weights[gx].x * weights[gy].y
                 
                cell_x = cell_idx + ti.Vector([gx - 1, gy - 1, 0]) 
                cell_index = ti.cast(cell_x.y, ti.i32) * grid_res + ti.cast(cell_x.x, ti.i32)
                
                dist = (cell_x - p.x) + ti.Vector([0.5, 0.5, 0.5])
                
                if cell_index < 0 or cell_index >= num_cells :
                    continue 
                 
                   
                weighted_velocity = grid[cell_index].v * weight
                
                # term = ti.Matrix([
                #     [weighted_velocity.x * dist.x, weighted_velocity.y * dist.x, weighted_velocity.z * dist.x],  # 첫 번째 행
                #     [weighted_velocity.x * dist.y, weighted_velocity.y * dist.y, weighted_velocity.z * dist.y],
                #     [weighted_velocity.x * dist.z, weighted_velocity.y * dist.z, weighted_velocity.z * dist.z],
                # ])
                 
                #B += term
                
                p.v += weighted_velocity
        
        #p.C = B * 4
        p.x += p.v * dt
        
        p.x.x = ti.max(ti.min(p.x.x, right), left)
        p.x.y = ti.max(ti.min(p.x.y, top), bottom)
        p.x.z = ti.max(ti.min(p.x.z, grid_res - 2), 1)
        
        particles[i] = p  
        

  
   
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