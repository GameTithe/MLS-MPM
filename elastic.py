import taichi as ti
import numpy as np

# ti.init(arch=ti.cpu, debug=True, log_level=ti.TRACE)
# ti.init(arch=ti.vulkan, debug=True, log_level=ti.TRACE)
# ti.init(arch=ti.vulkan, debug=True)
# ti.init(arch=ti.cpu, debug=True)
ti.init(arch=ti.cuda, debug=True)
# ti.init(arch=ti.cuda, default_fp=ti.f32, default_ip=ti.i32)
# ti.init(arch=ti.cpu, cpu_max_num_threads=1, debug=True)

vec3 = ti.math.vec3
ivec3 = ti.math.ivec3
vec2 = ti.math.vec2
mat2 = ti.math.mat2
 
Particle = ti.types.struct(
    pos = vec3,
    vel = vec3,
    mass = ti.f32,
    volume_0 = ti.f32,
    C = mat2,
    F = mat2 
)
 
Cell = ti.types.struct(
    vel = vec2,
    mass = ti.f32  
) 

grid_res = 64
cell_count = grid_res * grid_res
grid = Cell.field( shape=(grid_res * grid_res) )
 

num  = 64
particle_count = num * num
particles = Particle.field(shape=(particle_count, ))  

debug = ti.field(dtype=vec3, shape=(1,))

#equation of state
eos_stiffness = 10.0
eos_power = 4

#fluid parameters
rest_density = 4.0
dynamic_viscosity = 0.1

@ti.func
def ClearGrid():
    for i in grid:
        cell = grid[i]
        
        cell.mass = 0.0
        cell.vel = ti.Vector([0.0, 0.0]) 
        
        grid[i] = cell

@ti.kernel
def init_simulation():
    ClearGrid()
     

    for i in particles: 
        
        x = i % num
        y = i // num
        particles[i].pos = vec3([grid_res/2.0 + x*0.5 - 16.0, grid_res/2.0 + y* 0.5 - 16.0, 1]) 
        particles[i].vel = vec3([0, 0, 0])
        particles[i].mass = 1.0
        particles[i].volume_0 = 0.0
        particles[i].C = ti.Matrix.zero(dt=ti.f32, n=2, m=2)
        particles[i].F = ti.Matrix.identity(dt=ti.f32, n=2)
         
    
    P2G()
        
    for i in particles:
        particle = particles[i]
        cell_idx = ivec3([int(particle.pos.x), int(particle.pos.y), int(particle.pos.z)])
        cell_diff = (particle.pos - cell_idx) - 0.5
        weights = [0.5 * pow(0.5 - cell_diff, 2), 0.75 - pow(cell_diff, 2), 0.5 * pow(0.5 + cell_diff, 2)]

        density = 0.0
        for gy in ti.static(range(-1, 2)):
            for gx in ti.static(range(-1, 2)):
                
                weight = weights[gx + 1][0] * weights[gy + 1][1]
                cell_x = ivec3([cell_idx.x + gx, cell_idx.y + gy, cell_idx.z])
                c_idx = cell_x.y * grid_res + cell_x.x
                
                if 0 <= c_idx < cell_count :
                    density += grid[c_idx].mass  * weight

        volume = particle.mass / density
        particle.volume_0 = volume
        particles[i].volume_0 = particle.volume_0
        
@ti.func
def P2G():
    
    for i in particles:
        p = particles[i]
        
        # deformation gradient
        F = p.F
        
        # determinant of deformation gradient
        J = ti.math.determinant(F)
        
        # MPM Cours, page 46
        volume = p.volume_0 * J
        
        # Neo-Hookean model (MPM course equation 48)
        # P(Piola kirchoff)를 구하고, F(deformation gradient)를 사용해서..
        F_T = ti.Matrix.transpose(F)
        F_inv_T = ti.math.inverse(F_T)
        F_minus_F_inv_T = F - F_inv_T

        P_term_0 = elastic_mu * F_minus_F_inv_T
        P_term_1 = elastic_lambda * ti.math.log(J) * F_inv_T
        P = P_term_0 + P_term_1
        
        # cauchy stress (MPM course equation 38)
        # P(piola kirchoff)를 이용해서 응력을 구한다
        stress = (1.0 / J) * P @ F_T
        
        eq_16_term_0 = -volume * 4 * stress * dt
        
        cell_idx = ivec3([int(p.pos.x), int(p.pos.y), int(p.pos.z)])
        cell_diff = (p.pos - cell_idx) - 0.5
        
        weights = [0.5 * pow(0.5 - cell_diff, 2), 0.75 - pow(cell_diff, 2), 0.5 * pow(0.5 + cell_diff, 2)]

        for gy in ti.static(range(-1, 2)):
            for gx in ti.static(range(-1, 2)):
                weight = weights[gx + 1][0] * weights[gy + 1][1]
                cell_x = ivec3([cell_idx.x + gx, cell_idx.y + gy, cell_idx.z])

                c_idx = cell_x.y * grid_res + cell_x.x
                # 구역 벗어나면 particles 값이 고장남
                # 디버깅 항상 키자
                if 0 <= c_idx < cell_count: 
                    cell_dist = (cell_x - p.pos) + 0.5
                     
                    
                    Q = p.C @ cell_dist.xy

                    weighted_mass = weight * p.mass
                    grid[c_idx].mass += weighted_mass

                    grid[c_idx].vel += weighted_mass * (p.vel.xy + Q)                
                    momentum = (eq_16_term_0 * weight) @ cell_dist.xy
                    grid[c_idx].vel += momentum

@ti.func
def P2G_1() : 
    for i in particles : 
        p = particles[i] 
               
        cell_idx = ivec3([int(p.pos.x), int(p.pos.y), int(p.pos.z)])
        cell_diff = (p.pos - cell_idx) - 0.5 
        weights = [0.5 * pow(0.5 - cell_diff, 2), 0.75 - pow(cell_diff, 2), 0.5 * pow(0.5 + cell_diff, 2)]

        C = p.C
        for gy in ti.static(range(-1, 2)):
            for gx in ti.static(range(-1, 2)): 
                weight = weights[gx + 1][0] * weights[gy + 1][1]
                cell_x = ivec3([cell_idx.x + gx, cell_idx.y + gy, cell_idx.z]) 
                c_idx = cell_x.y * grid_res + cell_x.x

                if 0 <= c_idx < cell_count:  
                    cell_dist = (cell_x - p.pos) + 0.5 
                    Q = C @ cell_dist.xy 
                    
                    weighted_mass = weight * p.mass
                     
                    grid[c_idx].mass += weighted_mass 
                    grid[c_idx].vel += weighted_mass * (p.vel.xy + Q)        

        
@ti.func 
def P2G_2():
    for i in particles:
        p = particles[i] 
        cell_idx = ivec3([int(p.pos.x), int(p.pos.y), int(p.pos.z)])
        cell_diff = (p.pos - cell_idx) - 0.5
        
        weights = [0.5 * pow(0.5 - cell_diff, 2), 0.75 - pow(cell_diff, 2), 0.5 * pow(0.5 + cell_diff, 2)]

        density = 0.0
        for gy in ti.static(range(-1, 2)):
            for gx in ti.static(range(-1, 2)): 
                weight = weights[gx + 1][0] * weights[gy + 1][1]
                cell_x = ivec3([cell_idx.x + gx, cell_idx.y + gy, cell_idx.z])
                c_idx = cell_x.y * grid_res + cell_x.x

                if 0 <= c_idx < cell_count: 
                    density += grid[c_idx].mass * weight
        
        volume = p.mass / density
        
        # end goal, constitutive equation for isotropic fluid: 
        # stress = -pressure * I + viscosity * (velocity_gradient + velocity_gradient_transposed)
 
 
        # Tait equation of state. i clamped it as a bit of a hack.
        # clamping helps prevent particles absorbing into each other with negative pressures
               
        pressure = ti.math.max(-0.1, eos_stiffness * (ti.math.pow(density / rest_density, eos_power) - 1))
        
        stress = mat2([-pressure, 0], [0, -pressure])
        
        dudv = p.C
        strain = dudv
        
        #trace = strain.c1.x + strain.c0.y
        trace = strain[0, 0] + strain[1,1]
        strain[1,1] = strain[0,0] = trace

        viscosity_term = dynamic_viscosity * strain
        stress += viscosity_term
                
        eq_16_term_0 = -volume * 4 * stress * dt
        
        for gy in ti.static(range(-1, 2)):
            for gx in ti.static(range(-1, 2)): 
                    weight = weights[gx + 1][0] * weights[gy + 1][1]
                    cell_x = ivec3([cell_idx.x + gx, cell_idx.y + gy, cell_idx.z])
                    c_idx = cell_x.y * grid_res + cell_x.x
                    
                    if 0 <= c_idx < cell_count:   
                        cell_dist = (cell_x - p.pos) + 0.5
                        momentum = (eq_16_term_0 * weight ) @ cell_dist.xy
                        grid[c_idx].vel += momentum 
                        
                    
 
@ti.func
def gridUpdate():
    for i in grid:
        x = i % grid_res
        y = i // grid_res
        
        if(grid[i].mass > 0.0):
            grid[i].vel /= grid[i].mass
            grid[i].vel += gravity.xy * dt
            
            if x < 2 or x >= grid_res - 2:
                grid[i].vel.x = 0.0
            if y < 2 or y >= grid_res - 2:
                grid[i].vel.y = 0.0 
         

@ti.func
def G2P():
    for i in particles:
        #particle = Particle()
        particle = particles[i]
        
        particle.vel.fill(0.0)
        
        cell_idx = ivec3([int(particle.pos.x), int(particle.pos.y), int(particle.pos.z)])
        cell_diff = (particle.pos - cell_idx) - 0.5
        
        weights = [0.5 * pow(0.5 - cell_diff, 2), 0.75 - pow(cell_diff, 2), 0.5 * pow(0.5 + cell_diff, 2)]
        B = ti.Matrix.zero(dt=ti.f32, n=2, m=2)

        for gy in ti.static(range(-1, 2)):
            for gx in ti.static(range(-1, 2)):
                weight = weights[gx + 1][0] * weights[gy + 1][1]
                cell_x = ivec3([cell_idx.x + gx, cell_idx.y + gy, cell_idx.z])
                
                c_idx = cell_x.y * grid_res + cell_x.x

                if 0 <= c_idx < cell_count:
                    cell_dist = (cell_x - particle.pos) + 0.5
                    weighted_velocity = grid[c_idx].vel * weight

                    term = weighted_velocity.outer_product(cell_dist.xy)
                        
                    B += term
                    
                    particle.vel += vec3(weighted_velocity.x, weighted_velocity.y, 0.0)
        
        particle.C = B * 4
        particle.pos += particle.vel * dt
        particle.pos = clip_vec3(particle.pos, 1, grid_res - 2)
        
        # Fp_new = ti.Matrix.identity(dt=ti.f32, n=2)
        # Fp_new += particle.C * dt
        # Fp_new = Fp_new @ particle.F 
        # particle.F = Fp_new
         
        particles[i].pos = particle.pos
        
        x_n = particle.pos.xy + particle.vel.xy
        wall_min = 3
        wall_max = grid_res - 4
        
        if x_n.x < wall_min:
            particle.vel.x += wall_min - x_n.x
        if x_n.x > wall_max :
            particle.vel.x += wall_max - x_n.x
        if x_n.y < wall_min :
            particle.vel.y += wall_min - x_n.y
        if x_n.y > wall_max :
            particle.vel.y += wall_max - x_n.y
            
        particles[i] = particle

@ti.func
def clip_vec3(v: vec3, min_val: ti.f32, max_val: ti.f32) -> vec3:
    if v.x <= min_val:
        v.x = min_val  
    if v.x >= max_val:
        v.x = max_val
    
    if v.y <= min_val:
        v.y = min_val  
    if v.y >= max_val:
        v.y = max_val
    
    if v.z <= min_val:
        v.z = min_val  
    if v.z >= max_val:
        v.z = max_val
        
    return v

@ti.kernel
def simulation():
    ClearGrid()
    #P2G()
    P2G_1()
    P2G_2()
    gridUpdate()
    G2P()
    # print("simulation last", particles[0, 0].pos)

window = ti.ui.Window("Taichi Cloth Simulation on GGUI", (1024, 1024), vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

# Lame parameters for stress-strain relationship
elastic_mu = 0.1
elastic_lambda = 100.0
# fake liquid setting
# elastic_mu = 0.1
# elastic_lambda = 100.0

#gravity = vec3(0.0, -9.8, 0.0)
gravity = vec3(0.0, -0.3, 0.0)

dt = 0.2
current_t = 0.0 

n = 2
num_triangles = (n - 1) * (n - 1) * 2
indices = ti.field(int, shape=num_triangles * 3)
vertices = ti.Vector.field(3, dtype=float, shape=n * n)
colors = ti.Vector.field(3, dtype=float, shape=n * n)

@ti.kernel
def update_vertices():
    for i, j in ti.ndrange(n, n):
        vertices[i * n + j] = vec3(i*64, j*64, 0)

@ti.kernel
def initialize_mesh_indices():
    for i, j in ti.ndrange(n - 1, n - 1):
        quad_id = (i * (n - 1)) + j
        # 1st triangle of the square
        indices[quad_id * 6 + 0] = i * n + j
        indices[quad_id * 6 + 1] = (i + 1) * n + j
        indices[quad_id * 6 + 2] = i * n + (j + 1)
        # 2nd triangle of the square
        indices[quad_id * 6 + 3] = (i + 1) * n + j + 1
        indices[quad_id * 6 + 4] = i * n + (j + 1)
        indices[quad_id * 6 + 5] = (i + 1) * n + j

    for i, j in ti.ndrange(n, n):
        if (i // 4 + j // 4) % 2 == 0:
            colors[i * n + j] = (0.22, 0.72, 0.52)
        else:
            colors[i * n + j] = (1, 0.334, 0.52)

while window.running:
    
    if current_t < dt:
        # Reset
        init_simulation()
        update_vertices()
        initialize_mesh_indices()

    simulation()
       
    current_t += dt

    camera.position(32.0, 32.0, 64*2)
    camera.lookat(32.0, 32.0, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(10, 48, 48), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.mesh(vertices,
               indices=indices,
               per_vertex_color=colors,
               two_sided=True)

   
    scene.particles(particles.pos  , radius=0.2, color=(0.5, 0.42, 0.8))
    canvas.scene(scene)
    window.show()
    
    