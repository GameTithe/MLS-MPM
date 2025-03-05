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

@ti.dataclass
class Particle:
    pos: vec3
    vel: vec3
    mass: ti.f32
    volume_0: ti.f32
    C: mat2 #0으로 초기화 필요
    F: mat2 #대각 1로 초기화 필요
    
    def print(self):
        print(f"pos: {self.pos}")
        print(f"vel: {self.vel}")
        print(f"mass: {self.mass}")
        print(f"volume_0: {self.volume_0}")
        print(f"C: {self.C}")
        print(f"F: {self.F}")
    
@ti.dataclass
class Cell:
    vel: vec2
    mass: ti.f32
    
gridSize = 64
grid_v = ti.field(dtype=vec2, shape=(gridSize, gridSize)) # vx, vy
grid_m = ti.field(dtype=ti.f32, shape=(gridSize, gridSize)) # mass
# particles = ti.field(dtype=Particle, shape=(4096))

num  = 64
particle_count = num * num
particles = Particle.field(shape=(particle_count, ))
#grid = Cell.field(shape = (grid_res * grid_res))
visual_pos = ti.Vector.field(3, dtype=ti.f32, shape= particle_count)


debug = ti.field(dtype=vec3, shape=(1,))

@ti.func
def reset_grid():
    grid_v.fill(0)
    grid_m.fill(0)
    # for i, j in grid:
    #     grid[i, j] = vec3([0, 0, 0])

@ti.kernel
def init_simulation():
    reset_grid()
    
    # init
    # for i, j in particles:
    #     particles[i, j].pos = vec3([gridSize/2.0 + i*0.5 - 16.0, gridSize/2.0 + j*0.5 - 16.0, 1]) 
    #     particles[i, j].vel = vec3([0, 0, 0])
    #     particles[i, j].mass = 1.0
    #     particles[i, j].volume_0 = 0.0
    #     particles[i, j].C = ti.Matrix.zero(dt=ti.f32, n=2, m=2)
    #     particles[i, j].F = ti.Matrix.identity(dt=ti.f32, n=2)
    #     #print("[", i , ",", j, "]", particles[i,j].pos)

    for i in particles: 
        
        x = i % num
        y = i // num
        particles[i].pos = vec3([gridSize/2.0 + x*0.5 - 16.0, gridSize/2.0 + y* 0.5 - 16.0, 1]) 
        particles[i].vel = vec3([0, 0, 0])
        particles[i].mass = 1.0
        particles[i].volume_0 = 0.0
        particles[i].C = ti.Matrix.zero(dt=ti.f32, n=2, m=2)
        particles[i].F = ti.Matrix.identity(dt=ti.f32, n=2)
         
    
    P2G()
    
    # set volume_0
    # for i, j in particles:
    #     particle = particles[i, j]
    #     cell_idx = ivec3([int(particle.pos.x), int(particle.pos.y), int(particle.pos.z)])
    #     cell_diff = (particle.pos - cell_idx) - 0.5
    #     weights = [0.5 * pow(0.5 - cell_diff, 2), 0.75 - pow(cell_diff, 2), 0.5 * pow(0.5 + cell_diff, 2)]

    #     density = 0.0
    #     for gy in ti.static(range(-1, 2)):
    #         for gx in ti.static(range(-1, 2)):
    #             weight = weights[gx + 1][0] * weights[gy + 1][1]
    #             cell_x = ivec3([cell_idx.x + gx, cell_idx.y + gy, cell_idx.z])
    #             if 0 <= cell_x.x < gridSize and 0 <= cell_x.y < gridSize:
    #                 density += grid_m[cell_x.x, cell_x.y] * weight

    #     volume = particle.mass / density
    #     particle.volume_0 = volume
    #     particles[i, j].volume_0 = particle.volume_0
    
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
                if 0 <= cell_x.x < gridSize and 0 <= cell_x.y < gridSize:
                    density += grid_m[cell_x.x, cell_x.y] * weight

        volume = particle.mass / density
        particle.volume_0 = volume
        particles[i].volume_0 = particle.volume_0
    
        
        
@ti.func
def P2G():
    # 다른 코드에서도 datarace 무시하고 짜는거 같으니 일단 무시하고 짜보자
    
    for i in particles:
        particle = particles[i]
        F = particle.F
        
        # deformation gradient
        J = ti.math.determinant(F)
        volume = particle.volume_0 * J
        
        # Neo-Hookean model
        F_T = ti.Matrix.transpose(F)
        F_inv_T = ti.math.inverse(F_T)
        F_minus_F_inv_T = F - F_inv_T

        P_term_0 = elastic_mu * F_minus_F_inv_T
        P_term_1 = elastic_lambda * ti.math.log(J) * F_inv_T
        P = P_term_0 + P_term_1
        
        # cauchy stress
        stress = (1.0 / J) * P @ F_T
        # stress = (1.0 / J) * ti.math.dot(P, F_T)

        eq_16_term_0 = -volume * 4 * stress * dt

        cell_idx = ivec3([int(particle.pos.x), int(particle.pos.y), int(particle.pos.z)])
        cell_diff = (particle.pos - cell_idx) - 0.5
        
        weights = [0.5 * pow(0.5 - cell_diff, 2), 0.75 - pow(cell_diff, 2), 0.5 * pow(0.5 + cell_diff, 2)]

        for gy in ti.static(range(-1, 2)):
            for gx in ti.static(range(-1, 2)):
                weight = weights[gx + 1][0] * weights[gy + 1][1]
                cell_x = ivec3([cell_idx.x + gx, cell_idx.y + gy, cell_idx.z])

                # 구역 벗어나면 particles 값이 고장남
                # 디버깅 항상 키자
                if 0 <= cell_x.x < gridSize and 0 <= cell_x.y < gridSize:
                    cell_dist = (cell_x - particle.pos) + 0.5
                    # cell_dist_xy = vec2([cell_dist.x, cell_dist.y])
                    
                    Q = particle.C @ cell_dist.xy

                    weighted_mass = weight * particle.mass
                    grid_m[cell_x.x, cell_x.y] += weighted_mass

                    grid_v[cell_x.x, cell_x.y] += weighted_mass * (particle.vel.xy + Q)                
                    momentum = (eq_16_term_0 * weight) @ cell_dist.xy
                    grid_v[cell_x.x, cell_x.y] += momentum


    # for i, j in particles:
    #     particle = particles[i, j]
    #     F = particle.F
        
    #     # deformation gradient
    #     J = ti.math.determinant(F)
    #     volume = particle.volume_0 * J
        
    #     # Neo-Hookean model
    #     F_T = ti.Matrix.transpose(F)
    #     F_inv_T = ti.math.inverse(F_T)
    #     F_minus_F_inv_T = F - F_inv_T

    #     P_term_0 = elastic_mu * F_minus_F_inv_T
    #     P_term_1 = elastic_lambda * ti.math.log(J) * F_inv_T
    #     P = P_term_0 + P_term_1
        
    #     # cauchy stress
    #     stress = (1.0 / J) * P @ F_T
    #     # stress = (1.0 / J) * ti.math.dot(P, F_T)

    #     eq_16_term_0 = -volume * 4 * stress * dt

    #     cell_idx = ivec3([int(particle.pos.x), int(particle.pos.y), int(particle.pos.z)])
    #     cell_diff = (particle.pos - cell_idx) - 0.5
        
    #     weights = [0.5 * pow(0.5 - cell_diff, 2), 0.75 - pow(cell_diff, 2), 0.5 * pow(0.5 + cell_diff, 2)]

    #     for gy in ti.static(range(-1, 2)):
    #         for gx in ti.static(range(-1, 2)):
    #             weight = weights[gx + 1][0] * weights[gy + 1][1]
    #             cell_x = ivec3([cell_idx.x + gx, cell_idx.y + gy, cell_idx.z])

    #             # 구역 벗어나면 particles 값이 고장남
    #             # 디버깅 항상 키자
    #             if 0 <= cell_x.x < gridSize and 0 <= cell_x.y < gridSize:
    #                 cell_dist = (cell_x - particle.pos) + 0.5
    #                 # cell_dist_xy = vec2([cell_dist.x, cell_dist.y])
                    
    #                 Q = particle.C @ cell_dist.xy

    #                 weighted_mass = weight * particle.mass
    #                 grid_m[cell_x.x, cell_x.y] += weighted_mass

    #                 grid_v[cell_x.x, cell_x.y] += weighted_mass * (particle.vel.xy + Q)                
    #                 momentum = (eq_16_term_0 * weight) @ cell_dist.xy
    #                 grid_v[cell_x.x, cell_x.y] += momentum

@ti.func
def gridUpdate():
    for x, y in grid_v:
        if (grid_m[x, y] > 0.0):
            grid_v[x, y] /= grid_m[x, y]
            grid_v[x, y] += gravity.xy * dt
            
            if x < 2 or x >= gridSize - 2:
                grid_v[x, y].x = 0.0
            if y < 2 or y >= gridSize - 2:
                grid_v[x, y].y = 0.0

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


                if 0 <= cell_x.x < gridSize and 0 <= cell_x.y < gridSize:
                    cell_dist = (cell_x - particle.pos) + 0.5
                    weighted_velocity = grid_v[cell_x.x, cell_x.y] * weight

                    term = weighted_velocity.outer_product(cell_dist.xy)
                        
                    B += term
                    
                    particle.vel += vec3(weighted_velocity.x, weighted_velocity.y, 0.0)
        
        particle.C = B * 4
        particle.pos += particle.vel * dt
        particle.pos = clip_vec3(particle.pos, 1, gridSize - 2)
        Fp_new = ti.Matrix.identity(dt=ti.f32, n=2)
        Fp_new += particle.C * dt
        Fp_new = Fp_new @ particle.F
         
        particle.F = Fp_new
        particles[i] = particle
        particles[i].pos = particle.pos

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
    reset_grid()
    P2G()
    gridUpdate()
    G2P()
    # print("simulation last", particles[0, 0].pos)

window = ti.ui.Window("Taichi Cloth Simulation on GGUI", (1024, 1024), vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

# Lame parameters for stress-strain relationship
elastic_mu = 20.0
elastic_lambda = 10.0
# # fake liquid setting
# elastic_mu = 0.1
# elastic_lambda = 100.0


# gravity = vec3(0.0, -9.8, 0.0)
gravity = vec3(0.0, -0.3, 0.0)

dt = 0.1
current_t = 0.0


@ti.kernel
def visualization():
    for i in particles:
        x = i % num
        y = i // num
        visual_pos[y * num + x] = particles[i].pos

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
    
    # print("pos:", particles[0, 0].pos, "vel:", particles[0, 0].vel)
    # print(particles[0, 0].print())
    
    # print("step end\n")
    # break
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

    visualization()
    scene.particles(visual_pos, radius=0.2, color=(0.5, 0.42, 0.8))
    canvas.scene(scene)
    window.show()
    
    