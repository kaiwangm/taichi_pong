import taichi as ti
import numpy as np
import math

ti.init()

@ti.data_oriented
class PoogGame:
    def __init__(self, n, dt):
        self.n = n
        self.dt = dt
        self.x = ti.Vector.field(2, dtype=ti.f32, shape=n)
        self.v = ti.Vector.field(2, dtype=ti.f32, shape=n)
        self.g = ti.Vector([-0.0, -2.6])
        
        self.n_ret = 36
        self.ret = ti.Vector.field(2, dtype=ti.f32, shape=36)
        self.ret_use = ti.Vector.field(1, dtype=ti.uint8, shape=36)

        self.score = ti.Vector.field(1, dtype=ti.f32, shape=1)

    @ti.kernel
    def initialize_game(self):
        for i in range(self.n):
            self.x[i] = [0.5, 0.3] + self.random_vector(0.2)
            self.v[i] = [0.0, 0.0] + self.random_vector(0.2)
        
        ret_offset = ti.Vector([0.2, 0.5])
        for i in range(self.n_ret -  1):
            self.ret[i] = ti.Vector([ti.floor(i / 5), i % 5]) * 0.1 + ret_offset
        
        self.ret[self.n_ret -  1] = ti.Vector([0.5, 0.2])
        for i in range(self.n_ret):
            self.ret_use[i] = ti.Vector([1])

        self.score[0][0] = 0.0

    @staticmethod
    @ti.func
    def random_vector(radius):
        theta = ti.random() * 2 * math.pi
        r = ti.random() * radius
        return r * ti.Vector([ti.cos(theta), ti.sin(theta)])
    
    @ti.kernel
    def integrate(self):
        for i in range(self.n):

            if self.x[i][0] < 0.0:
                self.x[i][0] = 0.0
                self.v[i][0] = -self.v[i][0] 
                self.v[i] = self.v[i] + self.random_vector(0.2)
                self.v[i] = self.v[i] * 0.9

            if self.x[i][1] < 0.0:
                self.x[i][1] = 0.0
                self.v[i][1] = -self.v[i][1]
                self.v[i] = self.v[i] + self.random_vector(0.2)
                self.v[i] = self.v[i] * 1.0
            
            if self.x[i][0] > 1.0:
                self.x[i][0] = 1.0
                self.v[i][0] = -self.v[i][0]
                self.v[i] = self.v[i] + self.random_vector(0.2)
                self.v[i] = self.v[i] * 0.9

            if self.x[i][1] > 1.0:
                self.x[i][1] = 1.0
                self.v[i][1] = -self.v[i][1]
                self.v[i] = self.v[i] + self.random_vector(0.2)
                self.v[i] = self.v[i] * 0.9

            
            for j in range(self.n_ret):    
                pos = self.ret[j]
                tl = pos + [-0.03, 0.01]
                br = pos + [0.03, -0.01]

                if self.ret_use[j][0] == 1:
                    if self.x[i][0] > tl[0] and self.x[i][0] < br[0] and self.x[i][1] > br[1] and self.x[i][1] < tl[1]:

                        if self.x[i][0] > tl[0] and self.x[i][0] < br[0]:
                            self.v[i][0] = -self.v[i][0]

                        if self.x[i][1] > br[1] and self.x[i][1] < tl[1]:
                            self.v[i][1] = -self.v[i][1]

                        self.v[i] = self.v[i] + self.random_vector(0.2)

                        if j != self.n_ret - 1:
                            self.ret_use[j][0] = 0
                            self.score[0][0] += 1.0
                        else:
                            self.v[i] = self.v[i] * 1.2
            
            self.v[i] += self.dt * self.g
            if self.v[i].norm() > 2.0:
                self.v[i] = self.v[i].normalized() * 2.0

            self.x[i] += self.dt * self.v[i]


    def render(self, gui):
        gui.text("{:0>2d}".format(int(self.score[0][0])), pos = [2.0/8.0, 0.9], font_size = 400, color = 0xEFDAD7)
        gui.circles(self.x.to_numpy(), radius=5, color = 0xFF6363)

        for i in range(self.n_ret):
            pos = self.ret[i].to_numpy()
            tl = pos + np.array([-0.03, 0.01])
            br = pos + np.array([0.03, -0.01])

            if self.ret_use[i][0] == 1:
                #gui.circle(pos, radius=3, color = 0x00ffff)
                gui.rect(tl, br, radius=3, color = 0xFFAD60)

    def process_input(self, gui):
        for e in gui.get_events():
            if e.key == ti.GUI.SPACE:
                self.initialize_game()

            if e.key == ti.GUI.MOVE:
                mouse_x, mouse_y = e.pos
                self.ret[self.n_ret -  1][0] = 0.8 * self.ret[self.n_ret -  1][0] + 0.2 * mouse_x


game = PoogGame(3, 0.0001)
game.initialize_game()

gui = ti.GUI("Poog Game", background_color = 0x1A1A40, res = (800, 800))

while gui.running:
    game.process_input(gui)
    for i in range(100):
        game.integrate()
    game.render(gui)
    gui.show()