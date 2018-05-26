#!/usr/bin/env python3

"""SmoothLife -- Conway's GoL in continuous space

Re-written in Python using the speedups of Numpy

Todo:
    Better integration methods
    Fancy UI
    Refactor. OO design always fails because of nested variable access

Explanations and code (in other language) here:
https://0fps.net/2012/11/19/conways-game-of-life-for-curved-surfaces-part-1/
https://arxiv.org/pdf/1111.1567.pdf
https://jsfiddle.net/mikola/aj2vq/
https://www.youtube.com/watch?v=KJe9H6qS82I
"""

import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

# # Necessary for writing video
# from skvideo.io import FFmpegWriter
# from matplotlib import cm


class Rules:
    # Birth range
    B1 = 0.278
    B2 = 0.365
    # Survival range
    D1 = 0.267
    D2 = 0.445
    # Sigmoid widths
    N = 0.028
    M = 0.147

    # B1 = 0.257
    # B2 = 0.336
    # D1 = 0.365
    # D2 = 0.549
    # N = 0.028
    # M = 0.147

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)  # Set variables from constructor

    @staticmethod
    def sigma(x, a, alpha):
        """Logistic function on x

        Transition around a with steepness alpha
        """
        return 1.0 / (1.0 + np.exp(-4.0 / alpha * (x - a)))

    def sigma2(self, x, a, b):
        """Logistic function on x between a and b"""
        return self.sigma(x, a, self.N) * (1.0 - self.sigma(x, b, self.N))

    @staticmethod
    def lerp(a, b, t):
        """Linear intererpolate t:[0,1] from a to b"""
        return (1.0 - t) * a + t * b

    def s(self, n, m):
        """State transition function"""
        alive = self.sigma(m, 0.5, self.M)
        return self.sigma2(n, self.lerp(self.B1, self.D1, alive), self.lerp(self.B2, self.D2, alive))


def logistic2d(size, radius, roll=True, logres=None):
    """Create a circle with blurred edges

    Set roll=False to have the circle centered in the middle of the
    matrix. Default is to center at the extremes (best for convolution).

    The transition width of the blur scales with the size of the grid.
    I'm not actually sure of the math behind it, but it's what was presented
    in the code from:
    https://0fps.net/2012/11/19/conways-game-of-life-for-curved-surfaces-part-1/
    """
    y, x = size
    # Get coordinate values of each point
    yy, xx = np.mgrid[:y, :x]
    # Distance between each point and the center
    radiuses = np.sqrt((xx - x/2)**2 + (yy - y/2)**2)
    # Scale factor for the transition width
    if logres is None:
        logres = math.log(min(*size), 2)
    with np.errstate(over="ignore"):
        # With big radiuses, the exp overflows,
        # but 1 / (1 + inf) == 0, so it's fine
        logistic = 1 / (1 + np.exp(logres * (radiuses - radius)))
    if roll:
        logistic = np.roll(logistic, y//2, axis=0)
        logistic = np.roll(logistic, x//2, axis=1)
    return logistic


class Multipliers:
    """Kernel convulution for neighbor integral"""

    INNER_RADIUS = 7.0
    OUTER_RADIUS = INNER_RADIUS * 3.0

    def __init__(self, size, inner_radius=INNER_RADIUS, outer_radius=OUTER_RADIUS):
        inner = logistic2d(size, inner_radius)
        outer = logistic2d(size, outer_radius)
        annulus = outer - inner

        # Scale each kernel so the sum is 1
        inner /= np.sum(inner)
        annulus /= np.sum(annulus)

        # Precompute the FFT's
        self.M = np.fft.fft2(inner)
        self.N = np.fft.fft2(annulus)


class SmoothLife:
    def __init__(self, height, width):
        self.width = width
        self.height = height

        self.multipliers = Multipliers((height, width))
        self.rules = Rules()

        self.clear()
        # self.esses = [None] * 3
        # self.esses_count = 0

    def clear(self):
        """Zero out the field"""
        self.field = np.zeros((self.height, self.width))
        # self.esses_count = 0

    def step(self):
        """Do timestep and return field"""

        # To sum up neighbors, do kernel convolutions
        # by multiplying in the frequency domain
        # and converting back to spacial domain
        field_ = np.fft.fft2(self.field)
        M_buffer_ = field_ * self.multipliers.M
        N_buffer_ = field_ * self.multipliers.N
        M_buffer = np.real(np.fft.ifft2(M_buffer_))
        N_buffer = np.real(np.fft.ifft2(N_buffer_))

        # Apply transition rules
        s = self.rules.s(N_buffer, M_buffer)
        nextfield = s

        # Trying some things with smooth time stepping....
        # Not yet working well....
        # s0 = s - M_buffer
        # s1, s2, s3 = self.esses

        # if self.esses_count == 0:
        #     delta = s0
        # elif self.esses_count == 1:
        #     delta = (3 * s0 - s1) / 2
        # elif self.esses_count == 2:
        #     delta = (23 * s0 - 16 * s1 + 5 * s2) / 12
        # else:  # self.esses_count == 3:
        #     delta = (55 * s0 - 59 * s1 + 37 * s2 - 9 * s3) / 24

        # self.esses = [s0] + self.esses[:-1]
        # if self.esses_count < 3:
        #     self.esses_count += 1
        # dt = 0.1
        # nextfield = self.field + dt * delta

        # mode = 0  # timestep mode (0 for discrete)
        # dt = 0.9  # timestep
        # # Apply timestep
        # nextfield = self._step(mode, self.field, s, M_buffer, dt)

        self.field = np.clip(nextfield, 0, 1)
        return self.field

    def _step(self, mode, f, s, m, dt):
        """State transition options

        SmoothLifeAll/SmoothLifeSDL/shaders/snm2D.frag
        """
        if mode == 0:  # Discrete time step
            return s

        # Or use a solution to the differential equation
        elif mode == 1:
            return f + dt*(2*s - 1)
        elif mode == 2:
            return f + dt*(s - f)
        elif mode == 3:
            return m + dt*(2*s - 1)
        elif mode == 4:
            return m + dt*(s - m)

    def add_speckles(self, count=None, intensity=1):
        """Populate field with random living squares

        If count unspecified, do a moderately dense fill

        I suggest using a smaller count when using continuous time
        updating instead of discrete because continuous tends to converge.
        """
        if count is None:
            # count = 200 worked well for a 128x128 grid and INNER_RADIUS 7
            # scale according to area and INNER_RADIUS
            count = 200 * (self.width * self.height) / (128 * 128)
            count *= (7.0 / self.multipliers.INNER_RADIUS) ** 2
            count = int(count)
        for i in range(count):
            radius = int(self.multipliers.INNER_RADIUS)
            r = np.random.randint(0, self.height - radius)
            c = np.random.randint(0, self.width - radius)
            self.field[r:r+radius, c:c+radius] = intensity
        # self.esses_count = 0



def show_animation():
    w = 1 << 9
    h = 1 << 9
    # w = 1920
    # h = 1080
    sl = SmoothLife(h, w)
    sl.add_speckles()
    sl.step()

    fig = plt.figure()
    # Nice color maps: viridis, plasma, gray, binary, seismic, gnuplot
    im = plt.imshow(sl.field, animated=True,
                    cmap=plt.get_cmap("viridis"), aspect="equal")

    def animate(*args):
        im.set_array(sl.step())
        return (im, )

    ani = animation.FuncAnimation(fig, animate, interval=60, blit=True)
    plt.show()


def save_animation():
    w = 1 << 8
    h = 1 << 8
    # w = 1920
    # h = 1080
    sl = SmoothLife(h, w)
    sl.add_speckles()

    # Matplotlib shoves a horrible border on animation saves.
    # We'll do it manually. Ugh

    from skvideo.io import FFmpegWriter
    from matplotlib import cm

    fps = 10
    frames = 100
    w = FFmpegWriter("smoothlife.mp4", inputdict={"-r": str(fps)})
    for i in range(frames):
        frame = cm.viridis(sl.field)
        frame *= 255
        frame = frame.astype("uint8")
        w.writeFrame(frame)
        sl.step()
    w.close()

    # Also, webm output isn't working for me,
    # so I have to manually convert. Ugh
    # ffmpeg -i smoothlife.mp4 -c:v libvpx -b:v 2M smoothlife.webm


if __name__ == '__main__':
    show_animation()
    # save_animation()
