#!/usr/bin/env python3

"""SmoothLife -- Conway's GoL in continuous space

Re-written in Python using the speedups of Numpy

Notes on porting:
    Raffler's original paper uses M to represent the inner cell area
    and N to represent the outer cell area.

    `BasicRules` is simplified and has some good comments.
    `ExtensiveRules` is much more extensive but a little harder to follow.

Thank you to tscheepers for:
    Fixing the initialization in add_speckles to be better at scaling.
    Adding the rules and transition function that work with smooth timesteps.

Todo:
    Fancy UI
    Refactor. OO design always fails because of nested variable access

Explanations and code (in other language) here:
https://arxiv.org/pdf/1111.1567.pdf
https://0fps.net/2012/11/19/conways-game-of-life-for-curved-surfaces-part-1/
https://jsfiddle.net/mikola/aj2vq/
https://www.youtube.com/watch?v=KJe9H6qS82I
https://sourceforge.net/projects/smoothlife/
"""

import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

# # Necessary for writing video
# from skvideo.io import FFmpegWriter
# from matplotlib import cm


def logistic_threshold(x, x0, alpha):
    """Logistic function on x around x0 with transition width alpha

    Approximately:
        (x - alpha/2) < x0 : 0
        (x + alpha/2) > x0 : 1

    AKA snm2D.frag:func_smooth
    """
    return 1.0 / (1.0 + np.exp(-4.0 / alpha * (x - x0)))


def hard_threshold(x, x0):
    """x > x0 : 1 ? 0

    AKA snm2D.frag:func_hard
    """
    return np.greater(x, x0)


def linearized_threshold(x, x0, alpha):
    """Threshold x around x0 with a linear transition region alpha

    AKA snm2D.frag:func_linear
    """
    return np.clip((x - x0) / alpha + 0.5, 0, 1)


def logistic_interval(x, a, b, alpha):
    """Logistic function on x between a and b with transition width alpha

    Very approximately:
        x < a     : 0
        a < x < b : 1
        x > b     : 0

    AKA snm2D.frag:sigmoid_ab with sigtype==4
    """
    return logistic_threshold(x, a, alpha) * (1.0 - logistic_threshold(x, b, alpha))


def linearized_interval(x, a, b, alpha):
    """Function a<x<b with linearized threshold regions

    Very approximately:
        x < a     : 0
        a < x < b : 1
        x > b     : 0

    AKA snm2D.frag:sigmoid_ab with sigtype==1
    """
    return linearized_threshold(x, a, alpha) * (1.0 - linearized_threshold(x, b, alpha))


def lerp(a, b, t):
    """Linear intererpolate from a to b with t ranging [0,1]

    AKA: OpenGL mix
    """
    return (1.0 - t) * a + t * b


class BasicRules:
    # Birth range
    B1 = 0.278
    B2 = 0.365

    # Survival range
    D1 = 0.267
    D2 = 0.445

    # Sigmoid widths
    N = 0.028
    M = 0.147

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise ValueError("Unexpected attribute %s" % k)

    def s(self, n, m):
        """State transition function

        This corresponds to SmoothLifeSDL with:
            sigmode: 4
            sigtype: 4
            mixtype: 4
        """
        # Convert the local cell average `m` to a metric of how alive the local cell is.
        # We transition around 0.5 (0 is fully dead and 1 is fully alive).
        # The transition width is set by `self.M`
        aliveness = logistic_threshold(m, 0.5, self.M)
        # A fully dead cell will become alive if the neighbor density is between B1 and B2.
        # A fully alive cell will stay alive if the neighhbor density is between D1 and D2.
        # Interpolate between the two sets of thresholds depending on how alive/dead the cell is.
        threshold1 = lerp(self.B1, self.D1, aliveness)
        threshold2 = lerp(self.B2, self.D2, aliveness)
        # Now with the smoothness of `logistic_interval` determine if the neighbor density is
        # inside of the threshold to stay/become alive.
        new_aliveness = logistic_interval(n, threshold1, threshold2, self.N)
        return new_aliveness


class SmoothTimestepRules(BasicRules):
    # Birth range
    B1 = 0.254
    B2 = 0.312

    # Survival range
    D1 = 0.340
    D2 = 0.518

    def s(self, n, m):
        """State transition function

        This corresponds to SmoothLifeSDL with:
            sigmode: 2
            sigtype: 1
            mixtype: 0
        """
        aliveness = hard_threshold(m, 0.5)

        return np.where(
            aliveness,
            linearized_interval(n, self.D1, self.D2, self.N),
            linearized_interval(n, self.B1, self.B2, self.N),
        )


class ExtensiveRules(BasicRules):
    """Rules from Rafler's SmoothLifeSDL snm2D.frag"""

    sigmode = 0
    sigtype = 0
    mixtype = 0

    def sigmoid_ab(self, x, a, b):
        if self.sigtype == 0:
            return hard_threshold(x, a) * (1 - hard_threshold(x, b))
        elif self.sigtype == 1:
            return linearized_interval(x, a, b, self.N)
        elif self.sigtype == 4:
            return logistic_interval(x, a, b, self.N)
        else:
            raise NotImplementedError

    def sigmoid_mix(self, x, y, m):
        if self.mixtype == 0:
            intermediate = hard_threshold(m, 0.5)
        elif self.mixtype == 1:
            intermediate = linearized_threshold(m, 0.5, self.M)
        elif self.mixtype == 4:
            intermediate = logistic_threshold(m, 0.5, self.M)
        else:
            raise NotImplementedError
        return lerp(x, y, intermediate)

    def s(self, n, m):
        if self.sigmode == 1:
            b_thresh = self.sigmoid_ab(n, self.B1, self.B2)
            d_thresh = self.sigmoid_ab(n, self.D1, self.D2)
            return lerp(b_thresh, d_thresh, m)
        elif self.sigmode == 2:
            b_thresh = self.sigmoid_ab(n, self.B1, self.B2)
            d_thresh = self.sigmoid_ab(n, self.D1, self.D2)
            return self.sigmoid_mix(b_thresh, d_thresh, m)
        elif self.sigmode == 3:
            threshold1 = lerp(self.B1, self.D1, m)
            threshold2 = lerp(self.B2, self.D2, m)
            return self.sigmoid_ab(n, threshold1, threshold2)
        elif self.sigmode == 4:
            threshold1 = self.sigmoid_mix(self.B1, self.D1, m)
            threshold2 = self.sigmoid_mix(self.B2, self.D2, m)
            return self.sigmoid_ab(n, threshold1, threshold2)
        else:
            raise NotImplementedError


def antialiased_circle(size, radius, roll=True, logres=None):
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
    radiuses = np.sqrt((xx - x / 2) ** 2 + (yy - y / 2) ** 2)
    # Scale factor for the transition width
    if logres is None:
        logres = math.log(min(*size), 2)
    with np.errstate(over="ignore"):
        # With big radiuses, the exp overflows,
        # but 1 / (1 + inf) == 0, so it's fine
        logistic = 1 / (1 + np.exp(logres * (radiuses - radius)))
    if roll:
        logistic = np.roll(logistic, y // 2, axis=0)
        logistic = np.roll(logistic, x // 2, axis=1)
    return logistic


class Multipliers:
    """Kernel convulution for neighbor integral"""

    INNER_RADIUS = 7.0
    OUTER_RADIUS = INNER_RADIUS * 3.0

    def __init__(self, size, inner_radius=INNER_RADIUS, outer_radius=OUTER_RADIUS):
        inner = antialiased_circle(size, inner_radius)
        outer = antialiased_circle(size, outer_radius)
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

        # BasicRules works best with mode=0, discrete.
        # SmoothTimestepRules works best with other modes.

        # self.rules = BasicRules()
        # self.rules = SmoothTimestepRules()
        # self.rules = ExtensiveRules(
        #     B1=0.254, B2=0.312, D1=0.340, D2=0.518,
        #     sigmode=2,
        #     sigtype=1,
        #     mixtype=0)
        self.rules = ExtensiveRules(
            B1=0.278, B2=0.365, D1=0.267, D2=0.445, sigmode=4, sigtype=4, mixtype=4
        )

        self.mode = 0  # timestep mode (0 for discrete)
        self.dt = 0.1

        self.clear()

    def clear(self):
        """Zero out the field"""
        self.field = np.zeros((self.height, self.width))
        self.esses = [None] * 3
        self.esses_count = 0

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

        # Apply timestep
        nextfield = self._step(self.mode, self.field, s, M_buffer, self.dt)

        self.field = np.clip(nextfield, 0, 1)
        return self.field

    def _step(self, mode, f, s, m, dt):
        """State transition options

        SmoothLifeAll/SmoothLifeSDL/shaders/snm2D.frag

        Plus mode 5, but I don't remember where I got the math for it.
        """
        if mode == 0:  # Discrete time step
            return s

        # Or use a solution to the differential equation
        elif mode == 1:
            return f + dt * (2 * s - 1)
        elif mode == 2:
            return f + dt * (s - f)
        elif mode == 3:
            return m + dt * (2 * s - 1)
        elif mode == 4:
            return m + dt * (s - m)
        elif mode == 5:
            s0 = s - m
            s1, s2, s3 = self.esses
            if self.esses_count == 0:
                delta = s0
            elif self.esses_count == 1:
                delta = (3 * s0 - s1) / 2
            elif self.esses_count == 2:
                delta = (23 * s0 - 16 * s1 + 5 * s2) / 12
            else:  # self.esses_count == 3:
                delta = (55 * s0 - 59 * s1 + 37 * s2 - 9 * s3) / 24
            self.esses = [s0] + self.esses[:-1]
            if self.esses_count < 3:
                self.esses_count += 1
            return f + dt * delta

    def add_speckles(self, count=None, intensity=1):
        """Populate field with random living squares

        If count unspecified, do a moderately dense fill
        """

        if count is None:
            count = int(
                self.width * self.height / ((self.multipliers.OUTER_RADIUS * 2) ** 2)
            )
        for i in range(count):
            radius = int(self.multipliers.OUTER_RADIUS)
            # radius = int(self.multipliers.INNER_RADIUS)
            r = np.random.randint(0, self.height - radius)
            c = np.random.randint(0, self.width - radius)
            self.field[r : r + radius, c : c + radius] = intensity
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
    im = plt.imshow(
        sl.field, animated=True, cmap=plt.get_cmap("viridis"), aspect="equal"
    )

    def animate(*args):
        im.set_array(sl.step())
        return (im,)

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


if __name__ == "__main__":
    show_animation()
    # save_animation()
