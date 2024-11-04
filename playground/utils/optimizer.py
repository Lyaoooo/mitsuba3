import mitsuba as mi
import drjit as dr
from contextlib import contextmanager
from collections import defaultdict
mi.set_variant('cuda_ad_rgb')

def taylor_A(x,nth=10):
    # Taylor expansion of sin(x)/x
    ans = mi.Float(0)
    denom = mi.Float(1)
    for i in range(nth+1):
        if i>0: denom *= (2*i)*(2*i+1)
        ans = ans+(-1)**i*x**(2*i)/denom
    return ans

def taylor_B(x,nth=10):
    # Taylor expansion of (1-cos(x))/x**2
    ans = mi.Float(0)
    denom = mi.Float(1)
    for i in range(nth+1):
        denom *= (2*i+1)*(2*i+2)
        ans = ans+(-1)**i*x**(2*i)/denom
    return ans

def taylor_C(x,nth=10):
    # Taylor expansion of (x-sin(x))/x**3
    ans = mi.Float(0)
    denom = mi.Float(1)
    for i in range(nth+1):
        denom *= (2*i+2)*(2*i+3)
        ans = ans+(-1)**i*x**(2*i)/denom
    return ans

def wx_to_w(wx):
    return mi.Point3f(wx[1][2],wx[2][0],wx[0][1])

Ex = mi.Matrix3f([[0, 0, 0],
                [0, 0, -1],
                [0, 1, 0]])
Ey = mi.Matrix3f([[0, 0, 1],
                [0, 0, 0],
                [-1, 0, 0]])
Ez = mi.Matrix3f([[0, -1, 0],
                [1, 0, 0],
                [0, 0, 0]])

def w_to_wx(w):
    wx = w.x * Ex + w.y * Ey + w.z * Ez
    return wx

thresh = 0.01
def SO3_to_so3(R,eps=1e-7):
    trace = dr.trace(R)
    # trace = R[0][0] + R[1][1] + R[2][2]
    theta = dr.acos(dr.clamp((trace - 1) / 2, -1 + eps, 1 - eps))[0] % dr.pi # ln(R) will explode if theta==pi
    if theta < thresh:
        A = taylor_A(theta)
    else:
        A = dr.sin(theta) / theta
    lnR = 1/(2*A+1e-8)*(R-dr.transpose(R))
    w0,w1,w2 = lnR[2,1],lnR[0,2],lnR[1,0]
    w = mi.Point3f(w0[0],w1[0],w2[0])
    return w

def so3_to_SO3(wx):
    wx = mi.Matrix3f(wx)
    w = wx_to_w(wx)
    theta = dr.norm(w)
    I = dr.identity(mi.Matrix3f)
    # if theta -> 0
    if theta[0] < thresh:
        A = taylor_A(theta)
        B = taylor_B(theta)
    else:
        A = dr.sin(theta) / theta
        B = (1-dr.cos(theta))/(theta**2)
    R = I+A*wx+B*wx@wx
    return R

def SE3_to_se3(Rt,eps=1e-8):
    R = mi.Matrix3f(Rt)
    t = mi.Transform4f(Rt).translation()
    w = SO3_to_so3(R)
    wx = w_to_wx(w)
    theta = dr.norm(w)
    I = dr.identity(mi.Matrix3f)

    if theta[0] < thresh:
        A = taylor_A(theta)
        B = taylor_B(theta)
    else:
        A = dr.sin(theta) / theta
        B = (1-dr.cos(theta))/(theta**2)
    invV = I-0.5*wx+(1-A/(2*B))/(theta**2+1e-8)*wx@wx
    u = (invV@t)
    wu = mi.Float(w[0][0],w[1][0],w[2][0],u[0][0],u[1][0],u[2][0])
    return wu  

def se3_to_SE3(w,u):
    wx = w_to_wx(w)
    theta = dr.norm(w)
    I = dr.identity(mi.Matrix3f)
    if theta[0] < thresh:
        A = taylor_A(theta)
        B = taylor_B(theta)
        C = taylor_C(theta)
    else:
        A = dr.sin(theta) / theta
        B = (1-dr.cos(theta))/(theta**2)
        C = (theta-dr.sin(theta))/(theta**3)
    R = I+A*wx+B*wx@wx
    V = I+B*wx+C*wx@wx
    t = V @ u
    Rt = mi.Transform4f.translate(t) @ mi.Matrix4f(R)
    return Rt
class R_Adam(mi.ad.Optimizer):
    """
    Implements the Adam optimizer presented in the paper *Adam: A Method for
    Stochastic Optimization* by Kingman and Ba, ICLR 2015.

    When optimizing many variables (e.g. a high resolution texture) with
    momentum enabled, it may be beneficial to restrict state and variable
    updates to the entries that received nonzero gradients in the current
    iteration (``mask_updates=True``).
    In the context of differentiable Monte Carlo simulations, many of those
    variables may not be observed at each iteration, e.g. when a surface is
    not visible from the current camera. Gradients for unobserved variables
    will remain at zero by default.
    If we do not take special care, at each new iteration:

    1. Momentum accumulated at previous iterations (potentially very noisy)
       will keep being applied to the variable.
    2. The optimizer's state will be updated to incorporate ``gradient = 0``,
       even though it is not an actual gradient value but rather lack of one.

    Enabling ``mask_updates`` avoids these two issues. This is similar to
    `PyTorch's SparseAdam optimizer <https://pytorch.org/docs/1.9.0/generated/torch.optim.SparseAdam.html>`_.
    """
    def __init__(self, lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                 mask_updates=False, mode = 0, params: dict=None):
        """
        Parameter ``lr``:
            learning rate

        Parameter ``beta_1``:
            controls the exponential averaging of first order gradient moments

        Parameter ``beta_2``:
            controls the exponential averaging of second order gradient moments

        Parameter ``mask_updates``:
            if enabled, parameters and state variables will only be updated in a
            given iteration if it received nonzero gradients in that iteration

        Parameter ``mode``:
            mode = 0: Original Adam
            mode = 1: Uniform Adam
            if enabled, the optimizer will use the 'UniformAdam' variant of Adam
            [Nicolet et al. 2021], where the update rule uses the *maximum* of
            the second moment estimates at the current step instead of the
            per-element second moments.
            mode = 2: Vector Adam

        Parameter ``params`` (:py:class:`dict`):
            Optional dictionary-like object containing parameters to optimize.
        """
        assert 0 <= beta_1 < 1 and 0 <= beta_2 < 1 \
            and lr > 0 and epsilon > 0

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.mask_updates = mask_updates
        self.mode = mode
        self.t = defaultdict(lambda: 0)
        super().__init__(lr, params)

    def step(self):
        """Take a gradient step"""
        for k, p in self.variables.items():
            self.t[k] += 1
            lr_scale = dr.sqrt(1 - self.beta_2 ** self.t[k]) / (1 - self.beta_1 ** self.t[k])
            lr_scale = dr.opaque(dr.detached_t(mi.Float), lr_scale, shape=1)

            lr_t = self.lr_v[k] * lr_scale
            g_p = dr.grad(p)


            m_tp, v_tp = self.state[k]

            if(type(p) == type(mi.Matrix3f())):
                with dr.suspend_grad():
                    inv_p = dr.inverse(p)

                    g_p_o = inv_p @ g_p
                    g_p_o_proj = (g_p_o - dr.transpose(g_p_o))/2

                    g_p_w = wx_to_w(g_p_o_proj)


                m_t = self.beta_1 * m_tp + (1 - self.beta_1) * g_p_w
                if self.mode == 2:
                    v_t = self.beta_2 * v_tp + (1 - self.beta_2) * dr.sqr(dr.norm(g_p_w))
                else :
                    v_t = self.beta_2 * v_tp + (1 - self.beta_2) * dr.sqr(g_p_w)

                if self.mask_updates:
                    nonzero = dr.neq(g_p_o_proj, 0.)
                    m_t = dr.select(nonzero, m_t, m_tp)
                    v_t = dr.select(nonzero, v_t, v_tp)
                self.state[k] = (m_t, v_t)
                dr.schedule(self.state[k])


                if self.mode == 1:
                    step = lr_t[0] * m_t / (dr.sqrt(dr.max(v_t)) + self.epsilon)
                else:
                    step = lr_t[0] * m_t / ((dr.sqrt(v_t) + self.epsilon))

                if self.mask_updates:
                    step = dr.select(nonzero, step, 0.)

                g_p_o_m = so3_to_SO3(w_to_wx(-step))
                u = dr.detach(p) @ g_p_o_m
                    

            else :
                shape = dr.shape(g_p)
                if shape == 0:
                    continue
                elif shape != dr.shape(self.state[k][0]):
                    # Reset state if data size has changed
                    self.reset(k)
                
                m_t = self.beta_1 * m_tp + (1 - self.beta_1) * g_p
                if self.mode == 2:
                    v_t = self.beta_2 * v_tp + (1 - self.beta_2) * dr.sqr(dr.norm(g_p))
                else:
                    v_t = self.beta_2 * v_tp + (1 - self.beta_2) * dr.sqr(g_p)
                if self.mask_updates:
                    nonzero = dr.neq(g_p, 0.)
                    m_t = dr.select(nonzero, m_t, m_tp)
                    v_t = dr.select(nonzero, v_t, v_tp)
                self.state[k] = (m_t, v_t)
                dr.schedule(self.state[k])

                if self.mode == 1:
                    step = lr_t * m_t / (dr.sqrt(dr.max(v_t)) + self.epsilon)
                else:
                    step = lr_t * m_t / (dr.sqrt(v_t) + self.epsilon)
                if self.mask_updates:
                    step = dr.select(nonzero, step, 0.)
                u = dr.detach(p) - step
            
            
            u = type(p)(u)
            dr.enable_grad(u)
            self.variables[k] = u
            dr.schedule(self.variables[k])

        dr.eval()

    def reset(self, key):
        """Zero-initializes the internal state associated with a parameter"""
        p = self.variables[key]
        shape = dr.shape(p) if p.IsTensor else dr.width(p)
        if (type(p) == type(mi.Matrix3f())):
            self.state[key] = (dr.zeros(mi.Point3f),
                                dr.zeros(mi.Point3f))
        else :
            self.state[key] = (dr.zeros(dr.detached_t(p), shape),
                                dr.zeros(dr.detached_t(p), shape))
        self.t[key] = 0

    def __repr__(self):
        return ('Adam[\n'
                '  variables = %s,\n'
                '  lr = %s,\n'
                '  betas = (%g, %g),\n'
                '  eps = %g\n'
                ']' % (list(self.keys()), dict(self.lr, default=self.lr_default),
                       self.beta_1, self.beta_2, self.epsilon))


class Q_Adam(mi.ad.Optimizer):
    """
    Implements the Adam optimizer presented in the paper *Adam: A Method for
    Stochastic Optimization* by Kingman and Ba, ICLR 2015.

    When optimizing many variables (e.g. a high resolution texture) with
    momentum enabled, it may be beneficial to restrict state and variable
    updates to the entries that received nonzero gradients in the current
    iteration (``mask_updates=True``).
    In the context of differentiable Monte Carlo simulations, many of those
    variables may not be observed at each iteration, e.g. when a surface is
    not visible from the current camera. Gradients for unobserved variables
    will remain at zero by default.
    If we do not take special care, at each new iteration:

    1. Momentum accumulated at previous iterations (potentially very noisy)
       will keep being applied to the variable.
    2. The optimizer's state will be updated to incorporate ``gradient = 0``,
       even though it is not an actual gradient value but rather lack of one.

    Enabling ``mask_updates`` avoids these two issues. This is similar to
    `PyTorch's SparseAdam optimizer <https://pytorch.org/docs/1.9.0/generated/torch.optim.SparseAdam.html>`_.
    """
    def __init__(self, lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                 mask_updates=False, mode = 0, params: dict=None):
        """
        Parameter ``lr``:
            learning rate

        Parameter ``beta_1``:
            controls the exponential averaging of first order gradient moments

        Parameter ``beta_2``:
            controls the exponential averaging of second order gradient moments

        Parameter ``mask_updates``:
            if enabled, parameters and state variables will only be updated in a
            given iteration if it received nonzero gradients in that iteration

        Parameter ``mode``:
            mode = 0: Original Adam
            mode = 1: Uniform Adam
            if enabled, the optimizer will use the 'UniformAdam' variant of Adam
            [Nicolet et al. 2021], where the update rule uses the *maximum* of
            the second moment estimates at the current step instead of the
            per-element second moments.
            mode = 2: Vector Adam

        Parameter ``params`` (:py:class:`dict`):
            Optional dictionary-like object containing parameters to optimize.
        """
        assert 0 <= beta_1 < 1 and 0 <= beta_2 < 1 \
            and lr > 0 and epsilon > 0

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.mask_updates = mask_updates
        self.mode = mode
        self.t = defaultdict(lambda: 0)
        super().__init__(lr, params)

    def step(self):
        """Take a gradient step"""
        for k, p in self.variables.items():
            self.t[k] += 1
            lr_scale = dr.sqrt(1 - self.beta_2 ** self.t[k]) / (1 - self.beta_1 ** self.t[k])
            lr_scale = dr.opaque(dr.detached_t(mi.Float), lr_scale, shape=1)

            lr_t = self.lr_v[k] * lr_scale
            g_p = dr.grad(p)


            m_tp, v_tp = self.state[k]

            if('q' in k):
                with dr.suspend_grad():
                    g_p_proj = g_p - dr.dot(p,g_p) * p

                m_t = self.beta_1 * m_tp + (1 - self.beta_1) * g_p_proj
                if self.mode == 2:
                    v_t = self.beta_2 * v_tp + (1 - self.beta_2) * dr.sqr(dr.norm(g_p_proj))
                else :
                    v_t = self.beta_2 * v_tp + (1 - self.beta_2) * dr.sqr(g_p_proj)

                if self.mask_updates:
                    nonzero = dr.neq(g_p_proj, 0.)
                    m_t = dr.select(nonzero, m_t, m_tp)
                    v_t = dr.select(nonzero, v_t, v_tp)
                self.state[k] = (m_t, v_t)
                dr.schedule(self.state[k])


                if self.mode == 1:
                    step = lr_t[0] * m_t / (dr.sqrt(dr.max(v_t)) + self.epsilon)
                else:
                    step = lr_t[0] * m_t / ((dr.sqrt(v_t) + self.epsilon))

                if self.mask_updates:
                    step = dr.select(nonzero, step, 0.)

                u = dr.detach(p) - step
                u = dr.normalize(u)
                    

            else :
                shape = dr.shape(g_p)
                if shape == 0:
                    continue
                elif shape != dr.shape(self.state[k][0]):
                    # Reset state if data size has changed
                    self.reset(k)
                
                m_t = self.beta_1 * m_tp + (1 - self.beta_1) * g_p
                if self.mode == 2:
                    v_t = self.beta_2 * v_tp + (1 - self.beta_2) * dr.sqr(dr.norm(g_p))
                else:
                    v_t = self.beta_2 * v_tp + (1 - self.beta_2) * dr.sqr(g_p)
                if self.mask_updates:
                    nonzero = dr.neq(g_p, 0.)
                    m_t = dr.select(nonzero, m_t, m_tp)
                    v_t = dr.select(nonzero, v_t, v_tp)
                self.state[k] = (m_t, v_t)
                dr.schedule(self.state[k])

                if self.mode == 1:
                    step = lr_t * m_t / (dr.sqrt(dr.max(v_t)) + self.epsilon)
                else:
                    step = lr_t * m_t / (dr.sqrt(v_t) + self.epsilon)
                if self.mask_updates:
                    step = dr.select(nonzero, step, 0.)
                u = dr.detach(p) - step
            
            
            u = type(p)(u)
            dr.enable_grad(u)
            self.variables[k] = u
            dr.schedule(self.variables[k])

        dr.eval()

    def reset(self, key):
        """Zero-initializes the internal state associated with a parameter"""
        p = self.variables[key]
        shape = dr.shape(p) if p.IsTensor else dr.width(p)

        self.state[key] = (dr.zeros(dr.detached_t(p), shape),
                            dr.zeros(dr.detached_t(p), shape))
        self.t[key] = 0

    def __repr__(self):
        return ('Adam[\n'
                '  variables = %s,\n'
                '  lr = %s,\n'
                '  betas = (%g, %g),\n'
                '  eps = %g\n'
                ']' % (list(self.keys()), dict(self.lr, default=self.lr_default),
                       self.beta_1, self.beta_2, self.epsilon))

