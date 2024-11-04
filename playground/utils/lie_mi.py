import drjit as dr
import mitsuba as mi
import numpy as np

thresh = 0.01

mi.set_variant('llvm_ad_rgb')

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

def taylor_A(x,nth=10):
    # Taylor expansion of sin(x)/x
    ans = 0
    denom = 1.
    for i in range(nth+1):
        if i>0: denom *= (2*i)*(2*i+1)
        ans = ans+(-1)**i*x**(2*i)/denom
    return ans

def taylor_B(x,nth=10):
    # Taylor expansion of (1-cos(x))/x**2
    ans = 0
    denom = 1.
    for i in range(nth+1):
        denom *= (2*i+1)*(2*i+2)
        ans = ans+(-1)**i*x**(2*i)/denom
    return ans

def taylor_C(x,nth=10):
    # Taylor expansion of (x-sin(x))/x**3
    ans = 0
    denom = 1.
    for i in range(nth+1):
        denom *= (2*i+2)*(2*i+3)
        ans = ans+(-1)**i*x**(2*i)/denom
    return ans


def so3_to_SO3(w):
    wx = w_to_wx(w)
    theta = dr.norm(w)
    I = dr.identity(mi.Matrix3f)
    # if theta -> 0
    if theta[0] < thresh :
        A = taylor_A(theta)
        B = taylor_B(theta)
    else:
        A = dr.sin(theta) / theta
        B = (1-dr.cos(theta))/(theta**2)
    R = I+A*wx+B*wx@wx
    return R


def SO3_to_so3(R,eps=1e-7):
    # print(R)
    # trace = dr.trace(R)
    trace = R[0][0] + R[1][1] + R[2][2]
    # print(R)
    theta = dr.acos(dr.clamp((trace - 1) / 2, -1 + eps, 1 - eps))[0] % dr.pi # ln(R) will explode if theta==pi
    if theta < thresh:
        A = taylor_A(theta)
    else:
        A = dr.sin(theta) / theta
    lnR = 1/(2*A+1e-8)*(R-dr.transpose(R))
    w0,w1,w2 = lnR[2,1],lnR[0,2],lnR[1,0]
    w = mi.Point3f(w0[0],w1[0],w2[0])
    # print(R)
    return w


def se3_to_SE3(wu):
    w = wu[:3]
    u = mi.Vector3f(wu[3],wu[4],wu[5])
    wx = w_to_wx(w)
    theta = dr.norm(w)
    I = dr.identity(mi.Matrix3f)
    # A = taylor_A(theta)
    # B = taylor_B(theta)
    # C = taylor_C(theta)
    A = dr.sin(theta) / theta
    B = (1-dr.cos(theta))/(theta**2)
    C = (theta-dr.sin(theta))/(theta**3)
    R = I+A*wx+B*wx@wx
    V = I+B*wx+C*wx@wx
    t = V @ u
    #  用translation
    Rt = mi.Matrix4f(R[0][0],R[1][0],R[2][0],t[0],R[0][1],R[1][1],R[2][1],t[1],R[0][2],R[1][2],R[2][2],t[2],0,0,0,1)
    return Rt



def SE3_to_se3(Rt,eps=1e-8):
    R = mi.Matrix3f(Rt[0][0],Rt[1][0],Rt[2][0],Rt[0][1],Rt[1][1],Rt[2][1],Rt[0][2],Rt[1][2],Rt[2][2])
    t = mi.Vector3f(Rt[3][0],Rt[3][1],Rt[3][2])
    w = SO3_to_so3(R)
    wx = w_to_wx(w)
    theta = dr.norm(w)
    I = dr.identity(mi.Matrix3f)
    # A = taylor_A(theta)
    # B = taylor_B(theta)
    A = dr.sin(theta) / theta
    B = (1-dr.cos(theta))/(theta**2)
    invV = I-0.5*wx+(1-A/(2*B))/(theta**2+1e-8)*wx@wx
    u = (invV@t)
    wu = mi.Float(w[0],w[1],w[2],u[0][0],u[1][0],u[2][0])
    return wu  


def so2_to_SO2(theta):

    R = mi.Matrix2f(np.cos(theta),-np.sin(theta),np.sin(theta),np.cos(theta))
    return R

def SO2_to_so2(R):
    theta = dr.atan2(R[1][0],R[0][0])
    return theta


def se2_to_SE2(utheta):
    u = mi.Vector2f(utheta[0],utheta[1])
    theta = utheta[2]
    G = mi.Matrix2f(0,-1,1,0)    
    theta_x = theta * G
    I = dr.identity(mi.Matrix2f)
    A = taylor_A(theta)
    B = theta * taylor_B(theta)
    V = mi.Matrix2f(A, -B,B, A)
    R = so2_to_SO2(theta)
    t = V @ u
    Rt = mi.Matrix3f(R[0][0],R[1][0],t[0],R[0][1],R[1][1],t[1],0,0,1)
    return Rt


def SE2_to_se2(Rt):

    R = mi.Matrix2f(Rt[0][0],Rt[0][1],Rt[1][0],Rt[1][1])
    t = mi.Vector2f(Rt[2][0],Rt[2][1])
    theta = SO2_to_so2(R)
    A = taylor_A(theta)
    B = theta * taylor_B(theta)
    invV = mi.Matrix2f(A, B, -B, A)/(A**2 + B**2)
    u = invV@t
    utheta = mi.Float(u[0][0],u[1][0],theta[0])
    return utheta  