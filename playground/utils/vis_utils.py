import numpy as np
import meshplot as mp
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import plotly.graph_objects as go





def create_e_process(points):
    e_process = []
    for i in range(points.shape[0]-1):
        e_process.append([i,i+1])
    e_process = np.array(e_process)
    return e_process

def create_e_dir(points):
    e = []
    for i in range(points.shape[0]):
        e.append([i,i+points.shape[0]])
    e = np.array(e)
    return e

def create_e_to_o(points):
    e_origin = []
    for i in range(points.shape[0]):
        e_origin.append([points.shape[0],i])
    e_origin = np.array(e_origin)
    return e_origin

def create_e_camera1(points):
    e = []
    for i in range(points.shape[0]):
        e.append([i,i + points.shape[0]])
        e.append([i,i + 2*points.shape[0]])
        e.append([i,i + 3*points.shape[0]])
        e.append([i,i + 4*points.shape[0]])
    e = np.array(e)
    return e

def create_e_camera2(points):
    e_camera = []
    for i in range(points.shape[0]):
        e_camera.append([i + points.shape[0],i + 2*points.shape[0]])
        e_camera.append([i + 2*points.shape[0],i + 3*points.shape[0]])
        e_camera.append([i + 3*points.shape[0],i + 4*points.shape[0]])
        e_camera.append([i + 4*points.shape[0],i + points.shape[0]])
    e_camera = np.array(e_camera)
    return e_camera

def create_camera(pos, pos2, ups, rs, coef, reduce_scale):
    c1 = pos2 + ups * coef + rs * coef
    c2 = pos2 + ups * coef - rs * coef
    c3 = pos2 - ups * coef - rs * coef
    c4 = pos2 - ups * coef + rs * coef

    pos_reduce = pos[::reduce_scale]
    c1 = c1[::reduce_scale]
    c2 = c2[::reduce_scale]
    c3 = c3[::reduce_scale]
    c4 = c4[::reduce_scale]
    
    v_all = np.vstack([pos_reduce,c1,c2,c3,c4])
    return v_all, pos_reduce

def vis_result_matrix(record_p,record_r,ref_R,ref_T,coef = 0.01, reduce_scale = 1):
    # preparation
    record_p = record_p[::reduce_scale]
    record_r = record_r[::reduce_scale]
    dir_vec = mi.Point3f(0,0,1)
    up_vec = mi.Point3f(0,1,0)
    r_vec = mi.Point3f(1,0,0)

    directions = [ri @ dir_vec for ri in record_r]
    ups = [ri @ up_vec for ri in record_r]
    rs = [ri @ r_vec for ri in record_r]

    directions = np.array(directions).reshape(len(record_p),3)
    ups = np.array(ups).reshape(len(record_p),3)
    rs = np.array(rs).reshape(len(record_p),3)
    pos = np.array(record_p)
    ref_T = np.array(ref_T).reshape(1,3)
    
    e_process = create_e_process(pos)
    
    start = pos[0].reshape(1,3)
    end = pos[pos.shape[0]-1].reshape(1,3)
    # # plot position
    # p = mp.plot(pos[1:pos.shape[0]-1],shading={"point_size": 0.03})
    # p.add_points(start,shading={"point_color": "red","point_size": 0.1})
    # p.add_points(end,shading={"point_color": "blue","point_size": 0.1})
    # p.add_points(ref_T,shading={"point_color": "black","point_size": 0.1})
    # p.add_edges(pos, e_process, shading={"line_color": "red"})

    origin = np.array([0,0,0])
    e_origin = create_e_to_o(directions)

    start_dir = directions[0].reshape(1,3)
    end_dir = directions[directions.shape[0]-1].reshape(1,3)
    dir_ref = ref_R @ dir_vec
    dir_ref = np.array(dir_ref).reshape(1,3)

    # coor = np.array([[0,0,0],[0,0,1],[0,1,0],[1,0,0]])
    # coor_edge = np.array([[0,1],[0,2],[0,3]])
    # # plot directions
    # p = mp.plot(directions[1:directions.shape[0]-1],shading={"point_color": "green","point_size": 0.03})
    # p.add_points(origin,shading={"point_color": "orange","point_size": 0.03})
    # p.add_points(start_dir,shading={"point_color": "green","point_size": 0.1})
    # p.add_points(end_dir,shading={"point_color": "blue","point_size": 0.1})
    # p.add_points(dir_ref,shading={"point_color": "black","point_size": 0.1})
    # p.add_edges(directions, e_process, shading={"line_color": "green"})
    # p.add_edges(np.vstack((directions,origin)), e_origin, shading={"line_color": "orange"})
    # p.add_edges(coor, coor_edge,shading={"line_color": "blue"})
    # p.add_points(coor,shading={"point_color":"blue"})

    pos2 = pos + directions * coef * 2
    pos2_ref = ref_T + dir_ref * coef * 2
    
    e_dir = create_e_dir(pos)
    e_dir_ref = create_e_dir(ref_T)

    # plot all
    p = mp.plot(pos[1:pos.shape[0]-1],shading={"point_size": 0.03})
    p.add_points(start,shading={"point_color": "red","point_size": 0.1})
    p.add_points(end,shading={"point_color": "blue","point_size": 0.1})
    p.add_points(ref_T,shading={"point_color": "black","point_size": 0.05})
    p.add_edges(pos, e_process, shading={"line_color": "red"})
    p.add_edges(np.vstack([pos,pos2]), e_dir, shading={"line_color": "purple"})
    p.add_points(pos2,shading={"point_color": "red","point_size": 0.05})
    p.add_points(pos2_ref,shading={"point_color": "black","point_size": 0.1})
    p.add_edges(np.vstack([ref_T,pos2_ref]),e_dir_ref,shading={"point_color": "black"})
    return


