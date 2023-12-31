import torch
import math
import numpy as np
import cv2
from .box_utils import jaccard

INF=1e8

#------------------------------------------------------------------------------------------------------------------------------------
#        DATA CONVERSION IN VARIOUS FORMAT (BOX, CARTESIAN POLYGON (poly), POLAR POLYGON (polar), MASK)
#------------------------------------------------------------------------------------------------------------------------------------

def poly2polar(polygons,centers,num_rays=360):
    no_pts = polygons[:,:,0] == -1
    num_polys, _ , _ = polygons.shape
    v1 = polygons-centers[:,None,:] # (num_polys, max_pts, 2)
    # Here, a1 correspond to real angle (between 0 and 360°)
    a1 = ((torch.atan2(v1[:,:,0],v1[:,:,1])* 180/math.pi))%360 # (num_polys, max_pts)
    # We multiply a1 by num_rays/360 and select relative intenger to get its position in the output vector o1
    a1 = (torch.round(a1*num_rays/360) % num_rays) * torch.logical_not(no_pts) + no_pts * num_rays # (num_polys, max_pts)
    d1 = torch.sqrt(v1[:,:,0]**2+v1[:,:,1]**2) * torch.logical_not(no_pts)  # (num_polys, max_pts)

    # Sort distances in order to get for same angle only the maximum distance
    o1 = torch.zeros((num_polys,num_rays+1),dtype=d1.dtype)
    o1.scatter_reduce_(1,a1.long(),d1,reduce='amax')
    return o1[:,:-1]


def polar2poly(points, distances):
    '''Decode distance prediction to polygons
    Args:
        points (Tensor): Shape (num_points, 2), [x, y].
        distances (Tensor): Shape (num_points, num_rays)
    Returns:
        Tensor: polygons in cartesian coordinates.
    '''

    num_points = points.shape[:-1]
    num_rays = distances.shape[-1]
    angles = torch.arange(0, 360, 360//num_rays)/180*math.pi
    points = points[..., None].expand(points.shape + (num_rays,)).swapaxes(-2,-1)
    c_x, c_y = points[..., 0], points[..., 1]

    sin = torch.sin(angles)
    cos = torch.cos(angles)

    sin = sin[None,:].expand(np.prod(num_points),num_rays).reshape(num_points + (num_rays,))
    cos = cos[None,:].expand(np.prod(num_points),num_rays).reshape(num_points + (num_rays,))
    
    x = distances * sin + c_x
    y = distances * cos + c_y
    
    res = torch.stack([x, y], axis=-1)

    return res

def poly2bbox(polygons):
    '''Get min bbox from each polygon
    Args:
        polygons (Tensor): Shape (n, num_points, 2)
    Returns:
        bboxes (Tensor): Shape (n, 4)
    '''
    xmin=torch.min(polygons[...,0],dim=-1)[0]
    xmax=torch.max(polygons[...,0],dim=-1)[0]
    ymin=torch.min(polygons[...,1],dim=-1)[0]
    ymax=torch.max(polygons[...,1],dim=-1)[0]
    return torch.stack([xmin,ymin,xmax,ymax],axis=1)

@torch.jit.script
def sanitize_poly_coordinates(_polygons, w:int, h:int, padding:int=1, cast:bool=True):
    """
    Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0, and x2 <= image_size.
    Also converts from relative to absolute coordinates and casts the results to long tensors.

    If cast is false, the result won't be cast to longs.
    Warning: this does things in-place behind the scenes so copy if necessary.
    """
    a,b,c = _polygons.shape
    no_points = _polygons==-1
    _polygons[:,0,:] = _polygons[:,0,:] * w
    _polygons[:,1,:] = _polygons[:,1,:] * h
    if cast:
        _polygons = _polygons.long()
    _polygons[no_points] = -1
    return _polygons

def is_inside_polygon(points, polygons, valid_mask, angle_max=180, dist_min=0, debug=False):
    """
      Check for all points in valid_mask if they are inside there respective GT polygons or not
      points : (num_points,2)
      polygons : (num_gts, max_pts, 2)
      valid_mask : (num_points, num_gts)
    """
    first_pt_polygon = polygons[:,0:1,:].expand(polygons.shape)
    no_pts = polygons==-1
    completed_polygons = polygons.clone()
    completed_polygons[no_pts] = first_pt_polygon[no_pts]
    inside_polygon_mask = valid_mask.new_zeros(valid_mask.shape) # (num_points, num_gts)
    
    # Premier tri avec les points qui ne sont pas dans la bbox du polygon
    x1 = completed_polygons[:,:,0].min(dim=1)[0]
    x2 = completed_polygons[:,:,0].max(dim=1)[0]
    y1 = completed_polygons[:,:,1].min(dim=1)[0]
    y2 = completed_polygons[:,:,1].max(dim=1)[0]
    
    points = points[:,None,:].expand(valid_mask.shape[0],valid_mask.shape[1],2)
    masks_left  = points[:,:,0] >= x1.view(1, -1)
    masks_right = points[:,:,0] <  x2.view(1, -1)
    masks_up    = points[:,:,1] >= y1.view(1, -1)
    masks_down  = points[:,:,1] <  y2.view(1, -1)
    
    crop_mask = masks_left * masks_right * masks_up * masks_down
    valid_mask = valid_mask * crop_mask

    indices_valid = valid_mask.nonzero() # (num_valid_pts, 2)
    P = points[indices_valid[:,0]][:,0,None,:] # (num_valid_pts, 1, 2) => Coords des points valides
    A = completed_polygons[indices_valid[:,1],:,:] # (num_valid_pts, max_pts, 2) => polygones gt associés aux points valides
    B = torch.roll(A, 1, dims=1) #(num_valid_pts, max_pts, 2)

    angles = ((torch.atan2(B[:,:,0]-P[:,:,0],B[:,:,1]-P[:,:,1])-torch.atan2(A[:,:,0]-P[:,:,0],A[:,:,1]-P[:,:,1]))*180/math.pi)%360
    # Calcul distance entre les points et les sommets du polygone pour éliminer ceux bcp trop proches
    if dist_min>0:
        P = P.expand(P.shape[0],A.shape[1],2)
        D = torch.sqrt((P[:,:,0]-A[:,:,0])**2+(P[:,:,1]-A[:,:,1])**2)
        valid_D = D.min(dim=1)[0] > dist_min
    else:
        valid_D = torch.ones((P.shape[0])).bool()
    inside_poly = torch.logical_and(torch.all(angles < angle_max, dim=1),valid_D)
    inside_polygon_mask[indices_valid[:,0],indices_valid[:,1]] = inside_poly

    return inside_polygon_mask


def poly2mask(masks, polygons, ind_convex_pts=None, min_value=-100):
    """
    "Crop" predicted masks by zeroing out everything not in the box from predicted polygon.
    Here, zeroing is replace by -100 in order to apply a sigmoid after.
    
    Args:
        - masks should be a size [h, w, n] tensor of masks
        - polygons should be a size [n, 2, num_points] tensor of polygon coords in relative point form
    """

    h, w, _ = masks.size()
    polygons = sanitize_poly_coordinates(polygons, w, h, cast=False)
    polygons = polygons.moveaxis(2,1).flip((1))
    if ind_convex_pts is not None:
      ind_convex_pts = ind_convex_pts.flip((1))
   
    X,Y = torch.meshgrid(torch.Tensor(range(w)),torch.Tensor(range(h)))
    points = torch.stack([X.reshape(-1),Y.reshape(-1)],axis=1)

    # Process polygons to symplify them
    if ind_convex_pts is not None: 
      num_pts = ind_convex_pts.sum(dim=1)
      max_pts = num_pts.max()
      new_polygons = torch.ones((polygons.shape[0],max_pts,2)) * -1

      for i in range(polygons.shape[0]):
        new_polygons[i,:num_pts[i],:] = polygons[i, ind_convex_pts[i,:], :]
    else:
      new_polygons = polygons

    valid_mask = torch.ones((h*w,new_polygons.shape[0])).bool()

    ind_points_mask = is_inside_polygon(points, new_polygons, valid_mask) # (h*w,1)

    crop_mask = torch.ones_like(masks).bool()
    
    crop_mask[points[:,1].long(),points[:,0].long(),:] = ind_points_mask

    mask_min = torch.ones_like(masks) * min_value

    neg_crop_mask = torch.logical_not(crop_mask)

    mask_min = mask_min * neg_crop_mask

    return masks * crop_mask.float() + mask_min
   

def polar2mask(centers, distances, img_size):
    """
    centers: (num_points, 2)
    distances: (num_points, num_rays)
    img_size: tuple (h,w)

    masks: (num_points, w, h)
    """
    to_reshape = False

    if centers.ndim!=2:
        to_reshape = True
        orig_shape = centers.shape[:-1]
        centers = centers.reshape(-1,centers.shape[-1])
        distances = distances.reshape(-1,distances.shape[-1])
    else:
        orig_shape = None

    num_points = distances.shape[0]
    num_rays = distances.shape[1]
    
    # Get 360 rays 
    if num_rays<360:
        assert 360%num_rays == 0
        step = 360//num_rays
        distances_split = distances.split(1,dim=1)
        new_distances = [[d,torch.zeros((num_points,step-1)).to(distances.dtype)] for d in distances_split]
        new_distances = torch.cat([inner for outer in new_distances for inner in outer],dim=1)
        distances = new_distances.clone()

    # Fill 0 values of distances (with polar2convex function and all positive values as convex indices)
    distances = polar2convex(distances.clone(), distances>0)
    # Get mask points coordinates
    h,w = img_size
    Y,X = torch.meshgrid(torch.Tensor(range(w))/w,torch.Tensor(range(h))/h)
    coords_mask = torch.stack([X.reshape(-1),Y.reshape(-1)],axis=1)
    num_coords = coords_mask.shape[0]
    coords_mask = coords_mask[:,None,:].expand(num_coords,num_points,2)
    
    # Get angles and distances between coords_mask and angles
    v1 =  coords_mask - centers[None,:].expand(num_coords,num_points,2)
    a1 = ((torch.atan2(v1[:,:,0],v1[:,:,1])* 180/math.pi))%360
    a1 = torch.round(a1)%360
    #a1 = torch.round(a1*num_rays/360) % num_rays
    d1 = torch.sqrt(v1[:,:,0]**2+v1[:,:,1]**2)  
    
    # Compare coord_masks distances with polygon distances depending on their angles with centers
    a1 = a1.flatten()
    indices = torch.Tensor(range(num_points))[None,:].expand(num_coords,num_points).flatten().long()
    coords_max = distances[indices, a1.long()].reshape(num_coords, num_points)
    masks = (d1 <= coords_max).reshape(w,h,num_points) #.moveaxis(2,0)
    if to_reshape:
        masks = masks.reshape((w,h)+orig_shape)
    return masks


#------------------------------------------------------------------------------------------------------------------------------------
#        POLYGON TO CONVEX POLYGON IN POLAR FORMAT
#------------------------------------------------------------------------------------------------------------------------------------

def detect_convex_indices(distances):
    # distances size (n,p)
    n,p = distances.shape
    theta = 2/p*math.pi
    distances = torch.clamp(distances,1e-8)
    A = distances[:,:,None].expand(n,p,p)
    #B = torch.stack([torch.roll(distances, -i, dims=1) for i in range(p)],axis=1)
    indices = ((torch.arange(0,p)[None,:].expand(p,p)+torch.arange(0,p)[:,None].expand(p,p))%p)[None,:,:].expand(n,p,p)
    B = torch.take_along_dim(A,indices,1)
    alpha = theta * torch.minimum(torch.arange(0,p),torch.arange(p,0,-1))[None,None,:].expand(n,p,p)
    beta = math.pi/2 - alpha/2 + torch.atan((B-A)/(B+A) * 1/torch.tan(alpha/2))
    beta_left = beta[:,:,1:p//2]
    beta_right = beta[:,:,1+p//2:]
    beta_max = beta_left.max(dim=2)[0]+beta_right.max(dim=2)[0]
    return beta_max < math.pi

def get_coeff(indices):
    n,p = indices.shape
    y = torch.arange(0,p)[None,:].expand(n,p)
    c = 1+y-torch.cummax(indices*y,dim=1)[0]
    last_col = torch.tile(c[:,-1],(p,1)).T
    first_value_zero = torch.tile(indices[:,0]==0,(p,1)).T
    zero_indices = torch.cummax(indices,dim=1)[0]==0
    c = c + last_col * torch.logical_and(first_value_zero,zero_indices)
    return c

def polar2convex(distances, convex_indices=None):

    if convex_indices is None:
        convex_indices = detect_convex_indices(distances)
    distances *= convex_indices
    n,p = distances.shape
    theta = 2/distances.shape[1]*math.pi

    coeffs = get_coeff(torch.cat([torch.roll(convex_indices,1,dims=1),torch.roll(convex_indices,-1,dims=1).flip(-1)],axis=0))
    AB = torch.take_along_dim(torch.cat([distances,distances.flip(-1)],axis=0),(torch.arange(0,p)[None,:].expand(2*n,p)-coeffs)%p,-1).reshape(-1,p)

    alpha = coeffs[:n]*theta
    beta = coeffs[n:].flip(-1)*theta
    A = AB[:n]
    B = AB[n:].flip(-1)

    gamma = math.pi/2 - (alpha+beta)/2 + torch.atan((B-A)/(torch.clamp((B+A),min=1e-3)*torch.tan((alpha+beta)/2)))

    new_distances = A*torch.sin(gamma)/torch.sin(alpha+gamma)
    # Nouveau tri : les rayons où alpha+beta > 180° sont mis à zéro. Ca concerne les points qui ne sont pas dans le polygone, et où il y a des valeurs calculées qui sont fortement positives ou négatives dûes au calcul.
    big_angle = (alpha+beta) >=  (179/180 * math.pi) # - 2 * theta)  #?!
    new_distances = new_distances.nan_to_num() * torch.logical_not(big_angle) + torch.ones_like(new_distances)*1e-4 * big_angle
    new_distances = new_distances * torch.logical_not(convex_indices) + distances * convex_indices
    return new_distances


#------------------------------------------------------------------------------------------------------------------------------------
#        LABEL ENCODING IN POLAR REPRESENTATION 
#------------------------------------------------------------------------------------------------------------------------------------


def get_min_enclosing_circle(polygons):
    '''
    Get centerpoint and radius of polygons minimum enclosing circles
    We rely on OpenCV function instead of coding a new one for Pytorch, my bad

    Parameters:
        polygons (Tensor): Shape (num_polys, max_pts, 2)

    Return: 
        centerpoints (Tensor): Shape (num_polys, 2)
        radius (Tensor): Shape (num_polys)
    '''
    num_polys = polygons.shape[0]
    data = [cv2.minEnclosingCircle(polygons[i,polygons[i,:,0]>0,:].cpu().numpy()) for i in range(num_polys)]
    radius = torch.Tensor([d[1] for d in data])
    centerpoints = torch.Tensor([list(d[0]) for d in data])
    return centerpoints, radius

def get_mask_sample_region(gt_bb, mask_center, strides, img_size, num_points_per, gt_xys, radius=1):
    w,h = img_size 
    #no gt
    if mask_center[:, 0, 0].sum() == 0:
        return mask_center.new_zeros(mask_center[:, :, 0].shape, dtype=torch.uint8)

    beg = 0

    num_p,num_g,_ = mask_center.shape

    pts_strides = torch.cat([torch.ones((n_p))*strides[level]*radius for level, n_p in enumerate(num_points_per)])
    pts_strides = torch.stack([pts_strides/w,pts_strides/h],axis=-1)[:,None,:].expand(num_p, num_g, 2) # (num_points, num_gts, 2)

    center_gt = torch.cat([- torch.maximum(mask_center-pts_strides,gt_bb[...,0:2]),
                               torch.minimum(mask_center+pts_strides,gt_bb[...,2:4])],axis=-1)

    center_bbox = center_gt + gt_xys * torch.Tensor([1,1,-1,-1])

    inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0  # (num_points, num_gts) 
    # Si toutes les valeurs sont positives sur une ligne, cela signifie que le point est valide

    return inside_gt_bbox_mask


def polar_target(gt_bboxes, gt_polygons, gt_labels, points, regress_ranges, num_rays, img_size, crowd_boxes=None, radius=1.0, force_gt_attribute=False):
    """
    Match les vérités terrains avec les points (encodage de la GT dans le format de la prediction)
    
    Parameters:
        gt_bboxes: Tensor (num_gts, 4) / Ground truth bboxes
        gt_polygons: Tensor (num_gts, max_pts, 2) / Ground truth polygons
        gt_labels: Tensor (num_gts, 1) / Ground truth labels
        points: Tensor (num_points, 2) / Center points
        regress_ranges : Tensor (num_points, 2) / Points regress ranges (min,max)
    
    """

    num_points = points.size(0)
    num_gts = gt_labels.size(0)
    if num_gts == 0:
        return gt_labels.new_zeros(num_points), \
               gt_bboxes.new_zeros((num_points, 4)), \
               gt_polygons.new_zeros((num_points, num_rays)), \
               gt_labels.new_zeros((num_points)), \
               gt_labels.new_zeros((num_points)), \
               gt_labels.new_zeros((num_points, num_rays))
    
    w,h = img_size

    regress_ranges = regress_ranges[:, None, :].expand(
        num_points, num_gts, 2)                                 # (num_points, num_gts, 2)

    gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)  # (num_points, num_gts, 4)
 
    xys = torch.cat([points, points],axis=-1)[:, None, :].expand(num_points, num_gts, 4)
    bbox_targets = (xys - gt_bboxes)*torch.Tensor([h,w,-h,-w])
    
    # Get centerpoints for each GT polygon
    polygon_centers, polygon_mec_radius = get_min_enclosing_circle(gt_polygons*torch.Tensor([h,w])) # (num_gts, 2) & (num_gts)
    polygon_centers = polygon_centers[None,:,:].expand(num_points, num_gts, 2) # (num_points, num_gts, 2)
    polygon_mec_radius = polygon_mec_radius[None,:].expand(num_points, num_gts) # (num_points, num_gts)

    # Get distances between points and polygon_centers
    pt_to_center_gt = torch.sqrt((polygon_centers[:,:,0]-points[:,None,0]*h)**2 + (polygon_centers[:,:,1]-points[:,None,1]*w)**2)   

    # Compute centerness
    cent_gt = (1-pt_to_center_gt/polygon_mec_radius).clamp(0).square()

    #-------------------------------------------------------------------------------------------------------------------------------------------------------------
    # condition1: inside a gt bbox

    # La fonction en dessous cherche à vérifier quels points sont valides pour accueillir chaque gt. On prend chaque pt, on trace un carré
    # de côté 2*stride*radius et on voit si le centre de la gt est dans ce carré. Ca donne normalement pour chaque gt plusieurs propositions de pts

    inside_mec = pt_to_center_gt<=polygon_mec_radius*radius

    # condition2: limit the regression range for each location
    # Ici, pour chaque propal, on compare la bbox de chaque gt avec la capacité max de régression du pt [(-1,64),(64,128),etc.]. 
    # On prend ceux où l'area rentre dans la range
    
    max_regress_distance = bbox_targets.max(-1)[0] # (num_points, num_gts) => Distance maximale entre les limites de la gt (xmin, xmax, ymin, ymax) et le point central

    inside_regress_range = (                       # (num_points, num_gts) => Matrice booleen où pour chaque pt avec sa regress range, on dit si la gt est valide ou non.
        max_regress_distance >= regress_ranges[..., 0]) & (
        max_regress_distance <= regress_ranges[..., 1])


    all_inside = torch.logical_and(inside_mec, inside_regress_range)

    # condition3 : inside a gt polygon
    # On part du principe que les polygones GT sont convexes. On va donc déterminer pour chaque proposition de centre s'il est dans le polygon via un calcul d'angle...
    # On simplifie le calcul en analysant uniquement les points déjà potentiellement valides par les 2 premières conditions...

    #if inside_polygon:
    inside_gt_polygon = is_inside_polygon(points, gt_polygons, all_inside, angle_max= 179, dist_min = 0.005) #-2*(360//num_rays)

    #condition3bis: Si on a un point avec une centerness supérieure à un seuil prédéfini, il est automatiquement valide même s'il n'est pas dans un polygone.
    # Cela permet de gérer les polygones longs et fins où peu voire aucun point n'est à l'intérieur

    near_center = cent_gt>0.9 

    all_inside = torch.logical_and(torch.logical_or(inside_gt_polygon,near_center),all_inside)
    cent_gt = cent_gt * torch.logical_or(inside_gt_polygon,near_center)

    # Filter centerness
    cent_gt = cent_gt * inside_regress_range
    full_cent_gt = cent_gt.clone()
    max_cent_gt, argmax_cent_gt = cent_gt.max(dim=0)

    cent_gt[all_inside.logical_not()] = 0

    # TODO : match les gts avec zero pt valide avec leur pt avec la + grde centerness
    gt_without_pts = all_inside.sum(dim=0)==0

    
    if gt_without_pts.any() and force_gt_attribute:
        cent_gt[argmax_cent_gt,torch.arange(num_gts)][gt_without_pts] = max_cent_gt[gt_without_pts]
        all_inside[argmax_cent_gt,torch.arange(num_gts)][gt_without_pts] = True

    pts_with_gt = torch.any(all_inside,1) # (num_points)

    _, max_cent_pt_inds = cent_gt.max(dim=1) #pt_to_center_gt.min(dim=1) #cent_gt.max(dim=1) # On match le pt valide avec la gt où il y a la + grosse centerness

    labels = torch.ones((num_points)).long()*-1
    labels[pts_with_gt] = gt_labels[max_cent_pt_inds[pts_with_gt]]
 
    ind_pts_with_gt = torch.arange(num_points)[pts_with_gt]
    final_bbox_targets = torch.zeros((num_points, 4))
    final_bbox_targets[pts_with_gt] = bbox_targets[ind_pts_with_gt, max_cent_pt_inds[pts_with_gt]]

    mec_radius_targets = torch.zeros((num_points))
    mec_radius_targets[pts_with_gt] = polygon_mec_radius[ind_pts_with_gt, max_cent_pt_inds[pts_with_gt]]

    pos_inds = (labels!=-1).nonzero().reshape(-1)       # (num_gts) => Indices dans labels où il y a les gts (position en terme de pt)

    labels+=1

    
    # NEW STEP : GERER LES CROWD BOXES
    if crowd_boxes is not None:
        # On considère que le pt est à ignorer si la box max créée a une Intersection sur l'Aire de 0.5 min avec une des annotations à ignorer
        num_crowds = crowd_boxes.shape[0]
        crowd_boxes = crowd_boxes[None,:,:].expand(num_points, num_crowds, 4)
        xys = torch.cat([points, points],axis=-1)[:, None, :].expand(num_points, num_crowds, 4)
        regress_offsets = torch.clamp(regress_ranges[:,0,1],0,1024)[:,None,None].expand(num_points,num_crowds,4)
        regress_offsets = regress_offsets/torch.Tensor([-w,-h,w,h])
        xys = torch.clamp(xys+regress_offsets,0,1)
        crowd_iou = jaccard(xys,crowd_boxes,iscrowd=True)
        crowd_max_overlaps = crowd_iou.max(1)[0][:,0]
        labels[torch.logical_and(labels<=0,crowd_max_overlaps>=0.5)] = -1
    
    # Compute polygon distances in polar representation
    poly_targets = torch.zeros(num_points, num_rays).float()  # (num_points, num_rays)
    ind_ext_targets = torch.zeros(num_points, num_rays).bool()
    pos_poly_ids = max_cent_pt_inds[pos_inds] # (num_gts)
    distances_targets = poly2polar(gt_polygons[pos_poly_ids,:,:], points[pos_inds], num_rays)


    convex_targets = polar2convex(distances_targets, distances_targets>0) #convex_indices)

    ind_targets = max_cent_pt_inds
    if pts_with_gt.any():
      convex_targets = torch.clamp(convex_targets, min=0)
      poly_targets[pos_inds,:] = convex_targets
      ind_ext_targets[pos_inds,:] = distances_targets>0
      
    return labels, final_bbox_targets, poly_targets, ind_targets, full_cent_gt.max(dim=1)[0]


#------------------------------------------------------------------------------------------------------------------------------------
#        FUNCTIONS FOR CHANGING THE NUMBER OF RAYS OF POLYGONS IN POLAR FORMAT
#------------------------------------------------------------------------------------------------------------------------------------

def prime_factors(n):
    assert n>0 and type(n)==int
    factors = []
    if n==1:
        return [1]
    i = 2
    while n!=1:
        if n%i == 0:
            factors.append(i)
            n = n//i
        else:
            i+=1
    return factors

def enlarge_num_rays(distances,R):
    d_shape = distances.shape[:-1]
    num_rays = distances.shape[-1]
    step = R//num_rays
    distances_split = distances.split(1,dim=-1)
    pad = torch.zeros((d_shape+(step-1,)),dtype=distances.dtype)
    new_distances = [[d,pad] for d in distances_split]
    new_distances = torch.cat([inner for outer in new_distances for inner in outer],dim=-1)
    new_distances = polar2convex(new_distances.clone(), new_distances>0)    
    return new_distances

def reduce_num_rays(distances,R):
    num_rays = distances.shape[-1]
    step = num_rays//R
    distances = distances.split(step,dim=-1)
    return torch.stack([torch.min(d,axis=-1)[0] for d in distances],dim=-1)    

def change_num_rays(distances,R):
    num_rays = distances.shape[-1]
    if R == num_rays:
        return distances
    if R > num_rays and R%num_rays==0:
        return enlarge_num_rays(distances,R)
    elif R < num_rays and num_rays%R==0:
        return reduce_num_rays(distances,R)
    else:
        f_R = prime_factors(R)
        f_nr = prime_factors(num_rays)
        factor = np.prod([i for i in f_R if not i in f_nr or f_nr.remove(i)])
        return reduce_num_rays(enlarge_num_rays(distances,factor*num_rays),R)
    
#------------------------------------------------------------------------------------------------------------------------------------
#        FUNCTIONS FOR COMPUTING INTERSECTION OVER UNION OF POLYGONS
#------------------------------------------------------------------------------------------------------------------------------------

def polarArea(distances):
    num_rays = distances.shape[-1]
    theta = 2/num_rays * math.pi   
    A = distances
    B = torch.roll(distances,1,dims=-1)
    C = (A.square()+B.square()-2*A*B*math.cos(theta)).sqrt()
    S = (A+B+C)/2
    T = (S.clamp(0)*(S-A).clamp(0)*(S-B).clamp(0)*(S-C).clamp(0)).sqrt()
    return T.sum(dim=-1)


def limit_distances(points,distances):
    dist_shape = distances.shape[:-1]
    num_rays = distances.shape[-1]
    num_points = math.prod(dist_shape)
    frame_limits = torch.Tensor([[[0,0],[1,0],[1,1],[0,1]]]).expand(num_points,4,2)
    poly_limits = poly2polar(frame_limits,points.reshape(-1,2),num_rays)
    poly_limits = polar2convex(poly_limits,poly_limits>0).reshape(dist_shape+(num_rays,))
    return torch.minimum(distances,poly_limits)

def project_polygon(A,p_A,p_B):
    # Project polygon A (distances) with center p_A to a new center p_B
    num_rays = A.shape[-1]
    theta = 360//num_rays

    num_p_A = p_A.shape[0]

    v_AB = p_B-p_A # (num_p_A,2)
    d_AB = v_AB.square().sum(-1).sqrt()[...,None].expand(num_p_A,num_rays)
    angle_AB = (torch.round((torch.atan2(v_AB[...,0],v_AB[...,1])* 180/math.pi))%360)[...,None].expand(num_p_A,num_rays) # Angle qui indique position de p_B par rapport à p_A

    alpha = ((torch.arange(0,360,theta, dtype=A.dtype)-180-angle_AB)%360 - 180) /180*math.pi
    D = torch.clamp(A.square()+d_AB.square()-2*A*d_AB*torch.cos(torch.abs(alpha)),0).sqrt()
    yota = torch.sign(alpha+0.001) * (math.pi/2 - torch.abs(alpha)/2 + torch.atan(((A-d_AB)/torch.clamp(A+d_AB,1e-3))*1/torch.tan(torch.abs(alpha)/2)))*180/math.pi
    # We want 0° angles to be considered as positive values
    yota = ((180 + angle_AB - torch.round(yota)).long()%360) // theta

    return D, yota.long()


def poly_iou(boxes,polygons,points,R=None,idx=None, box_iou_thresh=0.05):
          
    iou = jaccard(boxes,boxes) # Premier tri : toutes les duos de bbox ayant une intersection

    if idx is not None:
        to_test = torch.zeros_like(iou)
        num_classes, top_k = idx.shape
        to_test[idx[:,None,:].expand(num_classes,top_k,top_k).flatten(),idx[:,:,None].expand(num_classes,top_k,top_k).flatten()]=1
        iou = iou*to_test # Deuxième tri : On teste uniquement les duos de polygones figurant dans le test d'un des classes 

    iou.triu_(diagonal=1) # Troisième tri : matrice triangulaire pour éviter de calculer 2x le même duo (ainsi que les duos d'un même polygone)

    indices = torch.argwhere(iou>box_iou_thresh)

    poly_to_test = indices.unique()
    polygons_bis = polygons[poly_to_test]
    points_bis = points[poly_to_test]

    polygons_bis = limit_distances(points_bis,polygons_bis)

    p_shape = polygons_bis.shape
    polygons_bis = polar2convex(polygons_bis, detect_convex_indices(polygons_bis))

    if R is not None:
        if p_shape[-1] != R:
            polygons_bis = change_num_rays(polygons_bis,R)

    area_polygons = polarArea(polygons_bis)

    polygons = torch.zeros((polygons.shape[0],R),dtype=polygons_bis.dtype)
    polygons[poly_to_test] = polygons_bis
    area_polygons = torch.zeros((polygons.shape[0]))
    area_polygons[poly_to_test] = polarArea(polygons_bis)

    A = polygons[indices[:,0]]
    B = polygons[indices[:,1]]
    p_A = points[indices[:,0]]
    p_B = points[indices[:,1]]
    area_A = area_polygons[indices[:,0]]
    area_B = area_polygons[indices[:,1]]
    
    num_p_A, num_rays = A.shape



    AB = torch.cat([A,B],axis=0)
    p_AB = torch.cat([p_A,p_B],axis=0)
    
    dist_proj, angles_proj = project_polygon(AB,
                                                p_AB,
                                                torch.roll(p_AB,num_p_A,0))


    pts_in = (dist_proj <= torch.take_along_dim(torch.roll(AB,num_p_A,0), angles_proj,-1))


    sum_coords = (polar2poly(p_AB,AB) * pts_in[...,None]).sum(-2) 

    nb_pts_inter = (pts_in[:num_p_A].sum(-1) + pts_in[num_p_A:].sum(-1))[...,None] # (num_p_A)
    
    p_C = (sum_coords[:num_p_A]+sum_coords[num_p_A:]) / torch.clamp(nb_pts_inter,1)

    dist_proj, angles_proj = project_polygon(AB, p_AB, p_C.tile(2,1))

    dist_proj = pts_in * dist_proj + torch.logical_not(pts_in)*10

    C = torch.ones((num_p_A,num_rays),dtype=dist_proj.dtype)*10

    dist_proj = torch.cat([dist_proj[:num_p_A],dist_proj[num_p_A:]],axis=1)
    angles_proj = torch.cat([angles_proj[:num_p_A],angles_proj[num_p_A:]],axis=1)

    C.scatter_reduce_(-1,angles_proj,dist_proj,reduce='amin')

    C *= (nb_pts_inter>2)
    C[C==10] = 0
    C = torch.clamp(polar2convex(C,C>0),0)

    area_C = polarArea(C)
    iou_poly = area_C / (area_A+area_B-area_C)
    iou[indices[:,0],indices[:,1]] = iou_poly

    return iou.clamp(0,1)
