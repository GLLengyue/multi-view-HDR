import cv2
import numpy as np


im0 = cv2.imread('im0.png',0)
im1 = cv2.imread('im1.png',0)

ims=np.array([im0, im1])
orig_rows, orig_cols = im0.shape

MARGIN_SIZE = 2 #thus window_size = 5 = 2*margin_size+1, has 25 points
delta_s = 2.6
delta_r = 14.0

def I(p):
    p
    return ims[int(p[0]),int(p[1]),int(p[2])]

def I_prime(p):
    ret = np.log(I(p))
    return ret

def w(t, p):
    ret = np.exp(-np.square(t-p).sum()/(2*np.square(delta_s))-np.square(I_prime(t)-I_prime(p))/(2*np.square(delta_r))) 
    return ret

def I_tilde(q, p):
    s = np.square(1+2*MARGIN_SIZE)
    tmp1 = np.zeros(s)
    tmp2 = np.zeros(s)
    for i in range(-MARGIN_SIZE, MARGIN_SIZE+1):
        for j in range(-MARGIN_SIZE, MARGIN_SIZE+1):
            n = i*5 + j
            t = p+[0,i,j]
            tmp1[n] = w(t, p)
            tmp2[n] = I_prime(t)
        
    ret = I_prime(q)-np.dot(tmp1, tmp2)/np.sum(tmp1)
    return ret

def NCC(p, fp):
    s1 = 0
    s2 = 0
    s3 = 0
    # for q in W(p):
    for i in range(-MARGIN_SIZE, MARGIN_SIZE+1):
        for j in range(-MARGIN_SIZE, MARGIN_SIZE+1):
            q = p+[0,i,j]
            w_l = w(q, p)
            w_r = w(q+fp, p+fp)
            I_l = I_tilde(q, p)
            I_r = I_tilde(q+fp, p+fp)
            s1 += w_l*w_r*I_l*I_l*I_r
            s2 += np.square(w_l*I_l)
            s3 += np.square(w_r*I_r)
    ret = s1/(np.sqrt(s2)*np.sqrt(s3))
    return ret

def E_d(f):
    tmp = 0
    for i in range(2, orig_rows-2):
        for j in range(2, orig_cols-2):
            fp = f[i][j]
            p = np.array([0, i, j])
            try:
                tmp += 1-NCC(p, fp)
            except IndexError:
                print(p)
                print(fp)
    return tmp


f = np.ones([orig_rows, orig_cols, 3])

print(E_d(f))

