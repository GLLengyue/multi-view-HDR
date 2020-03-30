import cv2
import numpy as np

MARGIN_SIZE = 2 #thus window_size = 5 = 2*margin_size+1, has 25 points
MAX_D = 64 #max disparity


im0 = cv2.imread('im2.ppm',0)
im1 = cv2.imread('im6.ppm',0)
orig_rows, orig_cols = im0.shape

avg_im0 = cv2.blur(im0, (2*MARGIN_SIZE+1, 2*MARGIN_SIZE+1))
avg_im1 = cv2.blur(im1, (2*MARGIN_SIZE+1, 2*MARGIN_SIZE+1))

ims=np.array([im0, im1])
avg_ims = np.array([avg_im0, avg_im1])

# buffer w value, Space-time trade-off
# buffer_w = np.zeros([2,100,100,100,100])-1
# cur = np.array([0,0,0])

def I(p):
    ret = ims[int(p[0]),int(p[1]),int(p[2])]
    return ret

def I_tilde(q, p):
    ret = int(I(q)) - int(avg_ims[int(p[0]),int(p[1]),int(p[2])])
    return ret

def NCC(p, fp):
    s1 = 0
    s2 = 0
    s3 = 0
    # for q in W(p):
    for i in range(-MARGIN_SIZE, MARGIN_SIZE+1):
        for j in range(-MARGIN_SIZE, MARGIN_SIZE+1):
            q = p+[0,i,j]
            I_l = int(I_tilde(q, p))
            I_r = int(I_tilde(q+fp, p+fp))
            s1 = s1 + I_l*I_r
            s2 += np.square(I_l)
            s3 += np.square(I_r)
    ret = s1/(np.sqrt(s2*s3))
    return ret

def E_d(f):
    sum = 0
    tmp = 0
    for i in range(MARGIN_SIZE, orig_rows-MARGIN_SIZE):
        for j in range(MARGIN_SIZE, orig_cols-MARGIN_SIZE):
            print(i, j)
            fp = f[i][j]
            p = np.array([0, i, j])
            try:
                tmp = 1-NCC(p, fp)
                if (tmp is np.nan):
                    return sum
            except IndexError:
                pass
                tmp = 0
            sum = sum + tmp
    return sum

def disparity():
    f = np.ones([orig_rows, orig_cols, 3])

    for i in range(MARGIN_SIZE, orig_rows-MARGIN_SIZE):
        for j in range(MARGIN_SIZE, orig_cols-MARGIN_SIZE):
            print(i, j)
            p = np.array([0, i, j])
            l_ncc = 0
            fp = np.array([1, 0, 0])
            for m in range(-1, 2):
                for n in range(0, MAX_D):
                    t_fp = np.array([1, m, n])
                    try:
                        t_ncc = NCC(p, t_fp)
                    except IndexError:
                        t_ncc = 0
                    if t_ncc > l_ncc:
                        l_ncc = t_ncc
                        fp = t_fp
            f[i][j] = fp
    return f


raw_d = disparity()
# np.savetxt("raw_d.csv", raw_d, delimiter=',')

d = np.sum(raw_d, axis=2)
np.savetxt("d.csv", d, delimiter=',')

disp = cv2.normalize(d, d, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                          dtype=cv2.CV_8U)
np.savetxt("disp.csv", disp, delimiter=',')

