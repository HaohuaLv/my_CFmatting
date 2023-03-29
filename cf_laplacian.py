import numpy as np
from scipy.signal import convolve2d
from scipy.sparse import csr_matrix, linalg



def _cf_laplacian(image: np.ndarray, trimap: np.ndarray, win_size: int = 3, epsilon: float = 1):

    def __Lk_func(image, row, col):
        patch = image[row:row+win_size, col:col+win_size].reshape(-1, 3)
        Gk = np.zeros((win_size*(win_size+1), 4))
        Gk[0:win_size**2, 0:3] = patch
        Gk[0:win_size**2, 3] = 1
        Gk[win_size**2:, 0:3] = np.diag(np.full(win_size, epsilon**0.5))

        temp = np.linalg.inv(np.matmul(Gk.transpose(), Gk))
        temp = np.matmul(Gk, temp)
        temp = np.matmul(temp, Gk.transpose())

        Gk_bar = temp -np.eye(win_size*(win_size+1))
        Lk = np.matmul(Gk_bar.transpose(), Gk_bar)
        return Lk[:win_size**2, :win_size**2]

    def __k(i: int, r: int, c: int, w:int):
        r_win = i // win_size
        c_win = i % win_size
        ki = (r + r_win) * w + c + c_win
        return ki

    h, w = image.shape[:2]
    L = np.zeros((h, w, win_size+2, win_size+2))
    b = np.zeros((h, w))
    
    # (9, 3, 3)
    shift_kernel = np.fliplr(np.eye(win_size**2)).reshape(win_size**2, win_size, win_size)

    is_unknown = np.bitwise_and(trimap >= 0.1, trimap <= 0.9)
    is_known = ~is_unknown
    is_unknown_flatten = is_unknown.flatten()
    kernel_dilate = np.ones((win_size, win_size), dtype=int)
    contain_unknown = convolve2d(is_unknown.astype(int), kernel_dilate, mode='valid').astype(np.bool8)
    
    for row in range(h-win_size+1):
        for col in range(w-win_size+1):
            if contain_unknown[row, col]:
                lk = __Lk_func(image, row, col) # (9, 9)

                lk = lk.reshape(win_size**2, win_size, win_size)        # (9, 9) -> (9, 3, 3)
                lk_conv = np.zeros((win_size**2, win_size+2, win_size+2))
                for channel in range(win_size**2):
                    lk_conv[channel] = convolve2d(lk[channel], shift_kernel[channel], mode='full')
                lk_conv = lk_conv.reshape(win_size, win_size, win_size+2, win_size+2)
                L[row:row+win_size, col:col+win_size] += lk_conv

                b[row:row+win_size, col:col+win_size] -= np.sum((lk.reshape(win_size, win_size, -1) * is_unknown[row:row+win_size, col:col+win_size, np.newaxis])
                                                                * (trimap[row:row+win_size, col:col+win_size]*is_known[row:row+win_size, col:col+win_size]).reshape(-1), 
                                                                axis=-1)


    index_i = np.arange(h*w).reshape(-1, 1).repeat((win_size+2)**2, axis=-1).reshape(h, w, win_size+2, win_size+2)
    index_kernel = np.arange(start=-(win_size+1)//2, stop=(win_size+1)//2+1).reshape(1, -1).repeat(win_size+2, axis=0) + np.arange(start=-(win_size+1)//2, stop=(win_size+1)//2+1).reshape(-1, 1).repeat(win_size+2, axis=1) * w
    index_j = index_i + index_kernel
    


    shift_kernel_2 = np.fliplr(np.eye((win_size+2)**2)).reshape((win_size+2)**2, win_size+2, win_size+2)
    j_is_unknown = np.zeros((h, w, (win_size+2)**2))
    for channel in range((win_size+2)**2):
        j_is_unknown[:, :, channel] = convolve2d(is_unknown, shift_kernel_2[channel], mode='same')
    j_is_unknown = j_is_unknown.reshape(h, w, win_size+2, win_size+2).astype(np.bool8)
    j_is_unknown[is_known] = False

    L11_values = L[j_is_unknown].reshape(-1)
    L11_idxi = index_i[j_is_unknown].reshape(-1)
    L11_idxj = index_j[j_is_unknown].reshape(-1)
    


    unknown_indexes = np.where(is_unknown_flatten == True)[0]
    idx_map = dict(zip(unknown_indexes, range(len(unknown_indexes))))

    rows_idx = [idx_map[t] for t in L11_idxi]
    cols_idx = [idx_map[t] for t in L11_idxj]
    # values = list(L.values())

    L11 = csr_matrix((L11_values, (rows_idx, cols_idx)), shape=(len(unknown_indexes), len(unknown_indexes)))
    b1 = b[is_unknown]

    solu = linalg.spsolve(L11, b1).clip(0, 1)
    alpha = trimap.copy()
    alpha[is_unknown] = solu

    return alpha
                    


            








def cf_laplacian(image: np.ndarray, trimap: np.ndarray, **kwargs):
    return _cf_laplacian(image, trimap, **kwargs)


if __name__ == '__main__':
    import time
    start = time.time()
    import cv2
    image = cv2.imread('./lemur.png')
    trimap = cv2.imread('./lemur_trimap.png', cv2.IMREAD_GRAYSCALE) / 255
    alpha = cf_laplacian(image, trimap)
    end = time.time()
    print(end-start)
    print(alpha.shape)
    cv2.imshow('', alpha)
    cv2.waitKey()
    cv2.destroyAllWindows()
