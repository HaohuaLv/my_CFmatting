import cv2
import numpy as np
from cf_laplacian import cf_laplacian

# 定义全局变量
drawing = False
brush_size = 100
prev_x, prev_y = None, None
fill = 50

def render(img_matting, trimap, show_brush=True):

    # brush_mask = np.zeros(img_matting.shape[:2])
    # cv2.circle(brush_mask, (prev_x, prev_y), brush_size, (fill,), -1)

    trimap = cv2.cvtColor(trimap, cv2.COLOR_GRAY2BGR)
    preview_img = cv2.addWeighted(img_matting, 1, trimap, 1, 0)

    if show_brush:
        cv2.circle(preview_img, (prev_x, prev_y), brush_size, (fill, fill, fill), -1)

    return preview_img

def draw_circle(event, x, y, flags, param):
    global prev_x, prev_y, drawing, brush_size, fill

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        prev_x, prev_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(trimap, (x, y), brush_size, (fill,), -1)
            cv2.line(trimap, (prev_x, prev_y), (x, y), (fill,), brush_size)
        prev_x, prev_y = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(trimap, (x, y), brush_size, (fill,), -1)

    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            brush_size += 2
        else:
            brush_size -= 2
        if brush_size < 2:
            brush_size = 2



img_org = cv2.imread('lemur.png')
img_matting = img_org.copy()
trimap = np.zeros(img_org.shape[:2], dtype=np.uint8)


cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

while True:
    # 显示图片和预览圆形)
    preview_img = render(img_matting, trimap)
    cv2.imshow('image', preview_img)

    # 处理键盘事件
    k = cv2.waitKey(1) & 0xFF

    if k == 9:
        # 进行GrabCut算法分割
        alpha = cf_laplacian(img_org, trimap/255)
        img_matting = (alpha[:, :, np.newaxis] * img_matting).astype(np.uint8)

        preview_img = render(img_matting, trimap)
        cv2.imshow('image', preview_img)
        cv2.imshow('alpha', alpha)
        cv2.imshow('img_matting', img_matting)
        GC_mask_current = np.ones(img_org.shape[:2], dtype=np.uint8)*4

    elif k == 27:
        break

    elif k == ord('3'):
        fill = 0
    elif k == ord('1'):
        fill = 50
    elif k == ord('2'):
        fill = 255

    elif k == ord('s'):
        alpha = (alpha*255).astype(np.uint8)
        output = cv2.cvtColor(img_org, cv2.COLOR_BGR2BGRA)
        output[:, :, 3] = alpha
        
        cv2.imwrite('image_matting.png', output)
        cv2.imwrite('alpha.png', alpha)
        cv2.imwrite('trimap.png', trimap)
        break
        

# 关闭所有窗口
cv2.destroyAllWindows()

