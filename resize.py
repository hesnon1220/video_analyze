import cv2

pic_path = r"F:\work\2023-05-16_152344.png"


image = cv2.imread(pic_path)
w,h,_ = image.shape
print(w,h)
re_image = cv2.resize( image , (int(w*4),int(h*4)) ,interpolation=cv2.INTER_LANCZOS4 )
out_image = cv2.fastNlMeansDenoisingColored(re_image,None,10,10,7,21)

cv2.imwrite( r"F:\work\out5.png",out_image)