import os

img_root = '/media/hansansui/han-cv/deep_sort_paddle/test'
for dir in os.listdir(img_root):
    img_path = '{}/{}/img1'.format(img_root, dir)
    length = len(os.listdir(img_path))
    with open('{}/{}.txt'.format(img_root, dir), 'w') as f:
        for i in range(1, length+1):
            i = str('%06d'%i)
            f.write('{}/{}.jpg\n'.format(img_path, i))


