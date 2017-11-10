import sys, os, re, traceback
from os.path import isfile
from counter import Counter
from ops.fliph import FlipH
from ops.flipv import FlipV
from skimage.io import imread, imsave

OPERATIONS = [Rotate, FlipH, FlipV, Translate, Noise, Zoom, Blur]

AUGMENTED_FILE_REGEX = re.compile('^.*(__.+)+\\.[^\\.]+$')

thread_pool = None
counter = None

def process(dir, file, op_lists):
    thread_pool.apply_async(work, (dir, file, op_lists))

if __name__ == '__main__':    ## Here Here only this part is needed for you
    if len(sys.argv) < 3:
        print '입력 값 갯수 오류'
        sys.exit(1)

    image_dir = sys.argv[1]
    if not os.path.isdir(image_dir):
        print '위치 정보 오류'
        sys.exit(2)

    op_codes = sys.argv[2:]
