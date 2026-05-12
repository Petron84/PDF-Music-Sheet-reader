import cv2
import os


def inspectandaddfromaside():
    one = 1
    dir = 'media\\aside'
    files = os.listdir(dir)
    for file in files:
        print(file)
        img = cv2.imread(f'media\\aside\\{file}')
        #get current time in seconds
        randomtime = int(os.path.getctime(f'media\\aside\\{file}'))
        newfilename = 'added_t'+ str(randomtime) + '_' + file
        with open('actionlog.txt', 'a') as f:
           f.write(f"{newfilename},0\n")
           
        cv2.imwrite(f'media\\actionmodeldataset_v2\\{newfilename}', img)


        
        
inspectandaddfromaside()