import cv2 as cv
import os

dirlist = os.listdir('media\\reallines')
for fileindex in range(len(dirlist)):
    if fileindex < 4750:
        continue 
    dirlistlen = len(dirlist)
    if dirlist[fileindex].endswith('.png'):
        tempimg = cv.imread(f'media\\reallines\\{dirlist[fileindex]}', cv.IMREAD_GRAYSCALE)

        while True:
            cv.imshow(f'sorting {fileindex}/{dirlistlen}', cv.resize(tempimg, (0,0), fx=0.5, fy=0.5))
            key = cv.waitKey(0)
            match key:
                case 84 | 116: # T keys, treble clef
                    cv.imwrite('media\\clefdata\\lines\\treble\\' + str(fileindex) + '.png', tempimg)
                    cv.destroyAllWindows()
                    break
                
                case 66 | 98: # B keys, bass clef
                    cv.imwrite('media\\clefdata\\lines\\bass\\' + str(fileindex) + '.png', tempimg)
                    cv.destroyAllWindows()
                    break
                
                case 68 | 100: # D keys, discard
                    print("discarded")
                    cv.destroyAllWindows()
                    break
                
                case _:
                    print("Invalid key, please press T for treble, B for bass, or D to discard.")
                    cv.destroyAllWindows()
                    continue
        
        print(f"Processed {fileindex} out of {dirlistlen} files.")