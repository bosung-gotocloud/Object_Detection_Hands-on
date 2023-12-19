import boto3
import io
import decimal
import sys
import os
from PIL import Image, ImageDraw


def findFace(srcName, targetName, similar):
    left = 0
    top = 0

    client = boto3.client('rekognition')

    srcimage = Image.open(open(srcName,'rb'))
    targetimage = Image.open(open(targetName,'rb'))

    srcstream = io.BytesIO()
    srcimage.save(srcstream,format=srcimage.format)
    srcimage_binary = srcstream.getvalue()

    targetstream = io.BytesIO()
    targetimage.save(targetstream,format=targetimage.format)
    targetimage_binary = targetstream.getvalue()

    srcwidth, srcheight = srcimage.size
    targetwidth, targetheight = targetimage.size

    response = client.compare_faces(SourceImage={'Bytes':srcimage_binary},
                                    TargetImage={'Bytes':targetimage_binary},
                                    SimilarityThreshold=similar)

    srcdraw = ImageDraw.Draw(srcimage)

    srcimageface = response['SourceImageFace']
    
    box = srcimageface['BoundingBox']
    left = srcwidth * box['Left']
    top = srcheight * box['Top']
    srcdraw.rectangle([left,top, left + (srcwidth * box['Width']), 
                    top +(srcheight * box['Height'])], outline='green')
    srcdraw.rectangle([left+1,top+1, left + (srcwidth * box['Width']-1), 
                    top +(srcheight * box['Height']-1)], outline='green')
#    srcimage.show()

    ar = float(srcwidth) / srcheight
    shrinkw = float(0.1*targetwidth)
    shrinkh = float(shrinkw)/ar
    srcimage = srcimage.resize((int(shrinkw),int(shrinkh)),Image.ANTIALIAS)

    targetdraw = ImageDraw.Draw(targetimage)

    facefound = False
    for label in response['FaceMatches']:
        facefound = True
        similarity_ratio = decimal.Decimal(label['Similarity'])

        facedetail = label['Face']
        foundbox = facedetail['BoundingBox']
        foundleft = targetwidth * foundbox['Left']
        foundtop = targetheight * foundbox['Top']
        targetdraw.text((foundleft,foundtop-10), 'Similarity :'+str(round(similarity_ratio,2)),fill='red')
        targetdraw.rectangle([foundleft,foundtop, foundleft + (targetwidth * foundbox['Width']), 
                              foundtop +(targetheight * foundbox['Height'])], outline='red')
        targetdraw.rectangle([foundleft+1,foundtop+1, foundleft + (targetwidth * foundbox['Width']-1), 
                              foundtop +(targetheight * foundbox['Height'])-1], outline='red')
        confidence = decimal.Decimal(facedetail['Confidence'])

    for unmatched in response['UnmatchedFaces']:
        box = unmatched['BoundingBox']
        left = targetwidth * box['Left']
        top = targetheight * box['Top']
        targetdraw.text((left,top-10), 'Unmatched',fill='green')
        targetdraw.rectangle([left,top, left + (targetwidth * box['Width']), 
                              top +(targetheight * box['Height'])], outline='green')
        targetdraw.rectangle([left+1,top+1, left + (targetwidth * box['Width']-1), 
                              top +(targetheight * box['Height'])-1], outline='green')

    if (facefound) :
        name, ext = os.path.splitext(targetName)
        srcn, ext = os.path.splitext(srcName)
        outputfileName = srcn+'_found_in_'+name+ext
        targetimage.paste(srcimage,(0,0))
        targetimage.save(outputfileName, 'JPEG')
        targetimage.show()

    return facefound

if __name__ == "__main__":
    argv_len = len(sys.argv)

    if argv_len > 3:
        sourcefileName = sys.argv[1]
        targetfileName = sys.argv[2]
        similarity = float(sys.argv[3])
        ret = findFace(sourcefileName, targetfileName, similarity)
        if ret:
            print ('Found matches')
        else :
            print ('No match found')
    else:
        print('Usage: find-face-in-picture.py [Face filename] [Picture filename] [Similarity(%)]')
        sys.exit(1)
