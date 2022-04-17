import os
from turtle import color, width
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person, QualityForRecognition

# This key will serve all examples in this document.
KEY = "466d12125fad4a25afbc1de73eb0ca16"

# This endpoint will be used in all examples in this quickstart.
ENDPOINT = "https://demo1422.cognitiveservices.azure.com/"


# Create an authenticated FaceClient.
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))


# Detect a face in an image that contains a single face
single_face_image_url = 'https://d2ubrtwy6ww54e.cloudfront.net/www.uvmhealth.org/assets/2020-11/uvmhn-staying-healthy-coronavirus-man-wearing-mask.jpg'
single_image_name = os.path.basename(single_face_image_url)
# We use detection model 3 to get better performance.
detected_faces = face_client.face.detect_with_url(url=single_face_image_url, detection_model='detection_01' , return_face_attributes={"glasses"})
detected_masks = face_client.face.detect_with_url(url=single_face_image_url, detection_model='detection_03' , return_face_attributes={"mask"}) 

if not detected_faces:
    raise Exception('No face detected from image {}'.format(single_image_name))

# Convert width height to a point in a rectangle
def getRectangle(faceDictionary):
    rect = faceDictionary.face_rectangle
    left = rect.left
    top = rect.top
    right = left + rect.width
    bottom = top + rect.height
    
    return ((left, top), (right, bottom))

def drawFaceRectangles() :
# Download the image from the url
    response = requests.get(single_face_image_url)
    img = Image.open(BytesIO(response.content))

# For each face returned use the face rectangle and draw a red box.
    print('Drawing rectangle around face... see popup for results.')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('Roboto-Bold.ttf', size=25)
    for face in detected_faces:
        attributes =face.face_attributes        
        print(attributes.glasses[:])
        draw.rectangle(getRectangle(face), outline='red',width=10) 
        draw.text((10,10),text = attributes.glasses[:],font=font)       
    for mask in detected_masks:
        mask_att = mask.face_attributes
        text = "Mask Type: "+str(mask_att.mask.type)+"\n Mouth and nose covered:"+str(mask_att.mask.nose_and_mouth_covered)
        draw.text((10,30),text = text,font=font) 
        if mask_att.mask.type == "MaskType.no_mask":
            print("no mask")
        else:
            print("mask Found")

        
        # print()
        
# Display the image in the default image browser.
    img.show()

# Uncomment this to show the face rectangles.
drawFaceRectangles()