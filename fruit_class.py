import cv2 
from ultralytics import YOLO
import pandas as pd 

def capture_image():
    #get image from external USB camera from cv2? 
    camera_id = 1 #first external camera 

    #open the camera 
    cap = cv2.VideoCapture(camera_id)

    #set the camera resolution, 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    ret, frame = cap.read()
    if not ret:
        print("Error: No image captured")
        return None
    
    img_path = "fruit_image.jpg"
    #save the image 
    cv2.imwrite(img_path, frame)
    return img_path, frame 

def fruit_detect(frame, image_path):
    #find the object
    model = YOLO('yolo11n.pt')
    #if fruit is detect search on the dataset
    inferences = model(image_path) 

    for i in inferences: 
        bounding_boxes = i.boxes 
        if bounding_boxes is not None: 
            for bounding_box in bounding_boxes:
                #get bounding box cooordinates 
                x1, y1, x2, y2 = map(int, bounding_box.xyxy[0].cpu.numpy()) 
                confidence = bounding_box.conf[0].cpu.numpy()
                class_id = int(bounding_box.cls[0].cpu().numpy())
                class_name = model.names[class_id]

                #draw the bounding rectangle for fruit  
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else: 
            print("No object detected")
    save_path = "fruit.jpg" 
    cv2.imwrite(save_path, frame)
    return frame, class_name 
                
def find_fruit(class_name, csv_path):
    #clean up data 
    df = pd.read_csv(csv_path)
    df_clean = df.drop(['date','atlantaretail','chicagoretail','losangelesretail','newyorkretail','averagespread'], axis = 1)
    df_clean = df.to_csv("ProductPriceIndex.csv", index=False)

    match = df_clean[df_clean['productname'].str.lower() == class_name.lower()]
    if not match.empty:
        print("Farm price:", match['farmprice'].values[0])
        return match['farmprice'].values[0]
    else: 
        print("The price item does not exist.")


def main():
    frame, image_path = capture_image()
    if frame is None:
        return

    class_name = fruit_detect(frame, image_path)
    if class_name:
        find_fruit(class_name, "original_dataset.csv")

if __name__ == "__main__":
    main()







