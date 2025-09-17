from ultralytics import YOLO

if __name__ == "__main__":
        
    # Load a pretrained YOLO model
    model = YOLO("yolov3-tinyu.pt")

    model.train(
        data="D:/Dev/Object_Detection/datasets/data.yaml",       
        epochs=30,             
        imgsz=640,             
        batch=2,                
        device=0,
        pretrained=True,
        half=True,
        freeze=15
    )
