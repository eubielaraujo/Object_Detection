from ultralytics import YOLO
import cv2 as cv

def val_model():
    if __name__ == "__main__":
         # Carrega o modelo treinado
        '''Função para validar o modelo treinado'''
            
        model = YOLO("runs/detect/train/weights/best.pt")

        # Valida o modelo
        metrics = model.val(data="D:/Dev/Object_Detection/datasets/data.yaml", batch=16, imgsz=640)
        print(metrics)

def predict_model():
    '''Função para realizar inferência em uma imagem usando o modelo treinado'''

    model = YOLO("runs/detect/train/weights/best.pt")

    # Inferência em uma imagem
    results = model.predict("D:/Dev/Object_Detection/datasets/images.jpeg")
    for r in results:

        print(r.boxes.conf)

        # Exibir a imagem com as detecções
        annotated_frame = r.plot()
        cv.imshow("Detections", annotated_frame)
        cv.waitKey(0)
    cv.destroyAllWindows()

predict_model()
