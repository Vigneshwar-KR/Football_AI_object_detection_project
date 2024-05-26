
import torch
print(torch.__version__)
# print(torch.cuda.is_available())  # Should return True if GPU is available
# print(torch.cuda.current_device())  # Should return the index of the current GPU device
# print(torch.cuda.get_device_name(torch.cuda.current_device()))  # Should return the name of your GPU


from ultralytics import YOLO 

model = YOLO('models/best.pt')

results = model.predict('input/08fd33_5.mp4',save=True)
print(results[0])
print('=====================================')
for box in results[0].boxes:
    print(box)