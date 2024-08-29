from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import tensorflow as tf
import numpy as np

class ObjectDetectionApp(App):
    def build(self):
        self.camera = cv2.VideoCapture(0)
        self.model = tf.saved_model.load('./ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model')
        self.class_names = self.load_class_names('./mscoco_complete_label_map.pbtxt')

        layout = BoxLayout()
        self.image = Image()
        layout.add_widget(self.image)
        Clock.schedule_interval(self.update, 1.0 / 30.0)  # 30 fps
        return layout

    def load_class_names(self, path):
        class_names = {}
        with open(path, 'r') as f:
            current_id = None
            for line in f:
                if "id:" in line:
                    current_id = int(line.strip().split(' ')[-1])
                if "display_name:" in line:
                    display_name = line.strip().split('"')[1]
                    class_names[current_id] = display_name
        return class_names

    def update(self, dt):
        ret, frame = self.camera.read()
        if ret:
            input_tensor = tf.convert_to_tensor([frame], dtype=tf.uint8)
            detections = self.model(input_tensor)

            boxes = detections['detection_boxes'].numpy()
            classes = detections['detection_classes'].numpy().astype(np.int32)
            scores = detections['detection_scores'].numpy()

            self.draw_boxes(frame, boxes, classes, scores)
            buf = cv2.flip(frame, 0).tostring()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = texture

    def draw_boxes(self, frame, boxes, classes, scores, min_score_thresh=.5):
        for i in range(boxes.shape[1]):
            if scores[0, i] > min_score_thresh:
                box = boxes[0, i] * np.array([frame.shape[0], frame.shape[1], frame.shape[0], frame.shape[1]])
                class_id = classes[0, i]
                class_name = self.class_names.get(class_id, 'N/A')
                cv2.rectangle(frame, (int(box[1]), int(box[0])), (int(box[3]), int(box[2])), (0, 255, 0), 2)
                cv2.putText(frame, class_name, (int(box[1]), int(box[0] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

if __name__ == '__main__':
    ObjectDetectionApp().run()
