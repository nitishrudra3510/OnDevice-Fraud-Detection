from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
import random


class DemoUI(App):
    def build(self):
        layout = BoxLayout(orientation='vertical', padding=20, spacing=10)
        self.status = Label(text='Normal User', font_size=24)
        btn_normal = Button(text='Simulate Normal', size_hint=(1, 0.2))
        btn_anomaly = Button(text='Simulate Suspicious', size_hint=(1, 0.2))
        btn_normal.bind(on_press=lambda _: self.update_status(0.9))
        btn_anomaly.bind(on_press=lambda _: self.update_status(0.2))
        layout.add_widget(self.status)
        layout.add_widget(btn_normal)
        layout.add_widget(btn_anomaly)
        return layout

    def update_status(self, score: float):
        if score >= 0.6:
            self.status.text = 'Normal User'
            self.status.color = (0, 1, 0, 1)
        else:
            self.status.text = 'Suspicious Activity Detected → Locking Device'
            self.status.color = (1, 0, 0, 1)


if __name__ == '__main__':
    DemoUI().run()


