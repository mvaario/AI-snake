from pynput import keyboard

class KEYBOARD:

    def __init__(self):
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()
        self.left = False
        self.right = False
        self.up = False
        self.down = False

    def on_press(self, key):
        if key == keyboard.Key.up:
            self.up = True
            self.left = False
            self.right = False
            self.down = False
        if key == keyboard.Key.down:
            self.down = True
            self.left = False
            self.right = False
            self.up = False
        if key == keyboard.Key.left:
            self.left = True
            self.right = False
            self.up = False
            self.down = False
        if key == keyboard.Key.right:
            self.right = True
            self.left = False
            self.up = False
            self.down = False

    def on_release(self, key):
        if key == keyboard.Key.up:
            self.up = False
        if key == keyboard.Key.down:
            self.down = False
        if key == keyboard.Key.left:
            self.left = False
        if key == keyboard.Key.right:
            self.right = False

    def get_keys(self):
        action = None
        if self.up:
            action = 1
        if self.down:
            action = 0
        if self.left:
            action = 3
        if self.right:
            action = 2

        return action
