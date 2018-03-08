class MyClass:
    words = []

    def __init__(self, data=["baz"]): # think of this as a Constructor in Java
        self.words = data

    def count(self): # think of self as "this" in Java
        return len(self.words)
